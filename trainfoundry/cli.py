"""Agent-friendly multi-command TrainFoundry CLI."""

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from config import ConfigError, get_path, load_config
from fetch.catalog import SOURCE_GROUPS, SOURCES, get_source
from fetch.fetcher_service import FetcherService
from fetch.models import (
    CommonCrawlAcquisition,
    DatasetMeta,
    HttpAcquisition,
    HuggingFaceAcquisition,
    MinariAcquisition,
)
from fetch.reconcile_registry import reconcile
from fetch.registry import (
    DatasetRegistry,
    RegistryVerificationError,
    verify_record,
)
from fetch.verify_downloads import verify_all
from metadata import (
    MetadataConflictError,
    MetadataNotFoundError,
    MetadataNotInitializedError,
    MetadataValidationError,
)
from metadata.cli import add_metadata_commands
from trainfoundry import __version__
from trainfoundry.manifest import COMMAND_MANIFEST


class CliFailure(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        exit_code: int = 1,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code
        self.retryable = retryable


class AgentArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliFailure("INVALID_ARGUMENT", message, exit_code=2)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    output_format = _requested_output(arguments)
    command = _command_name(arguments)
    try:
        program = "trainfoundry" if argv is not None else Path(sys.argv[0]).name
        args = build_parser(prog=program).parse_args(arguments)
        command = args.command_name
        output_format = args.output
        with (
            contextlib.redirect_stdout(sys.stderr)
            if output_format == "json"
            else contextlib.nullcontext()
        ):
            data = args.handler(args)
        _emit(command, data, output_format)
        return 0
    except CliFailure as exc:
        _emit_error(command, exc, output_format)
        return exc.exit_code
    except ConfigError as exc:
        return _unexpected_error(
            command,
            output_format,
            "CONFIG_ERROR",
            exc,
            exit_code=3,
        )
    except FileNotFoundError as exc:
        return _unexpected_error(
            command,
            output_format,
            "DATA_NOT_FOUND",
            exc,
            exit_code=4,
        )
    except subprocess.CalledProcessError as exc:
        return _unexpected_error(
            command,
            output_format,
            "DOWNLOAD_FAILED",
            exc,
            exit_code=5,
            retryable=True,
        )
    except RegistryVerificationError as exc:
        return _unexpected_error(
            command,
            output_format,
            "VERIFICATION_FAILED",
            exc,
            exit_code=6,
        )
    except MetadataValidationError as exc:
        return _unexpected_error(
            command,
            output_format,
            "INVALID_ARGUMENT",
            exc,
            exit_code=2,
        )
    except MetadataNotInitializedError as exc:
        return _unexpected_error(
            command,
            output_format,
            "METADATA_NOT_INITIALIZED",
            exc,
            exit_code=4,
        )
    except MetadataNotFoundError as exc:
        return _unexpected_error(
            command,
            output_format,
            "DATA_NOT_FOUND",
            exc,
            exit_code=4,
        )
    except MetadataConflictError as exc:
        return _unexpected_error(
            command,
            output_format,
            "METADATA_CONFLICT",
            exc,
            exit_code=8,
        )
    except PermissionError as exc:
        return _unexpected_error(
            command,
            output_format,
            "PERMISSION_DENIED",
            exc,
            exit_code=7,
        )
    except Exception as exc:
        return _unexpected_error(
            command,
            output_format,
            "INTERNAL_ERROR",
            exc,
            exit_code=1,
        )


def build_parser(*, prog: str = "trainfoundry") -> AgentArgumentParser:
    parser = AgentArgumentParser(prog=prog)
    parser.add_argument("--version", action="version", version=__version__)
    root = parser.add_subparsers(dest="root_command", required=True)

    commands = root.add_parser(
        "commands",
        help="show the machine-readable command contract",
    )
    _leaf(commands, "commands", _handle_commands)

    fetch = root.add_parser("fetch", help="plan and acquire datasets")
    fetch_commands = fetch.add_subparsers(dest="fetch_command", required=True)

    fetch_list = fetch_commands.add_parser("list", help="list dataset sources")
    _leaf(fetch_list, "fetch.list", _handle_fetch_list)

    fetch_run = fetch_commands.add_parser("run", help="fetch dataset sources")
    fetch_run.add_argument("source_ids", nargs="*", choices=sorted(SOURCES))
    targets = fetch_run.add_mutually_exclusive_group()
    targets.add_argument("--group", choices=sorted(SOURCE_GROUPS))
    targets.add_argument("--all", action="store_true", dest="fetch_all")
    mode = fetch_run.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--force", action="store_true")
    _leaf(fetch_run, "fetch.run", _handle_fetch_run)

    fetch_reconcile = fetch_commands.add_parser(
        "reconcile",
        help="register existing text downloads",
    )
    _leaf(fetch_reconcile, "fetch.reconcile", _handle_fetch_reconcile)

    dataset = root.add_parser("dataset", help="inspect and verify dataset records")
    dataset_commands = dataset.add_subparsers(
        dest="dataset_command",
        required=True,
    )

    inspect = dataset_commands.add_parser("inspect", help="inspect one record")
    inspect.add_argument("source_id", choices=sorted(SOURCES))
    _leaf(inspect, "dataset.inspect", _handle_dataset_inspect)

    verify = dataset_commands.add_parser("verify", help="verify local files")
    verify.add_argument("source_id", nargs="?", choices=sorted(SOURCES))
    _leaf(verify, "dataset.verify", _handle_dataset_verify)

    config = root.add_parser("config", help="inspect local configuration")
    config_commands = config.add_subparsers(dest="config_command", required=True)

    config_show = config_commands.add_parser("show", help="show storage paths")
    _leaf(config_show, "config.show", _handle_config_show)

    config_validate = config_commands.add_parser(
        "validate",
        help="validate configuration and local tools",
    )
    _leaf(config_validate, "config.validate", _handle_config_validate)

    add_metadata_commands(root, _leaf)

    return parser


def _leaf(
    parser: argparse.ArgumentParser,
    command_name: str,
    handler: Callable[[argparse.Namespace], Any],
) -> None:
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="output format",
    )
    parser.set_defaults(command_name=command_name, handler=handler)


def _handle_commands(args: argparse.Namespace) -> dict[str, Any]:
    return COMMAND_MANIFEST


def _handle_fetch_list(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sources": [
            {
                "source_id": meta.source_id,
                "modality": meta.modality,
                "homepage": meta.homepage,
                "acquisition": type(meta.acquisition).__name__,
                "tool": _tool_name(meta),
            }
            for meta in SOURCES.values()
        ],
        "groups": {
            name: list(source_ids) for name, source_ids in SOURCE_GROUPS.items()
        },
    }


def _handle_fetch_run(args: argparse.Namespace) -> dict[str, Any]:
    metas = _selected_metas(args)
    service = FetcherService()
    if args.dry_run:
        return {
            "dry_run": True,
            "plans": [service.plan(meta).to_dict() for meta in metas],
        }

    outcomes = []
    for meta in metas:
        plan, record = service.fetch_with_plan(meta, force=args.force)
        outcomes.append(
            {
                "action": plan.action,
                "reason": plan.reason,
                "record": record.to_dict(),
            }
        )
    return {"dry_run": False, "datasets": outcomes}


def _handle_fetch_reconcile(args: argparse.Namespace) -> dict[str, Any]:
    return reconcile()


def _handle_dataset_inspect(args: argparse.Namespace) -> dict[str, Any]:
    record = DatasetRegistry().find(args.source_id)
    if record is None:
        raise CliFailure(
            "DATA_NOT_FOUND",
            f"No registry record for {args.source_id}",
            exit_code=4,
        )
    return {"record": record}


def _handle_dataset_verify(args: argparse.Namespace) -> dict[str, Any]:
    if args.source_id is None:
        try:
            return verify_all()
        except (ValueError, RegistryVerificationError) as exc:
            raise CliFailure(
                "VERIFICATION_FAILED",
                str(exc),
                exit_code=6,
            ) from exc

    record = DatasetRegistry().find(args.source_id)
    if record is None:
        raise CliFailure(
            "DATA_NOT_FOUND",
            f"No registry record for {args.source_id}",
            exit_code=4,
        )
    try:
        verify_record(record)
    except RegistryVerificationError as exc:
        raise CliFailure(
            "VERIFICATION_FAILED",
            str(exc),
            exit_code=6,
        ) from exc
    return {
        "source_id": args.source_id,
        "file_count": record["file_count"],
        "total_bytes": record["total_bytes"],
        "status": "verified",
    }


def _handle_config_show(args: argparse.Namespace) -> dict[str, Any]:
    configured = load_config("paths")["paths"]
    return {
        "configured": configured,
        "resolved": _resolved_paths(),
    }


def _handle_config_validate(args: argparse.Namespace) -> dict[str, Any]:
    resolved = _resolved_paths()
    curl = shutil.which("curl")
    if curl is None:
        raise CliFailure(
            "TOOL_NOT_FOUND",
            "curl is required for HTTP dataset downloads",
            exit_code=3,
        )
    return {
        "status": "valid",
        "python": sys.version.split()[0],
        "curl": curl,
        "paths": resolved,
    }


def _selected_metas(args: argparse.Namespace) -> list[DatasetMeta]:
    target_count = bool(args.source_ids) + bool(args.group) + args.fetch_all
    if target_count != 1:
        raise CliFailure(
            "INVALID_ARGUMENT",
            "provide source IDs, --group, or --all",
            exit_code=2,
        )
    if args.group:
        source_ids = SOURCE_GROUPS[args.group]
    elif args.fetch_all:
        source_ids = tuple(SOURCES)
    else:
        source_ids = args.source_ids
    return [get_source(source_id) for source_id in source_ids]


def _resolved_paths() -> dict[str, str]:
    keys = (
        "paths.text_path",
        "paths.multimodal_path",
        "paths.robot_path",
        "paths.registry_path",
        "paths.metadata_db_path",
    )
    return {key: str(get_path(key)) for key in keys}


def _tool_name(meta: DatasetMeta) -> str:
    acquisition = meta.acquisition
    if isinstance(acquisition, HttpAcquisition):
        return "curl"
    if isinstance(acquisition, HuggingFaceAcquisition):
        return "huggingface-datasets"
    if isinstance(acquisition, CommonCrawlAcquisition):
        return "common-crawl-resolver+curl"
    if isinstance(acquisition, MinariAcquisition):
        return "minari"
    return "unknown"


def _emit(command: str, data: Any, output_format: str) -> None:
    if output_format == "json":
        print(
            json.dumps(
                {
                    "ok": True,
                    "command": command,
                    "data": data,
                    "error": None,
                },
                ensure_ascii=False,
            )
        )
        return
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _emit_error(
    command: str,
    error: CliFailure,
    output_format: str,
) -> None:
    payload = {
        "ok": False,
        "command": command,
        "data": None,
        "error": {
            "code": error.code,
            "message": str(error),
            "retryable": error.retryable,
        },
    }
    stream = sys.stdout if output_format == "json" else sys.stderr
    print(
        json.dumps(
            payload if output_format == "json" else payload["error"],
            ensure_ascii=False,
            indent=None if output_format == "json" else 2,
        ),
        file=stream,
    )


def _unexpected_error(
    command: str,
    output_format: str,
    code: str,
    error: Exception,
    *,
    exit_code: int,
    retryable: bool = False,
) -> int:
    _emit_error(
        command,
        CliFailure(
            code,
            str(error),
            exit_code=exit_code,
            retryable=retryable,
        ),
        output_format,
    )
    return exit_code


def _requested_output(arguments: Sequence[str]) -> str:
    for index, argument in enumerate(arguments[:-1]):
        if argument == "--output":
            return arguments[index + 1]
    return "json" if "--output=json" in arguments else "text"


def _command_name(arguments: Sequence[str]) -> str:
    command_parts = [item for item in arguments if not item.startswith("-")][:2]
    return ".".join(command_parts) or "trainfoundry"
