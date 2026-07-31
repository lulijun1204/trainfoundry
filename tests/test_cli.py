import json

from trainfoundry.cli import build_parser, main
from trainfoundry.manifest import COMMAND_MANIFEST


def test_commands_returns_machine_readable_json(capsys):
    exit_code = main(["commands", "--output", "json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["command"] == "commands"
    assert any(
        command["name"] == "fetch.run" for command in payload["data"]["commands"]
    )
    assert payload["data"]["aliases"] == ["tf"]


def test_short_alias_has_matching_help_program_name():
    assert build_parser(prog="tf").format_help().startswith("usage: tf ")
    assert COMMAND_MANIFEST["aliases"] == ["tf"]


def test_fetch_list_returns_tools_and_groups(capsys):
    exit_code = main(["fetch", "list", "--output", "json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert len(payload["data"]["sources"]) == 7
    assert payload["data"]["groups"]["huggingface"]


def test_missing_fetch_target_returns_structured_usage_error(capsys):
    exit_code = main(["fetch", "run", "--output", "json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload == {
        "ok": False,
        "command": "fetch.run",
        "data": None,
        "error": {
            "code": "INVALID_ARGUMENT",
            "message": "provide source IDs, --group, or --all",
            "retryable": False,
        },
    }


def test_unknown_fetch_source_returns_structured_usage_error(capsys):
    exit_code = main(["fetch", "run", "unknown", "--output", "json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["error"]["code"] == "INVALID_ARGUMENT"
    assert "unknown source 'unknown'" in payload["error"]["message"]


def test_missing_registry_record_returns_not_found(capsys, monkeypatch):
    monkeypatch.setattr(
        "trainfoundry.cli.DatasetRegistry.find",
        lambda self, source_id: None,
    )

    exit_code = main(
        [
            "dataset",
            "inspect",
            "coco_2017_validation",
            "--output",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 4
    assert payload["error"]["code"] == "DATA_NOT_FOUND"
