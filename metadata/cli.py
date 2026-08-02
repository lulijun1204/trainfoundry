"""Argument parsers and handlers for metadata CLI commands."""

import argparse
from collections.abc import Callable
from typing import Any

from metadata.repository import (
    LINEAGE_RELATIONS,
    MODALITIES,
    OUTPUT_MODES,
    PRODUCER_TYPES,
    PURPOSES,
    RESULT_STATUSES,
    RUN_STATUSES,
    RUN_TYPES,
    SCHEMA_FORMATS,
    SPLITS,
    TRAINING_ROLES,
    VERSION_STAGES,
    VERSION_STATUSES,
    MetadataRepository,
)

LeafBuilder = Callable[
    [argparse.ArgumentParser, str, Callable[[argparse.Namespace], Any]],
    None,
]


def add_metadata_commands(
    root: argparse._SubParsersAction,
    leaf: LeafBuilder,
) -> None:
    """Register the complete metadata command tree on the unified CLI."""
    metadata = root.add_parser("metadata", help="manage local relational metadata")
    commands = metadata.add_subparsers(dest="metadata_command", required=True)

    init = commands.add_parser("init", help="initialize the local SQLite schema")
    leaf(init, "metadata.init", _handle_init)

    status = commands.add_parser("status", help="show local metadata DB status")
    leaf(status, "metadata.status", _handle_status)

    _add_dataset_commands(commands, leaf)
    _add_version_commands(commands, leaf)
    _add_run_commands(commands, leaf)
    _add_lineage_commands(commands, leaf)
    _add_quality_commands(commands, leaf)
    _add_annotation_commands(commands, leaf)
    _add_training_commands(commands, leaf)


def _add_dataset_commands(
    root: argparse._SubParsersAction,
    leaf: LeafBuilder,
) -> None:
    dataset = root.add_parser("dataset", help="manage Dataset metadata")
    commands = dataset.add_subparsers(dest="metadata_dataset_command", required=True)

    create = commands.add_parser("create", help="create a Dataset")
    create.add_argument("--dataset-id")
    create.add_argument("--namespace", required=True)
    create.add_argument("--name", required=True)
    create.add_argument("--purpose", choices=sorted(PURPOSES), required=True)
    create.add_argument(
        "--modality",
        action="append",
        choices=sorted(MODALITIES),
        required=True,
    )
    create.add_argument("--owner", required=True)
    create.add_argument("--source-uri")
    create.add_argument("--source-revision")
    create.add_argument("--license")
    create.add_argument("--permitted-use-json")
    create.add_argument("--region")
    create.add_argument("--contains-pii", action="store_true")
    create.add_argument("--retention-policy-json")
    create.add_argument("--access-policy-json", default="{}")
    leaf(create, "metadata.dataset.create", _handle_dataset_create)

    get = commands.add_parser("get", help="get one Dataset")
    get.add_argument("dataset_id")
    leaf(get, "metadata.dataset.get", _handle_dataset_get)

    list_command = commands.add_parser("list", help="list Datasets")
    list_command.add_argument("--namespace")
    list_command.add_argument("--owner")
    list_command.add_argument("--purpose", choices=sorted(PURPOSES))
    list_command.add_argument("--modality", choices=sorted(MODALITIES))
    _limit_argument(list_command)
    leaf(list_command, "metadata.dataset.list", _handle_dataset_list)

    update = commands.add_parser("update", help="update Dataset metadata")
    update.add_argument("dataset_id")
    update.add_argument("--namespace")
    update.add_argument("--name")
    update.add_argument("--purpose", choices=sorted(PURPOSES))
    update.add_argument("--modality", action="append", choices=sorted(MODALITIES))
    update.add_argument("--owner")
    update.add_argument("--source-uri")
    update.add_argument("--source-revision")
    update.add_argument("--license")
    update.add_argument("--permitted-use-json")
    update.add_argument("--region")
    update.add_argument(
        "--contains-pii",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    update.add_argument("--retention-policy-json")
    update.add_argument("--access-policy-json")
    leaf(update, "metadata.dataset.update", _handle_dataset_update)


def _add_version_commands(
    root: argparse._SubParsersAction,
    leaf: LeafBuilder,
) -> None:
    version = root.add_parser("version", help="manage DatasetVersion metadata")
    commands = version.add_subparsers(dest="metadata_version_command", required=True)

    create = commands.add_parser("create", help="create a DatasetVersion")
    create.add_argument("--version-id")
    create.add_argument("--dataset-id", required=True)
    create.add_argument("--version-number", type=int)
    create.add_argument("--stage", choices=sorted(VERSION_STAGES), required=True)
    create.add_argument(
        "--status",
        choices=sorted(VERSION_STATUSES),
        default="DRAFT",
    )
    create.add_argument("--storage-uri", required=True)
    create.add_argument("--storage-format", required=True)
    create.add_argument(
        "--schema-format",
        choices=sorted(SCHEMA_FORMATS),
        required=True,
    )
    create.add_argument("--schema-definition-json", required=True)
    create.add_argument("--schema-version")
    create.add_argument("--schema-digest", required=True)
    create.add_argument("--content-digest", required=True)
    create.add_argument("--split", choices=sorted(SPLITS))
    create.add_argument("--usage-tag", action="append")
    create.add_argument("--row-count", type=int)
    create.add_argument("--byte-size", type=int)
    create.add_argument("--created-by", required=True)
    leaf(create, "metadata.version.create", _handle_version_create)

    get = commands.add_parser("get", help="get one DatasetVersion")
    get.add_argument("version_id")
    leaf(get, "metadata.version.get", _handle_version_get)

    list_command = commands.add_parser("list", help="list DatasetVersions")
    list_command.add_argument("--dataset-id")
    list_command.add_argument("--stage", choices=sorted(VERSION_STAGES))
    list_command.add_argument("--status", choices=sorted(VERSION_STATUSES))
    list_command.add_argument("--split", choices=sorted(SPLITS))
    _limit_argument(list_command)
    leaf(list_command, "metadata.version.list", _handle_version_list)

    update = commands.add_parser(
        "update",
        help="update a DRAFT version or change lifecycle status",
    )
    update.add_argument("version_id")
    update.add_argument("--stage", choices=sorted(VERSION_STAGES))
    update.add_argument("--status", choices=sorted(VERSION_STATUSES))
    update.add_argument("--storage-uri")
    update.add_argument("--storage-format")
    update.add_argument("--schema-format", choices=sorted(SCHEMA_FORMATS))
    update.add_argument("--schema-definition-json")
    update.add_argument("--schema-version")
    update.add_argument("--schema-digest")
    update.add_argument("--content-digest")
    update.add_argument("--split", choices=sorted(SPLITS))
    update.add_argument("--usage-tag", action="append")
    update.add_argument("--row-count", type=int)
    update.add_argument("--byte-size", type=int)
    leaf(update, "metadata.version.update", _handle_version_update)


def _add_run_commands(
    root: argparse._SubParsersAction,
    leaf: LeafBuilder,
) -> None:
    run = root.add_parser("run", help="manage DatasetRun metadata")
    commands = run.add_subparsers(dest="metadata_run_command", required=True)

    create = commands.add_parser("create", help="create a DatasetRun")
    create.add_argument("--run-id")
    create.add_argument("--run-type", choices=sorted(RUN_TYPES), required=True)
    create.add_argument("--operator-name", required=True)
    create.add_argument("--operator-version", required=True)
    create.add_argument("--operator-fingerprint", required=True)
    create.add_argument("--params-json", default="{}")
    create.add_argument("--compute-key", required=True)
    create.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    create.add_argument(
        "--output-mode",
        choices=sorted(OUTPUT_MODES),
        required=True,
    )
    create.add_argument("--target-dataset-id")
    create.add_argument("--status", choices=sorted(RUN_STATUSES), default="PENDING")
    create.add_argument("--started-at")
    create.add_argument("--finished-at")
    create.add_argument("--error-message")
    create.add_argument("--created-by", required=True)
    leaf(create, "metadata.run.create", _handle_run_create)

    get = commands.add_parser("get", help="get one DatasetRun")
    get.add_argument("run_id")
    leaf(get, "metadata.run.get", _handle_run_get)

    list_command = commands.add_parser("list", help="list DatasetRuns")
    list_command.add_argument("--run-type", choices=sorted(RUN_TYPES))
    list_command.add_argument("--status", choices=sorted(RUN_STATUSES))
    list_command.add_argument("--target-dataset-id")
    _limit_argument(list_command)
    leaf(list_command, "metadata.run.list", _handle_run_list)

    update = commands.add_parser("update", help="update DatasetRun execution status")
    update.add_argument("run_id")
    update.add_argument("--status", choices=sorted(RUN_STATUSES), required=True)
    update.add_argument("--started-at")
    update.add_argument("--finished-at")
    update.add_argument("--error-message")
    leaf(update, "metadata.run.update", _handle_run_update)


def _add_lineage_commands(
    root: argparse._SubParsersAction,
    leaf: LeafBuilder,
) -> None:
    lineage = root.add_parser("lineage", help="manage DatasetLineage edges")
    commands = lineage.add_subparsers(dest="metadata_lineage_command", required=True)

    create = commands.add_parser("create", help="create a lineage edge")
    create.add_argument("--lineage-id")
    create.add_argument("--run-id", required=True)
    create.add_argument("--source-version-id", required=True)
    create.add_argument("--target-version-id", required=True)
    create.add_argument(
        "--relation-type",
        choices=sorted(LINEAGE_RELATIONS),
        required=True,
    )
    leaf(create, "metadata.lineage.create", _handle_lineage_create)

    trace = commands.add_parser("trace", help="trace version lineage recursively")
    trace.add_argument("version_id")
    trace.add_argument(
        "--direction",
        choices=("upstream", "downstream", "both"),
        default="both",
    )
    trace.add_argument("--max-depth", type=int, default=20)
    leaf(trace, "metadata.lineage.trace", _handle_lineage_trace)


def _add_quality_commands(
    root: argparse._SubParsersAction,
    leaf: LeafBuilder,
) -> None:
    quality = root.add_parser("quality", help="manage QualityResultSet metadata")
    commands = quality.add_subparsers(dest="metadata_quality_command", required=True)

    create = commands.add_parser("create", help="create a QualityResultSet")
    create.add_argument("--result-set-id")
    create.add_argument("--dataset-version-id", required=True)
    create.add_argument("--run-id", required=True)
    create.add_argument("--evaluator-name", required=True)
    create.add_argument("--evaluator-version", required=True)
    create.add_argument(
        "--status",
        choices=sorted(RESULT_STATUSES),
        default="PENDING",
    )
    create.add_argument(
        "--passed",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    create.add_argument("--summary-json", default="{}")
    create.add_argument("--passed-count", type=int)
    create.add_argument("--rejected-count", type=int)
    create.add_argument("--detail-uri")
    create.add_argument("--detail-digest")
    leaf(create, "metadata.quality.create", _handle_quality_create)

    get = commands.add_parser("get", help="get one QualityResultSet")
    get.add_argument("result_set_id")
    leaf(get, "metadata.quality.get", _handle_quality_get)

    list_command = commands.add_parser("list", help="list QualityResultSets")
    _result_filters(list_command)
    leaf(list_command, "metadata.quality.list", _handle_quality_list)

    update = commands.add_parser("update", help="update a QualityResultSet")
    update.add_argument("result_set_id")
    update.add_argument("--status", choices=sorted(RESULT_STATUSES), required=True)
    update.add_argument(
        "--passed",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    update.add_argument("--summary-json")
    leaf(update, "metadata.quality.update", _handle_quality_update)


def _add_annotation_commands(
    root: argparse._SubParsersAction,
    leaf: LeafBuilder,
) -> None:
    annotation = root.add_parser(
        "annotation",
        help="manage AnnotationResultSet metadata",
    )
    commands = annotation.add_subparsers(
        dest="metadata_annotation_command",
        required=True,
    )

    create = commands.add_parser("create", help="create an AnnotationResultSet")
    create.add_argument("--result-set-id")
    create.add_argument("--dataset-version-id", required=True)
    create.add_argument("--run-id", required=True)
    create.add_argument("--annotation-schema-version", required=True)
    create.add_argument(
        "--producer-type",
        choices=sorted(PRODUCER_TYPES),
        required=True,
    )
    create.add_argument("--producer-ref")
    create.add_argument(
        "--status",
        choices=sorted(RESULT_STATUSES),
        default="PENDING",
    )
    create.add_argument("--summary-json", default="{}")
    create.add_argument("--coverage", type=float)
    create.add_argument("--detail-uri")
    create.add_argument("--detail-digest")
    leaf(create, "metadata.annotation.create", _handle_annotation_create)

    get = commands.add_parser("get", help="get one AnnotationResultSet")
    get.add_argument("result_set_id")
    leaf(get, "metadata.annotation.get", _handle_annotation_get)

    list_command = commands.add_parser("list", help="list AnnotationResultSets")
    _result_filters(list_command)
    leaf(list_command, "metadata.annotation.list", _handle_annotation_list)

    update = commands.add_parser("update", help="update an AnnotationResultSet")
    update.add_argument("result_set_id")
    update.add_argument("--status", choices=sorted(RESULT_STATUSES), required=True)
    update.add_argument("--summary-json")
    update.add_argument("--coverage", type=float)
    leaf(update, "metadata.annotation.update", _handle_annotation_update)


def _add_training_commands(
    root: argparse._SubParsersAction,
    leaf: LeafBuilder,
) -> None:
    training = root.add_parser("training", help="manage TrainingRun metadata")
    commands = training.add_subparsers(
        dest="metadata_training_command",
        required=True,
    )

    create = commands.add_parser("create", help="create a TrainingRun")
    create.add_argument("--training-run-id")
    create.add_argument("--code-commit", required=True)
    create.add_argument("--config-digest", required=True)
    create.add_argument("--dataloader-config-json", default="{}")
    create.add_argument("--status", choices=sorted(RUN_STATUSES), default="PENDING")
    create.add_argument("--started-at")
    create.add_argument("--finished-at")
    create.add_argument("--model-artifact-uri")
    create.add_argument("--created-by", required=True)
    leaf(create, "metadata.training.create", _handle_training_create)

    get = commands.add_parser("get", help="get one TrainingRun")
    get.add_argument("training_run_id")
    leaf(get, "metadata.training.get", _handle_training_get)

    list_command = commands.add_parser("list", help="list TrainingRuns")
    list_command.add_argument("--status", choices=sorted(RUN_STATUSES))
    _limit_argument(list_command)
    leaf(list_command, "metadata.training.list", _handle_training_list)

    update = commands.add_parser("update", help="update TrainingRun status")
    update.add_argument("training_run_id")
    update.add_argument("--status", choices=sorted(RUN_STATUSES), required=True)
    update.add_argument("--started-at")
    update.add_argument("--finished-at")
    update.add_argument("--model-artifact-uri")
    leaf(update, "metadata.training.update", _handle_training_update)

    bind = commands.add_parser(
        "bind-version",
        help="bind a frozen DatasetVersion to a TrainingRun",
    )
    bind.add_argument("--training-run-id", required=True)
    bind.add_argument("--dataset-version-id", required=True)
    bind.add_argument("--role", choices=sorted(TRAINING_ROLES), required=True)
    bind.add_argument("--weight", type=float, default=1.0)
    bind.add_argument("--sampling-config-json")
    leaf(bind, "metadata.training.bind_version", _handle_training_bind)


def _handle_init(args: argparse.Namespace) -> dict[str, Any]:
    return MetadataRepository().initialize()


def _handle_status(args: argparse.Namespace) -> dict[str, Any]:
    return MetadataRepository().status()


def _handle_dataset_create(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset": MetadataRepository().create_dataset(
            dataset_id=args.dataset_id,
            namespace=args.namespace,
            name=args.name,
            purpose=args.purpose,
            modalities=args.modality,
            owner=args.owner,
            source_uri=args.source_uri,
            source_revision=args.source_revision,
            license=args.license,
            permitted_use=args.permitted_use_json,
            region=args.region,
            contains_pii=args.contains_pii,
            retention_policy=args.retention_policy_json,
            access_policy=args.access_policy_json,
        )
    }


def _handle_dataset_get(args: argparse.Namespace) -> dict[str, Any]:
    return {"dataset": MetadataRepository().get_dataset(args.dataset_id)}


def _handle_dataset_list(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "datasets": MetadataRepository().list_datasets(
            namespace=args.namespace,
            owner=args.owner,
            purpose=args.purpose,
            modality=args.modality,
            limit=args.limit,
        )
    }


def _handle_dataset_update(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset": MetadataRepository().update_dataset(
            args.dataset_id,
            **_provided(
                args,
                {
                    "namespace": "namespace",
                    "name": "name",
                    "purpose": "purpose",
                    "modality": "modalities",
                    "owner": "owner",
                    "source_uri": "source_uri",
                    "source_revision": "source_revision",
                    "license": "license",
                    "permitted_use_json": "permitted_use",
                    "region": "region",
                    "contains_pii": "contains_pii",
                    "retention_policy_json": "retention_policy",
                    "access_policy_json": "access_policy",
                },
            ),
        )
    }


def _handle_version_create(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "version": MetadataRepository().create_version(
            version_id=args.version_id,
            dataset_id=args.dataset_id,
            version_number=args.version_number,
            stage=args.stage,
            status=args.status,
            storage_uri=args.storage_uri,
            storage_format=args.storage_format,
            schema_format=args.schema_format,
            schema_definition=args.schema_definition_json,
            schema_version=args.schema_version,
            schema_digest=args.schema_digest,
            content_digest=args.content_digest,
            split=args.split,
            usage_tags=args.usage_tag,
            row_count=args.row_count,
            byte_size=args.byte_size,
            created_by=args.created_by,
        ).to_dict()
    }


def _handle_version_get(args: argparse.Namespace) -> dict[str, Any]:
    return {"version": MetadataRepository().get_version(args.version_id).to_dict()}


def _handle_version_list(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "versions": [
            version.to_dict()
            for version in MetadataRepository().list_versions(
                dataset_id=args.dataset_id,
                stage=args.stage,
                status=args.status,
                split=args.split,
                limit=args.limit,
            )
        ]
    }


def _handle_version_update(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "version": MetadataRepository().update_version(
            args.version_id,
            **_provided(
                args,
                {
                    "stage": "stage",
                    "status": "status",
                    "storage_uri": "storage_uri",
                    "storage_format": "storage_format",
                    "schema_format": "schema_format",
                    "schema_definition_json": "schema_definition",
                    "schema_version": "schema_version",
                    "schema_digest": "schema_digest",
                    "content_digest": "content_digest",
                    "split": "split",
                    "usage_tag": "usage_tags",
                    "row_count": "row_count",
                    "byte_size": "byte_size",
                },
            ),
        ).to_dict()
    }


def _handle_run_create(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "run": MetadataRepository().create_run(
            run_id=args.run_id,
            run_type=args.run_type,
            operator_name=args.operator_name,
            operator_version=args.operator_version,
            operator_fingerprint=args.operator_fingerprint,
            params=args.params_json,
            compute_key=args.compute_key,
            deterministic=args.deterministic,
            output_mode=args.output_mode,
            target_dataset_id=args.target_dataset_id,
            status=args.status,
            started_at=args.started_at,
            finished_at=args.finished_at,
            error_message=args.error_message,
            created_by=args.created_by,
        ).to_dict()
    }


def _handle_run_get(args: argparse.Namespace) -> dict[str, Any]:
    return {"run": MetadataRepository().get_run(args.run_id).to_dict()}


def _handle_run_list(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "runs": [
            run.to_dict()
            for run in MetadataRepository().list_runs(
                run_type=args.run_type,
                status=args.status,
                target_dataset_id=args.target_dataset_id,
                limit=args.limit,
            )
        ]
    }


def _handle_run_update(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "run": MetadataRepository().update_run(
            args.run_id,
            status=args.status,
            started_at=args.started_at,
            finished_at=args.finished_at,
            error_message=args.error_message,
        ).to_dict()
    }


def _handle_lineage_create(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "lineage": MetadataRepository().create_lineage(
            lineage_id=args.lineage_id,
            run_id=args.run_id,
            source_version_id=args.source_version_id,
            target_version_id=args.target_version_id,
            relation_type=args.relation_type,
        )
    }


def _handle_lineage_trace(args: argparse.Namespace) -> dict[str, Any]:
    return MetadataRepository().get_lineage(
        args.version_id,
        direction=args.direction,
        max_depth=args.max_depth,
    )


def _handle_quality_create(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "quality_result": MetadataRepository().create_quality_result(
            result_set_id=args.result_set_id,
            dataset_version_id=args.dataset_version_id,
            run_id=args.run_id,
            evaluator_name=args.evaluator_name,
            evaluator_version=args.evaluator_version,
            status=args.status,
            passed=args.passed,
            summary=args.summary_json,
            passed_count=args.passed_count,
            rejected_count=args.rejected_count,
            detail_uri=args.detail_uri,
            detail_digest=args.detail_digest,
        )
    }


def _handle_quality_get(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "quality_result": MetadataRepository().get_quality_result(
            args.result_set_id
        )
    }


def _handle_quality_list(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "quality_results": MetadataRepository().list_quality_results(
            dataset_version_id=args.dataset_version_id,
            run_id=args.run_id,
            status=args.status,
            limit=args.limit,
        )
    }


def _handle_quality_update(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "quality_result": MetadataRepository().update_quality_result(
            args.result_set_id,
            status=args.status,
            passed=args.passed,
            summary=args.summary_json,
        )
    }


def _handle_annotation_create(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "annotation_result": MetadataRepository().create_annotation_result(
            result_set_id=args.result_set_id,
            dataset_version_id=args.dataset_version_id,
            run_id=args.run_id,
            annotation_schema_version=args.annotation_schema_version,
            producer_type=args.producer_type,
            producer_ref=args.producer_ref,
            status=args.status,
            summary=args.summary_json,
            coverage=args.coverage,
            detail_uri=args.detail_uri,
            detail_digest=args.detail_digest,
        )
    }


def _handle_annotation_get(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "annotation_result": MetadataRepository().get_annotation_result(
            args.result_set_id
        )
    }


def _handle_annotation_list(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "annotation_results": MetadataRepository().list_annotation_results(
            dataset_version_id=args.dataset_version_id,
            run_id=args.run_id,
            status=args.status,
            limit=args.limit,
        )
    }


def _handle_annotation_update(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "annotation_result": MetadataRepository().update_annotation_result(
            args.result_set_id,
            status=args.status,
            summary=args.summary_json,
            coverage=args.coverage,
        )
    }


def _handle_training_create(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "training_run": MetadataRepository().create_training_run(
            training_run_id=args.training_run_id,
            code_commit=args.code_commit,
            config_digest=args.config_digest,
            dataloader_config=args.dataloader_config_json,
            status=args.status,
            started_at=args.started_at,
            finished_at=args.finished_at,
            model_artifact_uri=args.model_artifact_uri,
            created_by=args.created_by,
        )
    }


def _handle_training_get(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "training_run": MetadataRepository().get_training_run(
            args.training_run_id
        )
    }


def _handle_training_list(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "training_runs": MetadataRepository().list_training_runs(
            status=args.status,
            limit=args.limit,
        )
    }


def _handle_training_update(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "training_run": MetadataRepository().update_training_run(
            args.training_run_id,
            status=args.status,
            started_at=args.started_at,
            finished_at=args.finished_at,
            model_artifact_uri=args.model_artifact_uri,
        )
    }


def _handle_training_bind(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "binding": MetadataRepository().bind_training_version(
            training_run_id=args.training_run_id,
            dataset_version_id=args.dataset_version_id,
            role=args.role,
            weight=args.weight,
            sampling_config=args.sampling_config_json,
        )
    }


def _provided(
    args: argparse.Namespace,
    mapping: dict[str, str],
) -> dict[str, Any]:
    return {
        destination: value
        for source, destination in mapping.items()
        if (value := getattr(args, source)) is not None
    }


def _limit_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=100)


def _result_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-version-id")
    parser.add_argument("--run-id")
    parser.add_argument("--status", choices=sorted(RESULT_STATUSES))
    _limit_argument(parser)
