"""Machine-readable CLI command contract."""

COMMAND_MANIFEST = {
    "schema_version": "1.1",
    "program": "trainfoundry",
    "aliases": ["tf"],
    "commands": [
        {
            "name": "commands",
            "description": "Return this machine-readable command contract.",
            "mutating": False,
            "arguments": {},
        },
        {
            "name": "fetch.list",
            "description": "List declared dataset sources and acquisition tools.",
            "mutating": False,
            "arguments": {},
        },
        {
            "name": "fetch.run",
            "description": "Plan or acquire and register one or more datasets.",
            "mutating": True,
            "arguments": {
                "source_ids": {"type": "array", "required": False},
                "group": {
                    "type": "string",
                    "enum": ["huggingface", "non_text"],
                    "required": False,
                },
                "all": {"type": "boolean", "default": False},
                "dry_run": {"type": "boolean", "default": False},
                "force": {"type": "boolean", "default": False},
            },
        },
        {
            "name": "fetch.reconcile",
            "description": "Register previously downloaded text datasets.",
            "mutating": True,
            "arguments": {},
        },
        {
            "name": "dataset.inspect",
            "description": "Return one persisted dataset record.",
            "mutating": False,
            "arguments": {
                "source_id": {"type": "string", "required": True},
            },
        },
        {
            "name": "dataset.verify",
            "description": "Verify one dataset or the complete expected registry.",
            "mutating": False,
            "arguments": {
                "source_id": {"type": "string", "required": False},
            },
        },
        {
            "name": "config.show",
            "description": "Show configured and resolved storage paths.",
            "mutating": False,
            "arguments": {},
        },
        {
            "name": "config.validate",
            "description": "Validate configuration and required local tools.",
            "mutating": False,
            "arguments": {},
        },
        {
            "name": "metadata.init",
            "description": "Initialize the configured SQLite metadata schema.",
            "mutating": True,
            "arguments": {},
        },
        {
            "name": "metadata.status",
            "description": "Show metadata database path and schema status.",
            "mutating": False,
            "arguments": {},
        },
        {
            "name": "metadata.dataset.create",
            "description": "Create a governed logical Dataset.",
            "mutating": True,
            "arguments": {
                "namespace": {"type": "string", "required": True},
                "name": {"type": "string", "required": True},
                "purpose": {
                    "type": "string",
                    "enum": ["PRETRAIN", "SFT", "RL", "BENCHMARK"],
                    "required": True,
                },
                "modality": {"type": "array", "required": True},
                "owner": {"type": "string", "required": True},
            },
        },
        {
            "name": "metadata.dataset.get",
            "description": "Get one Dataset by dataset_id.",
            "mutating": False,
            "arguments": {
                "dataset_id": {"type": "string", "required": True},
            },
        },
        {
            "name": "metadata.dataset.list",
            "description": "List Datasets with optional filters.",
            "mutating": False,
            "arguments": {
                "namespace": {"type": "string", "required": False},
                "owner": {"type": "string", "required": False},
                "purpose": {"type": "string", "required": False},
                "modality": {"type": "string", "required": False},
                "limit": {"type": "integer", "default": 100},
            },
        },
        {
            "name": "metadata.dataset.update",
            "description": "Update Dataset identity or governance metadata.",
            "mutating": True,
            "arguments": {
                "dataset_id": {"type": "string", "required": True},
            },
        },
        {
            "name": "metadata.version.create",
            "description": "Create an immutable-capable DatasetVersion.",
            "mutating": True,
            "arguments": {
                "dataset_id": {"type": "string", "required": True},
                "stage": {"type": "string", "required": True},
                "storage_uri": {"type": "string", "required": True},
                "storage_format": {"type": "string", "required": True},
                "schema_format": {"type": "string", "required": True},
                "schema_definition_json": {
                    "type": "json",
                    "required": True,
                },
                "schema_digest": {"type": "sha256", "required": True},
                "content_digest": {"type": "sha256", "required": True},
                "created_by": {"type": "string", "required": True},
            },
        },
        {
            "name": "metadata.version.get",
            "description": "Get one DatasetVersion by version_id.",
            "mutating": False,
            "arguments": {
                "version_id": {"type": "string", "required": True},
            },
        },
        {
            "name": "metadata.version.list",
            "description": "List DatasetVersions with optional filters.",
            "mutating": False,
            "arguments": {
                "dataset_id": {"type": "string", "required": False},
                "stage": {"type": "string", "required": False},
                "status": {"type": "string", "required": False},
                "split": {"type": "string", "required": False},
                "limit": {"type": "integer", "default": 100},
            },
        },
        {
            "name": "metadata.version.update",
            "description": "Edit a DRAFT version or update version status.",
            "mutating": True,
            "arguments": {
                "version_id": {"type": "string", "required": True},
            },
        },
        {
            "name": "metadata.run.create",
            "description": "Create a DatasetRun execution record.",
            "mutating": True,
            "arguments": {
                "run_type": {"type": "string", "required": True},
                "operator_name": {"type": "string", "required": True},
                "operator_version": {"type": "string", "required": True},
                "operator_fingerprint": {"type": "sha256", "required": True},
                "compute_key": {"type": "sha256", "required": True},
                "output_mode": {"type": "string", "required": True},
                "created_by": {"type": "string", "required": True},
            },
        },
        {
            "name": "metadata.run.get",
            "description": "Get one DatasetRun by run_id.",
            "mutating": False,
            "arguments": {
                "run_id": {"type": "string", "required": True},
            },
        },
        {
            "name": "metadata.run.list",
            "description": "List DatasetRuns with optional filters.",
            "mutating": False,
            "arguments": {
                "run_type": {"type": "string", "required": False},
                "status": {"type": "string", "required": False},
                "target_dataset_id": {"type": "string", "required": False},
                "limit": {"type": "integer", "default": 100},
            },
        },
        {
            "name": "metadata.run.update",
            "description": "Update DatasetRun execution status.",
            "mutating": True,
            "arguments": {
                "run_id": {"type": "string", "required": True},
                "status": {"type": "string", "required": True},
            },
        },
        {
            "name": "metadata.lineage.create",
            "description": "Create a version-level DatasetLineage edge.",
            "mutating": True,
            "arguments": {
                "run_id": {"type": "string", "required": True},
                "source_version_id": {"type": "string", "required": True},
                "target_version_id": {"type": "string", "required": True},
                "relation_type": {"type": "string", "required": True},
            },
        },
        {
            "name": "metadata.lineage.trace",
            "description": "Trace upstream or downstream version lineage.",
            "mutating": False,
            "arguments": {
                "version_id": {"type": "string", "required": True},
                "direction": {
                    "type": "string",
                    "enum": ["upstream", "downstream", "both"],
                    "default": "both",
                },
                "max_depth": {"type": "integer", "default": 20},
            },
        },
        {
            "name": "metadata.quality.create",
            "description": "Create a QualityResultSet.",
            "mutating": True,
            "arguments": {
                "dataset_version_id": {"type": "string", "required": True},
                "run_id": {"type": "string", "required": True},
                "evaluator_name": {"type": "string", "required": True},
                "evaluator_version": {"type": "string", "required": True},
            },
        },
        {
            "name": "metadata.quality.get",
            "description": "Get one QualityResultSet.",
            "mutating": False,
            "arguments": {
                "result_set_id": {"type": "string", "required": True},
            },
        },
        {
            "name": "metadata.quality.list",
            "description": "List QualityResultSets.",
            "mutating": False,
            "arguments": {},
        },
        {
            "name": "metadata.quality.update",
            "description": "Update QualityResultSet status and summary.",
            "mutating": True,
            "arguments": {
                "result_set_id": {"type": "string", "required": True},
                "status": {"type": "string", "required": True},
            },
        },
        {
            "name": "metadata.annotation.create",
            "description": "Create an AnnotationResultSet.",
            "mutating": True,
            "arguments": {
                "dataset_version_id": {"type": "string", "required": True},
                "run_id": {"type": "string", "required": True},
                "annotation_schema_version": {
                    "type": "string",
                    "required": True,
                },
                "producer_type": {"type": "string", "required": True},
            },
        },
        {
            "name": "metadata.annotation.get",
            "description": "Get one AnnotationResultSet.",
            "mutating": False,
            "arguments": {
                "result_set_id": {"type": "string", "required": True},
            },
        },
        {
            "name": "metadata.annotation.list",
            "description": "List AnnotationResultSets.",
            "mutating": False,
            "arguments": {},
        },
        {
            "name": "metadata.annotation.update",
            "description": "Update AnnotationResultSet status and summary.",
            "mutating": True,
            "arguments": {
                "result_set_id": {"type": "string", "required": True},
                "status": {"type": "string", "required": True},
            },
        },
        {
            "name": "metadata.training.create",
            "description": "Create a TrainingRun.",
            "mutating": True,
            "arguments": {
                "code_commit": {"type": "string", "required": True},
                "config_digest": {"type": "sha256", "required": True},
                "created_by": {"type": "string", "required": True},
            },
        },
        {
            "name": "metadata.training.get",
            "description": "Get one TrainingRun and its dataset bindings.",
            "mutating": False,
            "arguments": {
                "training_run_id": {"type": "string", "required": True},
            },
        },
        {
            "name": "metadata.training.list",
            "description": "List TrainingRuns and dataset bindings.",
            "mutating": False,
            "arguments": {},
        },
        {
            "name": "metadata.training.update",
            "description": "Update TrainingRun status and artifact URI.",
            "mutating": True,
            "arguments": {
                "training_run_id": {"type": "string", "required": True},
                "status": {"type": "string", "required": True},
            },
        },
        {
            "name": "metadata.training.bind_version",
            "description": "Bind a frozen DatasetVersion to a TrainingRun.",
            "mutating": True,
            "arguments": {
                "training_run_id": {"type": "string", "required": True},
                "dataset_version_id": {"type": "string", "required": True},
                "role": {"type": "string", "required": True},
                "weight": {"type": "number", "default": 1.0},
            },
        },
    ],
}
