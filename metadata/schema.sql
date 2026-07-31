PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS metadata_schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL CHECK (length(trim(namespace)) > 0),
    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
    purpose TEXT NOT NULL CHECK (
        purpose IN ('PRETRAIN', 'SFT', 'RL', 'BENCHMARK')
    ),
    modalities_json TEXT NOT NULL CHECK (
        json_valid(modalities_json)
        AND json_type(modalities_json) = 'array'
        AND json_array_length(modalities_json) > 0
    ),
    owner TEXT NOT NULL CHECK (length(trim(owner)) > 0),
    source_uri TEXT,
    source_revision TEXT,
    license TEXT,
    permitted_use_json TEXT CHECK (
        permitted_use_json IS NULL OR json_valid(permitted_use_json)
    ),
    region TEXT,
    contains_pii INTEGER NOT NULL DEFAULT 0 CHECK (contains_pii IN (0, 1)),
    retention_policy_json TEXT CHECK (
        retention_policy_json IS NULL OR json_valid(retention_policy_json)
    ),
    access_policy_json TEXT NOT NULL DEFAULT '{}' CHECK (
        json_valid(access_policy_json)
        AND json_type(access_policy_json) = 'object'
    ),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    updated_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    UNIQUE (namespace, name)
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    version_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE RESTRICT,
    version_number INTEGER NOT NULL CHECK (version_number > 0),
    stage TEXT NOT NULL CHECK (
        stage IN ('RAW', 'PROCESSED', 'ANNOTATED', 'TRAINING_READY')
    ),
    status TEXT NOT NULL DEFAULT 'DRAFT' CHECK (
        status IN (
            'DRAFT',
            'COMMITTED',
            'PUBLISHED',
            'DEPRECATED',
            'INVALIDATED'
        )
    ),
    storage_uri TEXT NOT NULL CHECK (length(trim(storage_uri)) > 0),
    storage_format TEXT NOT NULL CHECK (length(trim(storage_format)) > 0),
    schema_format TEXT NOT NULL CHECK (
        schema_format IN ('ARROW', 'JSON_SCHEMA')
    ),
    schema_definition_json TEXT NOT NULL CHECK (
        json_valid(schema_definition_json)
        AND json_type(schema_definition_json) = 'object'
    ),
    schema_version TEXT,
    schema_digest TEXT NOT NULL CHECK (
        length(schema_digest) = 64
        AND lower(schema_digest) NOT GLOB '*[^0-9a-f]*'
    ),
    content_digest TEXT NOT NULL CHECK (
        length(content_digest) = 64
        AND lower(content_digest) NOT GLOB '*[^0-9a-f]*'
    ),
    split TEXT CHECK (split IS NULL OR split IN ('TRAIN', 'VALID', 'TEST')),
    usage_tags_json TEXT CHECK (
        usage_tags_json IS NULL
        OR (
            json_valid(usage_tags_json)
            AND json_type(usage_tags_json) = 'array'
        )
    ),
    row_count INTEGER CHECK (row_count IS NULL OR row_count >= 0),
    byte_size INTEGER CHECK (byte_size IS NULL OR byte_size >= 0),
    created_by TEXT NOT NULL CHECK (length(trim(created_by)) > 0),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    committed_at TEXT,
    UNIQUE (dataset_id, version_number)
);

CREATE TABLE IF NOT EXISTS dataset_runs (
    run_id TEXT PRIMARY KEY,
    run_type TEXT NOT NULL CHECK (
        run_type IN (
            'INGEST',
            'CLEAN',
            'NORMALIZE',
            'DEDUP',
            'FILTER',
            'CONVERT',
            'SCHEMA_MAPPING',
            'MATERIALIZE_ANNOTATION',
            'SPLIT',
            'MERGE',
            'QUALITY',
            'ANNOTATION'
        )
    ),
    operator_name TEXT NOT NULL CHECK (length(trim(operator_name)) > 0),
    operator_version TEXT NOT NULL CHECK (length(trim(operator_version)) > 0),
    operator_fingerprint TEXT NOT NULL CHECK (
        length(operator_fingerprint) = 64
        AND lower(operator_fingerprint) NOT GLOB '*[^0-9a-f]*'
    ),
    params_json TEXT NOT NULL DEFAULT '{}' CHECK (
        json_valid(params_json) AND json_type(params_json) = 'object'
    ),
    compute_key TEXT NOT NULL CHECK (
        length(compute_key) = 64
        AND lower(compute_key) NOT GLOB '*[^0-9a-f]*'
    ),
    deterministic INTEGER NOT NULL DEFAULT 1 CHECK (deterministic IN (0, 1)),
    output_mode TEXT NOT NULL CHECK (
        output_mode IN ('NEW_VERSION', 'NEW_DATASET', 'NO_DATA_VERSION')
    ),
    target_dataset_id TEXT REFERENCES datasets(dataset_id) ON DELETE RESTRICT,
    status TEXT NOT NULL DEFAULT 'PENDING' CHECK (
        status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED')
    ),
    started_at TEXT,
    finished_at TEXT,
    error_message TEXT,
    created_by TEXT NOT NULL CHECK (length(trim(created_by)) > 0),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    CHECK (
        output_mode = 'NO_DATA_VERSION'
        OR target_dataset_id IS NOT NULL
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_dataset_runs_successful_compute_key
ON dataset_runs(compute_key)
WHERE deterministic = 1 AND status = 'SUCCEEDED';

CREATE TABLE IF NOT EXISTS dataset_lineage (
    lineage_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES dataset_runs(run_id) ON DELETE RESTRICT,
    source_version_id TEXT NOT NULL
        REFERENCES dataset_versions(version_id) ON DELETE RESTRICT,
    target_version_id TEXT NOT NULL
        REFERENCES dataset_versions(version_id) ON DELETE RESTRICT,
    relation_type TEXT NOT NULL CHECK (
        relation_type IN (
            'DERIVED_FROM',
            'FILTERED_FROM',
            'SPLIT_FROM',
            'MERGED_FROM',
            'ANNOTATED_FROM'
        )
    ),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    CHECK (source_version_id <> target_version_id),
    UNIQUE (run_id, source_version_id, target_version_id, relation_type)
);

CREATE TABLE IF NOT EXISTS quality_result_sets (
    result_set_id TEXT PRIMARY KEY,
    dataset_version_id TEXT NOT NULL
        REFERENCES dataset_versions(version_id) ON DELETE RESTRICT,
    run_id TEXT NOT NULL REFERENCES dataset_runs(run_id) ON DELETE RESTRICT,
    evaluator_name TEXT NOT NULL CHECK (length(trim(evaluator_name)) > 0),
    evaluator_version TEXT NOT NULL CHECK (length(trim(evaluator_version)) > 0),
    status TEXT NOT NULL DEFAULT 'PENDING' CHECK (
        status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED')
    ),
    passed INTEGER CHECK (passed IS NULL OR passed IN (0, 1)),
    summary_json TEXT NOT NULL DEFAULT '{}' CHECK (
        json_valid(summary_json) AND json_type(summary_json) = 'object'
    ),
    passed_count INTEGER CHECK (passed_count IS NULL OR passed_count >= 0),
    rejected_count INTEGER CHECK (rejected_count IS NULL OR rejected_count >= 0),
    detail_uri TEXT,
    detail_digest TEXT CHECK (
        detail_digest IS NULL
        OR (
            length(detail_digest) = 64
            AND lower(detail_digest) NOT GLOB '*[^0-9a-f]*'
        )
    ),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);

CREATE TABLE IF NOT EXISTS annotation_result_sets (
    result_set_id TEXT PRIMARY KEY,
    dataset_version_id TEXT NOT NULL
        REFERENCES dataset_versions(version_id) ON DELETE RESTRICT,
    run_id TEXT NOT NULL REFERENCES dataset_runs(run_id) ON DELETE RESTRICT,
    annotation_schema_version TEXT NOT NULL CHECK (
        length(trim(annotation_schema_version)) > 0
    ),
    producer_type TEXT NOT NULL CHECK (
        producer_type IN ('HUMAN', 'MODEL', 'RULE', 'HYBRID')
    ),
    producer_ref TEXT,
    status TEXT NOT NULL DEFAULT 'PENDING' CHECK (
        status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED')
    ),
    summary_json TEXT NOT NULL DEFAULT '{}' CHECK (
        json_valid(summary_json) AND json_type(summary_json) = 'object'
    ),
    coverage REAL CHECK (coverage IS NULL OR (coverage >= 0 AND coverage <= 1)),
    detail_uri TEXT,
    detail_digest TEXT CHECK (
        detail_digest IS NULL
        OR (
            length(detail_digest) = 64
            AND lower(detail_digest) NOT GLOB '*[^0-9a-f]*'
        )
    ),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);

CREATE TABLE IF NOT EXISTS training_runs (
    training_run_id TEXT PRIMARY KEY,
    code_commit TEXT NOT NULL CHECK (length(trim(code_commit)) > 0),
    config_digest TEXT NOT NULL CHECK (
        length(config_digest) = 64
        AND lower(config_digest) NOT GLOB '*[^0-9a-f]*'
    ),
    dataloader_config_json TEXT NOT NULL DEFAULT '{}' CHECK (
        json_valid(dataloader_config_json)
        AND json_type(dataloader_config_json) = 'object'
    ),
    status TEXT NOT NULL DEFAULT 'PENDING' CHECK (
        status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED')
    ),
    started_at TEXT,
    finished_at TEXT,
    model_artifact_uri TEXT,
    created_by TEXT NOT NULL CHECK (length(trim(created_by)) > 0),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);

CREATE TABLE IF NOT EXISTS training_run_dataset_versions (
    training_run_id TEXT NOT NULL
        REFERENCES training_runs(training_run_id) ON DELETE RESTRICT,
    dataset_version_id TEXT NOT NULL
        REFERENCES dataset_versions(version_id) ON DELETE RESTRICT,
    role TEXT NOT NULL CHECK (role IN ('TRAIN', 'VALID', 'TEST', 'AUXILIARY')),
    weight REAL NOT NULL DEFAULT 1.0 CHECK (weight >= 0),
    sampling_config_json TEXT CHECK (
        sampling_config_json IS NULL
        OR (
            json_valid(sampling_config_json)
            AND json_type(sampling_config_json) = 'object'
        )
    ),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    PRIMARY KEY (training_run_id, dataset_version_id, role)
);

CREATE INDEX IF NOT EXISTS idx_dataset_versions_dataset
ON dataset_versions(dataset_id);

CREATE INDEX IF NOT EXISTS idx_dataset_runs_target
ON dataset_runs(target_dataset_id);

CREATE INDEX IF NOT EXISTS idx_dataset_lineage_source
ON dataset_lineage(source_version_id);

CREATE INDEX IF NOT EXISTS idx_dataset_lineage_target
ON dataset_lineage(target_version_id);

CREATE INDEX IF NOT EXISTS idx_dataset_lineage_run
ON dataset_lineage(run_id);

CREATE INDEX IF NOT EXISTS idx_quality_version
ON quality_result_sets(dataset_version_id);

CREATE INDEX IF NOT EXISTS idx_quality_run
ON quality_result_sets(run_id);

CREATE INDEX IF NOT EXISTS idx_annotation_version
ON annotation_result_sets(dataset_version_id);

CREATE INDEX IF NOT EXISTS idx_annotation_run
ON annotation_result_sets(run_id);

CREATE INDEX IF NOT EXISTS idx_training_bindings_version
ON training_run_dataset_versions(dataset_version_id);

CREATE TRIGGER IF NOT EXISTS protect_committed_dataset_version
BEFORE UPDATE OF
    dataset_id,
    version_number,
    stage,
    storage_uri,
    storage_format,
    schema_format,
    schema_definition_json,
    schema_version,
    schema_digest,
    content_digest,
    split
ON dataset_versions
WHEN OLD.status <> 'DRAFT' AND (
    NEW.dataset_id IS NOT OLD.dataset_id
    OR NEW.version_number IS NOT OLD.version_number
    OR NEW.stage IS NOT OLD.stage
    OR NEW.storage_uri IS NOT OLD.storage_uri
    OR NEW.storage_format IS NOT OLD.storage_format
    OR NEW.schema_format IS NOT OLD.schema_format
    OR NEW.schema_definition_json IS NOT OLD.schema_definition_json
    OR NEW.schema_version IS NOT OLD.schema_version
    OR NEW.schema_digest IS NOT OLD.schema_digest
    OR NEW.content_digest IS NOT OLD.content_digest
    OR NEW.split IS NOT OLD.split
)
BEGIN
    SELECT RAISE(
        ABORT,
        'committed DatasetVersion content, schema, storage, and identity are immutable'
    );
END;

CREATE TRIGGER IF NOT EXISTS validate_training_dataset_version_insert
BEFORE INSERT ON training_run_dataset_versions
WHEN (
    SELECT status
    FROM dataset_versions
    WHERE version_id = NEW.dataset_version_id
) NOT IN ('COMMITTED', 'PUBLISHED')
BEGIN
    SELECT RAISE(
        ABORT,
        'TrainingRun may only bind COMMITTED or PUBLISHED DatasetVersion'
    );
END;

CREATE TRIGGER IF NOT EXISTS validate_training_dataset_version_update
BEFORE UPDATE OF dataset_version_id ON training_run_dataset_versions
WHEN (
    SELECT status
    FROM dataset_versions
    WHERE version_id = NEW.dataset_version_id
) NOT IN ('COMMITTED', 'PUBLISHED')
BEGIN
    SELECT RAISE(
        ABORT,
        'TrainingRun may only bind COMMITTED or PUBLISHED DatasetVersion'
    );
END;

INSERT OR IGNORE INTO metadata_schema_migrations(version, name)
VALUES (1, 'initial_metadata_model');

PRAGMA user_version = 1;
