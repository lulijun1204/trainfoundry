# Local Metadata

TrainFoundry uses a project-local SQLite database for relational metadata. The
default database is:

```text
<project-root>/.trainfoundry/metadata.db
```

The path is managed by `paths.metadata_db_path` in `config/paths.toml`.
`.trainfoundry/` is ignored by Git; schema code and tests are committed, while
each checkout keeps its own metadata state.

The preferred CLI executable is `tf`; the longer `trainfoundry` command is kept
as a compatible alias. For example:

```bash
tf metadata status --output json
tf metadata init --output json
```

## Initialize

```bash
trainfoundry metadata status --output json
trainfoundry metadata init --output json
```

`metadata init` is idempotent. It enables SQLite foreign keys, WAL, and a
five-second busy timeout, then applies [`schema.sql`](schema.sql). The current
schema version is stored both in `PRAGMA user_version` and
`metadata_schema_migrations`.

## Data model

The schema follows the eight persistent entities from the data-model design:

```text
Dataset 1 ─────── N DatasetVersion
                       │ source / target
                       ▼
DatasetRun 1 ───── N DatasetLineage

DatasetVersion 1 ─ N QualityResultSet
DatasetVersion 1 ─ N AnnotationResultSet

TrainingRun 1 ──── N TrainingRunDatasetVersion N ──── 1 DatasetVersion
```

| SQLite table | Model |
| --- | --- |
| `datasets` | Stable logical dataset identity and governance |
| `dataset_versions` | Versioned content, storage, schema, split, and digest |
| `dataset_runs` | One ingest, processing, quality, or annotation execution |
| `dataset_lineage` | Version-level source → run → target edge |
| `quality_result_sets` | Quality summary with external detail URI |
| `annotation_result_sets` | Annotation summary with external detail URI |
| `training_runs` | Training reproduction and artifact metadata |
| `training_run_dataset_versions` | Training/version role and sampling binding |

Dataset is the smallest managed data unit. Files, rows, frames, samples,
episodes, shards, and manifests are not separate metadata entities; they remain
inside the data representation addressed by `storage_uri`.

## Enforced rules

SQLite constraints and triggers enforce the model rather than relying only on
CLI validation:

- `UNIQUE(namespace, name)` for Dataset identity.
- `UNIQUE(dataset_id, version_number)` for DatasetVersion.
- A successful deterministic DatasetRun has a unique `compute_key`.
- DatasetLineage rejects self-edges and duplicate semantic edges.
- Content, schema, storage, identity, stage, and split cannot change after a
  DatasetVersion leaves `DRAFT`; lifecycle `status` can still change.
- TrainingRun can bind only `COMMITTED` or `PUBLISHED` DatasetVersion records.
- JSON, enum, non-negative count, coverage, digest, and foreign-key constraints
  are checked in the database.

## Repository API

All database operations are aggregated behind
[`MetadataRepository`](repository.py):

```python
from metadata import MetadataRepository

repository = MetadataRepository()
repository.initialize()

dataset = repository.create_dataset(
    namespace="examples",
    name="dolly_sft",
    purpose="SFT",
    modalities=["TEXT"],
    owner="training-data",
)

datasets = repository.list_datasets(namespace="examples")
loaded = repository.get_dataset(dataset["dataset_id"])
```

The repository supports create/get/list/update operations for Dataset,
DatasetVersion, DatasetRun, QualityResultSet, AnnotationResultSet, and
TrainingRun. It also supports lineage edge creation and recursive tracing, plus
TrainingRunDatasetVersion binding.

DatasetVersion and DatasetRun operations return frozen domain models covering
every column in their respective tables. `DatasetVersion` is a persisted source
or materialization boundary; pipeline operators exchange a non-persistent
`ExecutionDataset` containing a lazy `BlockStream[pyarrow.RecordBatch]`
instead. Runtime batches are never metadata entities and are not stored in
SQLite. `PipelineExecutor` creates one metadata-only
`DatasetRun` for every operator invocation, and a Materializer creates a new
Lance DatasetVersion only at an explicit persistence boundary. Use each domain
model's `to_dict()` method at JSON boundaries such as custom CLI integrations.

Writes use transactions. Database integrity failures are translated into
domain errors:

| Error | Meaning |
| --- | --- |
| `MetadataNotInitializedError` | Run `metadata init` first |
| `MetadataNotFoundError` | Requested ID does not exist |
| `MetadataValidationError` | Public input is invalid |
| `MetadataConflictError` | Unique, foreign-key, immutable, or lifecycle rule failed |

## CLI

Every leaf command supports `--output json`.

### Dataset

```bash
trainfoundry metadata dataset create \
  --namespace examples \
  --name dolly_sft \
  --purpose SFT \
  --modality TEXT \
  --owner training-data \
  --output json

trainfoundry metadata dataset get <dataset-id> --output json
trainfoundry metadata dataset list --namespace examples --output json
trainfoundry metadata dataset update <dataset-id> \
  --owner new-owner \
  --output json
```

Repeat `--modality` for multimodal datasets:

```bash
--modality IMAGE --modality TEXT
```

Governance structures use JSON arguments such as
`--access-policy-json '{"readers":["team-a"]}'`.

### DatasetVersion

Schema and content digests are lowercase or uppercase SHA-256 hex strings:

```bash
trainfoundry metadata version create \
  --dataset-id <dataset-id> \
  --stage RAW \
  --storage-uri file:///datasets/dolly/v1 \
  --storage-format JSONL \
  --schema-format JSON_SCHEMA \
  --schema-definition-json '{"type":"object"}' \
  --schema-digest <64-character-sha256> \
  --content-digest <64-character-sha256> \
  --created-by training-data \
  --output json

trainfoundry metadata version list --dataset-id <dataset-id> --output json
trainfoundry metadata version update <version-id> \
  --status COMMITTED \
  --output json
```

Once committed, a command that tries to change storage, schema, digest, stage,
split, or identity fails with `METADATA_CONFLICT`.

### DatasetRun and lineage

```bash
trainfoundry metadata run create \
  --run-type CLEAN \
  --operator-name text-cleaner \
  --operator-version 1.0.0 \
  --operator-fingerprint <64-character-sha256> \
  --compute-key <64-character-sha256> \
  --output-mode NEW_VERSION \
  --target-dataset-id <dataset-id> \
  --created-by training-data \
  --output json

trainfoundry metadata lineage create \
  --run-id <run-id> \
  --source-version-id <source-version-id> \
  --target-version-id <target-version-id> \
  --relation-type DERIVED_FROM \
  --output json

trainfoundry metadata lineage trace <version-id> \
  --direction both \
  --max-depth 20 \
  --output json
```

### Quality, annotation, and training

```bash
trainfoundry metadata quality create --help
trainfoundry metadata annotation create --help
trainfoundry metadata training create --help
trainfoundry metadata training bind-version --help
```

Training bindings carry a role (`TRAIN`, `VALID`, `TEST`, or `AUXILIARY`),
weight, and optional sampling JSON. The target version must already be frozen.

## Code layout

| File | Responsibility |
| --- | --- |
| [`schema.sql`](schema.sql) | Tables, indexes, constraints, and triggers |
| [`database.py`](database.py) | Configured path, connections, init, schema status |
| [`repository.py`](repository.py) | Public metadata read/write and lineage API |
| [`cli.py`](cli.py) | Metadata command parsers and handlers |
| [`errors.py`](errors.py) | Stable metadata domain errors |

The fetch source registry remains a separate ingestion-completion record. The
relational metadata DB models long-lived Dataset identity, versions, operations,
lineage, evaluation, annotation, and training consumption.
