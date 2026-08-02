# Dataset validation operators

Concrete Text examples are intentionally step-by-step so the physical reads and
validation decisions remain visible before they are wrapped in operators:

- [`examples/data_processing/wikitext/`](examples/data_processing/wikitext/README.md):
  JSONL validation, WikiText cleanup, Arrow batching, Lance writing, and read-back;
- [`examples/data_processing/common_crawl/`](examples/data_processing/common_crawl/README.md):
  streaming gzip/WARC-WET file-format and conversion-record validation;
- [`examples/data_processing/coco2017/`](examples/data_processing/coco2017/README.md):
  ZIP/CRC checks, full JPEG decode, and COCO image/annotation consistency;
- [`examples/data_processing/ucf101/`](examples/data_processing/ucf101/README.md):
  RAR/ZIP checks, ffprobe/full-frame decode, official folds, and leakage checks.
- [`examples/data_processing/minari_pointmaze/`](examples/data_processing/minari_pointmaze/README.md):
  full HDF5 reads, Minari trajectory contracts, and PointMaze goal/reward semantics.

`pipeline.validate` contains read-only operators for RAW training data. A
`DatasetVersion` is only the persisted source boundary. `SourceAdapter` opens it
as an ephemeral `ExecutionDataset`; operators exchange that runtime value and
never create intermediate versions. `PipelineExecutor` owns `DatasetRun`
lifecycle, result persistence, sequencing, and failure recording.

## Execution model

`ExecutionDataset` contains a re-openable `BlockStream`, an optional Arrow
schema, and the source version identity. A `BlockStream` always yields
`pyarrow.RecordBatch`; this is the only physical unit visible to an operator.
There is no separate `DataBlock` or runtime type switch.

Raw adapters are ingestion boundaries. JSON/JSONL records become Arrow batches;
ZIP and directory entries become Arrow rows with `source_path`,
`archive_member`, `location`, and `byte_size`. A Lance source is scanned directly
as Arrow batches. Operators therefore remain independent of the original file
or container format.

`PipelineOperator` consumes one `ExecutionDataset` and returns `OperatorOutput`
with another ephemeral dataset and zero or more result drafts. A
`ValidationReport` is persisted as `QualityResultSet`; supplemental drafts can
represent profiles, safety reports, conversion reports, and dedup summaries.

Every operator invocation creates a metadata-only `DatasetRun` with
`NO_DATA_VERSION`; that row does not materialize training data. Only an explicit
`MaterializationSpec` invokes a format-specific Materializer, creates a
`NEW_VERSION` run, writes physical data, registers one DatasetVersion, and adds
version lineage. Lance is the canonical persisted format and the default
registry exposes only `LanceMaterializer`; `pylance` is a required dependency.

```text
RAW (ZIP / JSON / directory) --SourceAdapter--> BlockStream[RecordBatch]
Lance DatasetVersion --------Lance Scanner----> BlockStream[RecordBatch]
                                              --> PipelineOperator ...
                                              --> Lance Materializer
                                              --> Lance DatasetVersion
```

```python
from pipeline import MaterializationSpec, PipelineExecutor
from pipeline.validate import DataValidationOperator, FileValidationOperator

result = PipelineExecutor(repository).execute(
    repository.get_version(version_id),
    [FileValidationOperator(), DataValidationOperator()],
    created_by="training-data",
)
assert result.output_version is None

processed = PipelineExecutor(repository).materialize(
    result.output_data,
    MaterializationSpec(
        storage_uri="/datasets/processed.lance",
        stage="PROCESSED",
        created_by="training-data",
    ),
)
```

## Operators

### `FileValidationOperator`

The file operator follows the document's full-scan chain:

1. discover files and reconcile an optional manifest;
2. reject missing, non-regular, unreadable, and (by default) symlink entries;
3. identify real formats from magic bytes and content, then compare with
   `DatasetVersion.storage_format`;
4. stream every file to EOF, calculate SHA-256, and detect concurrent changes;
5. parse JSON/JSONL, verify gzip CRC and WARC boundaries, test ZIP members,
   inspect RAR with `bsdtar`, open HDF5, fully decode images, and probe videos;
6. reject ZIP traversal, links, encryption, duplicate members, and archive-bomb
   limits;
7. compare `byte_size` and the canonical file-manifest `content_digest`;
8. enforce configured provenance fields.

Known package contracts additionally reconcile COCO 2017 image IDs,
annotation/category references and bbox/polygon/keypoint geometry, and reconcile
UCF101 class labels and split references against RAR video members while
detecting train/test overlap within each official fold.

```python
from pipeline.validate import (
    FileExpectation,
    FileValidationOperator,
    FileValidationPolicy,
)

version = repository.get_version(version_id)
report = FileValidationOperator(
    FileValidationPolicy(
        expected_files=(
            FileExpectation("train.jsonl", byte_size=123, sha256="..."),
        ),
    )
).validate(version)
```

### `DataValidationOperator`

The data operator scans every JSON/JSONL record, JSON members in ZIP files, and
WARC/WET text records in gzip streams. Its deterministic baseline checks parsing,
UTF-8, object shape, a useful
JSON Schema subset, required fields, empty text, control characters, and local
image references (decode, dimensions, EXIF orientation).

Training-quality heuristics are configurable rather than silently applying one
English-corpus recipe to every dataset. Available gates include character and
word length, alphanumeric ratio, repeated-line ratio, repeated word n-grams,
exact duplicates, and language/confidence. Language gates require an injected
detector; `FastTextLanguageDetector` is provided as an optional adapter for the
same fastText-style approach used by common curation systems.

```python
from pipeline.validate import DataValidationOperator, DataValidationPolicy

report = DataValidationOperator(
    DataValidationPolicy(
        required_fields=("text",),
        require_text=True,
        min_words=3,
        max_repeated_ngram_ratio=0.5,
    )
).validate(version)
```

`ValidationReport.summary()` is safe to store in `QualityResultSet.summary`.
Detailed findings are bounded by `max_issues`; production executors should
write the full detail stream to an immutable JSONL/Parquet artifact and record
its URI and digest in the result set.

## Industry alignment

The operator split intentionally mirrors mature training-data systems:

- Data-Juicer separates formatters, filters, mappers, and deduplicators. These
  validators report facts and quality gates; they never clean records in place.
- NVIDIA NeMo Curator exposes configurable word/character length,
  alphanumeric, repetition, and language-identification filters. Thresholds
  here are also explicit because language and task distributions differ.
- Archive validation follows the standard safe-extraction posture: inspect
  paths without extracting, prohibit traversal and links, and cap expanded
  size/member count/compression ratio.

The Minari PointMaze example shows how robot-specific semantics remain outside
the common executor. It validates `T` versus `T + 1` sequence alignment, finite
values, declared Gymnasium spaces, final terminated/truncated flags, action
bounds, goal consistency, and reward/success alignment. Other robot datasets
should add task plugins for timestamps, sensor synchronization, calibration,
control frequency, and embodiment-specific kinematics instead of placing those
rules in the generic pipeline abstraction.
