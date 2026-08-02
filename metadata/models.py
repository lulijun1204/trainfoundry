"""Domain models for governed training-data metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


class DatasetVersionStage(StrEnum):
    RAW = "RAW"
    PROCESSED = "PROCESSED"
    ANNOTATED = "ANNOTATED"
    TRAINING_READY = "TRAINING_READY"


class DatasetVersionStatus(StrEnum):
    DRAFT = "DRAFT"
    COMMITTED = "COMMITTED"
    PUBLISHED = "PUBLISHED"
    DEPRECATED = "DEPRECATED"
    INVALIDATED = "INVALIDATED"


class SchemaFormat(StrEnum):
    ARROW = "ARROW"
    JSON_SCHEMA = "JSON_SCHEMA"


class DatasetSplit(StrEnum):
    TRAIN = "TRAIN"
    VALID = "VALID"
    TEST = "TEST"


class DatasetRunType(StrEnum):
    INGEST = "INGEST"
    CLEAN = "CLEAN"
    NORMALIZE = "NORMALIZE"
    DEDUP = "DEDUP"
    FILTER = "FILTER"
    CONVERT = "CONVERT"
    SCHEMA_MAPPING = "SCHEMA_MAPPING"
    MATERIALIZE_ANNOTATION = "MATERIALIZE_ANNOTATION"
    SPLIT = "SPLIT"
    MERGE = "MERGE"
    QUALITY = "QUALITY"
    ANNOTATION = "ANNOTATION"


class OutputMode(StrEnum):
    NEW_VERSION = "NEW_VERSION"
    NEW_DATASET = "NEW_DATASET"
    NO_DATA_VERSION = "NO_DATA_VERSION"


class DatasetRunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class DatasetVersion:
    """Complete domain representation of one ``dataset_versions`` row.

    Repository updates return a new instance.  The object itself is frozen so
    pipeline code cannot accidentally mutate metadata independently of SQLite.
    """

    version_id: str
    dataset_id: str
    version_number: int
    stage: DatasetVersionStage
    status: DatasetVersionStatus
    storage_uri: str
    storage_format: str
    schema_format: SchemaFormat
    schema_definition: dict[str, Any]
    schema_version: str | None
    schema_digest: str
    content_digest: str
    split: DatasetSplit | None
    usage_tags: tuple[str, ...] | None
    row_count: int | None
    byte_size: int | None
    created_by: str
    created_at: str
    committed_at: str | None

    def local_path(self) -> Path:
        """Resolve local storage while rejecting unsupported remote schemes."""
        parsed = urlparse(self.storage_uri)
        if parsed.scheme not in {"", "file"}:
            raise ValueError(
                f"unsupported storage URI scheme {parsed.scheme!r}; "
                "install an object-store reader adapter"
            )
        if parsed.scheme == "file" and parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"unsupported file URI authority: {parsed.netloc!r}")
        raw_path = unquote(parsed.path) if parsed.scheme else self.storage_uri
        if not raw_path:
            raise ValueError("storage_uri does not contain a path")
        return Path(raw_path)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible representation used by the CLI."""
        value = asdict(self)
        value["stage"] = self.stage.value
        value["status"] = self.status.value
        value["schema_format"] = self.schema_format.value
        value["split"] = self.split.value if self.split is not None else None
        value["usage_tags"] = (
            list(self.usage_tags) if self.usage_tags is not None else None
        )
        return value


@dataclass(frozen=True, slots=True)
class DatasetRun:
    """Complete domain representation of one ``dataset_runs`` row."""

    run_id: str
    run_type: DatasetRunType
    operator_name: str
    operator_version: str
    operator_fingerprint: str
    params: dict[str, Any]
    compute_key: str
    deterministic: bool
    output_mode: OutputMode
    target_dataset_id: str | None
    status: DatasetRunStatus
    started_at: str | None
    finished_at: str | None
    error_message: str | None
    created_by: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""
        value = asdict(self)
        value["run_type"] = self.run_type.value
        value["output_mode"] = self.output_mode.value
        value["status"] = self.status.value
        return value


__all__ = [
    "DatasetSplit",
    "DatasetRun",
    "DatasetRunStatus",
    "DatasetRunType",
    "DatasetVersion",
    "DatasetVersionStage",
    "DatasetVersionStatus",
    "OutputMode",
    "SchemaFormat",
]
