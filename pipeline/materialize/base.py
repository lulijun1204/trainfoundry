"""Contracts for promoting ephemeral execution data to persisted storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pyarrow as pa

from pipeline.data import ExecutionDataset


@dataclass(frozen=True, slots=True)
class MaterializationSpec:
    storage_uri: str
    stage: str
    created_by: str
    storage_format: str = "LANCE"
    dataset_id: str | None = None
    status: str = "COMMITTED"
    schema_format: str | None = None
    schema_definition: dict[str, Any] | None = None
    schema_version: str | None = None
    split: str | None = None
    usage_tags: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class MaterializedData:
    storage_uri: str
    storage_format: str
    content_digest: str
    row_count: int | None
    byte_size: int | None
    schema: pa.Schema | None = None


class Materializer(Protocol):
    formats: frozenset[str]
    name: str
    version: str

    def write(
        self,
        data: ExecutionDataset,
        spec: MaterializationSpec,
    ) -> MaterializedData:
        """Write physical data without registering relational metadata."""

    def fingerprint(self) -> str:
        """Return the implementation fingerprint used by DatasetRun."""


class UnsupportedMaterializationFormatError(ValueError):
    """Raised when no Materializer supports an output format."""
