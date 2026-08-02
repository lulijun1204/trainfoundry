"""Ephemeral Arrow data exchanged between pipeline operators."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from uuid import uuid4

import pyarrow as pa

from metadata import DatasetVersion


@dataclass(frozen=True, slots=True)
class BlockStream:
    """Re-openable, lazy stream whose only physical unit is RecordBatch."""

    factory: Callable[[], Iterator[pa.RecordBatch]]
    schema: pa.Schema | None = None

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        expected_schema = self.schema
        for batch in self.factory():
            if not isinstance(batch, pa.RecordBatch):
                raise TypeError(
                    "BlockStream factories must yield pyarrow.RecordBatch, "
                    f"got {type(batch).__qualname__}"
                )
            if expected_schema is None:
                expected_schema = batch.schema
            elif not batch.schema.equals(expected_schema):
                raise TypeError(
                    "BlockStream yielded incompatible Arrow schemas: "
                    f"expected {expected_schema}, got {batch.schema}"
                )
            yield batch


@dataclass(frozen=True, slots=True)
class ExecutionDataset:
    """Non-persistent Arrow dataset passed between PipelineOperators."""

    source_version: DatasetVersion
    batches: BlockStream
    schema: pa.Schema | None = None
    execution_data_id: str = field(
        default_factory=lambda: f"data_{uuid4().hex}"
    )
    execution_digest: str | None = None

    def __post_init__(self) -> None:
        if (
            self.schema is not None
            and self.batches.schema is not None
            and not self.schema.equals(self.batches.schema)
        ):
            raise ValueError("ExecutionDataset schema must match BlockStream schema")

    def derive(
        self,
        *,
        batches: BlockStream | None = None,
        schema: pa.Schema | None = None,
        execution_digest: str | None = None,
    ) -> ExecutionDataset:
        """Create another ephemeral Arrow value without materializing a version."""
        output_batches = batches or self.batches
        output_schema = schema
        if output_schema is None:
            output_schema = output_batches.schema or self.schema
        return ExecutionDataset(
            source_version=self.source_version,
            batches=output_batches,
            schema=output_schema,
            execution_digest=execution_digest,
        )

    def with_execution_digest(self, digest: str) -> ExecutionDataset:
        return replace(self, execution_digest=digest)


__all__ = ["BlockStream", "ExecutionDataset"]
