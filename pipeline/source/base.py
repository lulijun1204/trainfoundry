"""Source adapters turn persisted versions into ephemeral execution data."""

from __future__ import annotations

from typing import Protocol

from metadata import DatasetVersion
from pipeline.data import ExecutionDataset


class SourceAdapter(Protocol):
    formats: frozenset[str]

    def open(self, version: DatasetVersion) -> ExecutionDataset:
        """Open one persisted version without creating another version."""


class UnsupportedSourceFormatError(ValueError):
    """Raised when no source adapter understands a storage format."""
