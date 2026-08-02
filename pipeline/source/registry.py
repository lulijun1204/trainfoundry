"""Storage-format routing for SourceAdapters."""

from __future__ import annotations

from collections.abc import Iterable

from metadata import DatasetVersion
from pipeline.data import ExecutionDataset
from pipeline.source.adapters import (
    FileSourceAdapter,
    JsonSourceAdapter,
    LanceSourceAdapter,
)
from pipeline.source.base import SourceAdapter, UnsupportedSourceFormatError


class SourceAdapterRegistry:
    def __init__(self, adapters: Iterable[SourceAdapter] = ()) -> None:
        self._adapters: dict[str, SourceAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    @classmethod
    def default(cls) -> SourceAdapterRegistry:
        return cls((JsonSourceAdapter(), FileSourceAdapter(), LanceSourceAdapter()))

    def register(self, adapter: SourceAdapter) -> None:
        for storage_format in adapter.formats:
            self._adapters[storage_format.upper()] = adapter

    def open(self, version: DatasetVersion) -> ExecutionDataset:
        storage_format = version.storage_format.upper()
        try:
            adapter = self._adapters[storage_format]
        except KeyError as exc:
            raise UnsupportedSourceFormatError(
                f"no SourceAdapter registered for {storage_format}"
            ) from exc
        return adapter.open(version)
