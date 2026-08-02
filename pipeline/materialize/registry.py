"""Storage-format routing for Materializers."""

from __future__ import annotations

from collections.abc import Iterable

from pipeline.materialize.base import (
    Materializer,
    UnsupportedMaterializationFormatError,
)
from pipeline.materialize.writers import LanceMaterializer


class MaterializerRegistry:
    def __init__(self, materializers: Iterable[Materializer] = ()) -> None:
        self._materializers: dict[str, Materializer] = {}
        for materializer in materializers:
            self.register(materializer)

    @classmethod
    def default(cls) -> MaterializerRegistry:
        return cls((LanceMaterializer(),))

    def register(self, materializer: Materializer) -> None:
        for storage_format in materializer.formats:
            self._materializers[storage_format.upper()] = materializer

    def get(self, storage_format: str) -> Materializer:
        normalized = storage_format.upper()
        try:
            return self._materializers[normalized]
        except KeyError as exc:
            raise UnsupportedMaterializationFormatError(
                f"no Materializer registered for {normalized}"
            ) from exc
