"""Built-in local source adapters."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa

from metadata import DatasetVersion
from pipeline.data import BlockStream, ExecutionDataset

_FILE_REFERENCE_SCHEMA = pa.schema(
    [
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("archive_member", pa.string()),
        pa.field("location", pa.string(), nullable=False),
        pa.field("byte_size", pa.int64(), nullable=False),
    ]
)


class JsonSourceAdapter:
    formats = frozenset({"JSON", "JSONL"})

    def __init__(self, batch_size: int = 1024) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size

    def open(self, version: DatasetVersion) -> ExecutionDataset:
        root = version.local_path()
        stream = BlockStream(
            lambda: self._batches(root, version.storage_format.upper())
        )
        return ExecutionDataset(
            source_version=version,
            batches=stream,
            execution_digest=version.content_digest,
        )

    def _batches(
        self,
        root: Path,
        storage_format: str,
    ) -> Iterator[pa.RecordBatch]:
        files = _matching_files(root, storage_format)
        batch: list[Mapping[str, Any]] = []
        for path in files:
            for record in _records(path, storage_format):
                batch.append(record)
                if len(batch) >= self.batch_size:
                    yield pa.RecordBatch.from_pylist(batch)
                    batch = []
        if batch:
            yield pa.RecordBatch.from_pylist(batch)


class FileSourceAdapter:
    formats = frozenset(
        {
            "FILES",
            "DIRECTORY",
            "MIXED",
            "ZIP",
            "WARC_GZIP",
            "WARC_WET_GZIP",
            "MINARI",
            "HDF5",
        }
    )

    def __init__(self, batch_size: int = 1024) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size

    def open(self, version: DatasetVersion) -> ExecutionDataset:
        root = version.local_path()
        stream = BlockStream(
            lambda: self._batches(root, version.storage_format.upper()),
            schema=_FILE_REFERENCE_SCHEMA,
        )
        return ExecutionDataset(
            source_version=version,
            batches=stream,
            schema=_FILE_REFERENCE_SCHEMA,
            execution_digest=version.content_digest,
        )

    def _batches(
        self,
        root: Path,
        storage_format: str,
    ) -> Iterator[pa.RecordBatch]:
        rows: list[dict[str, Any]] = []
        if storage_format == "ZIP" or root.suffix.lower() == ".zip":
            with zipfile.ZipFile(root) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    rows.append(
                        {
                            "source_path": str(root),
                            "archive_member": member.filename,
                            "location": member.filename,
                            "byte_size": member.file_size,
                        }
                    )
                    if len(rows) >= self.batch_size:
                        yield pa.RecordBatch.from_pylist(
                            rows,
                            schema=_FILE_REFERENCE_SCHEMA,
                        )
                        rows = []
            if rows:
                yield pa.RecordBatch.from_pylist(
                    rows,
                    schema=_FILE_REFERENCE_SCHEMA,
                )
            return
        paths = [root] if root.is_file() else sorted(
            path for path in root.rglob("*") if path.is_file()
        )
        for path in paths:
            location = path.name if root.is_file() else path.relative_to(root).as_posix()
            rows.append(
                {
                    "source_path": str(path),
                    "archive_member": None,
                    "location": location,
                    "byte_size": path.stat().st_size,
                }
            )
            if len(rows) >= self.batch_size:
                yield pa.RecordBatch.from_pylist(
                    rows,
                    schema=_FILE_REFERENCE_SCHEMA,
                )
                rows = []
        if rows:
            yield pa.RecordBatch.from_pylist(rows, schema=_FILE_REFERENCE_SCHEMA)


class LanceSourceAdapter:
    formats = frozenset({"LANCE"})

    def open(self, version: DatasetVersion) -> ExecutionDataset:
        try:
            import lance
        except ImportError as exc:
            raise RuntimeError(
                "Lance input requires the optional 'pylance' package"
            ) from exc
        root = version.local_path()
        dataset = lance.dataset(root)

        def batches() -> Iterator[pa.RecordBatch]:
            yield from dataset.scanner().to_batches()

        return ExecutionDataset(
            source_version=version,
            batches=BlockStream(batches, schema=dataset.schema),
            schema=dataset.schema,
            execution_digest=version.content_digest,
        )


def _matching_files(root: Path, storage_format: str) -> list[Path]:
    suffix = ".jsonl" if storage_format == "JSONL" else ".json"
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob(f"*{suffix}") if path.is_file())


def _records(path: Path, storage_format: str) -> Iterator[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        if storage_format == "JSONL":
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"{path}:{line_number} must contain an object")
                yield value
            return
        value = json.load(stream)
    values = value if isinstance(value, list) else [value]
    for index, item in enumerate(values):
        if not isinstance(item, Mapping):
            raise ValueError(f"{path}: record {index} must contain an object")
        yield item
