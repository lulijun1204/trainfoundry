"""Atomic persistence for completed dataset metadata records."""

import json
from pathlib import Path
from typing import Any

from config import get_path
from fetch.artifacts import sha256_file
from fetch.models import DatasetRecord


class DatasetRegistry:
    """Read and upsert one record per dataset source."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path

    def read(self) -> list[dict[str, Any]]:
        path = self._path()
        if not path.exists():
            return []

        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Registry must contain a JSON array: {path}")
        return data

    def find(self, source_id: str) -> dict[str, Any] | None:
        return next(
            (record for record in self.read() if record.get("source_id") == source_id),
            None,
        )

    def upsert(self, record: DatasetRecord) -> Path:
        """Insert or replace a source record without creating duplicates."""
        records = self.read()
        serialized = record.to_dict()

        for index, existing in enumerate(records):
            if existing.get("source_id") == record.source_id:
                records[index] = serialized
                break
        else:
            records.append(serialized)

        records.sort(key=lambda item: str(item.get("source_id", "")))
        return self._write(records)

    def _write(self, records: list[dict[str, Any]]) -> Path:
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def _path(self) -> Path:
        return self.path or get_path("paths.registry_path")


class RegistryVerificationError(ValueError):
    """Raised when a persisted dataset record no longer matches local files."""


def verify_record(record: dict[str, Any]) -> None:
    """Verify the files, sizes, and checksums in one serialized record."""
    if record.get("status") != "complete":
        raise RegistryVerificationError("record status is not complete")
    files = record.get("files")
    if not isinstance(files, list):
        raise RegistryVerificationError("record files must be a list")
    if record.get("file_count") != len(files):
        raise RegistryVerificationError("file_count does not match files")

    actual_total_bytes = 0
    for file_record in files:
        path = Path(file_record["path"])
        if not path.is_file():
            raise RegistryVerificationError(f"file does not exist: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != file_record["bytes"]:
            raise RegistryVerificationError(
                f"size mismatch for {path}: {actual_bytes} != {file_record['bytes']}"
            )
        if sha256_file(path) != file_record["sha256"]:
            raise RegistryVerificationError(f"SHA-256 mismatch for {path}")
        actual_total_bytes += actual_bytes

    if actual_total_bytes != record.get("total_bytes"):
        raise RegistryVerificationError(
            f"total_bytes mismatch: {actual_total_bytes} != {record.get('total_bytes')}"
        )
