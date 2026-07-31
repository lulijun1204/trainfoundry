"""File metadata utilities shared by acquisition tools."""

from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

from fetch.models import FileRecord


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Calculate a SHA-256 checksum without loading the file into memory."""
    digest = sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_records(paths: Iterable[Path]) -> list[FileRecord]:
    """Return verified metadata records for a collection of files."""
    records = []
    for path in sorted(paths):
        if path.is_file():
            records.append(
                FileRecord(
                    path=str(path),
                    bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                )
            )
    return records
