"""Shared constants and result types for the Minari learning example."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "model_data/robot/D4RL/pointmaze/umaze-v2"
DATA_DIRECTORY = "data"
HDF5_FILE = "main_data.hdf5"
METADATA_FILE = "metadata.json"
EXPECTED_FILES = (HDF5_FILE, METADATA_FILE)
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    source_file: str
    location: str
    code: str
    severity: Literal["ERROR", "WARNING"]
    message: str


class IssueCollector:
    """Count every issue while retaining bounded, representative details."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("max_issue_samples must be positive")
        self.limit = limit
        self.error_counts: Counter[str] = Counter()
        self.warning_counts: Counter[str] = Counter()
        self.samples: list[ValidationIssue] = []

    def add(
        self,
        source: str,
        location: str,
        code: str,
        message: str,
        *,
        severity: Literal["ERROR", "WARNING"] = "ERROR",
    ) -> None:
        counts = self.error_counts if severity == "ERROR" else self.warning_counts
        counts[code] += 1
        sampled_codes = {(item.severity, item.code) for item in self.samples}
        if len(self.samples) < self.limit and (severity, code) not in sampled_codes:
            self.samples.append(
                ValidationIssue(source, location, code, severity, message)
            )

    @property
    def error_total(self) -> int:
        return sum(self.error_counts.values())


def digest_and_magic(path: Path, *, magic_length: int = 8) -> tuple[str, bytes]:
    """Read a file once to calculate its identity and inspect its real format."""
    digest = sha256()
    magic = b""
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            if not magic:
                magic = chunk[:magic_length]
            digest.update(chunk)
    return digest.hexdigest(), magic


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
