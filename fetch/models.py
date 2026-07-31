"""Declarative dataset requirements and completed metadata records."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OutputSpec:
    """Describe where a source should be stored."""

    config_key: str
    path_parts: tuple[str, ...] = ()


@dataclass(frozen=True)
class HttpFileRequest:
    """One directly downloadable HTTP file."""

    url: str
    output_name: str
    expected_bytes: int | None = None


@dataclass(frozen=True)
class HttpAcquisition:
    """A collection of files downloaded by the HTTP adapter."""

    files: tuple[HttpFileRequest, ...]
    revision: str = "default"


@dataclass(frozen=True)
class HuggingFaceAcquisition:
    """A logical dataset loaded through Hugging Face datasets."""

    repo_id: str
    config: str | None = None
    data_dir: str | None = None
    revision: str | None = None


@dataclass(frozen=True)
class CommonCrawlAcquisition:
    """A dynamically resolved WET file from Common Crawl."""

    latest: bool = True
    wet_file_index: int = 0


@dataclass(frozen=True)
class MinariAcquisition:
    """A logical offline-RL dataset loaded through Minari."""

    dataset_id: str


Acquisition = (
    HttpAcquisition
    | HuggingFaceAcquisition
    | CommonCrawlAcquisition
    | MinariAcquisition
)


@dataclass(frozen=True)
class DatasetMeta:
    """Download-time dataset metadata, independent from tool implementations."""

    source_id: str
    modality: str
    homepage: str
    license: str
    permitted_use: str
    contains_pii: str | bool
    retention_policy: str
    output: OutputSpec
    acquisition: Acquisition
    region: str = "global"
    purpose: str = "BENCHMARK"
    owner: str = "trainfoundry"
    namespace: str = "catalog"


@dataclass
class AcquisitionResult:
    """Tool-independent acquisition output used by the fetcher service."""

    files: list[Path]
    output: Path
    download_url: str | list[str]
    revision: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FileRecord:
    """Verified metadata for one local dataset file."""

    path: str
    bytes: int
    sha256: str


@dataclass
class DatasetRecord:
    """One completed dataset registry entry."""

    source_id: str
    modality: str
    homepage: str
    download_url: str | list[str]
    dataset_revision: str
    license: str
    permitted_use: str
    region: str
    contains_pii: str | bool
    retention_policy: str
    output: str
    downloaded_at: str
    files: list[FileRecord]
    total_bytes: int
    request_fingerprint: str = ""
    status: str = "complete"
    details: dict[str, Any] = field(default_factory=dict)
    metadata_dataset_id: str | None = None
    metadata_version_id: str | None = None

    @property
    def file_count(self) -> int:
        return len(self.files)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation used by the registry."""
        record = asdict(self)
        details = record.pop("details")
        record["file_count"] = self.file_count
        return {**details, **record}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DatasetRecord":
        """Restore a typed record while preserving tool-specific details."""
        field_names = {item.name for item in fields(cls)}
        ignored = {"file_count"}
        details = {
            key: item
            for key, item in value.items()
            if key not in field_names and key not in ignored
        }
        core = {
            key: item
            for key, item in value.items()
            if key in field_names and key not in {"files", "details"}
        }
        core.setdefault("request_fingerprint", "")
        return cls(
            **core,
            files=[FileRecord(**item) for item in value.get("files", [])],
            details=details,
        )


@dataclass
class FetchPlan:
    """Resolved, machine-readable plan for one dataset fetch."""

    source_id: str
    tool: str
    destination: str
    revision: str
    download_url: str | list[str]
    estimated_bytes: int | None
    action: str
    reason: str
    request_fingerprint: str
    requirements: dict[str, Any]
    acquisition: Acquisition = field(repr=False)
    existing_record: DatasetRecord | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "tool": self.tool,
            "destination": self.destination,
            "revision": self.revision,
            "download_url": self.download_url,
            "estimated_bytes": self.estimated_bytes,
            "action": self.action,
            "reason": self.reason,
            "request_fingerprint": self.request_fingerprint,
            "requirements": self.requirements,
        }
