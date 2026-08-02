"""DatasetVersion file-integrity validation.

The operator is intentionally read-only.  It streams every regular file,
validates container internals where supported, and returns a bounded quality
report without creating a new DatasetVersion.
"""

from __future__ import annotations

import gzip
import json
import math
import mimetypes
import os
import shutil
import stat
import subprocess
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, BinaryIO

from PIL import Image, UnidentifiedImageError

from metadata import DatasetRunType, DatasetVersion, DatasetVersionStage
from pipeline.base import (
    IssueSeverity,
    OperatorContext,
    OperatorInputError,
    OperatorOutput,
    ValidationReport,
    operator_fingerprint,
)
from pipeline.data import ExecutionDataset
from pipeline.validate.common import (
    IssueCollector,
    relative_location,
    unsafe_archive_member,
)
from pipeline.validate.packages import validate_known_package

_CHUNK_SIZE = 8 * 1024 * 1024
_IMAGE_FORMATS = {"JPEG", "PNG", "GIF", "WEBP", "TIFF", "BMP"}
_VIDEO_FORMATS = {"MP4", "QUICKTIME", "MATROSKA", "AVI"}
_ARCHIVE_SUFFIXES = {".zip", ".rar", ".tar", ".gz", ".tgz", ".7z"}
_GENERIC_DECLARED_FORMATS = {"FILES", "DIRECTORY", "MIXED"}


class _WarcFormatError(ValueError):
    pass


@dataclass(frozen=True)
class FileExpectation:
    """One manifest entry relative to the DatasetVersion storage root."""

    path: str
    byte_size: int | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.path or unsafe_archive_member(self.path):
            raise ValueError(f"unsafe manifest path: {self.path!r}")
        if self.byte_size is not None and self.byte_size < 0:
            raise ValueError("manifest byte_size must be non-negative")
        if self.sha256 is not None and (
            len(self.sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.sha256.lower())
        ):
            raise ValueError("manifest sha256 must contain 64 hexadecimal characters")


@dataclass(frozen=True)
class FileValidationPolicy:
    """Safety bounds and optional manifest/provenance contract."""

    expected_files: tuple[FileExpectation, ...] = ()
    reject_unexpected_files: bool = True
    allow_empty_files: bool = False
    reject_symlinks: bool = True
    verify_version_content_digest: bool = True
    max_archive_members: int = 1_000_000
    max_archive_uncompressed_bytes: int = 1 << 40
    max_archive_compression_ratio: float = 1_000.0
    max_nested_archive_depth: int = 1
    required_provenance_fields: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    max_issues: int = 10_000

    def __post_init__(self) -> None:
        if self.max_archive_members < 1:
            raise ValueError("max_archive_members must be positive")
        if self.max_archive_uncompressed_bytes < 1:
            raise ValueError("max_archive_uncompressed_bytes must be positive")
        if self.max_archive_compression_ratio <= 0:
            raise ValueError("max_archive_compression_ratio must be positive")
        if self.max_nested_archive_depth < 0:
            raise ValueError("max_nested_archive_depth must be non-negative")
        paths = [item.path for item in self.expected_files]
        if len(paths) != len(set(paths)):
            raise ValueError("expected_files contains duplicate paths")


@dataclass(frozen=True)
class _FileObservation:
    location: str
    byte_size: int
    sha256: str
    detected_format: str
    mime_type: str | None


class FileValidationOperator:
    """Validate manifests, physical bytes, real formats, and containers."""

    name = "file_validity"
    version = "1.0.0"
    run_type = DatasetRunType.QUALITY
    deterministic = True

    def __init__(self, policy: FileValidationPolicy | None = None) -> None:
        self.policy = policy or FileValidationPolicy()

    def fingerprint(self) -> str:
        return operator_fingerprint(self.name, self.version, self.policy)

    def parameters(self) -> dict[str, Any]:
        return asdict(self.policy)

    def run(
        self,
        input_data: ExecutionDataset,
        _context: OperatorContext,
    ) -> OperatorOutput:
        report = self.validate(input_data.source_version)
        return OperatorOutput(
            data=input_data,
            results=(report,),
        )

    def validate(self, input_version: DatasetVersion) -> ValidationReport:
        if input_version.stage is not DatasetVersionStage.RAW:
            raise OperatorInputError(
                f"{self.name} requires a RAW DatasetVersion, got {input_version.stage}"
            )
        root = input_version.local_path()
        issues = IssueCollector(self.policy.max_issues)
        if not root.exists() and not root.is_symlink():
            issues.add("FILE_MISSING", "storage_uri does not exist", location=str(root))
            return self._report(input_version, (), issues)

        paths = self._discover(root, issues)
        self._validate_manifest_paths(paths, root, issues)
        self._validate_provenance(issues)

        observations: list[_FileObservation] = []
        rejected_files: set[str] = set()
        for path in paths:
            location = relative_location(path, root)
            before = issues.error_count
            observation = self._inspect_file(
                path,
                location,
                input_version.storage_format.upper(),
                issues,
            )
            if observation is not None:
                observations.append(observation)
            if issues.error_count > before:
                rejected_files.add(location)

        validate_known_package(root, paths, issues)
        self._validate_manifest_values(observations, issues, rejected_files)
        self._validate_version_totals(input_version, observations, issues)
        rejected_files.update(
            issue.location
            for issue in issues.issues
            if issue.severity is IssueSeverity.ERROR and issue.location
        )
        checked = len(paths)
        rejected = min(checked, len(rejected_files))
        if issues.error_count and rejected == 0:
            rejected = 1
            checked = max(checked, 1)
        metrics = {
            "file_count": checked,
            "total_bytes": sum(item.byte_size for item in observations),
            "formats": dict(
                sorted(Counter(item.detected_format for item in observations).items())
            ),
            "manifest_entries": len(self.policy.expected_files),
        }
        passed = issues.error_count == 0
        return ValidationReport(
            dataset_version_id=input_version.version_id,
            evaluator_name=self.name,
            evaluator_version=self.version,
            passed=passed,
            checked_count=checked,
            passed_count=max(0, checked - rejected),
            rejected_count=rejected,
            error_count=issues.total_errors,
            warning_count=issues.total_warnings,
            issues=tuple(issues.issues),
            metrics=metrics,
            issue_counts=dict(issues.code_counts),
            truncated_issue_count=issues.truncated,
        )

    def _report(
        self,
        input_version: DatasetVersion,
        observations: tuple[_FileObservation, ...],
        issues: IssueCollector,
    ) -> ValidationReport:
        return ValidationReport(
            dataset_version_id=input_version.version_id,
            evaluator_name=self.name,
            evaluator_version=self.version,
            passed=False,
            checked_count=len(observations),
            passed_count=0,
            rejected_count=max(1, len(observations)),
            error_count=issues.total_errors,
            warning_count=issues.total_warnings,
            issues=tuple(issues.issues),
            issue_counts=dict(issues.code_counts),
            truncated_issue_count=issues.truncated,
        )

    def _discover(self, root: Path, issues: IssueCollector) -> list[Path]:
        if root.is_symlink():
            if self.policy.reject_symlinks:
                issues.add("SYMLINK_NOT_ALLOWED", "storage root is a symbolic link")
                return [root]
            return [root]
        if root.is_file():
            return [root]
        if not root.is_dir():
            issues.add("NOT_REGULAR_FILE", "storage root is not a file or directory")
            return []

        paths: list[Path] = []
        for current, directories, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            for name in sorted(directories):
                candidate = current_path / name
                if candidate.is_symlink():
                    paths.append(candidate)
            for name in sorted(files):
                paths.append(current_path / name)
        return sorted(paths, key=lambda item: relative_location(item, root))

    def _inspect_file(
        self,
        path: Path,
        location: str,
        declared_format: str,
        issues: IssueCollector,
    ) -> _FileObservation | None:
        try:
            metadata = path.lstat()
        except OSError as exc:
            issues.add("UNREADABLE", str(exc), location=location)
            return None
        if stat.S_ISLNK(metadata.st_mode):
            if self.policy.reject_symlinks:
                issues.add(
                    "SYMLINK_NOT_ALLOWED",
                    "symbolic links are not accepted as dataset files",
                    location=location,
                )
                return None
            try:
                metadata = path.stat()
            except OSError as exc:
                issues.add("UNREADABLE", str(exc), location=location)
                return None
        if not stat.S_ISREG(metadata.st_mode):
            issues.add("NOT_REGULAR_FILE", "path is not a regular file", location=location)
            return None
        if metadata.st_size == 0 and not self.policy.allow_empty_files:
            issues.add("EMPTY_FILE", "file is empty", location=location)

        try:
            digest, first_bytes = _stream_digest(path)
        except OSError as exc:
            issues.add("UNREADABLE", str(exc), location=location)
            return None
        after = path.stat()
        if (metadata.st_size, metadata.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            issues.add(
                "CONCURRENTLY_MODIFIED",
                "size or mtime changed while the file was read",
                location=location,
            )

        detected = _detect_format(path, first_bytes)
        mime_type = mimetypes.guess_type(path.name)[0]
        if detected == "UNKNOWN":
            issues.add(
                "UNKNOWN_FORMAT",
                "format could not be identified from magic bytes or content",
                location=location,
            )
        elif not _declared_format_matches(declared_format, detected):
            issues.add(
                "FORMAT_MISMATCH",
                f"declared {declared_format}, detected {detected}",
                location=location,
                details={"declared": declared_format, "detected": detected},
            )
        _validate_format(path, location, detected, declared_format, self.policy, issues)
        return _FileObservation(location, after.st_size, digest, detected, mime_type)

    def _validate_manifest_paths(
        self,
        paths: list[Path],
        root: Path,
        issues: IssueCollector,
    ) -> None:
        if not self.policy.expected_files:
            return
        expected = {item.path for item in self.policy.expected_files}
        actual = {relative_location(path, root) for path in paths}
        for missing in sorted(expected - actual):
            issues.add("FILE_MISSING", "manifest file is missing", location=missing)
        if self.policy.reject_unexpected_files:
            for unexpected in sorted(actual - expected):
                issues.add(
                    "UNEXPECTED_FILE",
                    "file is not declared by the manifest",
                    location=unexpected,
                )

    def _validate_manifest_values(
        self,
        observations: list[_FileObservation],
        issues: IssueCollector,
        rejected_files: set[str],
    ) -> None:
        expected = {item.path: item for item in self.policy.expected_files}
        for item in observations:
            contract = expected.get(item.location)
            if contract is None:
                continue
            if contract.byte_size is not None and contract.byte_size != item.byte_size:
                issues.add(
                    "SIZE_MISMATCH",
                    f"expected {contract.byte_size} bytes, got {item.byte_size}",
                    location=item.location,
                )
                rejected_files.add(item.location)
            if contract.sha256 is not None and contract.sha256.lower() != item.sha256:
                issues.add(
                    "DIGEST_MISMATCH",
                    "SHA-256 does not match the manifest",
                    location=item.location,
                )
                rejected_files.add(item.location)

    def _validate_version_totals(
        self,
        version: DatasetVersion,
        observations: list[_FileObservation],
        issues: IssueCollector,
    ) -> None:
        total_bytes = sum(item.byte_size for item in observations)
        if version.byte_size is not None and version.byte_size != total_bytes:
            issues.add(
                "TOTAL_SIZE_MISMATCH",
                f"DatasetVersion declares {version.byte_size} bytes, got {total_bytes}",
            )
        if self.policy.verify_version_content_digest and observations:
            actual = _manifest_digest(observations)
            if actual != version.content_digest.lower():
                issues.add(
                    "CONTENT_DIGEST_MISMATCH",
                    "DatasetVersion content_digest does not match the file manifest",
                    details={"actual": actual},
                )

    def _validate_provenance(self, issues: IssueCollector) -> None:
        for name in self.policy.required_provenance_fields:
            if self.policy.provenance.get(name) in (None, ""):
                issues.add(
                    "PROVENANCE_INCOMPLETE",
                    f"required provenance field is missing: {name}",
                    location=name,
                )


def _stream_digest(path: Path) -> tuple[str, bytes]:
    digest = sha256()
    first = b""
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            if not first:
                first = chunk[:8192]
            digest.update(chunk)
    return digest.hexdigest(), first


def _manifest_digest(observations: list[_FileObservation]) -> str:
    files = [
        {"path": item.location, "bytes": item.byte_size, "sha256": item.sha256}
        for item in sorted(observations, key=lambda value: value.location)
    ]
    encoded = json.dumps(
        {"files": files},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _detect_format(path: Path, sample: bytes) -> str:
    signatures = (
        (b"PK\x03\x04", "ZIP"),
        (b"PK\x05\x06", "ZIP"),
        (b"Rar!\x1a\x07", "RAR"),
        (b"\x1f\x8b", "GZIP"),
        (b"\x89HDF\r\n\x1a\n", "HDF5"),
        (b"PAR1", "PARQUET"),
        (b"\xff\xd8\xff", "JPEG"),
        (b"\x89PNG\r\n\x1a\n", "PNG"),
        (b"GIF87a", "GIF"),
        (b"GIF89a", "GIF"),
        (b"II*\x00", "TIFF"),
        (b"MM\x00*", "TIFF"),
        (b"BM", "BMP"),
    )
    for magic, name in signatures:
        if sample.startswith(magic):
            return name
    if sample.startswith(b"RIFF"):
        if sample[8:12] == b"WEBP":
            return "WEBP"
        if sample[8:12] == b"AVI ":
            return "AVI"
    if sample.startswith(b"\x1aE\xdf\xa3"):
        return "MATROSKA"
    if len(sample) >= 12 and sample[4:8] == b"ftyp":
        return "QUICKTIME" if sample[8:12].startswith(b"qt") else "MP4"
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "JSONL"
    if suffix == ".json":
        return "JSON"
    if suffix in {".txt", ".md", ".csv", ".tsv"}:
        return "TEXT"
    try:
        text = sample.decode("utf-8")
    except UnicodeDecodeError:
        return "UNKNOWN"
    stripped = text.lstrip("\ufeff \t\r\n")
    if stripped.startswith(("{", "[")):
        return "JSON"
    return "TEXT" if stripped else "UNKNOWN"


def _declared_format_matches(declared: str, detected: str) -> bool:
    if declared in _GENERIC_DECLARED_FORMATS:
        return True
    aliases = {
        "WARC_WET_GZIP": {"GZIP"},
        "WARC_GZIP": {"GZIP"},
        "MINARI": {"HDF5"},
        "MINARI_HDF5": {"HDF5"},
        "IMAGE": _IMAGE_FORMATS,
        "VIDEO": _VIDEO_FORMATS,
    }
    return detected in aliases.get(declared, {declared})


def _validate_format(
    path: Path,
    location: str,
    detected: str,
    declared: str,
    policy: FileValidationPolicy,
    issues: IssueCollector,
) -> None:
    try:
        if detected == "JSONL":
            _validate_jsonl(path)
        elif detected == "JSON":
            with path.open(encoding="utf-8-sig") as stream:
                json.load(stream)
        elif detected == "ZIP":
            _validate_zip(path, location, policy, issues)
        elif detected == "GZIP":
            _validate_gzip(path)
            if declared in {"WARC_WET_GZIP", "WARC_GZIP"}:
                _validate_warc_gzip(path)
        elif detected in _IMAGE_FORMATS:
            _validate_image(path)
        elif detected in _VIDEO_FORMATS:
            _validate_video(path)
        elif detected == "HDF5":
            _validate_hdf5(path)
        elif detected == "RAR":
            _validate_rar(path, location, policy, issues)
    except UnicodeDecodeError as exc:
        issues.add("INVALID_ENCODING", str(exc), location=location)
    except json.JSONDecodeError as exc:
        issues.add(
            "PARSE_ERROR",
            exc.msg,
            location=location,
            details={"line": exc.lineno, "column": exc.colno},
        )
    except (gzip.BadGzipFile, EOFError) as exc:
        issues.add("CRC_ERROR", str(exc), location=location)
    except (OSError, ValueError, UnidentifiedImageError, subprocess.SubprocessError) as exc:
        issues.add("CONTAINER_CORRUPT", str(exc), location=location)


def _validate_jsonl(path: Path) -> None:
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"line {line_number}: {exc.msg}", line, exc.pos
                ) from exc


def _validate_gzip(path: Path) -> None:
    with gzip.open(path, "rb") as stream:
        while stream.read(_CHUNK_SIZE):
            pass


def _validate_warc_gzip(path: Path) -> None:
    with gzip.open(path, "rb") as stream:
        _validate_warc_stream(stream)


def _validate_warc_stream(stream: BinaryIO) -> None:
    record_count = 0
    while True:
        line = stream.readline()
        while line in {b"\r\n", b"\n"}:
            line = stream.readline()
        if not line:
            break
        if not line.startswith(b"WARC/"):
            raise _WarcFormatError(f"invalid WARC record header: {line[:80]!r}")
        headers: dict[str, str] = {}
        while True:
            line = stream.readline()
            if not line:
                raise _WarcFormatError("truncated WARC headers")
            if line in {b"\r\n", b"\n"}:
                break
            try:
                name, value = line.decode("utf-8").split(":", 1)
            except (UnicodeDecodeError, ValueError) as exc:
                raise _WarcFormatError(
                    f"invalid WARC header: {line[:80]!r}"
                ) from exc
            headers[name.lower()] = value.strip()
        try:
            content_length = int(headers["content-length"])
        except (KeyError, ValueError) as exc:
            raise _WarcFormatError("WARC record has invalid Content-Length") from exc
        if content_length < 0:
            raise _WarcFormatError("WARC Content-Length cannot be negative")
        remaining = content_length
        while remaining:
            chunk = stream.read(min(_CHUNK_SIZE, remaining))
            if not chunk:
                raise _WarcFormatError("WARC record body is truncated")
            remaining -= len(chunk)
        record_count += 1
    if record_count == 0:
        raise _WarcFormatError("WARC archive contains no records")


def _validate_zip(
    path: Path,
    location: str,
    policy: FileValidationPolicy,
    issues: IssueCollector,
) -> None:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > policy.max_archive_members:
            issues.add(
                "ARCHIVE_BOMB",
                f"archive has {len(members)} members",
                location=location,
            )
        total_size = sum(item.file_size for item in members)
        if total_size > policy.max_archive_uncompressed_bytes:
            issues.add(
                "ARCHIVE_BOMB",
                f"uncompressed size {total_size} exceeds the configured limit",
                location=location,
            )
        seen: set[str] = set()
        for member in members:
            if member.filename in seen:
                issues.add(
                    "DUPLICATE_ENTRY",
                    f"duplicate archive member: {member.filename}",
                    location=location,
                )
            seen.add(member.filename)
            if unsafe_archive_member(member.filename):
                issues.add(
                    "PATH_TRAVERSAL",
                    f"unsafe archive member: {member.filename}",
                    location=location,
                )
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                issues.add(
                    "ARCHIVE_LINK_NOT_ALLOWED",
                    f"archive member is a symbolic link: {member.filename}",
                    location=location,
                )
            if member.flag_bits & 0x1:
                issues.add(
                    "ENCRYPTED_ENTRY",
                    f"archive member is encrypted: {member.filename}",
                    location=location,
                )
            ratio = member.file_size / max(1, member.compress_size)
            if ratio > policy.max_archive_compression_ratio:
                issues.add(
                    "ARCHIVE_BOMB",
                    f"member compression ratio {ratio:.1f} exceeds the limit",
                    location=location,
                    details={"member": member.filename},
                )
            if (
                policy.max_nested_archive_depth == 0
                and Path(member.filename).suffix.lower() in _ARCHIVE_SUFFIXES
            ):
                issues.add(
                    "NESTED_ARCHIVE",
                    f"nested archive is not allowed: {member.filename}",
                    location=location,
                )
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"CRC failed for ZIP member: {corrupt}")


def _validate_rar(
    path: Path,
    location: str,
    policy: FileValidationPolicy,
    issues: IssueCollector,
) -> None:
    bsdtar = shutil.which("bsdtar")
    if bsdtar is None:
        issues.add(
            "VALIDATOR_UNAVAILABLE",
            "RAR validation requires bsdtar or a future RAR adapter",
            location=location,
        )
        return
    listed = subprocess.run(
        [bsdtar, "-tf", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    members = [line for line in listed.stdout.splitlines() if line]
    if len(members) > policy.max_archive_members:
        issues.add(
            "ARCHIVE_BOMB",
            f"archive has {len(members)} members",
            location=location,
        )
    for member in members:
        if unsafe_archive_member(member):
            issues.add(
                "PATH_TRAVERSAL",
                f"unsafe archive member: {member}",
                location=location,
            )
    subprocess.run(
        [bsdtar, "-xOf", str(path)],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _validate_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if image.width < 1 or image.height < 1:
            raise ValueError("image has invalid dimensions")
        orientation = image.getexif().get(274)
        if orientation is not None and orientation not in range(1, 9):
            raise ValueError(f"invalid EXIF orientation: {orientation}")
        image.load()


def _validate_video(path: Path) -> None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise ValueError("video validation requires ffprobe")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=codec_type,codec_name,width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(result.stdout)
    if not payload.get("streams"):
        raise ValueError("video container has no streams")


def _validate_hdf5(path: Path) -> None:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - declared project dependency
        raise ValueError("HDF5 validation requires h5py") from exc
    with h5py.File(path, "r") as container:
        datasets = []
        container.visititems(
            lambda _name, value: datasets.append(value)
            if isinstance(value, h5py.Dataset)
            else None
        )
        for dataset in datasets:
            if dataset.shape == ():
                dataset[()]
            elif dataset.chunks is not None:
                for selection in dataset.iter_chunks():
                    dataset[selection]
            elif dataset.shape and dataset.shape[0] > 0:
                # Contiguous datasets have no chunk iterator. Read bounded
                # slices along the leading axis to exercise all stored bytes.
                trailing_items = max(1, math.prod(dataset.shape[1:]))
                item_bytes = max(1, dataset.dtype.itemsize * trailing_items)
                step = max(1, _CHUNK_SIZE // item_bytes)
                for start in range(0, dataset.shape[0], step):
                    dataset[start : start + step]


__all__ = [
    "FileExpectation",
    "FileValidationOperator",
    "FileValidationPolicy",
]
