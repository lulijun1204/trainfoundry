"""Concrete Common Crawl WET file-format and record validation.

The implementation stays independent from PipelineOperator so learners can see
the physical gzip/WARC reads and every record-level decision directly.
"""

from __future__ import annotations

import gzip
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "model_data/text/common_crawl"
_GZIP_MAGIC = b"\x1f\x8b"
_READ_CHUNK_SIZE = 8 * 1024 * 1024
_SUPPORTED_WARC_VERSIONS = {"WARC/1.0", "WARC/1.1"}


@dataclass(frozen=True, slots=True)
class RawWarcRecord:
    source_file: str
    record_index: int
    version: str | None
    headers: dict[str, str]
    body: bytes
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class FileFormatIssue:
    source_file: str
    code: str
    message: str
    record_index: int | None = None


@dataclass(frozen=True, slots=True)
class FileObservation:
    source_file: str
    compressed_bytes: int
    sha256: str
    warc_records: int
    record_types: dict[str, int]


@dataclass(frozen=True, slots=True)
class FileFormatSummary:
    total_files: int
    valid_files: int
    invalid_files: int
    total_compressed_bytes: int
    total_warc_records: int
    record_type_counts: dict[str, int]
    issue_counts: dict[str, int]
    files: tuple[FileObservation, ...]
    issue_samples: tuple[FileFormatIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    source_file: str
    record_index: int
    code: str
    message: str
    url: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    total_records: int
    valid_records: int
    rejected_records: int
    skipped_records: int
    record_type_counts: dict[str, int]
    issue_counts: dict[str, int]
    issue_samples: tuple[ValidationIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_source_files(dataset_root: Path) -> tuple[Path, ...]:
    """Return every physical input file so unexpected formats are visible."""
    if dataset_root.is_file():
        return (dataset_root,)
    if not dataset_root.is_dir():
        return ()
    return tuple(sorted(path for path in dataset_root.rglob("*") if path.is_file()))


def iter_warc_records(path: Path) -> Iterator[RawWarcRecord]:
    """Stream WARC records from gzip without extracting the archive to disk."""
    source_file = path.name
    record_index = 0
    try:
        # gzip 流被持续读取到 EOF；除了解压数据，这也会触发 gzip CRC/长度校验。
        with gzip.open(path, "rb") as stream:
            while True:
                line = stream.readline()
                while line in {b"\r\n", b"\n"}:
                    line = stream.readline()
                if not line:
                    return

                record_index += 1
                version_bytes = line.rstrip(b"\r\n")
                try:
                    version = version_bytes.decode("ascii")
                except UnicodeDecodeError as exc:
                    yield _parse_error(
                        source_file,
                        record_index,
                        "INVALID_WARC_HEADER",
                        f"WARC version line is not ASCII: {exc}",
                    )
                    return
                if version not in _SUPPORTED_WARC_VERSIONS:
                    yield _parse_error(
                        source_file,
                        record_index,
                        "INVALID_WARC_VERSION",
                        f"unsupported WARC version line: {version!r}",
                    )
                    return

                # WARC 头以空行结束。字段名统一成小写，消除协议字段大小写差异。
                headers: dict[str, str] = {}
                while True:
                    line = stream.readline()
                    if not line:
                        yield _parse_error(
                            source_file,
                            record_index,
                            "TRUNCATED_WARC_HEADERS",
                            "WARC headers ended before the blank separator line",
                        )
                        return
                    if line in {b"\r\n", b"\n"}:
                        break
                    try:
                        name, value = line.decode("utf-8").split(":", 1)
                    except (UnicodeDecodeError, ValueError) as exc:
                        yield _parse_error(
                            source_file,
                            record_index,
                            "INVALID_WARC_HEADER",
                            str(exc),
                        )
                        return
                    normalized_name = name.strip().lower()
                    if not normalized_name:
                        yield _parse_error(
                            source_file,
                            record_index,
                            "INVALID_WARC_HEADER",
                            "WARC header name is empty",
                        )
                        return
                    headers[normalized_name] = value.strip()

                try:
                    content_length = int(headers["content-length"])
                except (KeyError, ValueError):
                    yield _parse_error(
                        source_file,
                        record_index,
                        "INVALID_CONTENT_LENGTH",
                        "Content-Length is missing or is not an integer",
                    )
                    return
                if content_length < 0:
                    yield _parse_error(
                        source_file,
                        record_index,
                        "INVALID_CONTENT_LENGTH",
                        "Content-Length cannot be negative",
                    )
                    return

                # WARC 没有依赖正文分隔符，而是依靠 Content-Length 切分 record。
                # 必须精确读取该字节数，否则下一条 record 的边界会整体错位。
                body = _read_exact(stream, content_length)
                if len(body) != content_length:
                    yield _parse_error(
                        source_file,
                        record_index,
                        "TRUNCATED_WARC_BODY",
                        f"expected {content_length} body bytes, got {len(body)}",
                    )
                    return
                yield RawWarcRecord(
                    source_file=source_file,
                    record_index=record_index,
                    version=version,
                    headers=headers,
                    body=body,
                )
    except (gzip.BadGzipFile, EOFError, OSError) as exc:
        yield _parse_error(
            source_file,
            max(1, record_index),
            "GZIP_CORRUPT",
            str(exc),
        )


def validate_file_format(
    dataset_root: Path,
    *,
    max_issue_samples: int = 10,
) -> FileFormatSummary:
    """Validate file identity, gzip integrity, and WARC record boundaries."""
    # Step 01 只关心物理文件是否可信：身份、摘要、容器完整性和 record 边界。
    # 正文是否适合训练留给 Step 02。
    paths = discover_source_files(dataset_root)
    issue_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    issue_samples: list[FileFormatIssue] = []
    observations: list[FileObservation] = []
    invalid_files = 0
    total_records = 0
    total_bytes = 0

    if not paths:
        issue = FileFormatIssue(str(dataset_root), "NO_FILES", "no input files found")
        _collect_file_issue(issue, issue_counts, issue_samples, max_issue_samples)

    for path in paths:
        # before 用来判断当前文件是否新增问题，从而按“文件”统计有效/无效。
        before = sum(issue_counts.values())
        file_size = path.stat().st_size
        total_bytes += file_size
        digest, magic = _digest_and_magic(path)
        file_types: Counter[str] = Counter()
        file_records = 0

        if not path.name.endswith(".warc.wet.gz"):
            _collect_file_issue(
                FileFormatIssue(
                    path.name,
                    "INVALID_FILE_SUFFIX",
                    "Common Crawl WET input must end with .warc.wet.gz",
                ),
                issue_counts,
                issue_samples,
                max_issue_samples,
            )
        if file_size == 0:
            _collect_file_issue(
                FileFormatIssue(path.name, "EMPTY_FILE", "input file is empty"),
                issue_counts,
                issue_samples,
                max_issue_samples,
            )
        elif magic != _GZIP_MAGIC:
            _collect_file_issue(
                FileFormatIssue(
                    path.name,
                    "INVALID_GZIP_MAGIC",
                    f"expected gzip magic {_GZIP_MAGIC!r}, got {magic!r}",
                ),
                issue_counts,
                issue_samples,
                max_issue_samples,
            )
        else:
            for record in iter_warc_records(path):
                if record.error_code is not None:
                    _collect_file_issue(
                        FileFormatIssue(
                            path.name,
                            record.error_code,
                            record.error_message or record.error_code,
                            record.record_index,
                        ),
                        issue_counts,
                        issue_samples,
                        max_issue_samples,
                    )
                    continue
                file_records += 1
                record_type = record.headers.get("warc-type")
                if not record_type:
                    _collect_file_issue(
                        FileFormatIssue(
                            path.name,
                            "MISSING_WARC_TYPE",
                            "WARC-Type header is required",
                            record.record_index,
                        ),
                        issue_counts,
                        issue_samples,
                        max_issue_samples,
                    )
                    record_type = "<missing>"
                file_types[record_type] += 1
                type_counts[record_type] += 1

        if file_records == 0 and magic == _GZIP_MAGIC:
            _collect_file_issue(
                FileFormatIssue(
                    path.name,
                    "NO_WARC_RECORDS",
                    "gzip stream contains no complete WARC records",
                ),
                issue_counts,
                issue_samples,
                max_issue_samples,
            )
        if sum(issue_counts.values()) > before:
            invalid_files += 1
        total_records += file_records
        observations.append(
            FileObservation(
                source_file=path.name,
                compressed_bytes=file_size,
                sha256=digest,
                warc_records=file_records,
                record_types=dict(sorted(file_types.items())),
            )
        )

    return FileFormatSummary(
        total_files=len(paths),
        valid_files=len(paths) - invalid_files,
        invalid_files=invalid_files,
        total_compressed_bytes=total_bytes,
        total_warc_records=total_records,
        record_type_counts=dict(sorted(type_counts.items())),
        issue_counts=dict(sorted(issue_counts.items())),
        files=tuple(observations),
        issue_samples=tuple(issue_samples),
    )


def validate_record(record: RawWarcRecord) -> ValidationIssue | None:
    """Apply the minimal validity contract for one WET conversion record."""
    if record.error_code is not None:
        return ValidationIssue(
            record.source_file,
            record.record_index,
            record.error_code,
            record.error_message or record.error_code,
        )

    # URL、Record-ID、时间共同构成网页正文的来源追踪信息。
    url = record.headers.get("warc-target-uri")
    if not url:
        return _record_issue(record, "MISSING_TARGET_URI", "WARC-Target-URI is required")
    parsed_url = urlsplit(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        return _record_issue(
            record,
            "INVALID_TARGET_URI",
            "WARC-Target-URI must be an absolute HTTP(S) URL",
            url,
        )
    if not record.headers.get("warc-record-id"):
        return _record_issue(
            record,
            "MISSING_RECORD_ID",
            "WARC-Record-ID is required for source traceability",
            url,
        )

    warc_date = record.headers.get("warc-date")
    if not warc_date:
        return _record_issue(record, "MISSING_WARC_DATE", "WARC-Date is required", url)
    try:
        parsed_date = datetime.fromisoformat(warc_date.replace("Z", "+00:00"))
    except ValueError:
        return _record_issue(
            record,
            "INVALID_WARC_DATE",
            "WARC-Date must be an ISO-8601 timestamp",
            url,
        )
    if parsed_date.tzinfo is None:
        return _record_issue(
            record,
            "INVALID_WARC_DATE",
            "WARC-Date must include a timezone",
            url,
        )

    content_type = record.headers.get("content-type", "")
    media_type = content_type.partition(";")[0].strip().lower()
    if media_type != "text/plain":
        return _record_issue(
            record,
            "INVALID_CONTENT_TYPE",
            f"WET conversion content must be text/plain, got {content_type!r}",
            url,
        )
    # WET conversion 正文按规范应是 text/plain；严格 UTF-8 解码，禁止静默替换坏字节。
    try:
        text = record.body.decode("utf-8")
    except UnicodeDecodeError as exc:
        return _record_issue(record, "INVALID_UTF8", str(exc), url)
    if not text.strip():
        return _record_issue(record, "EMPTY_TEXT", "conversion text is empty", url)

    invalid_controls = {
        character
        for character in text
        if unicodedata.category(character) == "Cc" and character not in "\n\r\t"
    }
    if invalid_controls:
        codepoints = ", ".join(
            f"U+{ord(character):04X}" for character in sorted(invalid_controls)
        )
        return _record_issue(
            record,
            "CONTROL_CHARACTER",
            f"text contains disallowed control characters: {codepoints}",
            url,
        )
    return None


def validate_dataset(
    dataset_root: Path,
    *,
    max_issue_samples: int = 10,
) -> ValidationSummary:
    """Validate every WET record while skipping non-conversion metadata records."""
    counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    samples: list[ValidationIssue] = []
    total = 0
    valid = 0
    rejected = 0
    skipped = 0

    paths = discover_source_files(dataset_root)
    if not paths:
        issue = ValidationIssue(
            str(dataset_root), 0, "NO_FILES", "no input files found"
        )
        _collect_validation_issue(issue, counts, samples, max_issue_samples)

    for path in paths:
        for record in iter_warc_records(path):
            total += 1
            if record.error_code is not None:
                rejected += 1
                issue = validate_record(record)
                assert issue is not None
                _collect_validation_issue(issue, counts, samples, max_issue_samples)
                continue

            record_type = record.headers.get("warc-type", "<missing>")
            type_counts[record_type] += 1
            # warcinfo 等 record 是容器元数据，不是训练文本：计入总数但单独 skipped。
            if record_type != "conversion":
                skipped += 1
                continue

            issue = validate_record(record)
            if issue is None:
                valid += 1
            else:
                rejected += 1
                _collect_validation_issue(issue, counts, samples, max_issue_samples)

    return ValidationSummary(
        total_records=total,
        valid_records=valid,
        rejected_records=rejected,
        skipped_records=skipped,
        record_type_counts=dict(sorted(type_counts.items())),
        issue_counts=dict(sorted(counts.items())),
        issue_samples=tuple(samples),
    )


def _read_exact(stream: BinaryIO, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = stream.read(min(_READ_CHUNK_SIZE, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _digest_and_magic(path: Path) -> tuple[str, bytes]:
    digest = sha256()
    magic = b""
    with path.open("rb") as stream:
        while chunk := stream.read(_READ_CHUNK_SIZE):
            if not magic:
                magic = chunk[:2]
            digest.update(chunk)
    return digest.hexdigest(), magic


def _parse_error(
    source_file: str,
    record_index: int,
    code: str,
    message: str,
) -> RawWarcRecord:
    return RawWarcRecord(
        source_file=source_file,
        record_index=record_index,
        version=None,
        headers={},
        body=b"",
        error_code=code,
        error_message=message,
    )


def _record_issue(
    record: RawWarcRecord,
    code: str,
    message: str,
    url: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        record.source_file,
        record.record_index,
        code,
        message,
        url,
    )


def _collect_file_issue(
    issue: FileFormatIssue,
    counts: Counter[str],
    samples: list[FileFormatIssue],
    limit: int,
) -> None:
    counts[issue.code] += 1
    sampled_codes = {sample.code for sample in samples}
    if len(samples) < limit and issue.code not in sampled_codes:
        samples.append(issue)


def _collect_validation_issue(
    issue: ValidationIssue,
    counts: Counter[str],
    samples: list[ValidationIssue],
    limit: int,
) -> None:
    counts[issue.code] += 1
    sampled_codes = {sample.code for sample in samples}
    if len(samples) < limit and issue.code not in sampled_codes:
        samples.append(issue)
