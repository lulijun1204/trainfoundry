"""Record-level validity and lightweight training-data quality checks."""

from __future__ import annotations

import gzip
import io
import json
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

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
from pipeline.validate.common import IssueCollector, relative_location

_DEFAULT_TEXT_FIELDS = (
    "text",
    "content",
    "instruction",
    "context",
    "response",
    "chosen",
    "rejected",
    "prompt",
    "completion",
    "messages.*.content",
)
_DEFAULT_IMAGE_FIELDS = ("image", "image_path", "images.*", "images.*.path")
_SUPPORTED_FORMATS = {
    "JSON",
    "JSONL",
    "FILES",
    "DIRECTORY",
    "MIXED",
    "WARC_GZIP",
    "WARC_WET_GZIP",
}


@dataclass(frozen=True)
class LanguagePrediction:
    language: str
    confidence: float


class LanguageDetector(Protocol):
    def detect(self, text: str) -> LanguagePrediction:
        """Return an ISO-style language label and confidence in [0, 1]."""


class FastTextLanguageDetector:
    """Optional adapter matching NeMo Curator/Data-Juicer language ID practice."""

    def __init__(self, model_path: str | Path) -> None:
        try:
            import fasttext
        except ImportError as exc:
            raise RuntimeError(
                "FastText language detection requires the optional fasttext package"
            ) from exc
        self._model = fasttext.load_model(str(model_path))

    def detect(self, text: str) -> LanguagePrediction:
        labels, scores = self._model.predict(text.replace("\n", " "), k=1)
        language = labels[0].removeprefix("__label__")
        return LanguagePrediction(language, float(scores[0]))


@dataclass(frozen=True)
class DataValidationPolicy:
    """Deterministic checks plus opt-in corpus-specific quality thresholds."""

    required_fields: tuple[str, ...] = ()
    non_empty_text_fields: tuple[str, ...] = ()
    text_fields: tuple[str, ...] = _DEFAULT_TEXT_FIELDS
    image_fields: tuple[str, ...] = _DEFAULT_IMAGE_FIELDS
    require_text: bool = False
    min_text_chars: int = 1
    max_text_chars: int | None = None
    min_words: int | None = None
    max_words: int | None = None
    min_alphanumeric_ratio: float | None = None
    max_control_character_ratio: float = 0.0
    max_repeated_line_ratio: float | None = None
    repeated_ngram_size: int = 3
    max_repeated_ngram_ratio: float | None = None
    expected_languages: tuple[str, ...] = ()
    min_language_confidence: float = 0.3
    check_exact_duplicates: bool = False
    min_image_width: int = 1
    min_image_height: int = 1
    max_rejected_ratio: float = 0.0
    max_issues: int = 10_000

    def __post_init__(self) -> None:
        if self.min_text_chars < 0:
            raise ValueError("min_text_chars must be non-negative")
        if self.max_text_chars is not None and self.max_text_chars < self.min_text_chars:
            raise ValueError("max_text_chars must be >= min_text_chars")
        for name in (
            "min_alphanumeric_ratio",
            "max_control_character_ratio",
            "max_repeated_line_ratio",
            "max_repeated_ngram_ratio",
            "min_language_confidence",
            "max_rejected_ratio",
        ):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.repeated_ngram_size < 1:
            raise ValueError("repeated_ngram_size must be positive")
        if self.min_image_width < 1 or self.min_image_height < 1:
            raise ValueError("minimum image dimensions must be positive")


@dataclass(frozen=True)
class _RecordEnvelope:
    value: Any | None
    location: str
    index: int
    parse_error: str | None = None


class _MetricAccumulator:
    def __init__(self) -> None:
        self.record_count = 0
        self.text_count = 0
        self.character_count = 0
        self.word_count = 0
        self.min_text_chars: int | None = None
        self.max_text_chars = 0
        self.language_counts: Counter[str] = Counter()

    def add_text(self, text: str) -> None:
        characters = len(text)
        self.text_count += 1
        self.character_count += characters
        self.word_count += _word_count(text)
        self.min_text_chars = (
            characters
            if self.min_text_chars is None
            else min(self.min_text_chars, characters)
        )
        self.max_text_chars = max(self.max_text_chars, characters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "text_value_count": self.text_count,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "min_text_chars": self.min_text_chars,
            "max_text_chars": self.max_text_chars if self.text_count else None,
            "mean_text_chars": (
                self.character_count / self.text_count if self.text_count else None
            ),
            "languages": dict(sorted(self.language_counts.items())),
        }


class DataValidationOperator:
    """Validate every parseable record and selected media references."""

    name = "data_validity"
    version = "1.0.0"
    run_type = DatasetRunType.QUALITY
    deterministic = True

    def __init__(
        self,
        policy: DataValidationPolicy | None = None,
        *,
        language_detector: LanguageDetector | None = None,
    ) -> None:
        self.policy = policy or DataValidationPolicy()
        self.language_detector = language_detector
        if self.policy.expected_languages and language_detector is None:
            raise ValueError(
                "expected_languages requires a LanguageDetector; use the FastText "
                "adapter for production language identification"
            )

    def fingerprint(self) -> str:
        detector = (
            type(self.language_detector).__qualname__
            if self.language_detector is not None
            else None
        )
        return operator_fingerprint(
            self.name,
            self.version,
            {"policy": self.policy, "language_detector": detector},
        )

    def parameters(self) -> dict[str, Any]:
        return {
            "policy": asdict(self.policy),
            "language_detector": (
                type(self.language_detector).__qualname__
                if self.language_detector is not None
                else None
            ),
        }

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
        declared = input_version.storage_format.upper()
        if declared not in _SUPPORTED_FORMATS:
            raise OperatorInputError(
                f"{self.name} currently supports record data in "
                f"{sorted(_SUPPORTED_FORMATS)}, got {declared}"
            )
        root = input_version.local_path()
        if not root.exists():
            raise OperatorInputError(f"storage path does not exist: {root}")

        issues = IssueCollector(self.policy.max_issues)
        metrics = _MetricAccumulator()
        rejected = 0
        seen_hashes: dict[str, tuple[str, int]] = {}
        records = _iter_records(root, declared)
        for envelope in records:
            metrics.record_count += 1
            error_count = issues.error_count
            if envelope.parse_error is not None:
                issues.add(
                    "PARSE_ERROR",
                    envelope.parse_error,
                    location=envelope.location,
                    record_index=envelope.index,
                )
            else:
                self._validate_record(
                    envelope,
                    root,
                    input_version.schema_definition,
                    metrics,
                    issues,
                    seen_hashes,
                )
            if issues.error_count > error_count:
                rejected += 1

        if metrics.record_count == 0:
            issues.add(
                "NO_RECORDS",
                "no JSON or JSONL records were found under storage_uri",
            )
        passed_records = max(0, metrics.record_count - rejected)
        rejected_ratio = rejected / metrics.record_count if metrics.record_count else 1.0
        passed = (
            metrics.record_count > 0
            and rejected_ratio <= self.policy.max_rejected_ratio
        )
        report_metrics = metrics.to_dict()
        report_metrics["rejected_ratio"] = rejected_ratio
        return ValidationReport(
            dataset_version_id=input_version.version_id,
            evaluator_name=self.name,
            evaluator_version=self.version,
            passed=passed,
            checked_count=metrics.record_count,
            passed_count=passed_records,
            rejected_count=rejected,
            error_count=issues.total_errors,
            warning_count=issues.total_warnings,
            issues=tuple(issues.issues),
            metrics=report_metrics,
            issue_counts=dict(issues.code_counts),
            truncated_issue_count=issues.truncated,
        )

    def _validate_record(
        self,
        envelope: _RecordEnvelope,
        root: Path,
        schema: Mapping[str, Any],
        metrics: _MetricAccumulator,
        issues: IssueCollector,
        seen_hashes: dict[str, tuple[str, int]],
    ) -> None:
        record = envelope.value
        location = envelope.location
        index = envelope.index
        if not isinstance(record, Mapping):
            issues.add(
                "TYPE_MISMATCH",
                "training-data record must be a JSON object",
                location=location,
                record_index=index,
                details={"actual_type": _json_type_name(record)},
            )
            return
        if not record:
            issues.add(
                "EMPTY_RECORD",
                "record is an empty object",
                location=location,
                record_index=index,
            )
        for field_name in self.policy.required_fields:
            values = _select_values(record, field_name)
            if not values or all(value in (None, "", [], {}) for value in values):
                issues.add(
                    "MISSING_REQUIRED_FIELD",
                    f"required field is missing or empty: {field_name}",
                    location=location,
                    record_index=index,
                    details={"field": field_name},
                )
        _validate_json_schema(record, schema, issues, location, index)

        text_values = list(_selected_strings(record, self.policy.text_fields))
        non_empty_text_values = [item for item in text_values if item[1].strip()]
        if self.policy.require_text and not non_empty_text_values:
            issues.add(
                "MISSING_TEXT",
                "record contains no non-empty configured text field",
                location=location,
                record_index=index,
            )
        for field_name, text in text_values:
            metrics.add_text(text)
            self._validate_text(field_name, text, location, index, metrics, issues)

        for field_name, value in _selected_strings(record, self.policy.image_fields):
            self._validate_image_reference(field_name, value, root, location, index, issues)

        if self.policy.check_exact_duplicates:
            digest = sha256(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            previous = seen_hashes.get(digest)
            if previous is not None:
                issues.add(
                    "EXACT_DUPLICATE",
                    "record is an exact duplicate",
                    location=location,
                    record_index=index,
                    details={
                        "first_location": previous[0],
                        "first_record_index": previous[1],
                    },
                )
            else:
                seen_hashes[digest] = (location, index)

    def _validate_text(
        self,
        field_name: str,
        text: str,
        location: str,
        index: int,
        metrics: _MetricAccumulator,
        issues: IssueCollector,
    ) -> None:
        stripped = text.strip()
        details = {"field": field_name}
        if not stripped:
            if (
                field_name in self.policy.non_empty_text_fields
                or field_name in self.policy.required_fields
            ):
                issues.add(
                    "EMPTY_TEXT",
                    f"text field is empty: {field_name}",
                    location=location,
                    record_index=index,
                    details=details,
                )
            return
        if len(text) < self.policy.min_text_chars:
            issues.add(
                "TEXT_TOO_SHORT",
                f"text has {len(text)} characters",
                location=location,
                record_index=index,
                details=details,
            )
        if self.policy.max_text_chars is not None and len(text) > self.policy.max_text_chars:
            issues.add(
                "TEXT_TOO_LONG",
                f"text has {len(text)} characters",
                location=location,
                record_index=index,
                details=details,
            )
        words = _word_count(text)
        if self.policy.min_words is not None and words < self.policy.min_words:
            issues.add(
                "WORD_COUNT_OUT_OF_RANGE",
                f"text has {words} words; minimum is {self.policy.min_words}",
                location=location,
                record_index=index,
                details=details,
            )
        if self.policy.max_words is not None and words > self.policy.max_words:
            issues.add(
                "WORD_COUNT_OUT_OF_RANGE",
                f"text has {words} words; maximum is {self.policy.max_words}",
                location=location,
                record_index=index,
                details=details,
            )
        alphanumeric_ratio = _alphanumeric_ratio(text)
        if (
            self.policy.min_alphanumeric_ratio is not None
            and alphanumeric_ratio < self.policy.min_alphanumeric_ratio
        ):
            issues.add(
                "LOW_ALPHANUMERIC_RATIO",
                f"alphanumeric ratio is {alphanumeric_ratio:.4f}",
                location=location,
                record_index=index,
                details=details,
            )
        control_ratio = _control_character_ratio(text)
        if control_ratio > self.policy.max_control_character_ratio:
            issues.add(
                "CONTROL_CHARACTER_RATIO",
                f"control-character ratio is {control_ratio:.4f}",
                location=location,
                record_index=index,
                details=details,
            )
        repeated_lines = _repeated_line_ratio(text)
        if (
            self.policy.max_repeated_line_ratio is not None
            and repeated_lines > self.policy.max_repeated_line_ratio
        ):
            issues.add(
                "REPEATED_LINE_RATIO",
                f"repeated-line ratio is {repeated_lines:.4f}",
                location=location,
                record_index=index,
                details=details,
            )
        repeated_ngrams = _repeated_ngram_ratio(
            text, self.policy.repeated_ngram_size
        )
        if (
            self.policy.max_repeated_ngram_ratio is not None
            and repeated_ngrams > self.policy.max_repeated_ngram_ratio
        ):
            issues.add(
                "REPEATED_NGRAM_RATIO",
                f"repeated {self.policy.repeated_ngram_size}-gram ratio is "
                f"{repeated_ngrams:.4f}",
                location=location,
                record_index=index,
                details=details,
            )
        if self.language_detector is not None:
            prediction = self.language_detector.detect(stripped)
            metrics.language_counts[prediction.language] += 1
            if self.policy.expected_languages and (
                prediction.language not in self.policy.expected_languages
                or prediction.confidence < self.policy.min_language_confidence
            ):
                issues.add(
                    "LANGUAGE_MISMATCH",
                    f"detected {prediction.language} at {prediction.confidence:.3f}",
                    location=location,
                    record_index=index,
                    details=details,
                )

    def _validate_image_reference(
        self,
        field_name: str,
        value: str,
        root: Path,
        location: str,
        index: int,
        issues: IssueCollector,
    ) -> None:
        if value.startswith(("http://", "https://", "s3://", "gs://")):
            issues.add(
                "REMOTE_MEDIA_NOT_PROBED",
                f"remote media reference was not fetched: {field_name}",
                severity=IssueSeverity.WARNING,
                location=location,
                record_index=index,
            )
            return
        media = Path(value)
        if not media.is_absolute():
            base = root if root.is_dir() else root.parent
            media = base / media
        if not media.is_file():
            issues.add(
                "MEDIA_REFERENCE_MISSING",
                f"referenced image does not exist: {value}",
                location=location,
                record_index=index,
                details={"field": field_name},
            )
            return
        try:
            with Image.open(media) as image:
                image.verify()
            with Image.open(media) as image:
                if (
                    image.width < self.policy.min_image_width
                    or image.height < self.policy.min_image_height
                ):
                    issues.add(
                        "IMAGE_DIMENSIONS",
                        f"image dimensions are {image.width}x{image.height}",
                        location=location,
                        record_index=index,
                        details={"field": field_name, "media": str(media)},
                    )
                orientation = image.getexif().get(274)
                if orientation is not None and orientation not in range(1, 9):
                    issues.add(
                        "INVALID_EXIF_ORIENTATION",
                        f"invalid EXIF orientation: {orientation}",
                        location=location,
                        record_index=index,
                        details={"field": field_name, "media": str(media)},
                    )
                image.load()
        except (OSError, UnidentifiedImageError) as exc:
            issues.add(
                "MEDIA_CORRUPT",
                str(exc),
                location=location,
                record_index=index,
                details={"field": field_name, "media": str(media)},
            )


def _iter_records(root: Path, declared_format: str) -> Iterator[_RecordEnvelope]:
    paths = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
    for path in paths:
        suffix = path.suffix.lower()
        location = relative_location(path, root)
        if suffix == ".jsonl":
            yield from _iter_jsonl(path, location)
        elif suffix == ".json":
            yield from _iter_json(path, location)
        elif suffix == ".zip":
            yield from _iter_zip_json(path, location)
        elif suffix == ".gz" and declared_format in {"WARC_GZIP", "WARC_WET_GZIP"}:
            yield from _iter_warc_gzip(path, location)


def _iter_jsonl(path: Path, location: str) -> Iterator[_RecordEnvelope]:
    try:
        stream = path.open("rb")
    except OSError as exc:
        yield _RecordEnvelope(None, location, 1, str(exc))
        return
    with stream:
        for line_number, raw_line in enumerate(stream, 1):
            try:
                line = raw_line.decode("utf-8-sig" if line_number == 1 else "utf-8")
            except UnicodeDecodeError as exc:
                yield _RecordEnvelope(None, location, line_number, str(exc))
                continue
            if not line.strip():
                yield _RecordEnvelope(
                    None,
                    location,
                    line_number,
                    "blank JSONL line",
                )
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                yield _RecordEnvelope(None, location, line_number, str(exc))
            else:
                yield _RecordEnvelope(value, location, line_number)


def _iter_json(path: Path, location: str) -> Iterator[_RecordEnvelope]:
    try:
        with path.open(encoding="utf-8-sig") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        yield _RecordEnvelope(None, location, 1, str(exc))
        return
    yield from _payload_records(payload, location)


def _iter_zip_json(
    path: Path,
    location: str,
) -> Iterator[_RecordEnvelope]:
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                suffix = Path(info.filename).suffix.lower()
                if info.is_dir() or suffix not in {".json", ".jsonl"}:
                    continue
                member_location = f"{location}!/{info.filename}"
                try:
                    with archive.open(info) as raw:
                        text = io.TextIOWrapper(raw, encoding="utf-8-sig")
                        if suffix == ".jsonl":
                            for line_number, line in enumerate(text, 1):
                                if not line.strip():
                                    yield _RecordEnvelope(
                                        None,
                                        member_location,
                                        line_number,
                                        "blank JSONL line",
                                    )
                                    continue
                                try:
                                    value = json.loads(line)
                                except json.JSONDecodeError as exc:
                                    yield _RecordEnvelope(
                                        None, member_location, line_number, str(exc)
                                    )
                                else:
                                    yield _RecordEnvelope(
                                        value, member_location, line_number
                                    )
                        else:
                            yield from _payload_records(json.load(text), member_location)
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    yield _RecordEnvelope(None, member_location, 1, str(exc))
    except (OSError, zipfile.BadZipFile) as exc:
        yield _RecordEnvelope(None, location, 1, f"container corrupt: {exc}")


def _payload_records(payload: Any, location: str) -> Iterator[_RecordEnvelope]:
    if isinstance(payload, list):
        for index, value in enumerate(payload, 1):
            yield _RecordEnvelope(value, location, index)
    else:
        yield _RecordEnvelope(payload, location, 1)


def _iter_warc_gzip(path: Path, location: str) -> Iterator[_RecordEnvelope]:
    record_index = 0
    try:
        with gzip.open(path, "rb") as stream:
            while True:
                line = stream.readline()
                while line in {b"\r\n", b"\n"}:
                    line = stream.readline()
                if not line:
                    break
                record_index += 1
                if not line.startswith(b"WARC/"):
                    yield _RecordEnvelope(
                        None,
                        location,
                        record_index,
                        f"invalid WARC record header: {line[:80]!r}",
                    )
                    return
                headers: dict[str, str] = {}
                while True:
                    line = stream.readline()
                    if not line:
                        yield _RecordEnvelope(
                            None, location, record_index, "truncated WARC headers"
                        )
                        return
                    if line in {b"\r\n", b"\n"}:
                        break
                    try:
                        name, value = line.decode("utf-8").split(":", 1)
                    except (UnicodeDecodeError, ValueError) as exc:
                        yield _RecordEnvelope(None, location, record_index, str(exc))
                        return
                    headers[name.lower()] = value.strip()
                try:
                    content_length = int(headers["content-length"])
                except (KeyError, ValueError) as exc:
                    yield _RecordEnvelope(None, location, record_index, str(exc))
                    return
                body = stream.read(content_length)
                if len(body) != content_length:
                    yield _RecordEnvelope(
                        None, location, record_index, "truncated WARC record body"
                    )
                    return
                try:
                    text = body.decode("utf-8")
                except UnicodeDecodeError as exc:
                    yield _RecordEnvelope(None, location, record_index, str(exc))
                    continue
                yield _RecordEnvelope(
                    {"text": text, "warc_headers": headers},
                    location,
                    record_index,
                )
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        yield _RecordEnvelope(None, location, max(1, record_index), str(exc))


def _select_values(value: Any, path: str) -> list[Any]:
    parts = path.split(".") if path else []
    current = [value]
    for part in parts:
        selected: list[Any] = []
        for item in current:
            if part == "*" and isinstance(item, Sequence) and not isinstance(item, str):
                selected.extend(item)
            elif isinstance(item, Mapping) and part in item:
                selected.append(item[part])
        current = selected
    return current


def _selected_strings(
    record: Mapping[str, Any], fields: Iterable[str]
) -> Iterator[tuple[str, str]]:
    for field_name in fields:
        for value in _select_values(record, field_name):
            if isinstance(value, str):
                yield field_name, value


def _validate_json_schema(
    value: Any,
    schema: Mapping[str, Any],
    issues: IssueCollector,
    location: str,
    index: int,
    path: str = "$",
) -> None:
    """Validate the deterministic JSON Schema subset used by current metadata."""
    if not schema:
        return
    expected = schema.get("type")
    if expected is not None and not _matches_json_type(value, expected):
        issues.add(
            "SCHEMA_TYPE_MISMATCH",
            f"{path} must be {expected}, got {_json_type_name(value)}",
            location=location,
            record_index=index,
            details={"path": path},
        )
        return
    if isinstance(value, Mapping):
        for required in schema.get("required", ()):
            if required not in value:
                issues.add(
                    "SCHEMA_REQUIRED_FIELD",
                    f"schema-required field is missing: {path}.{required}",
                    location=location,
                    record_index=index,
                    details={"path": f"{path}.{required}"},
                )
        properties = schema.get("properties", {})
        for name, child_schema in properties.items():
            if name in value and isinstance(child_schema, Mapping):
                _validate_json_schema(
                    value[name],
                    child_schema,
                    issues,
                    location,
                    index,
                    f"{path}.{name}",
                )
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for item_index, item in enumerate(value):
            _validate_json_schema(
                item,
                schema["items"],
                issues,
                location,
                index,
                f"{path}[{item_index}]",
            )


def _matches_json_type(value: Any, expected: str | list[str]) -> bool:
    choices = [expected] if isinstance(expected, str) else expected
    actual = _json_type_name(value)
    return actual in choices or (actual == "integer" and "number" in choices)


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


def _word_count(text: str) -> int:
    # Whitespace tokens plus individual CJK characters keeps the metric useful
    # for both spaced and non-spaced languages without a tokenizer dependency.
    count = 0
    in_word = False
    for character in text:
        if _is_cjk(character):
            count += 1
            in_word = False
        elif character.isalnum() or character == "_":
            if not in_word:
                count += 1
                in_word = True
        else:
            in_word = False
    return count


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _alphanumeric_ratio(text: str) -> float:
    non_space = [character for character in text if not character.isspace()]
    if not non_space:
        return 0.0
    return sum(character.isalnum() for character in non_space) / len(non_space)


def _control_character_ratio(text: str) -> float:
    if not text:
        return 0.0
    invalid = sum(
        unicodedata.category(character) == "Cc" and character not in "\n\r\t"
        for character in text
    )
    return invalid / len(text)


def _repeated_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    frequencies = Counter(lines)
    repeated = sum(count for count in frequencies.values() if count > 1)
    return repeated / len(lines)


def _repeated_ngram_ratio(text: str, size: int) -> float:
    words = [word.casefold() for word in text.split() if word]
    if len(words) < size:
        return 0.0
    ngrams = [tuple(words[index : index + size]) for index in range(len(words) - size + 1)]
    frequencies = Counter(ngrams)
    repeated = sum(count for count in frequencies.values() if count > 1)
    return repeated / len(ngrams)


__all__ = [
    "DataValidationOperator",
    "DataValidationPolicy",
    "FastTextLanguageDetector",
    "LanguageDetector",
    "LanguagePrediction",
]
