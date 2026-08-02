"""Concrete WikiText validation, normalization, and Lance IO.

This module intentionally avoids PipelineOperator and metadata abstractions so
each physical data-processing step is visible to learners.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "model_data/text/wikitext_2_raw"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "model_data/learning/wikitext_2_standardized.lance"
)
SPLITS = ("train", "validation", "test")
ARROW_SCHEMA = pa.schema(
    [
        pa.field("text", pa.string(), nullable=False),
        pa.field("source_split", pa.string(), nullable=False),
        pa.field("source_line", pa.int64(), nullable=False),
        pa.field("is_heading", pa.bool_(), nullable=False),
        pa.field("character_count", pa.int32(), nullable=False),
    ]
)


@dataclass(frozen=True, slots=True)
class RawRecord:
    split: str
    line_number: int
    raw_bytes: bytes
    value: Any | None
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    split: str
    line_number: int
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    total_records: int
    valid_records: int
    rejected_records: int
    issue_counts: dict[str, int]
    issue_samples: tuple[ValidationIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["issue_samples"] = [asdict(issue) for issue in self.issue_samples]
        return result


def iter_raw_records(dataset_root: Path) -> Iterator[RawRecord]:
    """Read every physical JSONL line and preserve parsing failures."""
    # 读取阶段不丢弃坏行：训练数据校验必须保留 split + 行号，才能回溯原始数据。
    for split in SPLITS:
        path = dataset_root / f"{split}.jsonl"
        with path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    yield RawRecord(
                        split,
                        line_number,
                        raw_line,
                        None,
                        "INVALID_UTF8",
                        str(exc),
                    )
                    continue
                if not line.strip():
                    yield RawRecord(
                        split,
                        line_number,
                        raw_line,
                        None,
                        "BLANK_JSONL_LINE",
                        "physical JSONL line is blank",
                    )
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    yield RawRecord(
                        split,
                        line_number,
                        raw_line,
                        None,
                        "INVALID_JSON",
                        str(exc),
                    )
                    continue
                yield RawRecord(split, line_number, raw_line, value)


def validate_record(record: RawRecord) -> ValidationIssue | None:
    """Apply the minimal validity contract used by this learning flow."""
    # 这里仅判断“能否成为文本训练样本”。长度、语言、重复率等属于质量策略，
    # 不应混入所有数据集都必须满足的硬有效性规则。
    if record.error_code is not None:
        return ValidationIssue(
            record.split,
            record.line_number,
            record.error_code,
            record.error_message or record.error_code,
        )
    if not isinstance(record.value, dict):
        return ValidationIssue(
            record.split,
            record.line_number,
            "NOT_AN_OBJECT",
            "each training record must be a JSON object",
        )
    if "text" not in record.value:
        return ValidationIssue(
            record.split,
            record.line_number,
            "MISSING_TEXT",
            "required field 'text' is missing",
        )
    text = record.value["text"]
    if not isinstance(text, str):
        return ValidationIssue(
            record.split,
            record.line_number,
            "TEXT_NOT_STRING",
            "field 'text' must be a string",
        )
    if not text.strip():
        return ValidationIssue(
            record.split,
            record.line_number,
            "EMPTY_TEXT",
            "empty WikiText separator rows are not training samples",
        )
    invalid_controls = [
        character
        for character in text
        if unicodedata.category(character) == "Cc"
        and character not in "\n\r\t"
    ]
    if invalid_controls:
        codepoints = ", ".join(
            f"U+{ord(character):04X}" for character in sorted(set(invalid_controls))
        )
        return ValidationIssue(
            record.split,
            record.line_number,
            "CONTROL_CHARACTER",
            f"text contains disallowed control characters: {codepoints}",
        )
    return None


def validate_dataset(
    dataset_root: Path,
    *,
    max_issue_samples: int = 10,
) -> ValidationSummary:
    # 全量扫描每一条物理记录，但只为每种错误保留一个样例，避免报告无限增长。
    counts: Counter[str] = Counter()
    samples: list[ValidationIssue] = []
    total = 0
    rejected = 0
    for record in iter_raw_records(dataset_root):
        total += 1
        issue = validate_record(record)
        if issue is None:
            continue
        rejected += 1
        counts[issue.code] += 1
        sampled_codes = {sample.code for sample in samples}
        if len(samples) < max_issue_samples and issue.code not in sampled_codes:
            samples.append(issue)
    return ValidationSummary(
        total_records=total,
        valid_records=total - rejected,
        rejected_records=rejected,
        issue_counts=dict(sorted(counts.items())),
        issue_samples=tuple(samples),
    )


def normalize_text(text: str) -> str:
    """Make the chosen WikiText representation explicit and deterministic."""
    # NFC 统一等价 Unicode 表示；WikiText 的 @-@ / @,@ / @.@ 是分词残留，
    # 标准化后恢复成更适合语言模型消费的自然文本。
    normalized = unicodedata.normalize("NFC", text).strip()
    return (
        normalized.replace(" @-@ ", "-")
        .replace(" @,@ ", ", ")
        .replace(" @.@ ", ". ")
    )


def is_heading(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) >= 5 and stripped.startswith("=") and stripped.endswith("=")


def iter_standardized_rows(dataset_root: Path) -> Iterator[dict[str, Any]]:
    """Filter invalid rows and map valid records to the canonical schema."""
    # clean + format 在同一次流式遍历中完成：坏样本跳过，好样本映射到固定字段。
    for record in iter_raw_records(dataset_root):
        if validate_record(record) is not None:
            continue
        raw_text = record.value["text"]
        text = normalize_text(raw_text)
        yield {
            "text": text,
            "source_split": record.split,
            "source_line": record.line_number,
            "is_heading": is_heading(text),
            "character_count": len(text),
        }


def iter_arrow_batches(
    dataset_root: Path,
    *,
    batch_size: int = 1024,
) -> Iterator[pa.RecordBatch]:
    """Convert standardized rows to bounded Arrow batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    # 批量构造 Arrow，避免先把整个语料装进内存；最后不足一批的数据也要输出。
    rows: list[dict[str, Any]] = []
    for row in iter_standardized_rows(dataset_root):
        rows.append(row)
        if len(rows) == batch_size:
            yield pa.RecordBatch.from_pylist(rows, schema=ARROW_SCHEMA)
            rows = []
    if rows:
        yield pa.RecordBatch.from_pylist(rows, schema=ARROW_SCHEMA)


def write_lance_dataset(
    dataset_root: Path,
    output_path: Path,
    *,
    batch_size: int = 1024,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Stream all standardized records into a new Lance dataset."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Lance 直接消费 RecordBatch 迭代器，因此标准化流水线中间不需要落盘。
    dataset = lance.write_dataset(
        iter_arrow_batches(dataset_root, batch_size=batch_size),
        output_path,
        schema=ARROW_SCHEMA,
        mode="overwrite" if overwrite else "create",
    )
    return {
        "action": "overwritten" if overwrite else "created",
        "output": str(output_path),
        "rows": dataset.count_rows(),
        "schema": str(dataset.schema),
    }


def inspect_lance_dataset(
    output_path: Path,
    *,
    limit: int = 5,
) -> dict[str, Any]:
    dataset = lance.dataset(output_path)
    rows = dataset.scanner(limit=limit).to_table().to_pylist()
    return {
        "path": str(output_path),
        "rows": dataset.count_rows(),
        "schema": str(dataset.schema),
        "sample": rows,
    }
