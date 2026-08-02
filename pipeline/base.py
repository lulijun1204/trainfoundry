"""Shared contracts for dataset pipeline operators."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol

from metadata import (
    DatasetRun,
    DatasetRunType,
    DatasetVersion,
)
from pipeline.data import ExecutionDataset


class OperatorInputError(ValueError):
    """Raised when an operator cannot consume the supplied execution data."""


class IssueSeverity(StrEnum):
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(frozen=True)
class ValidationIssue:
    """One deterministic validation finding."""

    code: str
    message: str
    severity: IssueSeverity = IssueSeverity.ERROR
    location: str | None = None
    record_index: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["severity"] = self.severity.value
        return {key: value for key, value in result.items() if value not in (None, {})}


@dataclass(frozen=True)
class ValidationReport:
    """Quality result draft that maps to one persisted QualityResultSet."""

    dataset_version_id: str
    evaluator_name: str
    evaluator_version: str
    passed: bool
    checked_count: int
    passed_count: int
    rejected_count: int
    error_count: int = 0
    warning_count: int = 0
    issues: tuple[ValidationIssue, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    issue_counts: Mapping[str, int] = field(default_factory=dict)
    truncated_issue_count: int = 0

    def summary(self) -> dict[str, Any]:
        """Return the bounded JSON summary stored in QualityResultSet."""
        counts_by_code = dict(self.issue_counts)
        if not counts_by_code:
            for issue in self.issues:
                counts_by_code[issue.code] = counts_by_code.get(issue.code, 0) + 1
        return {
            "checked_count": self.checked_count,
            "passed_count": self.passed_count,
            "rejected_count": self.rejected_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issue_counts": counts_by_code,
            "truncated_issue_count": self.truncated_issue_count,
            "metrics": dict(self.metrics),
        }

    def quality_result_kwargs(self, run_id: str) -> dict[str, Any]:
        """Map this report to MetadataRepository.create_quality_result()."""
        if not run_id.strip():
            raise ValueError("run_id must be non-empty")
        return {
            "dataset_version_id": self.dataset_version_id,
            "run_id": run_id,
            "evaluator_name": self.evaluator_name,
            "evaluator_version": self.evaluator_version,
            "status": "SUCCEEDED",
            "passed": self.passed,
            "summary": self.summary(),
            "passed_count": self.passed_count,
            "rejected_count": self.rejected_count,
        }


@dataclass(frozen=True)
class SupplementalResultDraft:
    """A typed non-quality result emitted by an operator.

    Specialized persistent result types can replace this runtime envelope when
    their query and lifecycle requirements justify a dedicated metadata table.
    """

    result_type: str
    subject_execution_data_id: str
    summary: Mapping[str, Any] = field(default_factory=dict)
    dataset_version_id: str | None = None
    detail_uri: str | None = None
    detail_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.result_type.strip():
            raise ValueError("result_type must be non-empty")
        if not self.subject_execution_data_id.strip():
            raise ValueError("subject_execution_data_id must be non-empty")


ResultDraft = ValidationReport | SupplementalResultDraft


@dataclass(frozen=True)
class OperatorOutput:
    """Ephemeral data and side results produced by one operator."""

    data: ExecutionDataset
    results: tuple[ResultDraft, ...] = ()

    @property
    def quality_results(self) -> tuple[ValidationReport, ...]:
        return tuple(
            result
            for result in self.results
            if isinstance(result, ValidationReport)
        )

    @property
    def supplemental_results(self) -> tuple[SupplementalResultDraft, ...]:
        return tuple(
            result
            for result in self.results
            if isinstance(result, SupplementalResultDraft)
        )


@dataclass(frozen=True, slots=True)
class OperatorContext:
    """Execution identity supplied by PipelineExecutor to an operator."""

    run_id: str
    created_by: str
    input_version: DatasetVersion


@dataclass(frozen=True)
class PipelineExecutionResult:
    """Aggregate output of one ordered PipelineExecutor invocation."""

    input_version: DatasetVersion
    output_data: ExecutionDataset
    output_version: DatasetVersion | None
    runs: tuple[DatasetRun, ...]
    results: tuple[ResultDraft, ...] = ()


class PipelineOperator(Protocol):
    """Hook invoked by PipelineExecutor; operators deliberately expose no execute()."""

    name: str
    version: str
    run_type: DatasetRunType
    deterministic: bool

    def run(
        self,
        input_data: ExecutionDataset,
        context: OperatorContext,
    ) -> OperatorOutput:
        """Compute ephemeral output data and zero or more result drafts."""

    def fingerprint(self) -> str:
        """Return a deterministic implementation and configuration digest."""

    def parameters(self) -> Mapping[str, Any]:
        """Return the normalized execution parameters persisted in DatasetRun."""


def operator_fingerprint(name: str, version: str, config: Any) -> str:
    """Hash a dataclass or JSON-compatible operator configuration."""
    value = asdict(config) if hasattr(config, "__dataclass_fields__") else config
    encoded = json.dumps(
        {"name": name, "version": version, "config": value},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
