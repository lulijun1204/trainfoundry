"""Internal helpers shared by validation operators."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath

from pipeline.base import IssueSeverity, ValidationIssue


class IssueCollector:
    """Bound detailed findings while retaining the number that was truncated."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("max_issues must be positive")
        self.limit = limit
        self.issues: list[ValidationIssue] = []
        self.truncated = 0
        self.total_errors = 0
        self.total_warnings = 0
        self.code_counts: Counter[str] = Counter()

    def add(
        self,
        code: str,
        message: str,
        *,
        severity: IssueSeverity = IssueSeverity.ERROR,
        location: str | None = None,
        record_index: int | None = None,
        details: dict | None = None,
    ) -> None:
        issue = ValidationIssue(
            code=code,
            message=message,
            severity=severity,
            location=location,
            record_index=record_index,
            details=details or {},
        )
        if severity is IssueSeverity.ERROR:
            self.total_errors += 1
        else:
            self.total_warnings += 1
        self.code_counts[code] += 1
        if len(self.issues) < self.limit:
            self.issues.append(issue)
        else:
            self.truncated += 1

    @property
    def error_count(self) -> int:
        return self.total_errors


def unsafe_archive_member(name: str) -> bool:
    """Detect absolute paths and traversal in POSIX or Windows archive names."""
    normalized = name.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(name)
    return (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
    )


def relative_location(path: Path, root: Path) -> str:
    if root.is_file() or root.is_symlink():
        return path.name
    return path.relative_to(root).as_posix()


def counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))
