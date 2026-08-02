"""Step 1 implementation: physical Minari package and HDF5 validation."""

from __future__ import annotations

import json
import math
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py

from .common import (
    CHUNK_SIZE,
    DATA_DIRECTORY,
    EXPECTED_FILES,
    HDF5_FILE,
    HDF5_MAGIC,
    METADATA_FILE,
    IssueCollector,
    ValidationIssue,
    dataclass_to_dict,
    digest_and_magic,
)


@dataclass(frozen=True, slots=True)
class Hdf5DatasetObservation:
    path: str
    shape: tuple[int, ...]
    dtype: str
    chunks: tuple[int, ...] | None
    compression: str | None
    checksum: bool


@dataclass(frozen=True, slots=True)
class FileFormatSummary:
    file_count: int
    valid_files: int
    invalid_files: int
    total_bytes: int
    archive_sha256: dict[str, str]
    hdf5_dataset_count: int
    hdf5_episode_groups: int
    hdf5_logical_bytes: int
    hdf5_inventory: tuple[Hdf5DatasetObservation, ...]
    error_counts: dict[str, int]
    warning_counts: dict[str, int]
    issue_samples: tuple[ValidationIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


def validate_file_format(
    dataset_root: Path,
    *,
    max_issue_samples: int = 10,
) -> FileFormatSummary:
    """Fully read the Minari package without interpreting trajectory semantics."""
    issues = IssueCollector(max_issue_samples)
    data_root = dataset_root / DATA_DIRECTORY
    paths = {name: data_root / name for name in EXPECTED_FILES}
    invalid_files: set[str] = set()
    digests: dict[str, str] = {}
    total_bytes = 0
    metadata: dict[str, Any] | None = None
    inventory: list[Hdf5DatasetObservation] = []
    dataset_count = 0
    logical_bytes = 0
    episode_groups = 0

    for name, path in paths.items():
        if not path.exists():
            issues.add(name, name, "MISSING_FILE", "required Minari file is missing")
            invalid_files.add(name)
            continue
        if path.is_symlink() or not path.is_file():
            issues.add(
                name, name, "UNSAFE_FILE_TYPE", "file must be a regular non-symlink"
            )
            invalid_files.add(name)
            continue
        mode = path.stat().st_mode
        if not stat.S_ISREG(mode):
            issues.add(name, name, "UNSAFE_FILE_TYPE", "file must be regular")
            invalid_files.add(name)
            continue
        total_bytes += path.stat().st_size
        digest, magic = digest_and_magic(path)
        digests[name] = digest
        before = issues.error_total
        if name == HDF5_FILE:
            if magic != HDF5_MAGIC:
                issues.add(
                    name, name, "INVALID_HDF5_MAGIC", f"unexpected magic {magic!r}"
                )
            else:
                try:
                    inventory, dataset_count, episode_groups, logical_bytes = (
                        _scan_hdf5(path, issues)
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    issues.add(name, name, "HDF5_CORRUPT", str(exc))
        else:
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
            except UnicodeDecodeError as exc:
                issues.add(name, name, "INVALID_UTF8", str(exc))
            except json.JSONDecodeError as exc:
                issues.add(name, name, "INVALID_JSON", str(exc))
            if metadata is not None:
                _validate_metadata_shape(metadata, issues)
        if issues.error_total > before:
            invalid_files.add(name)

    hdf5_path = paths[HDF5_FILE]
    if (
        isinstance(metadata, dict)
        and hdf5_path.is_file()
        and HDF5_FILE not in invalid_files
    ):
        before = issues.error_total
        _reconcile_metadata(hdf5_path, metadata, episode_groups, issues)
        if issues.error_total > before:
            invalid_files.update({HDF5_FILE, METADATA_FILE})

    present = sum(path.is_file() for path in paths.values())
    return FileFormatSummary(
        file_count=present,
        valid_files=max(0, present - len(invalid_files)),
        invalid_files=len(invalid_files),
        total_bytes=total_bytes,
        archive_sha256=dict(sorted(digests.items())),
        hdf5_dataset_count=dataset_count,
        hdf5_episode_groups=episode_groups,
        hdf5_logical_bytes=logical_bytes,
        hdf5_inventory=tuple(inventory),
        error_counts=dict(sorted(issues.error_counts.items())),
        warning_counts=dict(sorted(issues.warning_counts.items())),
        issue_samples=tuple(issues.samples),
    )


def _scan_hdf5(
    path: Path,
    issues: IssueCollector,
) -> tuple[list[Hdf5DatasetObservation], int, int, int]:
    """Read every dataset while retaining only bounded schema examples."""
    inventory: list[Hdf5DatasetObservation] = []
    inventory_paths: set[str] = set()
    dataset_count = 0
    logical_bytes = 0
    with h5py.File(path, "r") as container:
        _reject_external_links(container, issues)
        episode_groups = sum(
            name.startswith("episode_") and isinstance(value, h5py.Group)
            for name, value in container.items()
        )

        def visit(name: str, value: h5py.Group | h5py.Dataset) -> None:
            nonlocal dataset_count, logical_bytes
            if not isinstance(value, h5py.Dataset):
                return
            dataset_count += 1
            # Episode paths repeat thousands of times. Keep one example per physical
            # schema instead of returning a result object with 100k+ inventory rows.
            logical_path = re.sub(r"^episode_\d+/", "episode_*/", name)
            observation = Hdf5DatasetObservation(
                path=logical_path,
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                chunks=tuple(value.chunks) if value.chunks is not None else None,
                compression=value.compression,
                checksum=bool(value.fletcher32),
            )
            if logical_path not in inventory_paths:
                inventory_paths.add(logical_path)
                inventory.append(observation)
            logical_bytes += int(value.size * value.dtype.itemsize)
            _read_complete_dataset(value)

        container.visititems(visit)
    return inventory, dataset_count, episode_groups, logical_bytes


def _read_complete_dataset(dataset: h5py.Dataset) -> None:
    """Exercise every stored byte while keeping reads bounded."""
    if dataset.shape == ():
        dataset[()]
        return
    if dataset.chunks is not None:
        for selection in dataset.iter_chunks():
            dataset[selection]
        return
    if not dataset.shape or dataset.shape[0] == 0:
        return
    trailing_items = max(1, math.prod(dataset.shape[1:]))
    item_bytes = max(1, dataset.dtype.itemsize * trailing_items)
    step = max(1, CHUNK_SIZE // item_bytes)
    for start in range(0, dataset.shape[0], step):
        dataset[start : start + step]


def _reject_external_links(container: h5py.File, issues: IssueCollector) -> None:
    seen_groups: set[int] = set()

    def walk(group: h5py.Group) -> None:
        object_id = hash(group.id)
        if object_id in seen_groups:
            return
        seen_groups.add(object_id)
        for name in group:
            link = group.get(name, getlink=True)
            location = f"{group.name}/{name}".replace("//", "/")
            if isinstance(link, h5py.ExternalLink):
                issues.add(
                    HDF5_FILE,
                    location,
                    "EXTERNAL_LINK",
                    "external HDF5 links are not allowed",
                )
                continue
            if isinstance(link, h5py.SoftLink):
                issues.add(
                    HDF5_FILE, location, "SOFT_LINK", "soft HDF5 links are not allowed"
                )
                continue
            value = group.get(name)
            if isinstance(value, h5py.Group):
                walk(value)

    walk(container)


def _validate_metadata_shape(metadata: Any, issues: IssueCollector) -> None:
    if not isinstance(metadata, dict):
        issues.add(
            METADATA_FILE, "root", "METADATA_SCHEMA", "metadata must be an object"
        )
        return
    required = {
        "dataset_id": str,
        "data_format": str,
        "total_episodes": int,
        "total_steps": int,
        "observation_space": str,
        "action_space": str,
        "minari_version": str,
    }
    for field, expected_type in required.items():
        value = metadata.get(field)
        if not isinstance(value, expected_type) or isinstance(value, bool):
            issues.add(
                METADATA_FILE,
                field,
                "METADATA_SCHEMA",
                f"expected {expected_type.__name__}",
            )
    if metadata.get("data_format") != "hdf5":
        issues.add(
            METADATA_FILE,
            "data_format",
            "DATA_FORMAT_MISMATCH",
            repr(metadata.get("data_format")),
        )
    for field in ("observation_space", "action_space", "env_spec"):
        value = metadata.get(field)
        if not isinstance(value, str):
            continue
        try:
            json.loads(value)
        except json.JSONDecodeError as exc:
            issues.add(METADATA_FILE, field, "NESTED_JSON_INVALID", str(exc))


def _reconcile_metadata(
    path: Path,
    metadata: dict[str, Any],
    episode_groups: int,
    issues: IssueCollector,
) -> None:
    with h5py.File(path, "r") as container:
        attrs = dict(container.attrs)
    for field in ("total_episodes", "total_steps"):
        if (
            field in metadata
            and field in attrs
            and int(metadata[field]) != int(attrs[field])
        ):
            issues.add(
                HDF5_FILE,
                field,
                "METADATA_MISMATCH",
                f"json={metadata[field]}, hdf5={attrs[field]}",
            )
    if int(metadata.get("total_episodes", -1)) != episode_groups:
        issues.add(
            HDF5_FILE,
            "episodes",
            "EPISODE_COUNT_MISMATCH",
            f"metadata={metadata.get('total_episodes')}, groups={episode_groups}",
        )
    for field in ("observation_space", "action_space"):
        if (
            field in metadata
            and field in attrs
            and json.loads(metadata[field]) != json.loads(str(attrs[field]))
        ):
            issues.add(
                HDF5_FILE, field, "SPACE_MISMATCH", "metadata and HDF5 space differ"
            )
    metadata_id = metadata.get("dataset_id")
    hdf5_id = attrs.get("dataset_id")
    if (
        isinstance(metadata_id, str)
        and isinstance(hdf5_id, str)
        and not _dataset_ids_compatible(metadata_id, hdf5_id)
    ):
        issues.add(
            HDF5_FILE,
            "dataset_id",
            "DATASET_ID_MISMATCH",
            f"json={metadata_id}, hdf5={hdf5_id}",
        )
    metadata_version = metadata.get("minari_version")
    hdf5_version = attrs.get("minari_version")
    if (
        isinstance(metadata_version, str)
        and isinstance(hdf5_version, str)
        and _major_minor(metadata_version) != _major_minor(hdf5_version)
    ):
        issues.add(
            HDF5_FILE,
            "minari_version",
            "MINARI_VERSION_MISMATCH",
            f"json={metadata_version}, hdf5={hdf5_version}",
        )


def _dataset_ids_compatible(metadata_id: str, hdf5_id: str) -> bool:
    if metadata_id == hdf5_id:
        return True
    parts = metadata_id.split("/")
    return len(parts) >= 2 and "-".join(parts[-2:]) == hdf5_id


def _major_minor(value: str) -> tuple[int, int] | None:
    digits = []
    current = ""
    for character in value:
        if character.isdigit():
            current += character
        elif current:
            digits.append(int(current))
            current = ""
    if current:
        digits.append(int(current))
    return tuple(digits[:2]) if len(digits) >= 2 else None
