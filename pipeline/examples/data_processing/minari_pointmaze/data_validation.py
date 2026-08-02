"""Step 2 implementation: Minari trajectory and PointMaze semantic validation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .common import (
    DATA_DIRECTORY,
    HDF5_FILE,
    IssueCollector,
    ValidationIssue,
    dataclass_to_dict,
)

_EPISODE_NAME = re.compile(r"episode_(\d+)$")
_REWARD_ATTRS = (
    "rewards_max",
    "rewards_min",
    "rewards_mean",
    "rewards_std",
    "rewards_sum",
)


@dataclass(frozen=True, slots=True)
class DataValidationSummary:
    episode_count: int
    valid_episodes: int
    rejected_episodes: int
    transition_count: int
    episode_steps_min: int | None
    episode_steps_max: int | None
    episode_steps_mean: float | None
    return_min: float | None
    return_max: float | None
    return_mean: float | None
    successful_episodes: int
    success_rate: float | None
    action_min: tuple[float, ...]
    action_max: tuple[float, ...]
    duplicate_episodes: int
    episodes_with_reward_stats: int
    error_counts: dict[str, int]
    warning_counts: dict[str, int]
    issue_samples: tuple[ValidationIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


def validate_dataset(
    dataset_root: Path,
    *,
    max_issue_samples: int = 20,
) -> DataValidationSummary:
    """Scan every episode, keeping only per-dataset statistics in memory."""
    issues = IssueCollector(max_issue_samples)
    path = dataset_root / DATA_DIRECTORY / HDF5_FILE
    if not path.is_file():
        issues.add(HDF5_FILE, str(path), "MISSING_FILE", "main_data.hdf5 is missing")
        return _empty_summary(issues)

    steps: list[int] = []
    returns: list[float] = []
    rejected: set[str] = set()
    fingerprints: dict[str, str] = {}
    duplicate_count = 0
    reward_stats_count = 0
    successful = 0
    action_min: np.ndarray | None = None
    action_max: np.ndarray | None = None
    physical_episode_count = 0

    try:
        with h5py.File(path, "r") as container:
            observation_spec = _read_space(container, "observation_space", issues)
            action_spec = _read_space(container, "action_space", issues)
            names = sorted(
                (name for name in container if _EPISODE_NAME.fullmatch(name)),
                key=lambda name: int(_EPISODE_NAME.fullmatch(name).group(1)),  # type: ignore[union-attr]
            )
            physical_episode_count = len(names)
            unexpected = [
                name
                for name, value in container.items()
                if isinstance(value, h5py.Group) and not _EPISODE_NAME.fullmatch(name)
            ]
            for name in unexpected:
                issues.add(
                    HDF5_FILE,
                    name,
                    "UNEXPECTED_ROOT_GROUP",
                    "expected episode_<integer>",
                )

            seen_ids: set[int] = set()
            for name in names:
                before = issues.error_total
                result = _validate_episode(
                    container[name],
                    name,
                    observation_spec,
                    action_spec,
                    seen_ids,
                    issues,
                )
                if result is not None:
                    (
                        episode_steps,
                        episode_return,
                        episode_success,
                        actions,
                        fingerprint,
                        has_reward_stats,
                    ) = result
                    steps.append(episode_steps)
                    returns.append(episode_return)
                    successful += int(episode_success)
                    reward_stats_count += int(has_reward_stats)
                    flat_actions = actions.reshape(actions.shape[0], -1)
                    current_min = np.min(flat_actions, axis=0)
                    current_max = np.max(flat_actions, axis=0)
                    action_min = (
                        current_min
                        if action_min is None
                        else np.minimum(action_min, current_min)
                    )
                    action_max = (
                        current_max
                        if action_max is None
                        else np.maximum(action_max, current_max)
                    )
                    if fingerprint in fingerprints:
                        duplicate_count += 1
                        issues.add(
                            HDF5_FILE,
                            name,
                            "DUPLICATE_EPISODE",
                            f"identical to {fingerprints[fingerprint]}",
                            severity="WARNING",
                        )
                    else:
                        fingerprints[fingerprint] = name
                if issues.error_total > before:
                    rejected.add(name)

            _validate_root_totals(container, len(names), sum(steps), issues)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        issues.add(HDF5_FILE, "root", "HDF5_READ_ERROR", str(exc))
        return _empty_summary(issues)

    return DataValidationSummary(
        episode_count=physical_episode_count,
        valid_episodes=max(0, physical_episode_count - len(rejected)),
        rejected_episodes=len(rejected),
        transition_count=sum(steps),
        episode_steps_min=min(steps) if steps else None,
        episode_steps_max=max(steps) if steps else None,
        episode_steps_mean=float(np.mean(steps)) if steps else None,
        return_min=min(returns) if returns else None,
        return_max=max(returns) if returns else None,
        return_mean=float(np.mean(returns)) if returns else None,
        successful_episodes=successful,
        success_rate=(
            successful / physical_episode_count if physical_episode_count else None
        ),
        action_min=tuple(float(value) for value in action_min)
        if action_min is not None
        else (),
        action_max=tuple(float(value) for value in action_max)
        if action_max is not None
        else (),
        duplicate_episodes=duplicate_count,
        episodes_with_reward_stats=reward_stats_count,
        error_counts=dict(sorted(issues.error_counts.items())),
        warning_counts=dict(sorted(issues.warning_counts.items())),
        issue_samples=tuple(issues.samples),
    )


def _validate_episode(
    group: h5py.Group,
    name: str,
    observation_spec: dict[str, Any] | None,
    action_spec: dict[str, Any] | None,
    seen_ids: set[int],
    issues: IssueCollector,
) -> tuple[int, float, bool, np.ndarray, str, bool] | None:
    location = name
    episode_id = group.attrs.get("id")
    expected_id = int(_EPISODE_NAME.fullmatch(name).group(1))  # type: ignore[union-attr]
    if not isinstance(episode_id, (int, np.integer)) or int(episode_id) != expected_id:
        issues.add(
            HDF5_FILE,
            location,
            "EPISODE_ID_MISMATCH",
            f"name={expected_id}, attr={episode_id!r}",
        )
    elif int(episode_id) in seen_ids:
        issues.add(HDF5_FILE, location, "DUPLICATE_EPISODE_ID", str(episode_id))
    else:
        seen_ids.add(int(episode_id))

    required = ("observations", "actions", "rewards", "terminations", "truncations")
    missing = [field for field in required if field not in group]
    for field in missing:
        issues.add(
            HDF5_FILE,
            f"{location}/{field}",
            "MISSING_EPISODE_FIELD",
            "required by Minari",
        )
    if missing:
        return None

    observations = _load_leaves(group["observations"])
    actions = _load_leaves(group["actions"])
    infos = _load_leaves(group["infos"]) if "infos" in group else {}
    rewards = _read_vector(group["rewards"], f"{location}/rewards", issues)
    terminations = _read_vector(
        group["terminations"], f"{location}/terminations", issues
    )
    truncations = _read_vector(group["truncations"], f"{location}/truncations", issues)
    if rewards is None or terminations is None or truncations is None:
        return None

    action_lengths = {value.shape[0] for value in actions.values() if value.ndim > 0}
    if not actions or len(action_lengths) != 1:
        issues.add(
            HDF5_FILE,
            f"{location}/actions",
            "ACTION_LENGTH_MISMATCH",
            str(sorted(action_lengths)),
        )
        return None
    transition_count = action_lengths.pop()
    if transition_count < 1:
        issues.add(HDF5_FILE, location, "EMPTY_EPISODE", "episode has no transitions")
        return None

    _validate_leaf_lengths(
        observations, transition_count + 1, f"{location}/observations", issues
    )
    _validate_leaf_lengths(infos, transition_count + 1, f"{location}/infos", issues)
    for field, values in (
        ("rewards", rewards),
        ("terminations", terminations),
        ("truncations", truncations),
    ):
        if len(values) != transition_count:
            issues.add(
                HDF5_FILE,
                f"{location}/{field}",
                "TRANSITION_LENGTH_MISMATCH",
                f"expected {transition_count}, got {len(values)}",
            )

    _validate_finite(observations, f"{location}/observations", issues)
    _validate_finite(actions, f"{location}/actions", issues)
    _validate_finite(infos, f"{location}/infos", issues)
    if not np.issubdtype(rewards.dtype, np.number) or not np.all(np.isfinite(rewards)):
        issues.add(
            HDF5_FILE,
            f"{location}/rewards",
            "NON_FINITE_VALUE",
            "rewards must be finite numeric values",
        )
    _validate_flags(terminations, truncations, transition_count, location, issues)
    _validate_space(
        observations,
        observation_spec,
        transition_count + 1,
        f"{location}/observations",
        issues,
    )
    _validate_space(
        actions, action_spec, transition_count, f"{location}/actions", issues
    )
    _validate_pointmaze(observations, infos, rewards, location, issues)
    has_reward_stats = _validate_episode_attrs(
        group, transition_count, rewards, location, issues
    )

    action_array = actions.get("")
    if action_array is None or action_array.ndim < 1 or action_array.shape[0] == 0:
        issues.add(
            HDF5_FILE,
            f"{location}/actions",
            "ACTION_SCHEMA",
            "PointMaze requires one Box action array",
        )
        return None
    fingerprint = _fingerprint(
        observations, actions, infos, rewards, terminations, truncations
    )
    success = (
        bool(infos.get("success", np.array([False]))[-1])
        if "success" in infos
        else False
    )
    return (
        transition_count,
        float(np.sum(rewards, dtype=np.float64)),
        success,
        action_array,
        fingerprint,
        has_reward_stats,
    )


def _load_leaves(
    node: h5py.Group | h5py.Dataset, prefix: str = ""
) -> dict[str, np.ndarray]:
    if isinstance(node, h5py.Dataset):
        return {prefix: np.asarray(node[()])}
    leaves: dict[str, np.ndarray] = {}
    for name, value in node.items():
        child = f"{prefix}/{name}" if prefix else name
        leaves.update(_load_leaves(value, child))
    return leaves


def _read_vector(node: Any, location: str, issues: IssueCollector) -> np.ndarray | None:
    if not isinstance(node, h5py.Dataset):
        issues.add(
            HDF5_FILE,
            location,
            "EXPECTED_DATASET",
            "expected a one-dimensional dataset",
        )
        return None
    values = np.asarray(node[()])
    if values.ndim != 1:
        issues.add(HDF5_FILE, location, "EXPECTED_VECTOR", f"shape={values.shape}")
        return None
    return values


def _validate_leaf_lengths(
    leaves: dict[str, np.ndarray], expected: int, location: str, issues: IssueCollector
) -> bool:
    if not leaves:
        issues.add(HDF5_FILE, location, "EMPTY_STRUCTURE", "no leaf datasets")
    for name, values in leaves.items():
        actual = values.shape[0] if values.ndim else None
        if actual != expected:
            issues.add(
                HDF5_FILE,
                f"{location}/{name}",
                "SEQUENCE_LENGTH_MISMATCH",
                f"expected {expected}, got {actual}",
            )


def _validate_finite(
    leaves: dict[str, np.ndarray], location: str, issues: IssueCollector
) -> None:
    for name, values in leaves.items():
        if np.issubdtype(values.dtype, np.number) and not np.all(np.isfinite(values)):
            issues.add(
                HDF5_FILE,
                f"{location}/{name}",
                "NON_FINITE_VALUE",
                "contains NaN or infinity",
            )


def _validate_flags(
    terminations: np.ndarray,
    truncations: np.ndarray,
    size: int,
    location: str,
    issues: IssueCollector,
) -> None:
    for field, values in (("terminations", terminations), ("truncations", truncations)):
        if not np.issubdtype(values.dtype, np.bool_):
            issues.add(
                HDF5_FILE,
                f"{location}/{field}",
                "FLAG_DTYPE",
                f"expected bool, got {values.dtype}",
            )
    if len(terminations) != size or len(truncations) != size:
        return
    if np.any(terminations[:-1]) or np.any(truncations[:-1]):
        issues.add(
            HDF5_FILE,
            location,
            "EARLY_BOUNDARY",
            "termination/truncation may only mark the final transition",
        )
    if bool(terminations[-1]) == bool(truncations[-1]):
        issues.add(
            HDF5_FILE,
            location,
            "FINAL_BOUNDARY",
            "final transition must be exactly one of terminated or truncated",
        )


def _read_space(
    container: h5py.File, field: str, issues: IssueCollector
) -> dict[str, Any] | None:
    value = container.attrs.get(field)
    if not isinstance(value, str):
        issues.add(
            HDF5_FILE, field, "SPACE_MISSING", "serialized Gymnasium space is required"
        )
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        issues.add(HDF5_FILE, field, "SPACE_INVALID", str(exc))
        return None
    if not isinstance(parsed, dict):
        issues.add(HDF5_FILE, field, "SPACE_INVALID", "space must be an object")
        return None
    return parsed


def _validate_space(
    leaves: dict[str, np.ndarray],
    spec: dict[str, Any] | None,
    expected: int,
    location: str,
    issues: IssueCollector,
) -> None:
    if spec is None:
        return
    _validate_space_node(leaves, spec, "", expected, location, issues)


def _validate_space_node(
    leaves: dict[str, np.ndarray],
    spec: dict[str, Any],
    prefix: str,
    expected: int,
    location: str,
    issues: IssueCollector,
) -> None:
    space_type = spec.get("type")
    if space_type == "Dict":
        # Minari serializes Gymnasium Dict children under ``subspaces``.
        spaces = spec.get("subspaces", spec.get("spaces"))
        if not isinstance(spaces, dict):
            issues.add(
                HDF5_FILE, location, "SPACE_INVALID", "Dict.spaces must be an object"
            )
            return
        expected_prefixes = set(spaces)
        actual_prefixes = {
            name[len(prefix) :].lstrip("/").split("/", 1)[0]
            for name in leaves
            if name.startswith(prefix)
        }
        if actual_prefixes != expected_prefixes:
            issues.add(
                HDF5_FILE,
                location,
                "SPACE_KEYS_MISMATCH",
                f"expected={sorted(expected_prefixes)}, actual={sorted(actual_prefixes)}",
            )
        for name, child in spaces.items():
            if isinstance(child, dict):
                child_prefix = f"{prefix}/{name}" if prefix else name
                _validate_space_node(
                    leaves, child, child_prefix, expected, location, issues
                )
        return
    values = leaves.get(prefix)
    if values is None:
        issues.add(
            HDF5_FILE,
            f"{location}/{prefix}",
            "SPACE_LEAF_MISSING",
            "declared leaf is absent",
        )
        return
    if values.shape[0] != expected:
        return
    shape = tuple(spec.get("shape", ()))
    if tuple(values.shape[1:]) != shape:
        issues.add(
            HDF5_FILE,
            f"{location}/{prefix}",
            "SPACE_SHAPE_MISMATCH",
            f"expected={shape}, actual={values.shape[1:]}",
        )
    declared_dtype = spec.get("dtype")
    if declared_dtype is not None:
        try:
            if values.dtype != np.dtype(declared_dtype):
                issues.add(
                    HDF5_FILE,
                    f"{location}/{prefix}",
                    "SPACE_DTYPE_MISMATCH",
                    f"expected={declared_dtype}, actual={values.dtype}",
                )
        except TypeError:
            issues.add(
                HDF5_FILE,
                f"{location}/{prefix}",
                "SPACE_INVALID",
                f"invalid dtype {declared_dtype!r}",
            )
    if space_type == "Box":
        _validate_box_bounds(values, spec, f"{location}/{prefix}", issues)
    elif space_type == "Discrete":
        start, count = int(spec.get("start", 0)), int(spec.get("n", 0))
        if not np.issubdtype(values.dtype, np.integer) or np.any(
            (values < start) | (values >= start + count)
        ):
            issues.add(
                HDF5_FILE,
                f"{location}/{prefix}",
                "SPACE_BOUNDS",
                f"outside [{start}, {start + count})",
            )


def _validate_box_bounds(
    values: np.ndarray, spec: dict[str, Any], location: str, issues: IssueCollector
) -> None:
    try:
        low = np.asarray(spec["low"], dtype=np.float64)
        high = np.asarray(spec["high"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        issues.add(HDF5_FILE, location, "SPACE_INVALID", f"invalid Box bounds: {exc}")
        return
    tolerance = 1e-6
    if np.any(values < low - tolerance) or np.any(values > high + tolerance):
        issues.add(
            HDF5_FILE,
            location,
            "SPACE_BOUNDS",
            "value lies outside declared Box bounds",
        )


def _validate_pointmaze(
    observations: dict[str, np.ndarray],
    infos: dict[str, np.ndarray],
    rewards: np.ndarray,
    location: str,
    issues: IssueCollector,
) -> None:
    required_obs = {"achieved_goal", "desired_goal", "observation"}
    required_info = {"goal", "success"}
    for name in sorted(required_obs - observations.keys()):
        issues.add(
            HDF5_FILE,
            f"{location}/observations/{name}",
            "POINTMAZE_FIELD_MISSING",
            "required PointMaze observation",
        )
    for name in sorted(required_info - infos.keys()):
        issues.add(
            HDF5_FILE,
            f"{location}/infos/{name}",
            "POINTMAZE_FIELD_MISSING",
            "required PointMaze info",
        )
    if required_obs <= observations.keys():
        state, achieved = observations["observation"], observations["achieved_goal"]
        if (
            state.ndim != 2
            or state.shape[1] < 2
            or achieved.shape != state[:, :2].shape
            or not np.allclose(achieved, state[:, :2], rtol=0, atol=1e-8)
        ):
            issues.add(
                HDF5_FILE,
                location,
                "ACHIEVED_GOAL_MISMATCH",
                "achieved_goal must equal observation[..., :2]",
            )
    if (
        "desired_goal" in observations
        and "goal" in infos
        and (
            observations["desired_goal"].shape != infos["goal"].shape
            or not np.allclose(
                observations["desired_goal"], infos["goal"], rtol=0, atol=1e-8
            )
        )
    ):
        issues.add(
            HDF5_FILE,
            location,
            "DESIRED_GOAL_MISMATCH",
            "desired_goal must equal infos/goal",
        )
    if "success" in infos:
        success = infos["success"]
        if not np.issubdtype(success.dtype, np.bool_):
            issues.add(
                HDF5_FILE,
                f"{location}/infos/success",
                "SUCCESS_DTYPE",
                f"expected bool, got {success.dtype}",
            )
        if success.ndim != 1 or len(success) != len(rewards) + 1:
            issues.add(
                HDF5_FILE,
                f"{location}/infos/success",
                "SUCCESS_LENGTH_MISMATCH",
                "expected T + 1",
            )
        elif not np.array_equal(rewards.astype(bool), success[1:].astype(bool)):
            issues.add(
                HDF5_FILE,
                location,
                "REWARD_SUCCESS_MISMATCH",
                "reward[t] must equal success[t + 1]",
            )
    if not np.all(np.isin(rewards, (0, 1))):
        issues.add(
            HDF5_FILE,
            f"{location}/rewards",
            "POINTMAZE_REWARD",
            "expected sparse rewards in {0, 1}",
        )


def _validate_episode_attrs(
    group: h5py.Group,
    steps: int,
    rewards: np.ndarray,
    location: str,
    issues: IssueCollector,
) -> None:
    if int(group.attrs.get("total_steps", -1)) != steps:
        issues.add(
            HDF5_FILE,
            location,
            "EPISODE_STEPS_MISMATCH",
            f"attr={group.attrs.get('total_steps')}, actual={steps}",
        )
    expected = {
        "rewards_max": float(np.max(rewards)),
        "rewards_min": float(np.min(rewards)),
        "rewards_mean": float(np.mean(rewards)),
        "rewards_std": float(np.std(rewards)),
        "rewards_sum": float(np.sum(rewards)),
    }
    present = [field in group.attrs for field in _REWARD_ATTRS]
    if not any(present):
        # Cached summaries help discovery but are not transition data. Their absence
        # does not make an otherwise complete Minari episode unusable for training.
        return False
    if not all(present):
        issues.add(
            HDF5_FILE,
            location,
            "REWARD_STATS_PARTIAL",
            "reward summary attributes must be all present or all absent",
            severity="WARNING",
        )
    for field in _REWARD_ATTRS:
        if field not in group.attrs:
            continue
        actual = group.attrs.get(field)
        if not isinstance(actual, (int, float, np.number)) or not math.isclose(
            float(actual), expected[field], rel_tol=1e-6, abs_tol=1e-8
        ):
            issues.add(
                HDF5_FILE,
                f"{location}@{field}",
                "REWARD_STAT_MISMATCH",
                f"attr={actual!r}, actual={expected[field]}",
            )
    return all(present)


def _validate_root_totals(
    container: h5py.File, episodes: int, steps: int, issues: IssueCollector
) -> None:
    for field, actual in (("total_episodes", episodes), ("total_steps", steps)):
        declared = container.attrs.get(field)
        if not isinstance(declared, (int, np.integer)) or int(declared) != actual:
            issues.add(
                HDF5_FILE,
                field,
                "ROOT_TOTAL_MISMATCH",
                f"declared={declared!r}, actual={actual}",
            )


def _fingerprint(
    observations: dict[str, np.ndarray],
    actions: dict[str, np.ndarray],
    infos: dict[str, np.ndarray],
    rewards: np.ndarray,
    terminations: np.ndarray,
    truncations: np.ndarray,
) -> str:
    digest = sha256()
    for namespace, leaves in (
        ("observations", observations),
        ("actions", actions),
        ("infos", infos),
    ):
        for name, values in sorted(leaves.items()):
            digest.update(f"{namespace}/{name}:{values.dtype}:{values.shape}".encode())
            digest.update(np.ascontiguousarray(values).tobytes())
    for name, values in (
        ("rewards", rewards),
        ("terminations", terminations),
        ("truncations", truncations),
    ):
        digest.update(f"{name}:{values.dtype}:{values.shape}".encode())
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _empty_summary(issues: IssueCollector) -> DataValidationSummary:
    return DataValidationSummary(
        episode_count=0,
        valid_episodes=0,
        rejected_episodes=0,
        transition_count=0,
        episode_steps_min=None,
        episode_steps_max=None,
        episode_steps_mean=None,
        return_min=None,
        return_max=None,
        return_mean=None,
        successful_episodes=0,
        success_rate=None,
        action_min=(),
        action_max=(),
        duplicate_episodes=0,
        episodes_with_reward_stats=0,
        error_counts=dict(sorted(issues.error_counts.items())),
        warning_counts=dict(sorted(issues.warning_counts.items())),
        issue_samples=tuple(issues.samples),
    )
