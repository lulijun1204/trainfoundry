import json

import h5py
import numpy as np

from pipeline.examples.data_processing.minari_pointmaze.data_validation import (
    validate_dataset,
)
from pipeline.examples.data_processing.minari_pointmaze.file_validation import (
    validate_file_format,
)

OBSERVATION_SPACE = {
    "type": "Dict",
    "subspaces": {
        "achieved_goal": {
            "type": "Box",
            "dtype": "float64",
            "shape": [2],
            "low": [-100.0, -100.0],
            "high": [100.0, 100.0],
        },
        "desired_goal": {
            "type": "Box",
            "dtype": "float64",
            "shape": [2],
            "low": [-100.0, -100.0],
            "high": [100.0, 100.0],
        },
        "observation": {
            "type": "Box",
            "dtype": "float64",
            "shape": [4],
            "low": [-100.0] * 4,
            "high": [100.0] * 4,
        },
    },
}
ACTION_SPACE = {
    "type": "Box",
    "dtype": "float32",
    "shape": [2],
    "low": [-1.0, -1.0],
    "high": [1.0, 1.0],
}


def _write_fixture(root, *, duplicate=False):
    data_root = root / "data"
    data_root.mkdir(parents=True)
    observation_json = json.dumps(OBSERVATION_SPACE)
    action_json = json.dumps(ACTION_SPACE)
    metadata = {
        "dataset_id": "D4RL/pointmaze/umaze-v2",
        "data_format": "hdf5",
        "total_episodes": 2,
        "total_steps": 4,
        "observation_space": observation_json,
        "action_space": action_json,
        "env_spec": json.dumps({"id": "PointMaze_UMaze-v3"}),
        "minari_version": "0.4.3",
    }
    (data_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with h5py.File(data_root / "main_data.hdf5", "w") as container:
        container.attrs.update(
            {
                "dataset_id": "pointmaze-umaze-v2",
                "total_episodes": 2,
                "total_steps": 4,
                "observation_space": observation_json,
                "action_space": action_json,
                "minari_version": "~=0.4",
            }
        )
        for episode_id in range(2):
            offset = 0.0 if duplicate else float(episode_id)
            group = container.create_group(f"episode_{episode_id}")
            states = np.array(
                [
                    [offset, 0.0, 0.0, 0.0],
                    [offset + 0.5, 0.0, 0.0, 0.0],
                    [offset + 1.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
            goals = np.repeat(np.array([[offset + 1.0, 0.0]]), 3, axis=0)
            actions = np.array([[0.5, 0.0], [0.5, 0.0]], dtype=np.float32)
            rewards = np.array([0.0, 1.0], dtype=np.float64)
            group.create_dataset("actions", data=actions)
            group.create_dataset("rewards", data=rewards)
            group.create_dataset("terminations", data=np.array([False, False]))
            group.create_dataset("truncations", data=np.array([False, True]))
            observations = group.create_group("observations")
            observations.create_dataset("observation", data=states)
            observations.create_dataset("achieved_goal", data=states[:, :2])
            observations.create_dataset("desired_goal", data=goals)
            infos = group.create_group("infos")
            infos.create_dataset("goal", data=goals)
            infos.create_dataset("success", data=np.array([False, False, True]))
            infos.create_dataset("qpos", data=states[:, :2])
            infos.create_dataset("qvel", data=states[:, 2:])
            group.attrs.update(
                {
                    "id": episode_id,
                    "total_steps": 2,
                    "rewards_max": 1.0,
                    "rewards_min": 0.0,
                    "rewards_mean": 0.5,
                    "rewards_std": 0.5,
                    "rewards_sum": 1.0,
                }
            )


def test_minari_file_validation_reads_and_reconciles_package(tmp_path):
    root = tmp_path / "robot"
    _write_fixture(root)

    summary = validate_file_format(root)

    assert summary.file_count == 2
    assert summary.valid_files == 2
    assert summary.hdf5_episode_groups == 2
    assert summary.hdf5_dataset_count == 22
    assert set(summary.archive_sha256) == {"main_data.hdf5", "metadata.json"}
    assert summary.error_counts == {}


def test_minari_file_validation_reports_metadata_mismatch(tmp_path):
    root = tmp_path / "robot"
    _write_fixture(root)
    metadata_path = root / "data/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["total_episodes"] = 3
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    summary = validate_file_format(root)

    assert summary.error_counts == {
        "EPISODE_COUNT_MISMATCH": 1,
        "METADATA_MISMATCH": 1,
    }


def test_minari_data_validation_checks_trajectory_semantics(tmp_path):
    root = tmp_path / "robot"
    _write_fixture(root)

    summary = validate_dataset(root)

    assert summary.episode_count == 2
    assert summary.valid_episodes == 2
    assert summary.transition_count == 4
    assert summary.successful_episodes == 2
    assert summary.action_min == (0.5, 0.0)
    assert summary.action_max == (0.5, 0.0)
    assert summary.error_counts == {}
    assert summary.warning_counts == {}


def test_minari_data_validation_reports_temporal_and_pointmaze_errors(tmp_path):
    root = tmp_path / "robot"
    _write_fixture(root)
    with h5py.File(root / "data/main_data.hdf5", "r+") as container:
        episode = container["episode_1"]
        episode["actions"][0, 0] = 2.0
        episode["observations/achieved_goal"][0, 0] = 999.0
        episode["terminations"][0] = True
        episode["infos/success"][2] = False

    summary = validate_dataset(root)

    assert summary.valid_episodes == 1
    assert summary.rejected_episodes == 1
    assert summary.error_counts == {
        "ACHIEVED_GOAL_MISMATCH": 1,
        "EARLY_BOUNDARY": 1,
        "REWARD_SUCCESS_MISMATCH": 1,
        "SPACE_BOUNDS": 2,
    }


def test_minari_data_validation_profiles_duplicate_episodes_as_warning(tmp_path):
    root = tmp_path / "robot"
    _write_fixture(root, duplicate=True)

    summary = validate_dataset(root)

    assert summary.valid_episodes == 2
    assert summary.duplicate_episodes == 1
    assert summary.error_counts == {}
    assert summary.warning_counts == {"DUPLICATE_EPISODE": 1}


def test_minari_data_validation_accepts_absent_cached_reward_stats(tmp_path):
    root = tmp_path / "robot"
    _write_fixture(root)
    with h5py.File(root / "data/main_data.hdf5", "r+") as container:
        for field in (
            "rewards_max",
            "rewards_min",
            "rewards_mean",
            "rewards_std",
            "rewards_sum",
        ):
            del container["episode_1"].attrs[field]

    summary = validate_dataset(root)

    assert summary.valid_episodes == 2
    assert summary.episodes_with_reward_stats == 1
    assert summary.error_counts == {}
    assert summary.warning_counts == {}
