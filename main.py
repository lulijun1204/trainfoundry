"""Minimal demo for loading the downloaded Minari robot dataset."""

import os

from config import get_path

DATASET_ID = "D4RL/pointmaze/umaze-v2"


def main() -> None:
    os.environ["MINARI_DATASETS_PATH"] = str(get_path("paths.robot_path"))

    import minari

    dataset = minari.load_dataset(DATASET_ID)
    episode = next(dataset.iterate_episodes())

    print(f"dataset_id: {DATASET_ID}")
    print(f"total_steps: {dataset.total_steps}")
    print(f"total_episodes: {dataset.total_episodes}")
    print(f"observation_keys: {tuple(episode.observations.keys())}")
    print(f"actions_shape: {episode.actions.shape}")
    print(f"rewards_shape: {episode.rewards.shape}")


if __name__ == "__main__":
    main()
