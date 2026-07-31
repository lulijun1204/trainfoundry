"""Inspect a Minari dataset without coupling inspection to acquisition."""

from typing import Any


def inspect_minari_dataset(dataset: Any) -> dict[str, Any]:
    episode = next(dataset.iterate_episodes())
    observation_shapes = {
        key: list(value.shape) for key, value in episode.observations.items()
    }
    return {
        "total_steps": dataset.total_steps,
        "total_episodes": dataset.total_episodes,
        "schema": {
            "observations": observation_shapes,
            "actions": list(episode.actions.shape),
            "rewards": list(episode.rewards.shape),
            "terminations": list(episode.terminations.shape),
            "truncations": list(episode.truncations.shape),
        },
    }
