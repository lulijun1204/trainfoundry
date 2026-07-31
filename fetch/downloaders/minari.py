"""Minari acquisition adapter."""

import os
from pathlib import Path
from typing import Any

from fetch.models import MinariAcquisition


class MinariDownloader:
    """Download and load an offline-RL dataset through Minari."""

    def fetch(
        self,
        request: MinariAcquisition,
        destination: Path,
    ) -> tuple[Any, Path]:
        destination.mkdir(parents=True, exist_ok=True)
        os.environ["MINARI_DATASETS_PATH"] = str(destination)

        import minari

        dataset = minari.load_dataset(request.dataset_id, download=True)
        return dataset, destination / request.dataset_id
