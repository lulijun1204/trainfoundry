"""Hugging Face datasets acquisition adapter."""

from typing import Any

from fetch.models import HuggingFaceAcquisition


class HuggingFaceDownloader:
    """Load a logical dataset through the Hugging Face datasets client."""

    def fetch(self, request: HuggingFaceAcquisition) -> Any:
        from datasets import load_dataset

        kwargs = {}
        if request.data_dir is not None:
            kwargs["data_dir"] = request.data_dir
        if request.revision is not None:
            kwargs["revision"] = request.revision
        return load_dataset(request.repo_id, request.config, **kwargs)
