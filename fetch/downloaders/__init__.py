"""Acquisition tool adapters."""

from .http import CurlDownloader
from .huggingface import HuggingFaceDownloader
from .minari import MinariDownloader

__all__ = ["CurlDownloader", "HuggingFaceDownloader", "MinariDownloader"]
