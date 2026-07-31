"""Reconcile already-downloaded text sources into the shared registry."""

import json
from datetime import UTC, datetime

from config import get_path
from fetch.catalog import HUGGING_FACE_SOURCE_IDS, get_source
from fetch.fetcher_service import build_dataset_record, destination_for
from fetch.models import AcquisitionResult, HuggingFaceAcquisition
from fetch.registry import DatasetRegistry
from fetch.resolvers import CommonCrawlResolver


def reconcile() -> dict:
    registry = DatasetRegistry()
    existing = {
        record["source_id"]: record
        for record in registry.read()
        if isinstance(record.get("source_id"), str)
    }

    for source_id in HUGGING_FACE_SOURCE_IDS:
        spec = get_source(source_id)
        request = spec.acquisition
        if not isinstance(request, HuggingFaceAcquisition):
            raise TypeError(f"{source_id} is not a Hugging Face requirement")

        destination = destination_for(spec)
        files = list(destination.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No downloaded JSONL files found in {destination}")

        result = AcquisitionResult(
            files=files,
            output=destination,
            download_url=f"hf://datasets/{request.repo_id}",
            revision=request.revision or "default",
            details={"splits": sorted(path.stem for path in files)},
        )
        registry.upsert(
            build_dataset_record(
                spec,
                result,
                downloaded_at=_previous_timestamp(existing, source_id),
            )
        )

    text_root = get_path("paths.text_path")
    wet_files = [
        *text_root.glob("*.warc.wet.gz"),
        *(text_root / "common_crawl").glob("*.warc.wet.gz"),
    ]
    if not wet_files:
        raise FileNotFoundError(f"No Common Crawl WET file found in {text_root}")

    wet_file = max(wet_files, key=lambda path: path.stat().st_mtime)
    resolved = CommonCrawlResolver().identify(wet_file.name)
    revision, download_url = resolved or ("unknown", None)
    spec = get_source("common_crawl_wet")
    result = AcquisitionResult(
        files=[wet_file],
        output=wet_file,
        download_url=download_url or "unknown",
        revision=revision,
    )
    registry.upsert(
        build_dataset_record(
            spec,
            result,
            downloaded_at=datetime.fromtimestamp(
                wet_file.stat().st_mtime,
                tz=UTC,
            ).isoformat(),
        )
    )

    return {
        "registry": str(get_path("paths.registry_path")),
        "source_count": len(registry.read()),
    }


def execute() -> None:
    print(json.dumps(reconcile(), indent=2))


def _previous_timestamp(
    existing: dict[str, dict],
    source_id: str,
) -> str | None:
    timestamp = existing.get(source_id, {}).get("downloaded_at")
    return timestamp if isinstance(timestamp, str) else None


if __name__ == "__main__":
    execute()
