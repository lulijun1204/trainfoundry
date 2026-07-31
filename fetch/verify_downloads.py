"""Verify that all expected raw sources match the shared registry."""

import json
from typing import Any

from fetch.registry import DatasetRegistry, verify_record

EXPECTED_SOURCE_IDS = {
    "wikitext_2_raw",
    "common_crawl_wet",
    "dolly_15k",
    "hh_rlhf_helpful_base",
    "coco_2017_validation",
    "ucf101",
    "d4rl_pointmaze_umaze_minari",
}
REQUIRED_FIELDS = {
    "source_id",
    "homepage",
    "download_url",
    "dataset_revision",
    "license",
    "permitted_use",
    "region",
    "contains_pii",
    "retention_policy",
    "downloaded_at",
    "files",
    "file_count",
    "total_bytes",
    "status",
}


def verify_all() -> dict[str, Any]:
    records = DatasetRegistry().read()
    by_source = {record["source_id"]: record for record in records}
    actual_ids = set(by_source)
    if actual_ids != EXPECTED_SOURCE_IDS:
        raise ValueError(
            f"Unexpected registry sources: missing={EXPECTED_SOURCE_IDS - actual_ids}, "
            f"extra={actual_ids - EXPECTED_SOURCE_IDS}"
        )

    total_files = 0
    total_bytes = 0
    sources = []
    for source_id in sorted(EXPECTED_SOURCE_IDS):
        record = by_source[source_id]
        _verify_required_fields(record)
        verify_record(record)
        total_files += record["file_count"]
        total_bytes += record["total_bytes"]
        sources.append(
            {
                "source_id": source_id,
                "file_count": record["file_count"],
                "total_bytes": record["total_bytes"],
                "status": "verified",
            }
        )

    return {
        "source_count": len(records),
        "file_count": total_files,
        "total_bytes": total_bytes,
        "status": "all_verified",
        "sources": sources,
    }


def execute() -> None:
    print(json.dumps(verify_all(), indent=2))


def _verify_required_fields(record: dict[str, Any]) -> None:
    missing_fields = REQUIRED_FIELDS - set(record)
    if missing_fields:
        raise ValueError(
            f"{record.get('source_id')} is missing registry fields: {missing_fields}"
        )
    if record["status"] != "complete":
        raise ValueError(f"{record['source_id']} is not complete")


if __name__ == "__main__":
    execute()
