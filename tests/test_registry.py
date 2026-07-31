import json

from fetch.models import DatasetRecord
from fetch.registry import DatasetRegistry


def test_registry_upserts_without_duplicate_sources_and_sorts(tmp_path):
    registry_path = tmp_path / "source_registry.json"
    registry = DatasetRegistry(registry_path)

    registry.upsert(_record("zeta", "pending"))
    registry.upsert(_record("alpha", "complete"))
    registry.upsert(_record("zeta", "complete"))

    records = json.loads(registry_path.read_text(encoding="utf-8"))
    assert [record["source_id"] for record in records] == ["alpha", "zeta"]
    assert records[1]["status"] == "complete"
    assert records[1]["downloaded_at"]
    assert records[1]["file_count"] == 0
    assert not registry_path.with_suffix(".json.tmp").exists()


def _record(source_id: str, status: str) -> DatasetRecord:
    return DatasetRecord(
        source_id=source_id,
        modality="test",
        homepage="https://example.invalid",
        download_url="https://example.invalid/data",
        dataset_revision="v1",
        license="MIT",
        permitted_use="test",
        region="global",
        contains_pii=False,
        retention_policy="temporary",
        output="/tmp/data",
        downloaded_at="2026-01-01T00:00:00+00:00",
        files=[],
        total_bytes=0,
        status=status,
    )
