import pytest

from fetch.fetcher_service import FetcherService
from fetch.metadata_registrar import FetchMetadataRegistrar
from fetch.models import (
    DatasetMeta,
    HttpAcquisition,
    HttpFileRequest,
    OutputSpec,
)
from metadata import MetadataRepository


class FakeHttpDownloader:
    def __init__(self, content=b"downloaded"):
        self.calls = 0
        self.content = content

    def fetch(self, request, destination, *, force=False):
        self.calls += 1
        destination.mkdir(parents=True, exist_ok=True)
        outputs = []
        for item in request.files:
            output = destination / item.output_name
            output.write_bytes(self.content)
            outputs.append(output)
        return outputs


class FakeRegistry:
    def __init__(self, path):
        self.path = path
        self.records = []
        self.values = {}

    def find(self, source_id):
        return self.values.get(source_id)

    def upsert(self, record):
        self.records.append(record)
        self.values[record.source_id] = record.to_dict()
        return self.path


class FailingHttpDownloader:
    def fetch(self, request, destination, *, force=False):
        raise RuntimeError("download failed")


class FakeMetadataRegistrar:
    def __init__(self):
        self.calls = []

    def register(self, meta, record):
        self.calls.append((meta, record))
        return f"ds_{meta.source_id}", f"dv_{record.dataset_revision}"


def test_service_routes_meta_and_upserts_completed_record(
    tmp_path,
    monkeypatch,
):
    registry = FakeRegistry(tmp_path / "registry.json")
    monkeypatch.setattr("fetch.fetcher_service.get_path", lambda key: tmp_path)
    meta = DatasetMeta(
        source_id="example",
        modality="text",
        homepage="https://example.invalid",
        license="MIT",
        permitted_use="test",
        contains_pii=False,
        retention_policy="temporary",
        output=OutputSpec("paths.test", ("example",)),
        acquisition=HttpAcquisition(
            files=(
                HttpFileRequest(
                    url="https://example.invalid/data.bin",
                    output_name="data.bin",
                ),
            ),
            revision="v1",
        ),
    )
    downloader = FakeHttpDownloader()
    service = FetcherService(
        http=downloader,
        registry=registry,
        metadata_registrar=FakeMetadataRegistrar(),
    )

    record = service.fetch(meta)

    assert record.source_id == "example"
    assert record.dataset_revision == "v1"
    assert record.file_count == 1
    assert record.files[0].path == str(tmp_path / "example" / "data.bin")
    assert record.total_bytes == len(b"downloaded")
    assert registry.records == [record]
    assert record.request_fingerprint


def test_service_does_not_write_metadata_when_download_fails(
    tmp_path,
    monkeypatch,
):
    registry = FakeRegistry(tmp_path / "registry.json")
    monkeypatch.setattr("fetch.fetcher_service.get_path", lambda key: tmp_path)
    meta = DatasetMeta(
        source_id="failing",
        modality="text",
        homepage="https://example.invalid",
        license="MIT",
        permitted_use="test",
        contains_pii=False,
        retention_policy="temporary",
        output=OutputSpec("paths.test"),
        acquisition=HttpAcquisition(
            files=(
                HttpFileRequest(
                    url="https://example.invalid/data.bin",
                    output_name="data.bin",
                ),
            )
        ),
    )

    with pytest.raises(RuntimeError, match="download failed"):
        FetcherService(
            http=FailingHttpDownloader(),
            registry=registry,
            metadata_registrar=FakeMetadataRegistrar(),
        ).fetch(meta)

    assert registry.records == []


def test_service_skips_verified_duplicate_and_force_runs_again(
    tmp_path,
    monkeypatch,
):
    registry = FakeRegistry(tmp_path / "registry.json")
    downloader = FakeHttpDownloader()
    monkeypatch.setattr("fetch.fetcher_service.get_path", lambda key: tmp_path)
    meta = DatasetMeta(
        source_id="deduplicated",
        modality="text",
        homepage="https://example.invalid",
        license="MIT",
        permitted_use="test",
        contains_pii=False,
        retention_policy="temporary",
        output=OutputSpec("paths.test", ("deduplicated",)),
        acquisition=HttpAcquisition(
            files=(
                HttpFileRequest(
                    url="https://example.invalid/data.bin",
                    output_name="data.bin",
                ),
            ),
            revision="v1",
        ),
    )
    service = FetcherService(
        http=downloader,
        registry=registry,
        metadata_registrar=FakeMetadataRegistrar(),
    )

    first = service.fetch(meta)
    plan = service.plan(meta)
    second = service.fetch(meta)

    assert plan.action == "skip"
    assert second.to_dict() == first.to_dict()
    assert downloader.calls == 1
    assert len(registry.records) == 1

    forced_plan, _ = service.fetch_with_plan(meta, force=True)

    assert forced_plan.action == "force"
    assert downloader.calls == 2
    assert len(registry.records) == 2


def test_service_repairs_matching_record_when_local_file_is_corrupt(
    tmp_path,
    monkeypatch,
):
    registry = FakeRegistry(tmp_path / "registry.json")
    downloader = FakeHttpDownloader()
    monkeypatch.setattr("fetch.fetcher_service.get_path", lambda key: tmp_path)
    meta = DatasetMeta(
        source_id="repairable",
        modality="text",
        homepage="https://example.invalid",
        license="MIT",
        permitted_use="test",
        contains_pii=False,
        retention_policy="temporary",
        output=OutputSpec("paths.test", ("repairable",)),
        acquisition=HttpAcquisition(
            files=(
                HttpFileRequest(
                    url="https://example.invalid/data.bin",
                    output_name="data.bin",
                ),
            ),
        ),
    )
    service = FetcherService(
        http=downloader,
        registry=registry,
        metadata_registrar=FakeMetadataRegistrar(),
    )
    service.fetch(meta)
    record_path = tmp_path / "repairable" / "data.bin"
    record_path.write_bytes(b"corrupt")

    plan = service.plan(meta)

    assert plan.action == "repair"
    assert "size mismatch" in plan.reason


def test_service_registers_download_in_relational_metadata_idempotently(
    tmp_path,
    monkeypatch,
):
    registry = FakeRegistry(tmp_path / "registry.json")
    repository = MetadataRepository(tmp_path / "metadata.db")
    monkeypatch.setattr("fetch.fetcher_service.get_path", lambda key: tmp_path)
    meta = DatasetMeta(
        source_id="governed",
        modality="text",
        homepage="https://example.invalid/dataset",
        license="MIT",
        permitted_use="training",
        contains_pii=False,
        retention_policy="retain raw version",
        output=OutputSpec("paths.test", ("governed",)),
        acquisition=HttpAcquisition(
            files=(
                HttpFileRequest(
                    url="https://example.invalid/data.bin",
                    output_name="data.bin",
                ),
            ),
            revision="v1",
        ),
        purpose="PRETRAIN",
    )
    downloader = FakeHttpDownloader()
    service = FetcherService(
        http=downloader,
        registry=registry,
        metadata_registrar=FetchMetadataRegistrar(repository),
    )

    first = service.fetch(meta)
    second = service.fetch(meta)
    forced = service.fetch(meta, force=True)

    datasets = repository.list_datasets(namespace="catalog")
    versions = repository.list_versions(dataset_id=datasets[0]["dataset_id"])
    assert repository.status()["initialized"] is True
    assert len(datasets) == 1
    assert datasets[0]["name"] == "governed"
    assert datasets[0]["purpose"] == "PRETRAIN"
    assert len(versions) == 1
    assert versions[0]["status"] == "COMMITTED"
    assert versions[0]["byte_size"] == len(b"downloaded")
    assert first.metadata_dataset_id == datasets[0]["dataset_id"]
    assert first.metadata_version_id == versions[0]["version_id"]
    assert second.metadata_version_id == first.metadata_version_id
    assert forced.metadata_version_id == first.metadata_version_id
    assert len(registry.records) == 2

    downloader.content = b"changed"
    changed = service.fetch(meta, force=True)
    changed_versions = repository.list_versions(dataset_id=datasets[0]["dataset_id"])
    assert len(changed_versions) == 2
    assert changed.metadata_version_id != first.metadata_version_id
    assert len(registry.records) == 3
