"""Single service for acquiring datasets and persisting completed metadata."""

import json
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from config import get_path
from fetch.artifacts import file_records
from fetch.downloaders import CurlDownloader, HuggingFaceDownloader, MinariDownloader
from fetch.materializers import JsonlMaterializer, inspect_minari_dataset
from fetch.metadata_registrar import FetchMetadataRegistrar
from fetch.models import (
    AcquisitionResult,
    CommonCrawlAcquisition,
    DatasetMeta,
    DatasetRecord,
    FetchPlan,
    HttpAcquisition,
    HuggingFaceAcquisition,
    MinariAcquisition,
)
from fetch.registry import DatasetRegistry, RegistryVerificationError, verify_record
from fetch.resolvers import CommonCrawlResolver


class FetcherService:
    """Acquire datasets and persist registry plus governed metadata records."""

    def __init__(
        self,
        *,
        http: CurlDownloader | None = None,
        huggingface: HuggingFaceDownloader | None = None,
        minari: MinariDownloader | None = None,
        common_crawl: CommonCrawlResolver | None = None,
        jsonl: JsonlMaterializer | None = None,
        registry: DatasetRegistry | None = None,
        metadata_registrar: FetchMetadataRegistrar | None = None,
    ) -> None:
        self.http = http or CurlDownloader()
        self.huggingface = huggingface or HuggingFaceDownloader()
        self.minari = minari or MinariDownloader()
        self.common_crawl = common_crawl or CommonCrawlResolver()
        self.jsonl = jsonl or JsonlMaterializer()
        self.registry = registry or DatasetRegistry()
        self.metadata_registrar = metadata_registrar or FetchMetadataRegistrar()

    def plan(self, meta: DatasetMeta, *, force: bool = False) -> FetchPlan:
        """Resolve tools and determine whether work can be skipped safely."""
        destination = destination_for(meta)
        acquisition, revision, download_url = self._resolve(meta)
        fingerprint = _request_fingerprint(meta, acquisition)
        existing_value = self.registry.find(meta.source_id)
        existing_record = None
        action = "download"
        reason = "no matching completed registry record"

        if force:
            action = "force"
            reason = "forced by caller"
        elif (
            existing_value
            and existing_value.get("status") == "complete"
            and existing_value.get("request_fingerprint") == fingerprint
        ):
            try:
                verify_record(existing_value)
            except RegistryVerificationError as exc:
                action = "repair"
                reason = str(exc)
            else:
                action = "skip"
                reason = "matching record and local files verified"
                existing_record = DatasetRecord.from_dict(existing_value)

        tool, requirements, estimated_bytes = _describe_acquisition(
            acquisition,
            destination,
        )
        return FetchPlan(
            source_id=meta.source_id,
            tool=tool,
            destination=str(destination),
            revision=revision,
            download_url=download_url,
            estimated_bytes=estimated_bytes,
            action=action,
            reason=reason,
            request_fingerprint=fingerprint,
            requirements=requirements,
            acquisition=acquisition,
            existing_record=existing_record,
        )

    def fetch(self, meta: DatasetMeta, *, force: bool = False) -> DatasetRecord:
        """Acquire one dataset and upsert its metadata only after success."""
        plan = self.plan(meta, force=force)
        return self.execute_plan(meta, plan)

    def fetch_with_plan(
        self,
        meta: DatasetMeta,
        *,
        force: bool = False,
    ) -> tuple[FetchPlan, DatasetRecord]:
        """Fetch one dataset and return both the decision and final record."""
        plan = self.plan(meta, force=force)
        return plan, self.execute_plan(meta, plan)

    def execute_plan(
        self,
        meta: DatasetMeta,
        plan: FetchPlan,
    ) -> DatasetRecord:
        """Execute a previously resolved plan."""
        if plan.source_id != meta.source_id:
            raise ValueError(
                f"Plan source {plan.source_id!r} does not match {meta.source_id!r}"
            )
        if plan.action == "skip" and plan.existing_record is not None:
            record = plan.existing_record
            existing_metadata_ids = (
                record.metadata_dataset_id,
                record.metadata_version_id,
            )
            self._register_metadata(meta, record)
            if existing_metadata_ids != (
                record.metadata_dataset_id,
                record.metadata_version_id,
            ):
                self.registry.upsert(record)
            return record

        result = self._acquire(
            Path(plan.destination),
            plan.acquisition,
            plan.revision,
            plan.download_url,
            refresh=plan.action in {"force", "repair"},
        )
        record = build_dataset_record(
            meta,
            result,
            request_fingerprint=plan.request_fingerprint,
        )
        self._register_metadata(meta, record)
        self.registry.upsert(record)
        return record

    def _register_metadata(
        self,
        meta: DatasetMeta,
        record: DatasetRecord,
    ) -> None:
        dataset_id, version_id = self.metadata_registrar.register(meta, record)
        record.metadata_dataset_id = dataset_id
        record.metadata_version_id = version_id

    def fetch_many(
        self,
        metas: Iterable[DatasetMeta],
        *,
        force: bool = False,
    ) -> list[DatasetRecord]:
        return [self.fetch(meta, force=force) for meta in metas]

    def _resolve(
        self,
        meta: DatasetMeta,
    ) -> tuple[
        HttpAcquisition | HuggingFaceAcquisition | MinariAcquisition,
        str,
        str | list[str],
    ]:
        request = meta.acquisition
        if isinstance(request, CommonCrawlAcquisition):
            resolved = self.common_crawl.resolve(request)
            return (
                resolved.acquisition,
                resolved.revision,
                resolved.acquisition.files[0].url,
            )
        if isinstance(request, HttpAcquisition):
            return request, request.revision, [item.url for item in request.files]
        if isinstance(request, HuggingFaceAcquisition):
            return (
                request,
                request.revision or "default",
                f"hf://datasets/{request.repo_id}",
            )
        if isinstance(request, MinariAcquisition):
            return request, request.dataset_id, f"minari://{request.dataset_id}"
        raise TypeError(f"Unsupported acquisition request: {type(request).__name__}")

    def _acquire(
        self,
        destination: Path,
        request: HttpAcquisition | HuggingFaceAcquisition | MinariAcquisition,
        revision: str,
        download_url: str | list[str],
        *,
        refresh: bool,
    ) -> AcquisitionResult:
        if isinstance(request, HttpAcquisition):
            files = self.http.fetch(request, destination, force=refresh)
            return AcquisitionResult(
                files=files,
                output=destination,
                download_url=download_url,
                revision=revision,
            )

        if isinstance(request, HuggingFaceAcquisition):
            dataset = self.huggingface.fetch(request)
            files = self.jsonl.materialize(dataset, destination)
            return AcquisitionResult(
                files=files,
                output=destination,
                download_url=download_url,
                revision=revision,
                details={"splits": list(dataset.keys())},
            )

        if isinstance(request, MinariAcquisition):
            dataset, dataset_root = self.minari.fetch(request, destination)
            files = [path for path in dataset_root.rglob("*") if path.is_file()]
            return AcquisitionResult(
                files=files,
                output=dataset_root,
                download_url=download_url,
                revision=revision,
                details=inspect_minari_dataset(dataset),
            )

        raise TypeError(f"Unsupported acquisition request: {type(request).__name__}")


def build_dataset_record(
    meta: DatasetMeta,
    result: AcquisitionResult,
    *,
    downloaded_at: str | None = None,
    request_fingerprint: str = "",
) -> DatasetRecord:
    """Combine declared metadata with verified acquisition facts."""
    files = file_records(result.files)
    return DatasetRecord(
        source_id=meta.source_id,
        modality=meta.modality,
        homepage=meta.homepage,
        download_url=result.download_url,
        dataset_revision=result.revision,
        license=meta.license,
        permitted_use=meta.permitted_use,
        region=meta.region,
        contains_pii=meta.contains_pii,
        retention_policy=meta.retention_policy,
        output=str(result.output),
        downloaded_at=downloaded_at or datetime.now(UTC).isoformat(),
        files=files,
        total_bytes=sum(file.bytes for file in files),
        request_fingerprint=request_fingerprint,
        details=result.details,
    )


def destination_for(meta: DatasetMeta) -> Path:
    """Resolve a dataset requirement's configured output directory."""
    path = get_path(meta.output.config_key)
    return path.joinpath(*meta.output.path_parts)


def _request_fingerprint(
    meta: DatasetMeta,
    acquisition: HttpAcquisition | HuggingFaceAcquisition | MinariAcquisition,
) -> str:
    payload = {
        "source_id": meta.source_id,
        "declared_type": type(meta.acquisition).__name__,
        "resolved_type": type(acquisition).__name__,
        "acquisition": asdict(acquisition),
        "output": asdict(meta.output),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _describe_acquisition(
    acquisition: HttpAcquisition | HuggingFaceAcquisition | MinariAcquisition,
    destination: Path,
) -> tuple[str, dict, int | None]:
    if isinstance(acquisition, HttpAcquisition):
        files = [
            {
                "url": item.url,
                "output": str(destination / item.output_name),
                "expected_bytes": item.expected_bytes,
            }
            for item in acquisition.files
        ]
        sizes = [item.expected_bytes for item in acquisition.files]
        estimated_bytes = (
            sum(size for size in sizes if size is not None)
            if all(size is not None for size in sizes)
            else None
        )
        return "curl", {"files": files}, estimated_bytes
    if isinstance(acquisition, HuggingFaceAcquisition):
        return (
            "huggingface-datasets",
            {
                "repo_id": acquisition.repo_id,
                "config": acquisition.config,
                "data_dir": acquisition.data_dir,
                "revision": acquisition.revision,
                "materializer": "jsonl",
            },
            None,
        )
    return (
        "minari",
        {"dataset_id": acquisition.dataset_id},
        None,
    )
