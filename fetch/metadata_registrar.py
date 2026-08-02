"""Bridge completed fetch records into governed relational metadata."""

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from fetch.models import (
    CommonCrawlAcquisition,
    DatasetMeta,
    DatasetRecord,
    HttpAcquisition,
    HuggingFaceAcquisition,
    MinariAcquisition,
)
from metadata import MetadataRepository

_MODALITIES = {
    "text": ["TEXT"],
    "image": ["IMAGE"],
    "video": ["VIDEO"],
    "robot": ["ROBOT_STATE", "ACTION"],
}


class FetchMetadataRegistrar:
    """Idempotently register fetched datasets and immutable raw versions."""

    def __init__(self, repository: MetadataRepository | None = None) -> None:
        self.repository = repository or MetadataRepository()

    def register(
        self,
        meta: DatasetMeta,
        record: DatasetRecord,
    ) -> tuple[str, str]:
        """Create or reuse the Dataset and content-addressed DatasetVersion."""
        if not self.repository.status()["initialized"]:
            self.repository.initialize()

        dataset = self._find_dataset(meta)
        if dataset is None:
            dataset = self.repository.create_dataset(
                namespace=meta.namespace,
                name=meta.source_id,
                purpose=meta.purpose,
                modalities=_modalities(meta.modality),
                owner=meta.owner,
                source_uri=meta.homepage,
                source_revision=record.dataset_revision,
                license=meta.license,
                permitted_use={"policy": meta.permitted_use},
                region=meta.region,
                contains_pii=_contains_pii(meta.contains_pii),
                retention_policy={"policy": meta.retention_policy},
                access_policy={},
            )
        else:
            dataset = self._sync_dataset(dataset, meta, record)

        content_digest = _content_digest(record)
        storage_uri = Path(record.output).resolve().as_uri()
        existing = self.repository.find_version_by_content_digest(
            dataset["dataset_id"],
            content_digest,
            storage_uri,
        )
        if existing is not None:
            return dataset["dataset_id"], existing.version_id

        schema = _schema_definition(meta, record)
        version = self.repository.create_version(
            dataset_id=dataset["dataset_id"],
            stage="RAW",
            status="COMMITTED",
            storage_uri=storage_uri,
            storage_format=_storage_format(meta),
            schema_format="JSON_SCHEMA",
            schema_definition=schema,
            schema_version="fetch-v1",
            schema_digest=_json_digest(schema),
            content_digest=content_digest,
            byte_size=record.total_bytes,
            created_by="fetch-service",
        )
        return dataset["dataset_id"], version.version_id

    def _find_dataset(self, meta: DatasetMeta) -> dict[str, Any] | None:
        return self.repository.find_dataset(meta.namespace, meta.source_id)

    def _sync_dataset(
        self,
        dataset: dict[str, Any],
        meta: DatasetMeta,
        record: DatasetRecord,
    ) -> dict[str, Any]:
        desired = {
            "purpose": meta.purpose.upper(),
            "modalities": _modalities(meta.modality),
            "owner": meta.owner,
            "source_uri": meta.homepage,
            "source_revision": record.dataset_revision,
            "license": meta.license,
            "permitted_use": {"policy": meta.permitted_use},
            "region": meta.region,
            "contains_pii": _contains_pii(meta.contains_pii),
            "retention_policy": {"policy": meta.retention_policy},
        }
        changes = {
            name: value for name, value in desired.items() if dataset.get(name) != value
        }
        if not changes:
            return dataset
        return self.repository.update_dataset(dataset["dataset_id"], **changes)


def _modalities(modality: str) -> list[str]:
    try:
        return _MODALITIES[modality.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported metadata modality: {modality}") from exc


def _contains_pii(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() not in {"false", "no", "none", "absent"}


def _content_digest(record: DatasetRecord) -> str:
    root = Path(record.output)
    files = []
    for item in record.files:
        path = Path(item.path)
        try:
            relative_path = str(path.relative_to(root))
        except ValueError:
            relative_path = path.name
        files.append(
            {
                "path": relative_path,
                "bytes": item.bytes,
                "sha256": item.sha256,
            }
        )
    return _json_digest({"files": sorted(files, key=lambda item: item["path"])})


def _schema_definition(meta: DatasetMeta, record: DatasetRecord) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": True,
        "x-trainfoundry-modality": meta.modality,
    }
    if "schema" in record.details:
        schema["x-source-schema"] = record.details["schema"]
    return schema


def _storage_format(meta: DatasetMeta) -> str:
    acquisition = meta.acquisition
    if isinstance(acquisition, HuggingFaceAcquisition):
        return "JSONL"
    if isinstance(acquisition, MinariAcquisition):
        return "MINARI"
    if isinstance(acquisition, CommonCrawlAcquisition):
        return "WARC_WET_GZIP"
    if isinstance(acquisition, HttpAcquisition):
        return "FILES"
    raise TypeError(f"Unsupported acquisition request: {type(acquisition).__name__}")


def _json_digest(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()
