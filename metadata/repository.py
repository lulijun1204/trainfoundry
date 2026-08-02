"""Typed read/write operations for the local metadata model."""

import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from metadata.database import MetadataDatabase
from metadata.errors import (
    MetadataConflictError,
    MetadataNotFoundError,
    MetadataValidationError,
)
from metadata.models import (
    DatasetRun,
    DatasetRunStatus,
    DatasetRunType,
    DatasetSplit,
    DatasetVersion,
    DatasetVersionStage,
    DatasetVersionStatus,
    OutputMode,
    SchemaFormat,
)

PURPOSES = {"PRETRAIN", "SFT", "RL", "BENCHMARK"}
MODALITIES = {"TEXT", "IMAGE", "AUDIO", "VIDEO", "ROBOT_STATE", "ACTION"}
VERSION_STAGES = {"RAW", "PROCESSED", "ANNOTATED", "TRAINING_READY"}
VERSION_STATUSES = {
    "DRAFT",
    "COMMITTED",
    "PUBLISHED",
    "DEPRECATED",
    "INVALIDATED",
}
SPLITS = {"TRAIN", "VALID", "TEST"}
RUN_TYPES = {
    "INGEST",
    "CLEAN",
    "NORMALIZE",
    "DEDUP",
    "FILTER",
    "CONVERT",
    "SCHEMA_MAPPING",
    "MATERIALIZE_ANNOTATION",
    "SPLIT",
    "MERGE",
    "QUALITY",
    "ANNOTATION",
}
OUTPUT_MODES = {"NEW_VERSION", "NEW_DATASET", "NO_DATA_VERSION"}
RUN_STATUSES = {"PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"}
LINEAGE_RELATIONS = {
    "DERIVED_FROM",
    "FILTERED_FROM",
    "SPLIT_FROM",
    "MERGED_FROM",
    "ANNOTATED_FROM",
}
RESULT_STATUSES = {"PENDING", "RUNNING", "SUCCEEDED", "FAILED"}
PRODUCER_TYPES = {"HUMAN", "MODEL", "RULE", "HYBRID"}
TRAINING_ROLES = {"TRAIN", "VALID", "TEST", "AUXILIARY"}
SCHEMA_FORMATS = {"ARROW", "JSON_SCHEMA"}

_BOOL_FIELDS = {"contains_pii", "deterministic", "passed"}


class MetadataRepository:
    """Aggregate the public metadata operations behind one repository."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        database: MetadataDatabase | None = None,
    ) -> None:
        self.database = database or MetadataDatabase(path)

    def initialize(self) -> dict[str, Any]:
        return self.database.initialize()

    def status(self) -> dict[str, Any]:
        return self.database.status()

    def create_dataset(
        self,
        *,
        namespace: str,
        name: str,
        purpose: str,
        modalities: Iterable[str],
        owner: str,
        dataset_id: str | None = None,
        source_uri: str | None = None,
        source_revision: str | None = None,
        license: str | None = None,
        permitted_use: Any = None,
        region: str | None = None,
        contains_pii: bool = False,
        retention_policy: Any = None,
        access_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        dataset_id = dataset_id or _new_id("ds")
        purpose = _enum(purpose, PURPOSES, "purpose")
        normalized_modalities = _enum_list(modalities, MODALITIES, "modalities")
        values = (
            dataset_id,
            _required(namespace, "namespace"),
            _required(name, "name"),
            purpose,
            _json_array(normalized_modalities, "modalities"),
            _required(owner, "owner"),
            source_uri,
            source_revision,
            license,
            _optional_json(permitted_use, "permitted_use"),
            region,
            int(contains_pii),
            _optional_json(retention_policy, "retention_policy"),
            _json_object(access_policy or {}, "access_policy"),
        )
        with self._write() as connection:
            connection.execute(
                """
                INSERT INTO datasets (
                    dataset_id,
                    namespace,
                    name,
                    purpose,
                    modalities_json,
                    owner,
                    source_uri,
                    source_revision,
                    license,
                    permitted_use_json,
                    region,
                    contains_pii,
                    retention_policy_json,
                    access_policy_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            return self._get(connection, "datasets", "dataset_id", dataset_id)

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            return self._get(connection, "datasets", "dataset_id", dataset_id)

    def find_dataset(self, namespace: str, name: str) -> dict[str, Any] | None:
        """Find a Dataset by its governed natural identity."""
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM datasets WHERE namespace = ? AND name = ?",
                (namespace, name),
            ).fetchone()
        return _record(row) if row is not None else None

    def list_datasets(
        self,
        *,
        namespace: str | None = None,
        owner: str | None = None,
        purpose: str | None = None,
        modality: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if namespace is not None:
            clauses.append("namespace = ?")
            params.append(namespace)
        if owner is not None:
            clauses.append("owner = ?")
            params.append(owner)
        if purpose is not None:
            clauses.append("purpose = ?")
            params.append(_enum(purpose, PURPOSES, "purpose"))
        if modality is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM json_each(modalities_json) WHERE value = ?)"
            )
            params.append(_enum(modality, MODALITIES, "modality"))
        return self._list("datasets", clauses, params, "created_at, dataset_id", limit)

    def update_dataset(self, dataset_id: str, **changes: Any) -> dict[str, Any]:
        allowed = {
            "namespace",
            "name",
            "purpose",
            "modalities",
            "owner",
            "source_uri",
            "source_revision",
            "license",
            "permitted_use",
            "region",
            "contains_pii",
            "retention_policy",
            "access_policy",
        }
        values = _changes(changes, allowed)
        if "purpose" in values:
            values["purpose"] = _enum(values["purpose"], PURPOSES, "purpose")
        if "modalities" in values:
            values["modalities_json"] = _json_array(
                _enum_list(values.pop("modalities"), MODALITIES, "modalities"),
                "modalities",
            )
        for name in ("permitted_use", "retention_policy"):
            if name in values:
                values[f"{name}_json"] = _optional_json(values.pop(name), name)
        if "access_policy" in values:
            values["access_policy_json"] = _json_object(
                values.pop("access_policy"),
                "access_policy",
            )
        if "contains_pii" in values:
            values["contains_pii"] = int(bool(values["contains_pii"]))
        for name in ("namespace", "name", "owner"):
            if name in values:
                values[name] = _required(values[name], name)
        return self._update(
            "datasets",
            "dataset_id",
            dataset_id,
            values,
            touch_updated_at=True,
        )

    def create_version(
        self,
        *,
        dataset_id: str,
        stage: str,
        storage_uri: str,
        storage_format: str,
        schema_format: str,
        schema_definition: Mapping[str, Any],
        schema_digest: str,
        content_digest: str,
        created_by: str,
        version_id: str | None = None,
        version_number: int | None = None,
        status: str = "DRAFT",
        schema_version: str | None = None,
        split: str | None = None,
        usage_tags: Iterable[str] | None = None,
        row_count: int | None = None,
        byte_size: int | None = None,
    ) -> DatasetVersion:
        version_id = version_id or _new_id("dv")
        stage = _enum(stage, VERSION_STAGES, "stage")
        status = _enum(status, VERSION_STATUSES, "status")
        schema_format = _enum(schema_format, SCHEMA_FORMATS, "schema_format")
        split = _optional_enum(split, SPLITS, "split")
        _non_negative(row_count, "row_count")
        _non_negative(byte_size, "byte_size")
        with self._write() as connection:
            self._get(connection, "datasets", "dataset_id", dataset_id)
            if version_number is None:
                version_number = connection.execute(
                    """
                    SELECT COALESCE(MAX(version_number), 0) + 1
                    FROM dataset_versions
                    WHERE dataset_id = ?
                    """,
                    (dataset_id,),
                ).fetchone()[0]
            if version_number <= 0:
                raise MetadataValidationError("version_number must be positive")
            committed_at = (
                _database_now(connection) if status != "DRAFT" else None
            )
            connection.execute(
                """
                INSERT INTO dataset_versions (
                    version_id,
                    dataset_id,
                    version_number,
                    stage,
                    status,
                    storage_uri,
                    storage_format,
                    schema_format,
                    schema_definition_json,
                    schema_version,
                    schema_digest,
                    content_digest,
                    split,
                    usage_tags_json,
                    row_count,
                    byte_size,
                    created_by,
                    committed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    dataset_id,
                    version_number,
                    stage,
                    status,
                    _required(storage_uri, "storage_uri"),
                    _required(storage_format, "storage_format"),
                    schema_format,
                    _json_object(schema_definition, "schema_definition"),
                    schema_version,
                    _digest(schema_digest, "schema_digest"),
                    _digest(content_digest, "content_digest"),
                    split,
                    (
                        _json_array(list(usage_tags), "usage_tags")
                        if usage_tags is not None
                        else None
                    ),
                    row_count,
                    byte_size,
                    _required(created_by, "created_by"),
                    committed_at,
                ),
            )
            return _dataset_version(
                self._get(
                    connection,
                    "dataset_versions",
                    "version_id",
                    version_id,
                )
            )

    def get_version(self, version_id: str) -> DatasetVersion:
        with self.database.connect() as connection:
            return _dataset_version(
                self._get(
                    connection,
                    "dataset_versions",
                    "version_id",
                    version_id,
                )
            )

    def find_version_by_content_digest(
        self,
        dataset_id: str,
        content_digest: str,
        storage_uri: str | None = None,
    ) -> DatasetVersion | None:
        """Find a content-identical version, optionally at one storage URI."""
        digest = _digest(content_digest, "content_digest")
        storage_clause = " AND storage_uri = ?" if storage_uri is not None else ""
        params = (
            (dataset_id, digest, storage_uri)
            if storage_uri is not None
            else (dataset_id, digest)
        )
        with self.database.connect() as connection:
            row = connection.execute(
                f"""
                SELECT * FROM dataset_versions
                WHERE dataset_id = ? AND content_digest = ?
                {storage_clause}
                ORDER BY version_number
                LIMIT 1
                """,
                params,
            ).fetchone()
        return _dataset_version(_record(row)) if row is not None else None

    def list_versions(
        self,
        *,
        dataset_id: str | None = None,
        stage: str | None = None,
        status: str | None = None,
        split: str | None = None,
        limit: int = 100,
    ) -> list[DatasetVersion]:
        clauses = []
        params: list[Any] = []
        for name, value, choices in (
            ("stage", stage, VERSION_STAGES),
            ("status", status, VERSION_STATUSES),
            ("split", split, SPLITS),
        ):
            if value is not None:
                clauses.append(f"{name} = ?")
                params.append(_enum(value, choices, name))
        if dataset_id is not None:
            clauses.append("dataset_id = ?")
            params.append(dataset_id)
        return [
            _dataset_version(record)
            for record in self._list(
                "dataset_versions",
                clauses,
                params,
                "dataset_id, version_number",
                limit,
            )
        ]

    def update_version(self, version_id: str, **changes: Any) -> DatasetVersion:
        allowed = {
            "stage",
            "status",
            "storage_uri",
            "storage_format",
            "schema_format",
            "schema_definition",
            "schema_version",
            "schema_digest",
            "content_digest",
            "split",
            "usage_tags",
            "row_count",
            "byte_size",
        }
        values = _changes(changes, allowed)
        enums = {
            "stage": VERSION_STAGES,
            "status": VERSION_STATUSES,
            "schema_format": SCHEMA_FORMATS,
        }
        for name, choices in enums.items():
            if name in values:
                values[name] = _enum(values[name], choices, name)
        if "split" in values:
            values["split"] = _optional_enum(values["split"], SPLITS, "split")
        if "schema_definition" in values:
            values["schema_definition_json"] = _json_object(
                values.pop("schema_definition"),
                "schema_definition",
            )
        if "usage_tags" in values:
            usage_tags = values.pop("usage_tags")
            values["usage_tags_json"] = (
                _json_array(list(usage_tags), "usage_tags")
                if usage_tags is not None
                else None
            )
        for name in ("schema_digest", "content_digest"):
            if name in values:
                values[name] = _digest(values[name], name)
        for name in ("row_count", "byte_size"):
            if name in values:
                _non_negative(values[name], name)
        if "status" in values and values["status"] != "DRAFT":
            with self.database.connect() as connection:
                existing = self._get(
                    connection,
                    "dataset_versions",
                    "version_id",
                    version_id,
                )
            if existing["committed_at"] is None:
                values["committed_at"] = _timestamp_expression()
        return _dataset_version(
            self._update(
                "dataset_versions",
                "version_id",
                version_id,
                values,
            )
        )

    def create_run(
        self,
        *,
        run_type: str,
        operator_name: str,
        operator_version: str,
        operator_fingerprint: str,
        compute_key: str,
        output_mode: str,
        created_by: str,
        run_id: str | None = None,
        params: Mapping[str, Any] | None = None,
        deterministic: bool = True,
        target_dataset_id: str | None = None,
        status: str = "PENDING",
        started_at: str | None = None,
        finished_at: str | None = None,
        error_message: str | None = None,
    ) -> DatasetRun:
        run_id = run_id or _new_id("run")
        run_type = _enum(run_type, RUN_TYPES, "run_type")
        output_mode = _enum(output_mode, OUTPUT_MODES, "output_mode")
        status = _enum(status, RUN_STATUSES, "status")
        if output_mode != "NO_DATA_VERSION" and not target_dataset_id:
            raise MetadataValidationError(
                f"target_dataset_id is required for output_mode={output_mode}"
            )
        with self._write() as connection:
            connection.execute(
                """
                INSERT INTO dataset_runs (
                    run_id,
                    run_type,
                    operator_name,
                    operator_version,
                    operator_fingerprint,
                    params_json,
                    compute_key,
                    deterministic,
                    output_mode,
                    target_dataset_id,
                    status,
                    started_at,
                    finished_at,
                    error_message,
                    created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    run_type,
                    _required(operator_name, "operator_name"),
                    _required(operator_version, "operator_version"),
                    _digest(operator_fingerprint, "operator_fingerprint"),
                    _json_object(params or {}, "params"),
                    _digest(compute_key, "compute_key"),
                    int(deterministic),
                    output_mode,
                    target_dataset_id,
                    status,
                    started_at,
                    finished_at,
                    error_message,
                    _required(created_by, "created_by"),
                ),
            )
            return _dataset_run(
                self._get(connection, "dataset_runs", "run_id", run_id)
            )

    def get_run(self, run_id: str) -> DatasetRun:
        with self.database.connect() as connection:
            return _dataset_run(
                self._get(connection, "dataset_runs", "run_id", run_id)
            )

    def list_runs(
        self,
        *,
        run_type: str | None = None,
        status: str | None = None,
        target_dataset_id: str | None = None,
        limit: int = 100,
    ) -> list[DatasetRun]:
        clauses = []
        params: list[Any] = []
        if run_type is not None:
            clauses.append("run_type = ?")
            params.append(_enum(run_type, RUN_TYPES, "run_type"))
        if status is not None:
            clauses.append("status = ?")
            params.append(_enum(status, RUN_STATUSES, "status"))
        if target_dataset_id is not None:
            clauses.append("target_dataset_id = ?")
            params.append(target_dataset_id)
        return [
            _dataset_run(record)
            for record in self._list(
                "dataset_runs", clauses, params, "created_at, run_id", limit
            )
        ]

    def update_run(
        self,
        run_id: str,
        *,
        status: str,
        started_at: str | None = None,
        finished_at: str | None = None,
        error_message: str | None = None,
    ) -> DatasetRun:
        values: dict[str, Any] = {
            "status": _enum(status, RUN_STATUSES, "status"),
        }
        if started_at is not None:
            values["started_at"] = started_at
        if finished_at is not None:
            values["finished_at"] = finished_at
        if error_message is not None:
            values["error_message"] = error_message
        return _dataset_run(
            self._update("dataset_runs", "run_id", run_id, values)
        )

    def create_lineage(
        self,
        *,
        run_id: str,
        source_version_id: str,
        target_version_id: str,
        relation_type: str,
        lineage_id: str | None = None,
    ) -> dict[str, Any]:
        lineage_id = lineage_id or _new_id("lin")
        relation_type = _enum(
            relation_type,
            LINEAGE_RELATIONS,
            "relation_type",
        )
        with self._write() as connection:
            connection.execute(
                """
                INSERT INTO dataset_lineage (
                    lineage_id,
                    run_id,
                    source_version_id,
                    target_version_id,
                    relation_type
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    lineage_id,
                    run_id,
                    source_version_id,
                    target_version_id,
                    relation_type,
                ),
            )
            return self._get(
                connection,
                "dataset_lineage",
                "lineage_id",
                lineage_id,
            )

    def get_lineage(
        self,
        version_id: str,
        *,
        direction: str = "both",
        max_depth: int = 20,
    ) -> dict[str, Any]:
        if direction not in {"upstream", "downstream", "both"}:
            raise MetadataValidationError(
                "direction must be upstream, downstream, or both"
            )
        if max_depth < 1 or max_depth > 100:
            raise MetadataValidationError("max_depth must be between 1 and 100")
        self.get_version(version_id)
        result: dict[str, Any] = {
            "version_id": version_id,
            "direction": direction,
            "max_depth": max_depth,
        }
        with self.database.connect() as connection:
            if direction in {"upstream", "both"}:
                result["upstream"] = self._lineage_query(
                    connection,
                    version_id,
                    "upstream",
                    max_depth,
                )
            if direction in {"downstream", "both"}:
                result["downstream"] = self._lineage_query(
                    connection,
                    version_id,
                    "downstream",
                    max_depth,
                )
        return result

    def create_quality_result(
        self,
        *,
        dataset_version_id: str,
        run_id: str,
        evaluator_name: str,
        evaluator_version: str,
        result_set_id: str | None = None,
        status: str = "PENDING",
        passed: bool | None = None,
        summary: Mapping[str, Any] | None = None,
        passed_count: int | None = None,
        rejected_count: int | None = None,
        detail_uri: str | None = None,
        detail_digest: str | None = None,
    ) -> dict[str, Any]:
        result_set_id = result_set_id or _new_id("qrs")
        _non_negative(passed_count, "passed_count")
        _non_negative(rejected_count, "rejected_count")
        with self._write() as connection:
            connection.execute(
                """
                INSERT INTO quality_result_sets (
                    result_set_id,
                    dataset_version_id,
                    run_id,
                    evaluator_name,
                    evaluator_version,
                    status,
                    passed,
                    summary_json,
                    passed_count,
                    rejected_count,
                    detail_uri,
                    detail_digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result_set_id,
                    dataset_version_id,
                    run_id,
                    _required(evaluator_name, "evaluator_name"),
                    _required(evaluator_version, "evaluator_version"),
                    _enum(status, RESULT_STATUSES, "status"),
                    None if passed is None else int(passed),
                    _json_object(summary or {}, "summary"),
                    passed_count,
                    rejected_count,
                    detail_uri,
                    _optional_digest(detail_digest, "detail_digest"),
                ),
            )
            return self._get(
                connection,
                "quality_result_sets",
                "result_set_id",
                result_set_id,
            )

    def get_quality_result(self, result_set_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            return self._get(
                connection,
                "quality_result_sets",
                "result_set_id",
                result_set_id,
            )

    def list_quality_results(
        self,
        *,
        dataset_version_id: str | None = None,
        run_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self._list_results(
            "quality_result_sets",
            dataset_version_id,
            run_id,
            status,
            limit,
        )

    def update_quality_result(
        self,
        result_set_id: str,
        *,
        status: str,
        passed: bool | None = None,
        summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {
            "status": _enum(status, RESULT_STATUSES, "status"),
        }
        if passed is not None:
            values["passed"] = int(passed)
        if summary is not None:
            values["summary_json"] = _json_object(summary, "summary")
        return self._update(
            "quality_result_sets",
            "result_set_id",
            result_set_id,
            values,
        )

    def create_annotation_result(
        self,
        *,
        dataset_version_id: str,
        run_id: str,
        annotation_schema_version: str,
        producer_type: str,
        result_set_id: str | None = None,
        producer_ref: str | None = None,
        status: str = "PENDING",
        summary: Mapping[str, Any] | None = None,
        coverage: float | None = None,
        detail_uri: str | None = None,
        detail_digest: str | None = None,
    ) -> dict[str, Any]:
        result_set_id = result_set_id or _new_id("ars")
        if coverage is not None and not 0 <= coverage <= 1:
            raise MetadataValidationError("coverage must be between 0 and 1")
        with self._write() as connection:
            connection.execute(
                """
                INSERT INTO annotation_result_sets (
                    result_set_id,
                    dataset_version_id,
                    run_id,
                    annotation_schema_version,
                    producer_type,
                    producer_ref,
                    status,
                    summary_json,
                    coverage,
                    detail_uri,
                    detail_digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result_set_id,
                    dataset_version_id,
                    run_id,
                    _required(
                        annotation_schema_version,
                        "annotation_schema_version",
                    ),
                    _enum(producer_type, PRODUCER_TYPES, "producer_type"),
                    producer_ref,
                    _enum(status, RESULT_STATUSES, "status"),
                    _json_object(summary or {}, "summary"),
                    coverage,
                    detail_uri,
                    _optional_digest(detail_digest, "detail_digest"),
                ),
            )
            return self._get(
                connection,
                "annotation_result_sets",
                "result_set_id",
                result_set_id,
            )

    def get_annotation_result(self, result_set_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            return self._get(
                connection,
                "annotation_result_sets",
                "result_set_id",
                result_set_id,
            )

    def list_annotation_results(
        self,
        *,
        dataset_version_id: str | None = None,
        run_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self._list_results(
            "annotation_result_sets",
            dataset_version_id,
            run_id,
            status,
            limit,
        )

    def update_annotation_result(
        self,
        result_set_id: str,
        *,
        status: str,
        summary: Mapping[str, Any] | None = None,
        coverage: float | None = None,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {
            "status": _enum(status, RESULT_STATUSES, "status"),
        }
        if summary is not None:
            values["summary_json"] = _json_object(summary, "summary")
        if coverage is not None:
            if not 0 <= coverage <= 1:
                raise MetadataValidationError("coverage must be between 0 and 1")
            values["coverage"] = coverage
        return self._update(
            "annotation_result_sets",
            "result_set_id",
            result_set_id,
            values,
        )

    def create_training_run(
        self,
        *,
        code_commit: str,
        config_digest: str,
        created_by: str,
        training_run_id: str | None = None,
        dataloader_config: Mapping[str, Any] | None = None,
        status: str = "PENDING",
        started_at: str | None = None,
        finished_at: str | None = None,
        model_artifact_uri: str | None = None,
    ) -> dict[str, Any]:
        training_run_id = training_run_id or _new_id("tr")
        with self._write() as connection:
            connection.execute(
                """
                INSERT INTO training_runs (
                    training_run_id,
                    code_commit,
                    config_digest,
                    dataloader_config_json,
                    status,
                    started_at,
                    finished_at,
                    model_artifact_uri,
                    created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    training_run_id,
                    _required(code_commit, "code_commit"),
                    _digest(config_digest, "config_digest"),
                    _json_object(dataloader_config or {}, "dataloader_config"),
                    _enum(status, RUN_STATUSES, "status"),
                    started_at,
                    finished_at,
                    model_artifact_uri,
                    _required(created_by, "created_by"),
                ),
            )
            return self._training_run(connection, training_run_id)

    def get_training_run(self, training_run_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            return self._training_run(connection, training_run_id)

    def list_training_runs(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(_enum(status, RUN_STATUSES, "status"))
        records = self._list(
            "training_runs",
            clauses,
            params,
            "created_at, training_run_id",
            limit,
        )
        with self.database.connect() as connection:
            for record in records:
                record["dataset_versions"] = self._training_bindings(
                    connection,
                    record["training_run_id"],
                )
        return records

    def update_training_run(
        self,
        training_run_id: str,
        *,
        status: str,
        started_at: str | None = None,
        finished_at: str | None = None,
        model_artifact_uri: str | None = None,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {
            "status": _enum(status, RUN_STATUSES, "status"),
        }
        for name, value in (
            ("started_at", started_at),
            ("finished_at", finished_at),
            ("model_artifact_uri", model_artifact_uri),
        ):
            if value is not None:
                values[name] = value
        self._update(
            "training_runs",
            "training_run_id",
            training_run_id,
            values,
        )
        return self.get_training_run(training_run_id)

    def bind_training_version(
        self,
        *,
        training_run_id: str,
        dataset_version_id: str,
        role: str,
        weight: float = 1.0,
        sampling_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if weight < 0:
            raise MetadataValidationError("weight must be non-negative")
        role = _enum(role, TRAINING_ROLES, "role")
        with self._write() as connection:
            connection.execute(
                """
                INSERT INTO training_run_dataset_versions (
                    training_run_id,
                    dataset_version_id,
                    role,
                    weight,
                    sampling_config_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    training_run_id,
                    dataset_version_id,
                    role,
                    weight,
                    (
                        _json_object(sampling_config, "sampling_config")
                        if sampling_config is not None
                        else None
                    ),
                ),
            )
            row = connection.execute(
                """
                SELECT *
                FROM training_run_dataset_versions
                WHERE training_run_id = ?
                  AND dataset_version_id = ?
                  AND role = ?
                """,
                (training_run_id, dataset_version_id, role),
            ).fetchone()
            return _record(row)

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        try:
            with self.database.connect() as connection:
                yield connection
        except sqlite3.IntegrityError as exc:
            raise MetadataConflictError(str(exc)) from exc

    def _get(
        self,
        connection: sqlite3.Connection,
        table: str,
        id_field: str,
        value: str,
    ) -> dict[str, Any]:
        row = connection.execute(
            f"SELECT * FROM {table} WHERE {id_field} = ?",
            (value,),
        ).fetchone()
        if row is None:
            raise MetadataNotFoundError(
                f"No {table} record with {id_field}={value!r}"
            )
        return _record(row)

    def _list(
        self,
        table: str,
        clauses: list[str],
        params: list[Any],
        order_by: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        limit = _limit(limit)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.database.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM {table}{where} ORDER BY {order_by} LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [_record(row) for row in rows]

    def _update(
        self,
        table: str,
        id_field: str,
        value: str,
        changes: dict[str, Any],
        *,
        touch_updated_at: bool = False,
    ) -> dict[str, Any]:
        if not changes:
            raise MetadataValidationError("at least one update field is required")
        assignments = [f"{name} = ?" for name in changes]
        parameters = list(changes.values())
        if touch_updated_at:
            assignments.append(
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
            )
        with self._write() as connection:
            cursor = connection.execute(
                f"""
                UPDATE {table}
                SET {", ".join(assignments)}
                WHERE {id_field} = ?
                """,
                (*parameters, value),
            )
            if cursor.rowcount == 0:
                raise MetadataNotFoundError(
                    f"No {table} record with {id_field}={value!r}"
                )
            return self._get(connection, table, id_field, value)

    def _lineage_query(
        self,
        connection: sqlite3.Connection,
        version_id: str,
        direction: str,
        max_depth: int,
    ) -> list[dict[str, Any]]:
        if direction == "upstream":
            anchor = "target_version_id = ?"
            next_join = "edge.target_version_id = walk.source_version_id"
            path_value = "edge.source_version_id"
        else:
            anchor = "source_version_id = ?"
            next_join = "edge.source_version_id = walk.target_version_id"
            path_value = "edge.target_version_id"
        rows = connection.execute(
            f"""
            WITH RECURSIVE walk AS (
                SELECT
                    lineage_id,
                    run_id,
                    source_version_id,
                    target_version_id,
                    relation_type,
                    1 AS depth,
                    '|' || source_version_id || '|' ||
                        target_version_id || '|' AS path
                FROM dataset_lineage
                WHERE {anchor}

                UNION ALL

                SELECT
                    edge.lineage_id,
                    edge.run_id,
                    edge.source_version_id,
                    edge.target_version_id,
                    edge.relation_type,
                    walk.depth + 1,
                    walk.path || {path_value} || '|'
                FROM walk
                JOIN dataset_lineage AS edge ON {next_join}
                WHERE walk.depth < ?
                  AND instr(walk.path, '|' || {path_value} || '|') = 0
            )
            SELECT DISTINCT
                lineage_id,
                run_id,
                source_version_id,
                target_version_id,
                relation_type,
                depth
            FROM walk
            ORDER BY depth, lineage_id
            """,
            (version_id, max_depth),
        ).fetchall()
        return [_record(row) for row in rows]

    def _list_results(
        self,
        table: str,
        dataset_version_id: str | None,
        run_id: str | None,
        status: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if dataset_version_id is not None:
            clauses.append("dataset_version_id = ?")
            params.append(dataset_version_id)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(_enum(status, RESULT_STATUSES, "status"))
        return self._list(table, clauses, params, "created_at, result_set_id", limit)

    def _training_run(
        self,
        connection: sqlite3.Connection,
        training_run_id: str,
    ) -> dict[str, Any]:
        record = self._get(
            connection,
            "training_runs",
            "training_run_id",
            training_run_id,
        )
        record["dataset_versions"] = self._training_bindings(
            connection,
            training_run_id,
        )
        return record

    def _training_bindings(
        self,
        connection: sqlite3.Connection,
        training_run_id: str,
    ) -> list[dict[str, Any]]:
        rows = connection.execute(
            """
            SELECT *
            FROM training_run_dataset_versions
            WHERE training_run_id = ?
            ORDER BY role, dataset_version_id
            """,
            (training_run_id,),
        ).fetchall()
        return [_record(row) for row in rows]


def _record(row: sqlite3.Row | None) -> dict[str, Any]:
    if row is None:
        raise MetadataNotFoundError("Metadata record does not exist")
    result: dict[str, Any] = {}
    for key, value in dict(row).items():
        if key.endswith("_json"):
            result[key.removesuffix("_json")] = (
                json.loads(value) if value is not None else None
            )
        elif key in _BOOL_FIELDS:
            result[key] = bool(value) if value is not None else None
        else:
            result[key] = value
    return result


def _dataset_version(record: Mapping[str, Any]) -> DatasetVersion:
    """Hydrate the complete version domain model from a decoded SQLite row."""
    return DatasetVersion(
        version_id=record["version_id"],
        dataset_id=record["dataset_id"],
        version_number=record["version_number"],
        stage=DatasetVersionStage(record["stage"]),
        status=DatasetVersionStatus(record["status"]),
        storage_uri=record["storage_uri"],
        storage_format=record["storage_format"],
        schema_format=SchemaFormat(record["schema_format"]),
        schema_definition=dict(record["schema_definition"]),
        schema_version=record["schema_version"],
        schema_digest=record["schema_digest"],
        content_digest=record["content_digest"],
        split=DatasetSplit(record["split"]) if record["split"] is not None else None,
        usage_tags=(
            tuple(record["usage_tags"])
            if record["usage_tags"] is not None
            else None
        ),
        row_count=record["row_count"],
        byte_size=record["byte_size"],
        created_by=record["created_by"],
        created_at=record["created_at"],
        committed_at=record["committed_at"],
    )


def _dataset_run(record: Mapping[str, Any]) -> DatasetRun:
    """Hydrate the complete run domain model from a decoded SQLite row."""
    return DatasetRun(
        run_id=record["run_id"],
        run_type=DatasetRunType(record["run_type"]),
        operator_name=record["operator_name"],
        operator_version=record["operator_version"],
        operator_fingerprint=record["operator_fingerprint"],
        params=dict(record["params"]),
        compute_key=record["compute_key"],
        deterministic=record["deterministic"],
        output_mode=OutputMode(record["output_mode"]),
        target_dataset_id=record["target_dataset_id"],
        status=DatasetRunStatus(record["status"]),
        started_at=record["started_at"],
        finished_at=record["finished_at"],
        error_message=record["error_message"],
        created_by=record["created_by"],
        created_at=record["created_at"],
    )


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _required(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MetadataValidationError(f"{name} must be a non-empty string")
    return value


def _enum(value: str, choices: set[str], name: str) -> str:
    normalized = _required(value, name).upper()
    if normalized not in choices:
        expected = ", ".join(sorted(choices))
        raise MetadataValidationError(f"{name} must be one of: {expected}")
    return normalized


def _optional_enum(
    value: str | None,
    choices: set[str],
    name: str,
) -> str | None:
    return None if value is None else _enum(value, choices, name)


def _enum_list(
    values: Iterable[str],
    choices: set[str],
    name: str,
) -> list[str]:
    normalized = sorted({_enum(value, choices, name) for value in values})
    if not normalized:
        raise MetadataValidationError(f"{name} must contain at least one value")
    return normalized


def _json_value(value: Any, name: str) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise MetadataValidationError(f"{name} must be valid JSON: {exc}") from exc
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise MetadataValidationError(f"{name} must be JSON serializable") from exc


def _json_object(value: Any, name: str) -> str:
    serialized = _json_value(value, name)
    if not isinstance(json.loads(serialized), dict):
        raise MetadataValidationError(f"{name} must be a JSON object")
    return serialized


def _json_array(value: Any, name: str) -> str:
    serialized = _json_value(value, name)
    if not isinstance(json.loads(serialized), list):
        raise MetadataValidationError(f"{name} must be a JSON array")
    return serialized


def _optional_json(value: Any, name: str) -> str | None:
    return None if value is None else _json_value(value, name)


def _digest(value: str, name: str) -> str:
    normalized = _required(value, name).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise MetadataValidationError(f"{name} must be a 64-character SHA-256 hex")
    return normalized


def _optional_digest(value: str | None, name: str) -> str | None:
    return None if value is None else _digest(value, name)


def _non_negative(value: int | None, name: str) -> None:
    if value is not None and value < 0:
        raise MetadataValidationError(f"{name} must be non-negative")


def _limit(value: int) -> int:
    if value < 1 or value > 1000:
        raise MetadataValidationError("limit must be between 1 and 1000")
    return value


def _changes(changes: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    unknown = set(changes) - allowed
    if unknown:
        raise MetadataValidationError(
            f"unsupported update fields: {', '.join(sorted(unknown))}"
        )
    return {name: value for name, value in changes.items() if value is not None}


def _database_now(connection: sqlite3.Connection) -> str:
    return connection.execute(
        "SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
    ).fetchone()[0]


def _timestamp_expression() -> str:
    # The update path uses bound parameters. Generate the same UTC representation
    # here instead of injecting a SQL expression into a value slot.
    import datetime

    return datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")
