"""Governed streaming execution of dataset pipeline operators."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256

import pyarrow as pa

from metadata import (
    DatasetRun,
    DatasetRunStatus,
    DatasetRunType,
    DatasetVersion,
    MetadataRepository,
    OutputMode,
)
from pipeline.base import (
    OperatorContext,
    PipelineExecutionResult,
    PipelineOperator,
    ResultDraft,
    ValidationReport,
)
from pipeline.data import ExecutionDataset
from pipeline.materialize import MaterializationSpec, MaterializerRegistry
from pipeline.source import SourceAdapterRegistry


class PipelineExecutionError(RuntimeError):
    """Raised when an operator violates the runtime data contract."""


class PipelineExecutor:
    """Pass ephemeral ExecutionDataset values through ordered operators."""

    def __init__(
        self,
        repository: MetadataRepository | None = None,
        *,
        sources: SourceAdapterRegistry | None = None,
        materializers: MaterializerRegistry | None = None,
    ) -> None:
        self.repository = repository or MetadataRepository()
        self.sources = sources or SourceAdapterRegistry.default()
        self.materializers = materializers or MaterializerRegistry.default()

    def execute(
        self,
        input_version: DatasetVersion,
        operators: Sequence[PipelineOperator],
        *,
        created_by: str,
        materialization: MaterializationSpec | None = None,
    ) -> PipelineExecutionResult:
        """Execute without creating versions between operators."""
        persisted = self.repository.get_version(input_version.version_id)
        if persisted != input_version:
            raise PipelineExecutionError(
                "input_version does not match the persisted DatasetVersion"
            )

        current = self.sources.open(input_version)
        runs: list[DatasetRun] = []
        results: list[ResultDraft] = []
        for operator in operators:
            compute_key = _operator_compute_key(current, operator)
            run = self.repository.create_run(
                run_type=operator.run_type,
                operator_name=operator.name,
                operator_version=operator.version,
                operator_fingerprint=operator.fingerprint(),
                params=operator.parameters(),
                compute_key=compute_key,
                deterministic=operator.deterministic,
                output_mode=OutputMode.NO_DATA_VERSION,
                status=DatasetRunStatus.RUNNING,
                started_at=_now(),
                created_by=created_by,
            )
            context = OperatorContext(
                run_id=run.run_id,
                created_by=created_by,
                input_version=input_version,
            )
            try:
                output = operator.run(current, context)
                if output.data.source_version.version_id != input_version.version_id:
                    raise PipelineExecutionError(
                        f"operator {operator.name} changed the persisted source "
                        "identity of its ExecutionDataset"
                    )
                for result in output.results:
                    self._persist_result(run.run_id, result)
                current = output.data.with_execution_digest(compute_key)
                run = self.repository.update_run(
                    run.run_id,
                    status=DatasetRunStatus.SUCCEEDED,
                    finished_at=_now(),
                )
            except Exception as exc:
                self.repository.update_run(
                    run.run_id,
                    status=DatasetRunStatus.FAILED,
                    finished_at=_now(),
                    error_message=str(exc),
                )
                raise
            runs.append(run)
            results.extend(output.results)

        output_version = None
        if materialization is not None:
            output_version, materialization_run = self._materialize(
                current,
                materialization,
            )
            runs.append(materialization_run)

        return PipelineExecutionResult(
            input_version=input_version,
            output_data=current,
            output_version=output_version,
            runs=tuple(runs),
            results=tuple(results),
        )

    def materialize(
        self,
        data: ExecutionDataset,
        spec: MaterializationSpec,
    ) -> DatasetVersion:
        """Explicitly promote ephemeral data to a governed DatasetVersion."""
        version, _run = self._materialize(data, spec)
        return version

    def _materialize(
        self,
        data: ExecutionDataset,
        spec: MaterializationSpec,
    ) -> tuple[DatasetVersion, DatasetRun]:
        materializer = self.materializers.get(spec.storage_format)
        target_dataset_id = spec.dataset_id or data.source_version.dataset_id
        run = self.repository.create_run(
            run_type=DatasetRunType.CONVERT,
            operator_name=materializer.name,
            operator_version=materializer.version,
            operator_fingerprint=materializer.fingerprint(),
            params=asdict(spec),
            compute_key=_materialization_compute_key(data, spec, materializer.fingerprint()),
            deterministic=True,
            output_mode=OutputMode.NEW_VERSION,
            target_dataset_id=target_dataset_id,
            status=DatasetRunStatus.RUNNING,
            started_at=_now(),
            created_by=spec.created_by,
        )
        try:
            physical = materializer.write(data, spec)
            schema_definition, schema_format = _materialized_schema(
                data,
                spec,
                physical.schema,
            )
            version = self.repository.create_version(
                dataset_id=target_dataset_id,
                stage=spec.stage,
                status=spec.status,
                storage_uri=physical.storage_uri,
                storage_format=physical.storage_format,
                schema_format=schema_format,
                schema_definition=schema_definition,
                schema_version=spec.schema_version,
                schema_digest=_json_digest(schema_definition),
                content_digest=physical.content_digest,
                split=spec.split,
                usage_tags=spec.usage_tags,
                row_count=physical.row_count,
                byte_size=physical.byte_size,
                created_by=spec.created_by,
            )
            self.repository.create_lineage(
                run_id=run.run_id,
                source_version_id=data.source_version.version_id,
                target_version_id=version.version_id,
                relation_type="DERIVED_FROM",
            )
            run = self.repository.update_run(
                run.run_id,
                status=DatasetRunStatus.SUCCEEDED,
                finished_at=_now(),
            )
        except Exception as exc:
            self.repository.update_run(
                run.run_id,
                status=DatasetRunStatus.FAILED,
                finished_at=_now(),
                error_message=str(exc),
            )
            raise
        return version, run

    def _persist_result(self, run_id: str, result: ResultDraft) -> None:
        if isinstance(result, ValidationReport):
            self.repository.create_quality_result(
                **result.quality_result_kwargs(run_id)
            )


def _operator_compute_key(
    data: ExecutionDataset,
    operator: PipelineOperator,
) -> str:
    return _json_digest(
        {
            "upstream_digest": data.execution_digest,
            "source_version_id": data.source_version.version_id,
            "operator_fingerprint": operator.fingerprint(),
            "params": operator.parameters(),
        }
    )


def _materialization_compute_key(
    data: ExecutionDataset,
    spec: MaterializationSpec,
    materializer_fingerprint: str,
) -> str:
    return _json_digest(
        {
            "upstream_digest": data.execution_digest,
            "materializer_fingerprint": materializer_fingerprint,
            "spec": asdict(spec),
        }
    )


def _json_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _materialized_schema(
    data: ExecutionDataset,
    spec: MaterializationSpec,
    physical_schema: pa.Schema | None,
) -> tuple[dict[str, object], str]:
    if spec.schema_definition is not None:
        return (
            dict(spec.schema_definition),
            spec.schema_format or data.source_version.schema_format.value,
        )
    arrow_schema = physical_schema or data.schema
    if arrow_schema is not None:
        return (
            _arrow_schema_definition(arrow_schema),
            spec.schema_format or "ARROW",
        )
    return (
        dict(data.source_version.schema_definition),
        spec.schema_format or data.source_version.schema_format.value,
    )


def _arrow_schema_definition(schema: pa.Schema) -> dict[str, object]:
    return {
        "fields": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
                "metadata": _decoded_metadata(field.metadata),
            }
            for field in schema
        ],
        "metadata": _decoded_metadata(schema.metadata),
    }


def _decoded_metadata(
    metadata: dict[bytes, bytes] | None,
) -> dict[str, str]:
    if not metadata:
        return {}
    return {
        key.decode("utf-8", errors="replace"): value.decode(
            "utf-8",
            errors="replace",
        )
        for key, value in metadata.items()
    }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
