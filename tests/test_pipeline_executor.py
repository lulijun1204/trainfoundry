import json
import zipfile
from hashlib import sha256

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from metadata import (
    DatasetRunStatus,
    DatasetRunType,
    MetadataRepository,
)
from pipeline import (
    BlockStream,
    MaterializationSpec,
    OperatorOutput,
    PipelineExecutor,
    SourceAdapterRegistry,
    SupplementalResultDraft,
)
from pipeline.validate import DataValidationOperator


def _digest(value):
    return sha256(value.encode()).hexdigest()


def _version(repository, path, *, storage_format="JSONL", name="executor"):
    dataset = repository.create_dataset(
        namespace="tests",
        name=name,
        purpose="SFT",
        modalities=["TEXT"],
        owner="tests",
    )
    return repository.create_version(
        dataset_id=dataset["dataset_id"],
        stage="RAW",
        status="COMMITTED",
        storage_uri=path.resolve().as_uri(),
        storage_format=storage_format,
        schema_format="JSON_SCHEMA",
        schema_definition={"type": "object"},
        schema_digest=_digest("schema"),
        content_digest=_digest("content"),
        created_by="tests",
    )


class _ProfileOperator:
    name = "profile"
    version = "1.0.0"
    run_type = DatasetRunType.QUALITY
    deterministic = True

    def fingerprint(self):
        return _digest("profile-1.0.0")

    def parameters(self):
        return {"sample": False}

    def run(self, input_data, context):
        return OperatorOutput(
            data=input_data,
            results=(
                SupplementalResultDraft(
                    result_type="PROFILE",
                    subject_execution_data_id=input_data.execution_data_id,
                    summary={"records": 1},
                ),
            ),
        )


class _FailingOperator(_ProfileOperator):
    name = "failing"

    def fingerprint(self):
        return _digest("failing-1.0.0")

    def run(self, input_data, context):
        raise RuntimeError("operator failed")


class _UppercaseOperator(_ProfileOperator):
    name = "uppercase"
    run_type = DatasetRunType.CLEAN

    def fingerprint(self):
        return _digest("uppercase-1.0.0")

    def run(self, input_data, context):
        def batches():
            for batch in input_data.batches:
                index = batch.schema.get_field_index("text")
                yield batch.set_column(
                    index,
                    batch.schema.field(index),
                    pc.utf8_upper(batch.column(index)),
                )

        return OperatorOutput(
            data=input_data.derive(
                batches=BlockStream(batches, schema=input_data.schema),
            )
        )


def test_executor_governs_runs_quality_and_supplemental_results(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"text": "hello"}) + "\n", encoding="utf-8")
    repository = MetadataRepository(tmp_path / "metadata.db")
    repository.initialize()
    version = _version(repository, data)

    result = PipelineExecutor(repository).execute(
        version,
        [DataValidationOperator(), _UppercaseOperator(), _ProfileOperator()],
        created_by="tests",
    )

    assert result.input_version == version
    assert result.output_data.source_version == version
    assert result.output_version is None
    assert [run.status for run in result.runs] == [
        DatasetRunStatus.SUCCEEDED,
        DatasetRunStatus.SUCCEEDED,
        DatasetRunStatus.SUCCEEDED,
    ]
    output_batches = list(result.output_data.batches)
    assert all(isinstance(batch, pa.RecordBatch) for batch in output_batches)
    assert output_batches[0].to_pylist() == [{"text": "HELLO"}]
    assert len(result.results) == 2
    assert len(repository.list_quality_results(dataset_version_id=version.version_id)) == 1
    assert result.results[1].result_type == "PROFILE"
    assert repository.list_versions(dataset_id=version.dataset_id) == [version]


def test_executor_records_failed_dataset_run(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    repository = MetadataRepository(tmp_path / "metadata.db")
    repository.initialize()
    version = _version(repository, data)

    with pytest.raises(RuntimeError, match="operator failed"):
        PipelineExecutor(repository).execute(
            version,
            [_FailingOperator()],
            created_by="tests",
        )

    runs = repository.list_runs()
    assert len(runs) == 1
    assert runs[0].status is DatasetRunStatus.FAILED
    assert runs[0].error_message == "operator failed"


def test_executor_materializes_lance_only_at_explicit_boundary(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text('{"text":"hello"}\n', encoding="utf-8")
    repository = MetadataRepository(tmp_path / "metadata.db")
    repository.initialize()
    version = _version(repository, source)
    target = tmp_path / "processed.lance"

    result = PipelineExecutor(repository).execute(
        version,
        [_UppercaseOperator()],
        created_by="tests",
        materialization=MaterializationSpec(
            storage_uri=str(target),
            stage="PROCESSED",
            created_by="tests",
        ),
    )

    assert result.output_version is not None
    assert result.output_version.version_id != version.version_id
    assert result.output_version.storage_format == "LANCE"
    assert result.output_version.schema_format == "ARROW"
    reopened = SourceAdapterRegistry.default().open(result.output_version)
    assert [row for batch in reopened.batches for row in batch.to_pylist()] == [
        {"text": "HELLO"}
    ]
    assert len(repository.list_versions(dataset_id=version.dataset_id)) == 2
    assert [run.output_mode for run in result.runs] == [
        "NO_DATA_VERSION",
        "NEW_VERSION",
    ]


def test_zip_source_adapter_exposes_arrow_file_reference_records(tmp_path):
    archive = tmp_path / "raw.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("data/a.json", "{}")
        output.writestr("images/a.jpg", b"jpeg")
    repository = MetadataRepository(tmp_path / "metadata.db")
    repository.initialize()
    version = _version(
        repository,
        archive,
        storage_format="ZIP",
        name="raw-zip",
    )

    data = SourceAdapterRegistry.default().open(version)
    batches = list(data.batches)

    assert data.source_version == version
    assert all(isinstance(batch, pa.RecordBatch) for batch in batches)
    assert [
        row["archive_member"]
        for batch in batches
        for row in batch.to_pylist()
    ] == [
        "data/a.json",
        "images/a.jpg",
    ]


def test_block_stream_rejects_non_arrow_batches():
    stream = BlockStream(lambda: iter([{"text": "not-arrow"}]))

    with pytest.raises(TypeError, match="must yield pyarrow.RecordBatch"):
        list(stream)
