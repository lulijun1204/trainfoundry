from hashlib import sha256

import pytest

from metadata import MetadataConflictError, MetadataRepository
from metadata.database import DOMAIN_TABLES


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


@pytest.fixture
def repository(tmp_path):
    result = MetadataRepository(tmp_path / "metadata.db")
    result.initialize()
    return result


def _dataset(repository, name="example", namespace="tests"):
    return repository.create_dataset(
        namespace=namespace,
        name=name,
        purpose="SFT",
        modalities=["TEXT"],
        owner="test-team",
        permitted_use={"training": True},
        retention_policy={"days": 30},
        access_policy={"readers": ["test-team"]},
    )


def _version(repository, dataset_id, value="v1", status="DRAFT"):
    return repository.create_version(
        dataset_id=dataset_id,
        stage="RAW",
        status=status,
        storage_uri=f"file:///datasets/{value}",
        storage_format="JSONL",
        schema_format="JSON_SCHEMA",
        schema_definition={"type": "object", "properties": {}},
        schema_digest=_digest(f"{value}-schema"),
        content_digest=_digest(f"{value}-content"),
        created_by="tester",
    )


def _run(repository, dataset_id, value="run", status="PENDING"):
    return repository.create_run(
        run_type="CLEAN",
        operator_name="cleaner",
        operator_version="1.0",
        operator_fingerprint=_digest(f"{value}-operator"),
        compute_key=_digest(f"{value}-compute"),
        output_mode="NEW_VERSION",
        target_dataset_id=dataset_id,
        status=status,
        created_by="tester",
    )


def test_initialize_creates_all_documented_domain_tables(tmp_path):
    repository = MetadataRepository(tmp_path / "metadata.db")

    first = repository.initialize()
    second = repository.initialize()

    assert first["initialized"] is True
    assert first["schema_version"] == 1
    assert set(DOMAIN_TABLES).issubset(first["tables"])
    assert second == first


def test_dataset_crud_and_namespace_name_uniqueness(repository):
    dataset = _dataset(repository)

    loaded = repository.get_dataset(dataset["dataset_id"])
    assert loaded["modalities"] == ["TEXT"]
    assert loaded["contains_pii"] is False
    assert loaded["permitted_use"] == {"training": True}

    updated = repository.update_dataset(
        dataset["dataset_id"],
        owner="new-owner",
        access_policy={"readers": ["new-owner"]},
    )
    assert updated["owner"] == "new-owner"

    assert repository.list_datasets(owner="new-owner") == [updated]
    with pytest.raises(MetadataConflictError, match="UNIQUE"):
        _dataset(repository)


def test_committed_version_is_immutable_but_status_can_change(repository):
    dataset = _dataset(repository)
    version = _version(repository, dataset["dataset_id"])

    committed = repository.update_version(version["version_id"], status="COMMITTED")
    assert committed["committed_at"]

    with pytest.raises(MetadataConflictError, match="immutable"):
        repository.update_version(
            version["version_id"],
            storage_uri="file:///datasets/replaced",
        )

    published = repository.update_version(version["version_id"], status="PUBLISHED")
    assert published["status"] == "PUBLISHED"


def test_successful_deterministic_compute_key_is_unique(repository):
    dataset = _dataset(repository)
    first = _run(repository, dataset["dataset_id"], status="SUCCEEDED")
    assert first["deterministic"] is True

    with pytest.raises(MetadataConflictError, match="compute_key"):
        repository.create_run(
            run_type="CLEAN",
            operator_name="other",
            operator_version="2.0",
            operator_fingerprint=_digest("other-operator"),
            compute_key=first["compute_key"],
            output_mode="NEW_VERSION",
            target_dataset_id=dataset["dataset_id"],
            status="SUCCEEDED",
            created_by="tester",
        )


def test_recursive_lineage_traces_upstream_and_downstream(repository):
    dataset = _dataset(repository)
    first = _version(repository, dataset["dataset_id"], "v1")
    second = _version(repository, dataset["dataset_id"], "v2")
    third = _version(repository, dataset["dataset_id"], "v3")
    first_run = _run(repository, dataset["dataset_id"], "run-1")
    second_run = _run(repository, dataset["dataset_id"], "run-2")

    repository.create_lineage(
        run_id=first_run["run_id"],
        source_version_id=first["version_id"],
        target_version_id=second["version_id"],
        relation_type="DERIVED_FROM",
    )
    repository.create_lineage(
        run_id=second_run["run_id"],
        source_version_id=second["version_id"],
        target_version_id=third["version_id"],
        relation_type="FILTERED_FROM",
    )

    graph = repository.get_lineage(second["version_id"])

    assert graph["upstream"][0]["source_version_id"] == first["version_id"]
    assert graph["downstream"][0]["target_version_id"] == third["version_id"]
    assert (
        repository.get_lineage(third["version_id"], direction="upstream")[
            "upstream"
        ][1]["source_version_id"]
        == first["version_id"]
    )


def test_results_and_training_bindings_follow_version_rules(repository):
    dataset = _dataset(repository)
    version = _version(repository, dataset["dataset_id"])
    quality_run = repository.create_run(
        run_type="QUALITY",
        operator_name="quality",
        operator_version="1",
        operator_fingerprint=_digest("quality-operator"),
        compute_key=_digest("quality-compute"),
        output_mode="NO_DATA_VERSION",
        created_by="tester",
    )
    annotation_run = repository.create_run(
        run_type="ANNOTATION",
        operator_name="annotator",
        operator_version="1",
        operator_fingerprint=_digest("annotation-operator"),
        compute_key=_digest("annotation-compute"),
        output_mode="NO_DATA_VERSION",
        created_by="tester",
    )

    quality = repository.create_quality_result(
        dataset_version_id=version["version_id"],
        run_id=quality_run["run_id"],
        evaluator_name="quality",
        evaluator_version="1",
        status="SUCCEEDED",
        passed=True,
        summary={"score": 0.99},
    )
    annotation = repository.create_annotation_result(
        dataset_version_id=version["version_id"],
        run_id=annotation_run["run_id"],
        annotation_schema_version="labels-v1",
        producer_type="MODEL",
        coverage=0.9,
    )
    assert quality["passed"] is True
    assert annotation["coverage"] == 0.9

    training = repository.create_training_run(
        code_commit="abc123",
        config_digest=_digest("train-config"),
        dataloader_config={"batch_size": 8},
        created_by="tester",
    )
    with pytest.raises(MetadataConflictError, match="COMMITTED"):
        repository.bind_training_version(
            training_run_id=training["training_run_id"],
            dataset_version_id=version["version_id"],
            role="TRAIN",
        )

    repository.update_version(version["version_id"], status="COMMITTED")
    binding = repository.bind_training_version(
        training_run_id=training["training_run_id"],
        dataset_version_id=version["version_id"],
        role="TRAIN",
        weight=0.75,
    )
    loaded = repository.get_training_run(training["training_run_id"])
    assert binding["weight"] == 0.75
    assert loaded["dataset_versions"] == [binding]
