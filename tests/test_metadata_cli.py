import json
from hashlib import sha256

from trainfoundry.cli import main


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def test_metadata_cli_initializes_and_reads_written_records(
    tmp_path,
    monkeypatch,
    capsys,
):
    database_path = tmp_path / "metadata.db"
    monkeypatch.setattr(
        "metadata.database.get_path",
        lambda key: database_path,
    )

    assert main(["metadata", "init", "--output", "json"]) == 0
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["data"]["initialized"] is True

    assert (
        main(
            [
                "metadata",
                "dataset",
                "create",
                "--dataset-id",
                "ds_cli",
                "--namespace",
                "examples",
                "--name",
                "cli-dataset",
                "--purpose",
                "SFT",
                "--modality",
                "TEXT",
                "--owner",
                "cli-user",
                "--output",
                "json",
            ]
        )
        == 0
    )
    created = json.loads(capsys.readouterr().out)
    assert created["command"] == "metadata.dataset.create"
    assert created["data"]["dataset"]["dataset_id"] == "ds_cli"

    assert (
        main(
            [
                "metadata",
                "version",
                "create",
                "--version-id",
                "dv_cli",
                "--dataset-id",
                "ds_cli",
                "--stage",
                "RAW",
                "--storage-uri",
                "file:///tmp/cli",
                "--storage-format",
                "JSONL",
                "--schema-format",
                "JSON_SCHEMA",
                "--schema-definition-json",
                '{"type":"object"}',
                "--schema-digest",
                _digest("schema"),
                "--content-digest",
                _digest("content"),
                "--created-by",
                "cli-user",
                "--output",
                "json",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "metadata",
                "version",
                "get",
                "dv_cli",
                "--output",
                "json",
            ]
        )
        == 0
    )
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["data"]["version"]["schema_definition"] == {"type": "object"}


def test_metadata_cli_reports_uninitialized_database(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        "metadata.database.get_path",
        lambda key: tmp_path / "missing.db",
    )

    exit_code = main(
        [
            "metadata",
            "dataset",
            "get",
            "missing",
            "--output",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 4
    assert payload["error"]["code"] == "METADATA_NOT_INITIALIZED"
