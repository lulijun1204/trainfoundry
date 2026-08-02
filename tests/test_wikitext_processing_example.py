import json

import pyarrow as pa

from pipeline.examples.data_processing.wikitext.processing import (
    ARROW_SCHEMA,
    inspect_lance_dataset,
    iter_arrow_batches,
    normalize_text,
    validate_dataset,
    write_lance_dataset,
)


def _write_fixture(root):
    root.mkdir()
    (root / "train.jsonl").write_bytes(
        b'{"text":""}\n'
        b'{"text":" = Heading = \\n"}\n'
        b'{"text":"role @-@ playing"}\n'
        b'{invalid}\n'
    )
    (root / "validation.jsonl").write_text(
        json.dumps({"text": "valid"}) + "\n",
        encoding="utf-8",
    )
    (root / "test.jsonl").write_text(
        json.dumps({"other": "missing"}) + "\n",
        encoding="utf-8",
    )


def test_wikitext_validation_explains_rejected_rows(tmp_path):
    root = tmp_path / "raw"
    _write_fixture(root)

    summary = validate_dataset(root)

    assert summary.total_records == 6
    assert summary.valid_records == 3
    assert summary.rejected_records == 3
    assert summary.issue_counts == {
        "EMPTY_TEXT": 1,
        "INVALID_JSON": 1,
        "MISSING_TEXT": 1,
    }


def test_wikitext_standardizes_to_arrow_and_lance(tmp_path):
    root = tmp_path / "raw"
    _write_fixture(root)

    batches = list(iter_arrow_batches(root, batch_size=2))

    assert all(isinstance(batch, pa.RecordBatch) for batch in batches)
    assert all(batch.schema.equals(ARROW_SCHEMA) for batch in batches)
    assert [row["text"] for batch in batches for row in batch.to_pylist()] == [
        "= Heading =",
        "role-playing",
        "valid",
    ]

    output = tmp_path / "standardized.lance"
    written = write_lance_dataset(root, output, batch_size=2, overwrite=False)
    reopened = inspect_lance_dataset(output, limit=10)

    assert written["action"] == "created"
    assert written["rows"] == 3
    assert reopened["rows"] == 3
    assert [row["source_split"] for row in reopened["sample"]] == [
        "train",
        "train",
        "validation",
    ]

    overwritten = write_lance_dataset(root, output, batch_size=2)
    assert overwritten["action"] == "overwritten"
    assert overwritten["rows"] == 3


def test_wikitext_normalization_is_explicit_and_deterministic():
    assert normalize_text(" role @-@ playing ") == "role-playing"
