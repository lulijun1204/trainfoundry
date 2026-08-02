import gzip
import json

import pytest
from PIL import Image

from metadata import (
    DatasetVersion,
    DatasetVersionStage,
    DatasetVersionStatus,
    SchemaFormat,
)
from pipeline import OperatorInputError
from pipeline.validate import (
    DataValidationOperator,
    DataValidationPolicy,
    LanguagePrediction,
)


class _EnglishDetector:
    def detect(self, text):
        return LanguagePrediction("en", 0.99)


def _version(root, *, stage="RAW", schema=None, storage_format="JSONL"):
    return DatasetVersion(
        version_id="dv_records",
        dataset_id="ds_records",
        version_number=1,
        stage=DatasetVersionStage(stage),
        status=DatasetVersionStatus.COMMITTED,
        storage_uri=root.resolve().as_uri(),
        storage_format=storage_format,
        schema_format=SchemaFormat.JSON_SCHEMA,
        schema_definition=schema or {"type": "object"},
        schema_version=None,
        schema_digest="1" * 64,
        content_digest="0" * 64,
        split=None,
        usage_tags=None,
        row_count=None,
        byte_size=None,
        created_by="tests",
        created_at="2026-01-01T00:00:00Z",
        committed_at="2026-01-01T00:00:00Z",
    )


def _codes(report):
    return [issue.code for issue in report.issues]


def test_data_validator_checks_every_jsonl_record_and_schema(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text(
        "\n".join(
            [
                json.dumps({"text": "A valid training example", "label": 1}),
                json.dumps({"text": "", "label": 2}),
                json.dumps({"text": "missing label"}),
                "{broken",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    schema = {
        "type": "object",
        "required": ["text", "label"],
        "properties": {"text": {"type": "string"}, "label": {"type": "integer"}},
    }
    policy = DataValidationPolicy(required_fields=("text",), require_text=True)

    report = DataValidationOperator(policy).validate(_version(data, schema=schema))

    assert report.passed is False
    assert report.checked_count == 4
    assert report.passed_count == 1
    assert report.rejected_count == 3
    assert set(_codes(report)) >= {
        "EMPTY_TEXT",
        "SCHEMA_REQUIRED_FIELD",
        "PARSE_ERROR",
    }
    quality_kwargs = report.quality_result_kwargs("run_quality")
    assert quality_kwargs["dataset_version_id"] == "dv_records"
    assert quality_kwargs["status"] == "SUCCEEDED"
    assert quality_kwargs["rejected_count"] == 3


def test_data_validator_applies_configurable_industry_quality_metrics(tmp_path):
    data = tmp_path / "train.jsonl"
    repeated = "same words here same words here same words here"
    data.write_text(json.dumps({"text": repeated}) + "\n", encoding="utf-8")
    policy = DataValidationPolicy(
        require_text=True,
        min_words=3,
        min_alphanumeric_ratio=0.5,
        max_repeated_ngram_ratio=0.2,
        repeated_ngram_size=3,
        expected_languages=("zh",),
    )

    report = DataValidationOperator(
        policy, language_detector=_EnglishDetector()
    ).validate(_version(data))

    assert report.passed is False
    assert "REPEATED_NGRAM_RATIO" in _codes(report)
    assert "LANGUAGE_MISMATCH" in _codes(report)
    assert report.metrics["languages"] == {"en": 1}


def test_data_validator_allows_empty_optional_text_fields(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text(
        json.dumps({"instruction": "Explain gravity", "context": "", "response": "..."})
        + "\n",
        encoding="utf-8",
    )

    report = DataValidationOperator(
        DataValidationPolicy(require_text=True)
    ).validate(_version(data))

    assert report.passed is True
    assert "EMPTY_TEXT" not in _codes(report)


def test_data_validator_detects_duplicates_and_validates_image_references(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 6), color="red").save(image)
    record = {"text": "caption", "image": "image.png"}
    data = tmp_path / "train.jsonl"
    data.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )
    policy = DataValidationPolicy(check_exact_duplicates=True, require_text=True)

    report = DataValidationOperator(policy).validate(_version(tmp_path))

    assert report.checked_count == 2
    assert report.rejected_count == 1
    assert "EXACT_DUPLICATE" in _codes(report)
    assert "MEDIA_CORRUPT" not in _codes(report)


def test_data_validator_rejects_blank_lines_and_invalid_utf8(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_bytes(b'{"text":"ok"}\n\n\xff\n')

    report = DataValidationOperator().validate(_version(data))

    assert report.checked_count >= 2
    assert report.passed is False
    assert "PARSE_ERROR" in _codes(report)


def test_data_validator_requires_detector_for_language_gate():
    policy = DataValidationPolicy(expected_languages=("en",))

    with pytest.raises(ValueError, match="LanguageDetector"):
        DataValidationOperator(policy)


def test_data_validator_requires_raw_and_supported_format(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")

    with pytest.raises(OperatorInputError, match="RAW"):
        DataValidationOperator().validate(_version(data, stage="PROCESSED"))
    with pytest.raises(OperatorInputError, match="supports"):
        DataValidationOperator().validate(_version(data, storage_format="LANCE"))


def test_data_validator_keeps_exact_counts_when_issue_details_are_bounded(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text("\n" * 5, encoding="utf-8")
    policy = DataValidationPolicy(max_issues=2)

    report = DataValidationOperator(policy).validate(_version(data))

    assert len(report.issues) == 2
    assert report.truncated_issue_count == 3
    assert report.error_count == 5
    assert report.rejected_count == 5
    assert report.summary()["issue_counts"] == {"PARSE_ERROR": 5}


def test_data_validator_streams_warc_wet_records(tmp_path):
    archive = tmp_path / "sample.warc.wet.gz"
    bodies = [b"first training document", b"second training document"]
    with gzip.open(archive, "wb") as stream:
        for body in bodies:
            stream.write(
                b"WARC/1.0\r\n"
                b"WARC-Type: conversion\r\n"
                + f"Content-Length: {len(body)}\r\n".encode()
                + b"\r\n"
                + body
                + b"\r\n\r\n"
            )

    report = DataValidationOperator(
        DataValidationPolicy(require_text=True)
    ).validate(_version(archive, storage_format="WARC_WET_GZIP"))

    assert report.passed is True
    assert report.checked_count == 2
    assert report.metrics["text_value_count"] == 2
