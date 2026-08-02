import gzip
import json
import zipfile
from hashlib import sha256
from types import SimpleNamespace

import pytest

from metadata import (
    DatasetVersion,
    DatasetVersionStage,
    DatasetVersionStatus,
    SchemaFormat,
)
from pipeline import OperatorInputError
from pipeline.validate import (
    FileExpectation,
    FileValidationOperator,
    FileValidationPolicy,
)
from pipeline.validate.common import IssueCollector
from pipeline.validate.packages import validate_known_package


def _sha256(path):
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8192):
            digest.update(chunk)
    return digest.hexdigest()


def _content_digest(root, paths):
    files = []
    for path in paths:
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        files.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    payload = json.dumps(
        {"files": sorted(files, key=lambda item: item["path"])},
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode()).hexdigest()


def _version(root, paths, *, storage_format="JSONL", stage="RAW", byte_size=None):
    return DatasetVersion(
        version_id="dv_test",
        dataset_id="ds_test",
        version_number=1,
        stage=DatasetVersionStage(stage),
        status=DatasetVersionStatus.COMMITTED,
        storage_uri=root.resolve().as_uri(),
        storage_format=storage_format,
        schema_format=SchemaFormat.JSON_SCHEMA,
        schema_definition={"type": "object"},
        schema_version=None,
        schema_digest="1" * 64,
        content_digest=_content_digest(root, paths),
        split=None,
        usage_tags=None,
        row_count=None,
        byte_size=(
            sum(path.stat().st_size for path in paths)
            if byte_size is None
            else byte_size
        ),
        created_by="tests",
        created_at="2026-01-01T00:00:00Z",
        committed_at="2026-01-01T00:00:00Z",
    )


def _codes(report):
    return {issue.code for issue in report.issues}


def test_file_validator_streams_and_matches_manifest(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n{"text":"world"}\n', encoding="utf-8")
    version = _version(tmp_path, [data])
    policy = FileValidationPolicy(
        expected_files=(
            FileExpectation(
                path="train.jsonl",
                byte_size=data.stat().st_size,
                sha256=_sha256(data),
            ),
        )
    )

    report = FileValidationOperator(policy).validate(version)

    assert report.passed is True
    assert report.checked_count == 1
    assert report.passed_count == 1
    assert report.rejected_count == 0
    assert report.metrics["formats"] == {"JSONL": 1}
    assert report.summary()["error_count"] == 0


def test_file_validator_reports_parse_and_version_total_failures(tmp_path):
    data = tmp_path / "broken.jsonl"
    data.write_text('{"text":"ok"}\n{"text":\n', encoding="utf-8")
    version = _version(tmp_path, [data], byte_size=data.stat().st_size + 1)

    report = FileValidationOperator().validate(version)

    assert report.passed is False
    assert "PARSE_ERROR" in _codes(report)
    assert "TOTAL_SIZE_MISMATCH" in _codes(report)
    assert report.rejected_count == 1


def test_file_validator_rejects_zip_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.json", "{}")
    version = _version(tmp_path, [archive], storage_format="FILES")

    report = FileValidationOperator().validate(version)

    assert report.passed is False
    assert "PATH_TRAVERSAL" in _codes(report)
    assert report.rejected_count == 1


def test_file_validator_rejects_declared_format_mismatch(tmp_path):
    data = tmp_path / "data.json"
    data.write_text("{}", encoding="utf-8")
    version = _version(data, [data], storage_format="ZIP")

    report = FileValidationOperator().validate(version)

    assert report.passed is False
    assert "FORMAT_MISMATCH" in _codes(report)


def test_file_validator_requires_raw_version(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    version = _version(data, [data], stage="PROCESSED")

    with pytest.raises(OperatorInputError, match="RAW"):
        FileValidationOperator().validate(version)


def test_file_validator_checks_gzip_crc_and_warc_boundaries(tmp_path):
    archive = tmp_path / "sample.warc.wet.gz"
    body = b"training text"
    record = (
        b"WARC/1.0\r\n"
        b"WARC-Type: conversion\r\n"
        + f"Content-Length: {len(body)}\r\n".encode()
        + b"\r\n"
        + body
        + b"\r\n\r\n"
    )
    with gzip.open(archive, "wb") as stream:
        stream.write(record)
    version = _version(tmp_path, [archive], storage_format="WARC_WET_GZIP")

    report = FileValidationOperator().validate(version)

    assert report.passed is True
    assert report.metrics["formats"] == {"GZIP": 1}


def test_file_validator_reads_all_hdf5_datasets(tmp_path):
    h5py = pytest.importorskip("h5py")
    data = tmp_path / "episodes.hdf5"
    with h5py.File(data, "w") as container:
        container.create_dataset("observations", data=[[1.0, 2.0], [3.0, 4.0]])
        container.create_dataset("reward", data=1.0)
    version = _version(tmp_path, [data], storage_format="MINARI")

    report = FileValidationOperator().validate(version)

    assert report.passed is True
    assert report.metrics["formats"] == {"HDF5": 1}


def test_file_validator_checks_coco_image_annotation_contract(tmp_path):
    images = tmp_path / "val2017.zip"
    annotations = tmp_path / "annotations_trainval2017.zip"
    with zipfile.ZipFile(images, "w") as archive:
        archive.writestr("val2017/000000000001.jpg", b"placeholder")
    payload = {
        "images": [
            {"id": 1, "file_name": "000000000001.jpg", "width": 10, "height": 8}
        ],
        "categories": [{"id": 5, "name": "thing"}],
        "annotations": [
            {"id": 9, "image_id": 1, "category_id": 5, "bbox": [1, 1, 4, 3]}
        ],
    }
    with zipfile.ZipFile(annotations, "w") as archive:
        archive.writestr("annotations/instances_val2017.json", json.dumps(payload))
    version = _version(tmp_path, [annotations, images], storage_format="FILES")

    report = FileValidationOperator().validate(version)

    assert report.passed is True


def test_file_validator_rejects_broken_coco_references_and_geometry(tmp_path):
    images = tmp_path / "val2017.zip"
    annotations = tmp_path / "annotations_trainval2017.zip"
    with zipfile.ZipFile(images, "w") as archive:
        archive.writestr("val2017/another.jpg", b"placeholder")
    payload = {
        "images": [
            {"id": 1, "file_name": "missing.jpg", "width": 10, "height": 8}
        ],
        "categories": [{"id": 5, "name": "thing"}],
        "annotations": [
            {"id": 9, "image_id": 1, "category_id": 5, "bbox": [9, 7, 4, 3]}
        ],
    }
    with zipfile.ZipFile(annotations, "w") as archive:
        archive.writestr("annotations/instances_val2017.json", json.dumps(payload))
    version = _version(tmp_path, [annotations, images], storage_format="FILES")

    report = FileValidationOperator().validate(version)

    assert report.passed is False
    assert "REFERENCE_MISSING" in _codes(report)
    assert "ANNOTATION_OUT_OF_BOUNDS" in _codes(report)


def test_ucf101_contract_checks_video_references_labels_and_fold_leakage(
    tmp_path, monkeypatch
):
    videos = tmp_path / "UCF101.rar"
    videos.write_bytes(b"Rar!\x1a\x07\x01\x00")
    splits = tmp_path / "UCF101TrainTestSplits.zip"
    with zipfile.ZipFile(splits, "w") as archive:
        archive.writestr("ucfTrainTestlist/classInd.txt", "1 ApplyEyeMakeup\n")
        archive.writestr(
            "ucfTrainTestlist/trainlist01.txt",
            "ApplyEyeMakeup/v_One.avi 1\nApplyEyeMakeup/v_Missing.avi 2\n",
        )
        archive.writestr(
            "ucfTrainTestlist/testlist01.txt", "ApplyEyeMakeup/v_One.avi\n"
        )
    monkeypatch.setattr("pipeline.validate.packages.shutil.which", lambda _name: "bsdtar")
    monkeypatch.setattr(
        "pipeline.validate.packages.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="UCF-101/ApplyEyeMakeup/v_One.avi\n"
        ),
    )
    issues = IssueCollector(100)

    validate_known_package(tmp_path, [videos, splits], issues)

    assert issues.code_counts["REFERENCE_MISSING"] == 1
    assert issues.code_counts["LABEL_MISMATCH"] == 1
    assert issues.code_counts["SPLIT_LEAKAGE"] == 1
