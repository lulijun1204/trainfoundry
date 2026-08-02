import gzip

import pytest

from pipeline.examples.data_processing.common_crawl.processing import (
    RawWarcRecord,
    validate_dataset,
    validate_file_format,
    validate_record,
)


def _warc_record(record_type, body, **headers):
    values = {
        "WARC-Type": record_type,
        "WARC-Date": "2026-07-10T08:36:32Z",
        "WARC-Record-ID": "<urn:uuid:00000000-0000-0000-0000-000000000001>",
        "Content-Type": (
            "text/plain" if record_type == "conversion" else "application/warc-fields"
        ),
        **headers,
    }
    encoded_headers = b"".join(
        f"{name}: {value}\r\n".encode() for name, value in values.items()
    )
    return (
        b"WARC/1.0\r\n"
        + encoded_headers
        + f"Content-Length: {len(body)}\r\n".encode()
        + b"\r\n"
        + body
        + b"\r\n\r\n"
    )


def _write_gzip(path, *records):
    with gzip.open(path, "wb") as stream:
        for record in records:
            stream.write(record)


def test_common_crawl_file_format_validation_reads_complete_warc(tmp_path):
    archive = tmp_path / "sample.warc.wet.gz"
    _write_gzip(
        archive,
        _warc_record("warcinfo", b"publisher: Common Crawl"),
        _warc_record(
            "conversion",
            b"first training document",
            **{"WARC-Target-URI": "https://example.com/one"},
        ),
    )

    summary = validate_file_format(tmp_path)

    assert summary.total_files == 1
    assert summary.valid_files == 1
    assert summary.invalid_files == 0
    assert summary.total_warc_records == 2
    assert summary.record_type_counts == {"conversion": 1, "warcinfo": 1}
    assert summary.issue_counts == {}
    assert len(summary.files[0].sha256) == 64


def test_common_crawl_file_format_validation_reports_truncated_record(tmp_path):
    archive = tmp_path / "broken.warc.wet.gz"
    record = (
        b"WARC/1.0\r\n"
        b"WARC-Type: conversion\r\n"
        b"Content-Length: 20\r\n"
        b"\r\n"
        b"short"
    )
    _write_gzip(archive, record)

    summary = validate_file_format(tmp_path)

    assert summary.invalid_files == 1
    assert summary.total_warc_records == 0
    assert summary.issue_counts == {
        "NO_WARC_RECORDS": 1,
        "TRUNCATED_WARC_BODY": 1,
    }


def test_common_crawl_file_format_validation_checks_magic_and_gzip_crc(tmp_path):
    fake = tmp_path / "fake.warc.wet.gz"
    fake.write_bytes(b"not gzip")
    valid = tmp_path / "crc.warc.wet.gz"
    _write_gzip(
        valid,
        _warc_record(
            "conversion",
            b"text",
            **{"WARC-Target-URI": "https://example.com"},
        ),
    )
    damaged = bytearray(valid.read_bytes())
    damaged[-8] ^= 0xFF
    valid.write_bytes(damaged)

    summary = validate_file_format(tmp_path)

    assert summary.total_files == 2
    assert summary.invalid_files == 2
    assert summary.issue_counts["INVALID_GZIP_MAGIC"] == 1
    assert summary.issue_counts["GZIP_CORRUPT"] == 1


def test_common_crawl_data_validation_checks_conversion_records(tmp_path):
    archive = tmp_path / "sample.warc.wet.gz"
    _write_gzip(
        archive,
        _warc_record("warcinfo", b"publisher: Common Crawl"),
        _warc_record(
            "conversion",
            b"valid training text",
            **{"WARC-Target-URI": "https://example.com/valid"},
        ),
        _warc_record("conversion", b"missing URL"),
        _warc_record(
            "conversion",
            b"   \n",
            **{"WARC-Target-URI": "https://example.com/empty"},
        ),
        _warc_record(
            "conversion",
            b"invalid utf8: \xff",
            **{"WARC-Target-URI": "https://example.com/encoding"},
        ),
    )

    summary = validate_dataset(tmp_path)

    assert summary.total_records == 5
    assert summary.valid_records == 1
    assert summary.rejected_records == 3
    assert summary.skipped_records == 1
    assert summary.record_type_counts == {"conversion": 4, "warcinfo": 1}
    assert summary.issue_counts == {
        "EMPTY_TEXT": 1,
        "INVALID_UTF8": 1,
        "MISSING_TARGET_URI": 1,
    }


def test_common_crawl_data_validation_rejects_control_characters(tmp_path):
    archive = tmp_path / "sample.warc.wet.gz"
    _write_gzip(
        archive,
        _warc_record(
            "conversion",
            b"bad control: \x01",
            **{"WARC-Target-URI": "https://example.com/control"},
        ),
    )

    summary = validate_dataset(archive)

    assert summary.valid_records == 0
    assert summary.rejected_records == 1
    assert summary.issue_counts == {"CONTROL_CHARACTER": 1}


@pytest.mark.parametrize(
    ("headers", "expected_code"),
    [
        (
            {
                "warc-target-uri": "ftp://example.com/file",
                "warc-record-id": "<urn:uuid:1>",
                "warc-date": "2026-07-10T08:36:32Z",
                "content-type": "text/plain",
            },
            "INVALID_TARGET_URI",
        ),
        (
            {
                "warc-target-uri": "https://example.com",
                "warc-date": "2026-07-10T08:36:32Z",
                "content-type": "text/plain",
            },
            "MISSING_RECORD_ID",
        ),
        (
            {
                "warc-target-uri": "https://example.com",
                "warc-record-id": "<urn:uuid:1>",
                "content-type": "text/plain",
            },
            "MISSING_WARC_DATE",
        ),
        (
            {
                "warc-target-uri": "https://example.com",
                "warc-record-id": "<urn:uuid:1>",
                "warc-date": "not-a-date",
                "content-type": "text/plain",
            },
            "INVALID_WARC_DATE",
        ),
        (
            {
                "warc-target-uri": "https://example.com",
                "warc-record-id": "<urn:uuid:1>",
                "warc-date": "2026-07-10T08:36:32Z",
                "content-type": "application/octet-stream",
            },
            "INVALID_CONTENT_TYPE",
        ),
    ],
)
def test_common_crawl_data_validation_checks_wet_headers(headers, expected_code):
    record = RawWarcRecord(
        source_file="sample.warc.wet.gz",
        record_index=1,
        version="WARC/1.0",
        headers={"warc-type": "conversion", **headers},
        body=b"valid text",
    )

    issue = validate_record(record)

    assert issue is not None
    assert issue.code == expected_code
