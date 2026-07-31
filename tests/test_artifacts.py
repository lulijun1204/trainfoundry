from hashlib import sha256

from fetch.artifacts import file_records


def test_file_records(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"abc")
    second.write_bytes(b"defg")

    records = file_records([second, first])

    assert len(records) == 2
    assert sum(record.bytes for record in records) == 7
    assert records[0].path == str(first)
    assert records[0].bytes == 3
    assert records[0].sha256 == sha256(b"abc").hexdigest()
