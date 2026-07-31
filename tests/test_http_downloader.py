from subprocess import CalledProcessError

import pytest

from fetch.downloaders import http
from fetch.downloaders.http import CurlDownloader
from fetch.models import HttpFileRequest


def test_download_failure_without_partial_file_raises_process_error(
    tmp_path, monkeypatch
):
    class Result:
        returncode = 6

    monkeypatch.setattr(http.subprocess, "run", lambda *args, **kwargs: Result())

    with pytest.raises(CalledProcessError):
        CurlDownloader().fetch_file(
            HttpFileRequest(
                url="https://example.invalid/data.zip",
                output_name="data.zip",
            ),
            tmp_path,
            max_attempts=1,
        )


def test_download_resumes_after_interruption(tmp_path, monkeypatch):
    output = tmp_path / "data.zip"
    partial = tmp_path / "data.zip.part"
    calls = 0

    class Result:
        def __init__(self, returncode):
            self.returncode = returncode

    def fake_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            partial.write_bytes(b"abc")
            return Result(18)
        partial.write_bytes(b"abcdef")
        return Result(0)

    monkeypatch.setattr(http.subprocess, "run", fake_run)
    monkeypatch.setattr(http.time, "sleep", lambda seconds: None)

    result = CurlDownloader().fetch_file(
        HttpFileRequest(
            url="https://example.invalid/data.zip",
            output_name="data.zip",
            expected_bytes=6,
        ),
        tmp_path,
        max_attempts=2,
    )

    assert result == output
    assert output.read_bytes() == b"abcdef"
    assert not partial.exists()


def test_force_refresh_keeps_old_file_until_replacement_succeeds(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "data.zip"
    output.write_bytes(b"old")

    class Result:
        returncode = 0

    def fake_run(command, check):
        partial = command[command.index("--output") + 1]
        assert output.read_bytes() == b"old"
        with open(partial, "wb") as file:
            file.write(b"new-data")
        return Result()

    monkeypatch.setattr(http.subprocess, "run", fake_run)

    result = CurlDownloader().fetch_file(
        HttpFileRequest(
            url="https://example.invalid/data.zip",
            output_name="data.zip",
            expected_bytes=8,
        ),
        tmp_path,
        force=True,
    )

    assert result == output
    assert output.read_bytes() == b"new-data"
