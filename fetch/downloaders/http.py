"""Resumable HTTP downloads implemented with curl."""

import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path

from fetch.models import HttpAcquisition, HttpFileRequest


class CurlDownloader:
    """Fulfil HTTP file requirements with resumable curl processes."""

    def fetch(
        self,
        request: HttpAcquisition,
        destination: Path,
        *,
        force: bool = False,
    ) -> list[Path]:
        return [
            self.fetch_file(file_request, destination, force=force)
            for file_request in request.files
        ]

    def fetch_file(
        self,
        request: HttpFileRequest,
        destination: Path,
        *,
        max_attempts: int = 20,
        force: bool = False,
    ) -> Path:
        output = destination / request.output_name
        output.parent.mkdir(parents=True, exist_ok=True)

        if output.is_file() and not force:
            _validate_size(output, request.expected_bytes)
            print(f"Already downloaded: {output}", file=sys.stderr)
            return output

        if force:
            url_digest = sha256(request.url.encode("utf-8")).hexdigest()[:12]
            partial = output.with_name(f"{output.name}.{url_digest}.part")
        else:
            partial = output.with_name(f"{output.name}.part")
        if (
            request.expected_bytes is not None
            and partial.is_file()
            and partial.stat().st_size == request.expected_bytes
        ):
            partial.replace(output)
            return output

        for attempt in range(1, max_attempts + 1):
            command = [
                "curl",
                "--fail",
                "--location",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                request.url,
            ]
            result = subprocess.run(command, check=False)
            if result.returncode == 0:
                break

            partial_bytes = partial.stat().st_size if partial.is_file() else 0
            if (
                request.expected_bytes is not None
                and partial_bytes == request.expected_bytes
            ):
                break
            if attempt == max_attempts:
                raise subprocess.CalledProcessError(result.returncode, command)

            delay_seconds = min(2 ** (attempt - 1), 15)
            print(
                f"Download interrupted (attempt {attempt}/{max_attempts}); "
                f"resuming {partial_bytes} bytes in {delay_seconds}s",
                file=sys.stderr,
            )
            time.sleep(delay_seconds)

        _validate_size(partial, request.expected_bytes)
        partial.replace(output)
        return output


def _validate_size(path: Path, expected_bytes: int | None) -> None:
    if expected_bytes is None:
        return

    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Unexpected size for {path}: expected {expected_bytes}, got {actual_bytes}"
        )
