"""Lance materialization for Arrow execution data."""

from __future__ import annotations

from hashlib import sha256
from itertools import chain
from pathlib import Path
from urllib.parse import unquote, urlparse

from pipeline.base import operator_fingerprint
from pipeline.data import ExecutionDataset
from pipeline.materialize.base import MaterializationSpec, MaterializedData


class LanceMaterializer:
    formats = frozenset({"LANCE"})
    name = "lance_materializer"
    version = "1.0.0"

    def fingerprint(self) -> str:
        return operator_fingerprint(self.name, self.version, {})

    def write(
        self,
        data: ExecutionDataset,
        spec: MaterializationSpec,
    ) -> MaterializedData:
        try:
            import lance
        except ImportError as exc:
            raise RuntimeError(
                "Lance output requires the optional 'pylance' package"
            ) from exc
        path = _new_local_path(spec.storage_uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        batches = iter(data.batches)
        try:
            first = next(batches)
        except StopIteration as exc:
            raise ValueError("cannot infer a Lance schema from an empty dataset") from exc
        all_batches = chain((first,), batches)
        dataset = lance.write_dataset(
            all_batches,
            path,
            schema=first.schema,
            mode="create",
        )
        return MaterializedData(
            storage_uri=path.resolve().as_uri(),
            storage_format="LANCE",
            content_digest=_directory_digest(path),
            row_count=dataset.count_rows(),
            byte_size=sum(
                item.stat().st_size for item in path.rglob("*") if item.is_file()
            ),
            schema=first.schema,
        )


def _new_local_path(storage_uri: str) -> Path:
    parsed = urlparse(storage_uri)
    if parsed.scheme not in {"", "file"}:
        raise ValueError("built-in materializers currently support local paths only")
    if parsed.scheme == "file" and parsed.netloc not in {"", "localhost"}:
        raise ValueError(f"unsupported file URI authority: {parsed.netloc!r}")
    raw_path = unquote(parsed.path) if parsed.scheme else storage_uri
    if not raw_path:
        raise ValueError("storage_uri does not contain a path")
    path = Path(raw_path)
    if path.exists():
        raise FileExistsError(f"materialization target already exists: {path}")
    return path


def _directory_digest(path: Path) -> str:
    digest = sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()
