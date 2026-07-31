"""SQLite lifecycle and connection management for TrainFoundry metadata."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any

from config import get_path
from metadata.errors import MetadataNotInitializedError

SCHEMA_VERSION = 1
DOMAIN_TABLES = (
    "datasets",
    "dataset_versions",
    "dataset_runs",
    "dataset_lineage",
    "quality_result_sets",
    "annotation_result_sets",
    "training_runs",
    "training_run_dataset_versions",
)


class MetadataDatabase:
    """Own the configured SQLite file and initialize its schema."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or get_path("paths.metadata_db_path")

    def initialize(self) -> dict[str, Any]:
        """Create or upgrade the local database to the bundled schema."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        schema = files("metadata").joinpath("schema.sql").read_text(encoding="utf-8")
        with self._connect() as connection:
            current = connection.execute("PRAGMA user_version").fetchone()[0]
            if current > SCHEMA_VERSION:
                raise MetadataNotInitializedError(
                    f"Metadata database schema {current} is newer than supported "
                    f"schema {SCHEMA_VERSION}: {self.path}"
                )
            connection.executescript(schema)
        return self.status()

    def status(self) -> dict[str, Any]:
        """Return initialization status without creating an empty database."""
        if not self.path.is_file():
            return {
                "path": str(self.path),
                "initialized": False,
                "schema_version": 0,
                "expected_schema_version": SCHEMA_VERSION,
                "tables": [],
            }
        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT name
                    FROM sqlite_schema
                    WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                    """
                )
            ]
        initialized = version == SCHEMA_VERSION and all(
            table in tables for table in DOMAIN_TABLES
        )
        return {
            "path": str(self.path),
            "initialized": initialized,
            "schema_version": version,
            "expected_schema_version": SCHEMA_VERSION,
            "tables": tables,
        }

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open an initialized connection with integrity checks enabled."""
        status = self.status()
        if not status["initialized"]:
            raise MetadataNotInitializedError(
                f"Metadata database is not initialized: {self.path}; "
                "run `trainfoundry metadata init`"
            )
        with self._connect() as connection:
            yield connection

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
