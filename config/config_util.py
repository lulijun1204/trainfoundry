"""Utilities for reading project configuration from TOML files."""

import os
from copy import deepcopy
from functools import cache
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    try:
        import tomli as tomllib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Python 3.10 and earlier require `tomli`: pip install tomli"
        ) from exc


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PACKAGED_CONFIG_DIR = Path(__file__).resolve().parent


def _discover_locations() -> tuple[Path, Path]:
    configured_dir = os.environ.get("TRAINFOUNDRY_CONFIG_DIR")
    configured_root = os.environ.get("TRAINFOUNDRY_PROJECT_ROOT")
    if configured_dir:
        config_dir = Path(configured_dir).expanduser().resolve()
        project_root = (
            Path(configured_root).expanduser().resolve()
            if configured_root
            else config_dir.parent
        )
        return config_dir, project_root

    working_root = Path.cwd()
    working_config = working_root / "config"
    if (working_config / "paths.toml").is_file():
        return working_config, working_root

    if (PACKAGE_ROOT / "pyproject.toml").is_file():
        return PACKAGED_CONFIG_DIR, PACKAGE_ROOT

    project_root = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else Path.home() / ".local" / "share" / "trainfoundry"
    )
    return PACKAGED_CONFIG_DIR, project_root


CONFIG_DIR, PROJECT_ROOT = _discover_locations()
_MISSING = object()


class ConfigError(ValueError):
    """Raised when a configuration file or value is invalid."""


def _config_path(name: str) -> Path:
    """Turn a config name such as ``paths`` into a safe TOML file path."""
    path = Path(name)
    if path.name != name or path.suffix not in ("", ".toml"):
        raise ConfigError(f"Invalid config name: {name!r}")

    filename = name if path.suffix == ".toml" else f"{name}.toml"
    return CONFIG_DIR / filename


@cache
def _load_config(name: str) -> dict[str, Any]:
    path = _config_path(name)
    if not path.is_file():
        raise ConfigError(f"Config file does not exist: {path}")

    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc


def load_config(name: str) -> dict[str, Any]:
    """Load one config file.

    ``name`` may be ``"paths"`` or ``"paths.toml"``. A defensive copy is
    returned so callers cannot mutate the cached configuration.
    """
    return deepcopy(_load_config(name))


def get_by_key(key: str, default: Any = _MISSING) -> Any:
    """Read a value by dotted key.

    The first key segment identifies the TOML file. For example,
    ``paths.text_path`` is read from ``paths.toml``.
    """
    name, separator, _ = key.partition(".")
    if not separator:
        raise ConfigError(
            f"Config key must include its file name, for example 'paths.text_path': {key!r}"
        )

    value: Any = _load_config(name)
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            if default is not _MISSING:
                return default
            raise ConfigError(f"Missing config key {key!r} in {_config_path(name)}")
        value = value[part]
    return deepcopy(value)


def get_path(key: str) -> Path:
    """Read a path value and resolve relative paths from the project root."""
    value = get_by_key(key)
    if not isinstance(value, str):
        raise ConfigError(
            f"Path config {key!r} must be a string, got {type(value).__name__}"
        )

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
