from pathlib import Path

import pytest

from config import config_util


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "sample.toml").write_text(
        '[sample]\nrelative = "../data"\nabsolute = "/tmp/data"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(config_util, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_util, "PROJECT_ROOT", tmp_path)
    config_util._load_config.cache_clear()
    yield
    config_util._load_config.cache_clear()


def test_load_config_returns_defensive_copy(isolated_config):
    first = config_util.load_config("sample")
    first["sample"]["relative"] = "changed"

    assert config_util.load_config("sample")["sample"]["relative"] == "../data"


def test_get_by_key_and_resolve_relative_path(isolated_config, tmp_path):
    assert config_util.get_by_key("sample.relative") == "../data"
    assert config_util.get_path("sample.relative") == (tmp_path / "../data").resolve()
    assert config_util.get_path("sample.absolute") == Path("/tmp/data").resolve()


def test_rejects_unsafe_config_name(isolated_config):
    with pytest.raises(config_util.ConfigError, match="Invalid config name"):
        config_util.load_config("../sample")


def test_environment_overrides_config_and_project_locations(
    tmp_path,
    monkeypatch,
):
    config_dir = tmp_path / "custom-config"
    project_root = tmp_path / "custom-project"
    monkeypatch.setenv("TRAINFOUNDRY_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("TRAINFOUNDRY_PROJECT_ROOT", str(project_root))

    assert config_util._discover_locations() == (
        config_dir.resolve(),
        project_root.resolve(),
    )
