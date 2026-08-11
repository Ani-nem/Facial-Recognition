"""Tests for config resolution and precedence."""

from pathlib import Path

import pytest

from faceorg.config import DEFAULT_DB_PATH, Config


def test_defaults():
    c = Config.from_sources()
    assert c.model == "hog"
    assert c.tolerance == 0.55
    assert c.mode == "symlink"
    assert c.db == DEFAULT_DB_PATH.expanduser()
    assert isinstance(c.extensions, tuple)


def test_cli_override_and_none_ignored():
    c = Config.from_sources({"model": "cnn", "tolerance": None})
    assert c.model == "cnn"
    assert c.tolerance == 0.55  # None override ignored


def test_paths_expanded():
    c = Config.from_sources({"src": "~/Photos"})
    assert str(c.src).startswith(str(Path.home()))


def test_invalid_model_rejected():
    with pytest.raises(ValueError):
        Config.from_sources({"model": "bogus"})


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        Config.from_sources({"mode": "bogus"})


def test_toml_file_and_precedence(tmp_path):
    cfg_file = tmp_path / "faceorg.toml"
    cfg_file.write_text(
        """
[paths]
db = "/tmp/custom.db"
[detect]
model = "cnn"
[match]
tolerance = 0.5
[scan]
extensions = [".jpg", ".png"]
"""
    )
    c = Config.from_sources(config_path=cfg_file)
    assert c.model == "cnn"
    assert c.tolerance == 0.5
    assert str(c.db) == "/tmp/custom.db"
    assert c.extensions == (".jpg", ".png")

    # CLI beats file
    c2 = Config.from_sources({"model": "hog"}, config_path=cfg_file)
    assert c2.model == "hog"
    assert c2.tolerance == 0.5  # still from file


def test_missing_explicit_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.from_sources(config_path=tmp_path / "does-not-exist.toml")
