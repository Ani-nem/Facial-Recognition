"""End-to-end CLI tests via Typer's CliRunner, with a mocked encoder."""

import os

from typer.testing import CliRunner

import faceorg.ingest as ingest_mod
from faceorg import cli

runner = CliRunner()


def _db_path(tmp_path):
    return str(tmp_path / "faces.db")


def _patched_scan(fakes, monkeypatch):
    """Patch ingest.scan (as referenced by the CLI) to inject fakes."""
    orig = ingest_mod.scan

    def wrapper(config, db, **kw):
        kw.update(fakes)
        return orig(config, db, **kw)

    monkeypatch.setattr(ingest_mod, "scan", wrapper)


def test_version():
    res = runner.invoke(cli.app, ["version"])
    assert res.exit_code == 0
    assert "faceorg" in res.stdout


def test_status_empty(tmp_path):
    res = runner.invoke(cli.app, ["--db", _db_path(tmp_path), "status"])
    assert res.exit_code == 0
    assert "People: 0" in res.stdout


def test_scan_without_src_errors(tmp_path):
    res = runner.invoke(cli.app, ["--db", _db_path(tmp_path), "scan"])
    assert res.exit_code != 0


def test_full_flow(tmp_path, fake_library, monkeypatch):
    root, files, fakes = fake_library
    _patched_scan(fakes, monkeypatch)
    db = _db_path(tmp_path)

    res = runner.invoke(cli.app, ["--db", db, "scan", "--src", str(root)])
    assert res.exit_code == 0, res.stdout
    assert "5 processed" in res.stdout
    assert "3 new people" in res.stdout

    res = runner.invoke(cli.app, ["--db", db, "list", "--sort", "id"])
    assert res.exit_code == 0
    assert "person_001" in res.stdout

    # rename person 1
    res = runner.invoke(cli.app, ["--db", db, "rename", "1", "Alice"])
    assert res.exit_code == 0
    res = runner.invoke(cli.app, ["--db", db, "list", "--named"])
    assert "Alice" in res.stdout

    # show
    res = runner.invoke(cli.app, ["--db", db, "show", "1"])
    assert res.exit_code == 0
    assert "Alice" in res.stdout


def test_duplicate_name_rejected(tmp_path, fake_library, monkeypatch):
    root, files, fakes = fake_library
    _patched_scan(fakes, monkeypatch)
    db = _db_path(tmp_path)
    runner.invoke(cli.app, ["--db", db, "scan", "--src", str(root)])
    runner.invoke(cli.app, ["--db", db, "rename", "1", "Alice"])
    res = runner.invoke(cli.app, ["--db", db, "rename", "2", "Alice"])
    assert res.exit_code != 0


def test_min_photos_filter(tmp_path, fake_library, monkeypatch):
    root, files, fakes = fake_library
    _patched_scan(fakes, monkeypatch)
    db = _db_path(tmp_path)
    runner.invoke(cli.app, ["--db", db, "scan", "--src", str(root)])
    # X appears in only 1 photo; with --min-photos 2 it should drop out
    res = runner.invoke(cli.app, ["--db", db, "list", "--min-photos", "2"])
    assert res.exit_code == 0
    # A (3) and B (2) remain; the single-photo person is filtered
    lines = [ln for ln in res.stdout.splitlines() if "person_" in ln]
    assert len(lines) == 2


def test_merge_reduces_person_count(tmp_path, fake_library, monkeypatch):
    root, files, fakes = fake_library
    _patched_scan(fakes, monkeypatch)
    db = _db_path(tmp_path)
    runner.invoke(cli.app, ["--db", db, "scan", "--src", str(root)])
    res = runner.invoke(cli.app, ["--db", db, "merge", "2", "1"])
    assert res.exit_code == 0
    res = runner.invoke(cli.app, ["--db", db, "status"])
    assert "People: 2" in res.stdout
