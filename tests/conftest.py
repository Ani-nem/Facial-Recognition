"""Shared pytest fixtures.

Tests use a deterministic fake encoder (identity -> distinct 128-d vector) so
they run fast and require neither dlib nor real face images.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from faceorg.db import Database


@pytest.fixture
def db():
    """An in-memory Database with schema initialized."""
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def clock():
    """A monotonically increasing fake clock (avoids real time in tests)."""
    state = {"t": 1000.0}

    def _tick() -> float:
        state["t"] += 1.0
        return state["t"]

    return _tick


# Distinct, well-separated base vectors per identity.
_BASES = {
    "A": np.array([1.0] + [0.0] * 127),
    "B": np.array([0.0, 1.0] + [0.0] * 126),
    "C": np.array([0.0, 0.0, 1.0] + [0.0] * 125),
    "X": np.array([0.0, 0.0, 0.0, 1.0] + [0.0] * 124),
}


def embedding_for(identity: str, k: int = 0) -> np.ndarray:
    """Return a stable embedding for an identity, with tiny per-face jitter."""
    v = _BASES[identity].copy()
    v[10 + k] += 0.01  # well within default tolerance 0.6
    return v


@pytest.fixture
def emb():
    """Fixture exposing the deterministic embedding_for helper."""
    return embedding_for


@pytest.fixture
def fake_library(tmp_path):
    """Create a small photo library and return (root, files, fakes).

    ``files`` maps relative path -> list of identity strings present.
    ``fakes`` is a dict of loader/detector/encoder functions to inject into
    ingest.scan.
    """
    root = Path(tmp_path).resolve()
    (root / "beach").mkdir()
    (root / "party").mkdir()
    files = {
        "beach/a1.jpg": ["A"],
        "beach/a2.jpg": ["A"],
        "beach/ab.jpg": ["A", "B"],
        "party/b1.jpg": ["B"],
        "party/x1.jpg": ["X"],
    }
    for rel, _ids in files.items():
        (root / rel).write_bytes(b"fake-image-" + rel.encode())

    current = {"path": None}

    def loader(path):
        current["path"] = path
        return np.zeros((4, 4, 3))

    def detector(rgb, model, upsample):
        rel = os.path.relpath(current["path"], root)
        return [(0, 1, 1, 0)] * len(files[rel])

    def encoder(rgb, locations, jitters):
        rel = os.path.relpath(current["path"], root)
        return [embedding_for(ident, i) for i, ident in enumerate(files[rel])]

    fakes = {"loader_fn": loader, "detector_fn": detector, "encoder_fn": encoder}
    return root, files, fakes
