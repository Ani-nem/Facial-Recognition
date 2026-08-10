"""Lightweight row dataclasses returned by the DB layer.

These are plain data carriers (no ORM). Embeddings are exposed as numpy
arrays here; serialization to/from SQLite blobs lives in ``db.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ImageRow:
    id: int
    path: str
    size: int
    mtime: float
    content_hash: str | None
    status: str  # pending | processed | error | no_face
    n_faces: int
    error: str | None
    processed_at: float | None
    created_at: float


@dataclass
class Person:
    id: int
    name: str | None
    is_ignored: bool
    created_at: float
    updated_at: float

    @property
    def display_name(self) -> str:
        """Human-facing label: the assigned name, or person_NN if unnamed."""
        return self.name if self.name else f"person_{self.id:03d}"


@dataclass
class FaceRow:
    id: int
    image_id: int
    person_id: int | None
    embedding: np.ndarray  # shape (128,), float64
    top: int
    right: int
    bottom: int
    left: int
    match_distance: float | None
    detector: str
    created_at: float


@dataclass
class PersonSummary:
    """A person plus aggregate counts, for `list`/`status` output."""

    person: Person
    n_faces: int
    n_photos: int
    sample_path: str | None
