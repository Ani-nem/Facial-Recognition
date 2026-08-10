"""Image loading and face embedding.

Loads an image as RGB (via OpenCV, mirroring the original project's proven
BGR->RGB path) and computes 128-d encodings with ``face_recognition``. Heavy
imports are deferred to call time so tests can mock this module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from faceorg.detect import BBox


class ImageReadError(Exception):
    """Raised when an image file cannot be decoded."""


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image file as an RGB numpy array.

    Uses cv2.imread then converts BGR->RGB (face_recognition expects RGB).
    """
    import cv2

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ImageReadError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_faces(
    rgb: np.ndarray, locations: list[BBox], jitters: int = 1
) -> list[np.ndarray]:
    """Compute 128-d encodings for the given face locations in ``rgb``.

    Passing ``known_face_locations`` avoids a second detection pass inside
    face_recognition. Returns one encoding per location (order preserved).
    """
    import face_recognition

    return face_recognition.face_encodings(
        rgb, known_face_locations=locations, num_jitters=jitters
    )


def face_distance(known: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Euclidean distance from ``query`` to each row of ``known`` (N,128).

    Thin wrapper matching face_recognition.face_distance semantics, but
    implemented with numpy so matching works without the heavy import.
    Returns an empty array when ``known`` is empty.
    """
    if known.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    return np.linalg.norm(known - query, axis=1)
