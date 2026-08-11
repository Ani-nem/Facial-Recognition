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


def downscale(rgb: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is <= max_dimension.

    Returns (possibly-smaller image, scale) where ``scale`` is
    downscaled/original (<= 1.0). Detection runs on the smaller image; multiply
    resulting bbox coords by 1/scale to map back to the original. A
    max_dimension of 0 (or an image already small enough) is a no-op (scale 1).
    """
    if max_dimension <= 0:
        return rgb, 1.0
    import cv2

    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_dimension:
        return rgb, 1.0
    scale = max_dimension / longest
    resized = cv2.resize(
        rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA
    )
    return resized, scale


def rescale_bboxes(locations: list[BBox], scale: float) -> list[BBox]:
    """Map bboxes detected on a downscaled image back to original coords."""
    if scale == 1.0:
        return locations
    inv = 1.0 / scale
    return [
        (round(t * inv), round(r * inv), round(b * inv), round(l * inv))
        for (t, r, b, l) in locations
    ]


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
