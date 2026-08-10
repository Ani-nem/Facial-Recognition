"""Face detection — locate faces in an RGB image.

Wraps ``face_recognition.face_locations``. The heavy ``face_recognition``
import is deferred to call time so that modules importing this one (and tests
using a mocked encoder) don't require dlib to be installed.
"""

from __future__ import annotations

import numpy as np

# bbox is (top, right, bottom, left) — face_recognition's native ordering.
BBox = tuple[int, int, int, int]


def detect_faces(
    rgb: np.ndarray, model: str = "hog", upsample: int = 1
) -> list[BBox]:
    """Return bounding boxes for every face found in ``rgb``.

    :param rgb: RGB image array (H, W, 3).
    :param model: "hog" (fast, CPU) or "cnn" (accurate, GPU-preferred).
    :param upsample: number_of_times_to_upsample; higher finds smaller faces.
    """
    import face_recognition

    return face_recognition.face_locations(
        rgb, number_of_times_to_upsample=upsample, model=model
    )
