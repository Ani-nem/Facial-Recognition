"""Greedy online nearest-neighbor face-to-person assignment.

For each new face embedding, find the nearest already-known embedding (across
all people). If it's within ``tolerance`` (Euclidean), the face joins that
person; otherwise a new person is created. This is stable with respect to
existing person IDs/names — new photos attach to already-named people without
any global re-clustering — which is what the incremental/"self-learning"
workflow needs.

``Matcher`` keeps the (embeddings, person_ids) pool in memory for the duration
of a scan and appends to it as new faces are assigned, so we don't reload from
SQLite on every face.
"""

from __future__ import annotations

import numpy as np

from faceorg.db import Database
from faceorg.embed import face_distance


class Matcher:
    """In-memory greedy matcher backed by a Database for persistence."""

    def __init__(self, db: Database, tolerance: float, clock):
        """:param clock: zero-arg callable returning a timestamp (time.time)."""
        self.db = db
        self.tolerance = tolerance
        self._clock = clock
        matrix, person_ids = db.load_all_embeddings()
        # keep a growable list of rows for cheap appends; stack lazily
        self._embeddings: list[np.ndarray] = [row for row in matrix]
        self._person_ids: list[int] = list(person_ids)

    def assign(
        self,
        image_id: int,
        embedding: np.ndarray,
        bbox: tuple[int, int, int, int],
        detector: str,
    ) -> tuple[int, bool, float | None]:
        """Assign one face to a person, persisting the face row.

        :return: (person_id, created_new_person, match_distance)
        """
        person_id, distance = self._nearest(embedding)
        created_new = person_id is None

        now = self._clock()
        if created_new:
            person_id = self.db.create_person(when=now)

        self.db.add_face(
            image_id=image_id,
            person_id=person_id,
            embedding=embedding,
            bbox=bbox,
            detector=detector,
            created_at=now,
            match_distance=distance,
        )

        # extend the in-memory pool so subsequent faces can match this one
        self._embeddings.append(np.asarray(embedding, dtype=np.float64))
        self._person_ids.append(person_id)

        return person_id, created_new, distance

    def _nearest(self, embedding: np.ndarray) -> tuple[int | None, float | None]:
        """Nearest existing person within tolerance, else (None, None)."""
        if not self._embeddings:
            return None, None
        known = np.vstack(self._embeddings)
        dists = face_distance(known, np.asarray(embedding, dtype=np.float64))
        j = int(np.argmin(dists))
        best = float(dists[j])
        if best <= self.tolerance:
            return self._person_ids[j], best
        return None, best
