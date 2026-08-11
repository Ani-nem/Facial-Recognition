"""Batch re-clustering of all stored face embeddings.

Greedy online assignment (match.py) is order-dependent and prone to
single-linkage "chaining" at scale — on a 4,324-image / 158-person test it
merged distinct people badly. A batch pass with **agglomerative
average-linkage** scored far better (98.5% purity, ARI 0.990) and is used here.

``recluster`` recomputes clusters over every face, then rebuilds the persons
table so that each cluster becomes one person. Names the user already assigned
are carried over by majority vote: whichever new cluster contains the most
faces that previously belonged to "Alice" inherits the name "Alice".
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Callable

import numpy as np

from faceorg.db import Database


@dataclass
class ReclusterStats:
    n_faces: int
    n_clusters: int
    names_preserved: int


def recluster(
    db: Database,
    threshold: float,
    linkage: str = "average",
    clock: Callable[[], float] = None,
) -> ReclusterStats:
    """Re-cluster all faces and rebuild the persons table.

    :param threshold: agglomerative distance threshold (Euclidean).
    :param linkage: "average" (recommended), "complete", or "single".
    """
    import time

    if clock is None:
        clock = time.time

    face_ids, X = db.load_all_faces()
    if len(face_ids) == 0:
        return ReclusterStats(n_faces=0, n_clusters=0, names_preserved=0)

    labels = _cluster(X, threshold, linkage)

    face_to_cluster = {fid: int(lbl) for fid, lbl in zip(face_ids, labels)}

    # Carry over names by majority vote from the pre-recluster assignment.
    old_names = db.get_face_person_names()  # face_id -> name|None
    cluster_names = _assign_names(face_to_cluster, old_names)

    db.rebuild_persons(face_to_cluster, cluster_names, when=clock())

    return ReclusterStats(
        n_faces=len(face_ids),
        n_clusters=len(set(labels)),
        names_preserved=sum(1 for n in cluster_names.values() if n),
    )


def _cluster(X: np.ndarray, threshold: float, linkage: str) -> np.ndarray:
    """Return an integer cluster label per row of X.

    A single sample is trivially its own cluster (sklearn requires >= 2).
    """
    if X.shape[0] == 1:
        return np.array([0])

    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        linkage=linkage,
        metric="euclidean",
    )
    return model.fit_predict(X)


def _assign_names(
    face_to_cluster: dict[int, int], old_names: dict[int, str | None]
) -> dict[int, str | None]:
    """Decide a name for each new cluster by majority vote of old names.

    A name is assigned to exactly one cluster — the cluster holding the most
    faces that used to carry that name. Ties break deterministically by lowest
    cluster label. Prevents two clusters claiming the same name.
    """
    # For each old name, count faces per new cluster.
    name_cluster_counts: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for fid, cluster in face_to_cluster.items():
        name = old_names.get(fid)
        if name:
            name_cluster_counts[name][cluster] += 1

    # Assign each name to its dominant cluster; resolve collisions by count.
    # cluster -> (best_name, best_count)
    chosen: dict[int, tuple[str, int]] = {}
    for name, counts in name_cluster_counts.items():
        # dominant cluster for this name (tie -> lowest label)
        best_cluster = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        count = counts[best_cluster]
        current = chosen.get(best_cluster)
        if current is None or count > current[1]:
            chosen[best_cluster] = (name, count)

    return {cluster: nm for cluster, (nm, _) in chosen.items()}
