"""Tests for batch reclustering (agglomerative) and name carry-over."""

import numpy as np
import pytest

from faceorg.cluster import _assign_names, recluster
from faceorg.match import Matcher

# skip the sklearn-backed clustering test if sklearn isn't installed
sklearn = pytest.importorskip("sklearn")


def _seed_faces(db, clock, emb, layout):
    """Insert faces per `layout`: {image_rel: [identities]} using greedy first."""
    m = Matcher(db, tolerance=0.5, clock=clock)
    for rel, idents in layout.items():
        iid = db.insert_image(f"/p/{rel}", 1, 1.0, created_at=clock())
        for i, ident in enumerate(idents):
            m.assign(iid, emb(ident, i), (0, 1, 1, 0), "hog")


def test_recluster_separates_and_counts(db, clock, emb):
    # Two clearly distinct identities, several faces each.
    for i in range(4):
        iid = db.insert_image(f"/p/a{i}.jpg", 1, 1.0, created_at=clock())
        db.add_face(iid, None, emb("A", i), (0, 1, 1, 0), "hog", created_at=clock())
    for i in range(3):
        iid = db.insert_image(f"/p/b{i}.jpg", 1, 1.0, created_at=clock())
        db.add_face(iid, None, emb("B", i), (0, 1, 1, 0), "hog", created_at=clock())

    stats = recluster(db, threshold=0.55, linkage="average", clock=clock)
    assert stats.n_faces == 7
    assert stats.n_clusters == 2

    summaries = db.list_person_summaries()
    assert len(summaries) == 2
    # each cluster is a single identity
    per_person = db.iter_person_image_paths()
    assert sorted(len(v) for v in per_person.values()) == [3, 4]


def test_recluster_preserves_names(db, clock, emb):
    # seed A (named Alice) and B (named Bob) via greedy
    _seed_faces(db, clock, emb, {"a0.jpg": ["A"], "a1.jpg": ["A"], "b0.jpg": ["B"]})
    summaries = {s.sample_path: s for s in db.list_person_summaries()}
    # name them
    for s in db.list_person_summaries():
        paths = db.iter_person_image_paths()[s.person.id]
        label = "Alice" if "/p/a" in paths[0] else "Bob"
        db.rename_person(s.person.id, label, when=clock())

    stats = recluster(db, threshold=0.55, linkage="average", clock=clock)
    names = {s.person.name for s in db.list_person_summaries() if s.person.name}
    assert names == {"Alice", "Bob"}
    assert stats.names_preserved == 2


def test_recluster_empty_db(db, clock):
    stats = recluster(db, threshold=0.55, clock=clock)
    assert stats.n_faces == 0
    assert stats.n_clusters == 0


def test_assign_names_majority_vote_no_collision():
    # cluster 0 has 3 Alice + 1 Bob; cluster 1 has 2 Bob.
    face_to_cluster = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1}
    old_names = {1: "Alice", 2: "Alice", 3: "Alice", 4: "Bob", 5: "Bob", 6: "Bob"}
    result = _assign_names(face_to_cluster, old_names)
    # Alice -> cluster 0 (3 votes); Bob -> cluster 1 (2 votes) beats cluster 0 (1)
    assert result[0] == "Alice"
    assert result[1] == "Bob"


def test_assign_names_single_cluster_one_name_wins():
    # both names land in the same cluster -> only the majority name is kept
    face_to_cluster = {1: 0, 2: 0, 3: 0}
    old_names = {1: "Alice", 2: "Alice", 3: "Bob"}
    result = _assign_names(face_to_cluster, old_names)
    assert result[0] == "Alice"  # 2 beats 1
    assert list(result.values()).count("Bob") == 0
