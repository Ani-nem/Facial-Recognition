"""Tests for the greedy nearest-neighbor matcher."""

import numpy as np

from faceorg.embed import face_distance
from faceorg.match import Matcher


def test_face_distance_euclidean():
    known = np.array([[0.0] * 128, [1.0] * 128])
    query = np.array([0.0] * 128)
    d = face_distance(known, query)
    assert d.shape == (2,)
    assert d[0] == 0.0
    assert np.isclose(d[1], np.sqrt(128))


def test_face_distance_empty():
    assert face_distance(np.empty((0, 128)), np.zeros(128)).shape == (0,)


def test_first_face_creates_person(db, clock, emb):
    iid = db.insert_image("/p/a.jpg", 1, 1.0, created_at=clock())
    m = Matcher(db, tolerance=0.6, clock=clock)
    pid, created, dist = m.assign(iid, emb("A"), (0, 1, 1, 0), "hog")
    assert created is True
    assert dist is None
    assert db.get_person(pid) is not None


def test_same_identity_joins_person(db, clock, emb):
    iid = db.insert_image("/p/a.jpg", 1, 1.0, created_at=clock())
    m = Matcher(db, tolerance=0.6, clock=clock)
    pid1, c1, _ = m.assign(iid, emb("A", 0), (0, 1, 1, 0), "hog")
    pid2, c2, dist = m.assign(iid, emb("A", 1), (0, 1, 1, 0), "hog")
    assert c1 is True and c2 is False
    assert pid1 == pid2
    assert dist is not None and dist <= 0.6


def test_different_identity_creates_new_person(db, clock, emb):
    iid = db.insert_image("/p/a.jpg", 1, 1.0, created_at=clock())
    m = Matcher(db, tolerance=0.6, clock=clock)
    pidA, _, _ = m.assign(iid, emb("A"), (0, 1, 1, 0), "hog")
    pidB, created, _ = m.assign(iid, emb("B"), (0, 1, 1, 0), "hog")
    assert created is True
    assert pidA != pidB


def test_matcher_reloads_existing_state(db, clock, emb):
    """A fresh Matcher should match against people already in the DB."""
    iid = db.insert_image("/p/a.jpg", 1, 1.0, created_at=clock())
    m1 = Matcher(db, tolerance=0.6, clock=clock)
    pid, _, _ = m1.assign(iid, emb("A", 0), (0, 1, 1, 0), "hog")

    # new Matcher instance (simulates a later scan run)
    m2 = Matcher(db, tolerance=0.6, clock=clock)
    pid2, created, _ = m2.assign(iid, emb("A", 2), (0, 1, 1, 0), "hog")
    assert created is False
    assert pid2 == pid


def test_strict_tolerance_splits(db, clock, emb):
    """A tiny tolerance forces even the same identity into separate people."""
    iid = db.insert_image("/p/a.jpg", 1, 1.0, created_at=clock())
    m = Matcher(db, tolerance=0.001, clock=clock)
    pid1, _, _ = m.assign(iid, emb("A", 0), (0, 1, 1, 0), "hog")
    pid2, created, _ = m.assign(iid, emb("A", 5), (0, 1, 1, 0), "hog")
    assert created is True
    assert pid1 != pid2
