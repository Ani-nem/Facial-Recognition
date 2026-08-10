"""Tests for the SQLite persistence layer."""

import numpy as np

from faceorg.db import blob_to_enc, enc_to_blob


def test_blob_roundtrip():
    v = np.random.rand(128)
    out = blob_to_enc(enc_to_blob(v))
    assert out.shape == (128,)
    assert out.dtype == np.float64
    assert np.array_equal(out, v)


def test_image_insert_and_lookup(db, clock):
    iid = db.insert_image("/p/a.jpg", 100, 1.0, created_at=clock())
    img = db.get_image_by_path("/p/a.jpg")
    assert img.id == iid
    assert img.status == "pending"
    assert db.get_image_by_path("/nope.jpg") is None


def test_image_status_transitions(db, clock):
    iid = db.insert_image("/p/a.jpg", 100, 1.0, created_at=clock())
    db.mark_image_processed(iid, n_faces=2, processed_at=clock(), content_hash="h")
    assert db.get_image_by_path("/p/a.jpg").status == "processed"

    iid2 = db.insert_image("/p/b.jpg", 100, 1.0, created_at=clock())
    db.mark_image_processed(iid2, n_faces=0, processed_at=clock())
    assert db.get_image_by_path("/p/b.jpg").status == "no_face"

    iid3 = db.insert_image("/p/c.jpg", 100, 1.0, created_at=clock())
    db.mark_image_error(iid3, "boom", clock())
    assert db.get_image_by_path("/p/c.jpg").status == "error"


def test_face_add_and_load(db, clock):
    iid = db.insert_image("/p/a.jpg", 100, 1.0, created_at=clock())
    pid = db.create_person(when=clock())
    e = np.random.rand(128)
    db.add_face(iid, pid, e, (10, 20, 30, 5), "hog", created_at=clock(), match_distance=0.3)

    faces = db.get_faces_for_person(pid)
    assert len(faces) == 1
    assert faces[0].detector == "hog"
    assert np.allclose(faces[0].embedding, e)

    matrix, pids = db.load_all_embeddings()
    assert matrix.shape == (1, 128)
    assert pids == [pid]


def test_load_all_embeddings_excludes_unassigned(db, clock):
    matrix, pids = db.load_all_embeddings()
    assert matrix.shape == (0, 128)
    assert pids == []


def test_rename_and_unique_name(db, clock):
    pid = db.create_person(when=clock())
    db.rename_person(pid, "Alice", when=clock())
    assert db.get_person_by_name("Alice").id == pid
    assert db.get_person(pid).display_name == "Alice"


def test_merge_persons(db, clock):
    iid = db.insert_image("/p/a.jpg", 100, 1.0, created_at=clock())
    p1 = db.create_person(when=clock())
    p2 = db.create_person(when=clock())
    db.add_face(iid, p1, np.random.rand(128), (1, 2, 3, 4), "hog", created_at=clock())
    db.add_face(iid, p2, np.random.rand(128), (1, 2, 3, 4), "hog", created_at=clock())

    db.merge_persons(p2, p1)
    assert db.get_person(p2) is None
    assert len(db.get_faces_for_person(p1)) == 2


def test_requeue_cascades_faces(db, clock):
    iid = db.insert_image("/p/a.jpg", 100, 1.0, created_at=clock())
    iid2 = db.insert_image("/p/b.jpg", 100, 1.0, created_at=clock())
    pid = db.create_person(when=clock())
    db.add_face(iid, pid, np.random.rand(128), (1, 2, 3, 4), "hog", created_at=clock())
    db.add_face(iid2, pid, np.random.rand(128), (1, 2, 3, 4), "hog", created_at=clock())

    db.requeue_image(iid, 100, 5.0, "newhash")
    assert db.get_image_by_path("/p/a.jpg").status == "pending"
    remaining = db.get_faces_for_person(pid)
    assert len(remaining) == 1
    assert remaining[0].image_id == iid2


def test_ignore_flag(db, clock):
    pid = db.create_person(when=clock())
    db.set_ignored(pid, True, when=clock())
    assert db.get_person(pid).is_ignored is True
    db.set_ignored(pid, False, when=clock())
    assert db.get_person(pid).is_ignored is False


def test_person_summaries(db, clock):
    iid = db.insert_image("/p/a.jpg", 100, 1.0, created_at=clock())
    pid = db.create_person(when=clock())
    db.add_face(iid, pid, np.random.rand(128), (1, 2, 3, 4), "hog", created_at=clock())
    db.rename_person(pid, "Alice", when=clock())

    summaries = db.list_person_summaries()
    assert len(summaries) == 1
    s = summaries[0]
    assert s.person.name == "Alice"
    assert s.n_faces == 1
    assert s.n_photos == 1
    assert s.sample_path == "/p/a.jpg"
