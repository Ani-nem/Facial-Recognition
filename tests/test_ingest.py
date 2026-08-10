"""Tests for incremental scan orchestration."""

import os

from faceorg.config import Config
from faceorg import ingest


def _scan(root, db, fakes, clock, **kw):
    cfg = Config.from_sources({"src": str(root), "tolerance": 0.6})
    return ingest.scan(cfg, db, clock=clock, **fakes, **kw)


def test_scan_clusters_identities(db, clock, fake_library):
    root, files, fakes = fake_library
    stats = _scan(root, db, fakes, clock)
    assert stats.images_seen == 5
    assert stats.images_processed == 5
    assert stats.faces_found == 6  # ab.jpg has two faces
    # A(3 photos), B(2), X(1)
    counts = sorted((s.n_faces, s.n_photos) for s in db.list_person_summaries())
    assert counts == [(1, 1), (2, 2), (3, 3)]


def test_two_person_photo_links_two_people(db, clock, fake_library):
    root, files, fakes = fake_library
    _scan(root, db, fakes, clock)
    # the image with two identities should map to two distinct persons
    per_person = db.iter_person_image_paths()
    ab = str((root / "beach/ab.jpg"))
    owners = [pid for pid, paths in per_person.items() if ab in paths]
    assert len(owners) == 2


def test_incremental_skip(db, clock, fake_library):
    root, files, fakes = fake_library
    _scan(root, db, fakes, clock)
    stats2 = _scan(root, db, fakes, clock)
    assert stats2.images_processed == 0
    assert stats2.images_skipped == 5


def test_touch_is_skipped(db, clock, fake_library):
    root, files, fakes = fake_library
    _scan(root, db, fakes, clock)
    os.utime(root / "beach/a1.jpg", (9999.0, 9999.0))  # change mtime only
    stats = _scan(root, db, fakes, clock)
    assert stats.images_processed == 0


def test_edit_is_reprocessed(db, clock, fake_library):
    root, files, fakes = fake_library
    _scan(root, db, fakes, clock)
    (root / "beach/a1.jpg").write_bytes(b"totally-different-and-longer-content")
    stats = _scan(root, db, fakes, clock)
    assert stats.images_processed == 1


def test_force_reprocesses_all(db, clock, fake_library):
    root, files, fakes = fake_library
    _scan(root, db, fakes, clock)
    stats = _scan(root, db, fakes, clock, force=True)
    assert stats.images_processed == 5
    # identities unchanged -> still 3 people
    assert len(db.list_person_summaries()) == 3


def test_new_photo_auto_matches(db, clock, fake_library):
    root, files, fakes = fake_library
    _scan(root, db, fakes, clock)
    # add a new photo of an existing identity
    (root / "party/a3.jpg").write_bytes(b"new-photo")
    files["party/a3.jpg"] = ["A"]
    stats = _scan(root, db, fakes, clock)
    assert stats.images_processed == 1
    assert stats.new_persons == 0
    assert len(db.list_person_summaries()) == 3


def test_limit_bounds_processing(db, clock, fake_library):
    root, files, fakes = fake_library
    stats = _scan(root, db, fakes, clock, limit=2)
    assert stats.images_processed == 2
    # the rest remain pending for a later run
    remaining = _scan(root, db, fakes, clock)
    assert remaining.images_processed == 3
