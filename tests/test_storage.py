"""Tests for the apply (by-person tree) storage layer."""

import os
from pathlib import Path

import numpy as np

from faceorg.storage import apply, link_name, sanitize_name


def _make_photo(dirpath: Path, name: str) -> str:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    p.write_bytes(b"content-" + name.encode())
    return str(p.resolve())


def _add(db, clock, src_path, person_id, emb=None):
    iid = db.insert_image(src_path, 1, 1.0, created_at=clock())
    e = emb if emb is not None else np.random.rand(128)
    db.add_face(iid, person_id, e, (0, 1, 1, 0), "hog", created_at=clock())
    return iid


def test_sanitize_and_link_name():
    assert sanitize_name("Ada Lovelace") == "Ada_Lovelace"
    assert sanitize_name("../etc/passwd") == "etc_passwd"
    n1 = link_name("/a/IMG_1.jpg")
    n2 = link_name("/b/IMG_1.jpg")
    assert n1 != n2  # same basename, different path -> different link name
    assert link_name("/a/IMG_1.jpg") == n1  # deterministic


def test_symlinks_created_and_resolve(tmp_path, db, clock):
    src = _make_photo(tmp_path / "src", "a.jpg")
    pid = db.create_person(when=clock())
    _add(db, clock, src, pid)
    db.rename_person(pid, "Alice", when=clock())

    dest = tmp_path / "out"
    stats = apply(db, dest, mode="symlink")
    assert stats.created == 1
    links = list((dest / "by-person" / "Alice").iterdir())
    assert len(links) == 1
    link = links[0]
    assert link.is_symlink()
    assert os.path.realpath(link) == src
    assert link.exists()  # resolves


def test_multi_person_photo_links_into_each(tmp_path, db, clock):
    src = _make_photo(tmp_path / "src", "group.jpg")
    p1 = db.create_person(when=clock())
    p2 = db.create_person(when=clock())
    # same image, two faces, two people
    iid = db.insert_image(src, 1, 1.0, created_at=clock())
    db.add_face(iid, p1, np.random.rand(128), (0, 1, 1, 0), "hog", created_at=clock())
    db.add_face(iid, p2, np.random.rand(128), (0, 1, 1, 0), "hog", created_at=clock())

    dest = tmp_path / "out"
    apply(db, dest, mode="symlink")
    d1 = dest / "by-person" / f"person_{p1:03d}"
    d2 = dest / "by-person" / f"person_{p2:03d}"
    assert len(list(d1.iterdir())) == 1
    assert len(list(d2.iterdir())) == 1


def test_min_photos_filters_but_named_bypass(tmp_path, db, clock):
    # person A: 1 photo, unnamed -> filtered at min_photos=2
    a = db.create_person(when=clock())
    _add(db, clock, _make_photo(tmp_path / "src", "a.jpg"), a)
    # person B: 1 photo, named -> always emitted
    b = db.create_person(when=clock())
    _add(db, clock, _make_photo(tmp_path / "src", "b.jpg"), b)
    db.rename_person(b, "Bob", when=clock())

    dest = tmp_path / "out"
    stats = apply(db, dest, mode="symlink", min_photos=2)
    assert stats.people == 1  # only Bob
    assert (dest / "by-person" / "Bob").exists()
    assert not (dest / "by-person" / f"person_{a:03d}").exists()


def test_ignored_person_skipped(tmp_path, db, clock):
    a = db.create_person(when=clock())
    _add(db, clock, _make_photo(tmp_path / "src", "a.jpg"), a)
    db.set_ignored(a, True, when=clock())
    dest = tmp_path / "out"
    stats = apply(db, dest, mode="symlink")
    assert stats.people == 0


def test_idempotent_reapply(tmp_path, db, clock):
    src = _make_photo(tmp_path / "src", "a.jpg")
    pid = db.create_person(when=clock())
    _add(db, clock, src, pid)
    dest = tmp_path / "out"
    apply(db, dest, mode="symlink")
    stats2 = apply(db, dest, mode="symlink")
    assert stats2.created == 0
    assert stats2.replaced == 0
    assert stats2.skipped == 1


def test_rename_prunes_old_dir(tmp_path, db, clock):
    src = _make_photo(tmp_path / "src", "a.jpg")
    pid = db.create_person(when=clock())
    _add(db, clock, src, pid)
    dest = tmp_path / "out"
    apply(db, dest, mode="symlink", src_root=tmp_path / "src")
    old_dir = dest / "by-person" / f"person_{pid:03d}"
    assert old_dir.exists()

    db.rename_person(pid, "Alice", when=clock())
    apply(db, dest, mode="symlink", src_root=tmp_path / "src")
    assert (dest / "by-person" / "Alice").exists()
    assert not old_dir.exists()  # pruned


def test_dry_run_changes_nothing(tmp_path, db, clock):
    src = _make_photo(tmp_path / "src", "a.jpg")
    pid = db.create_person(when=clock())
    _add(db, clock, src, pid)
    dest = tmp_path / "out"
    stats = apply(db, dest, mode="symlink", dry_run=True)
    assert stats.created == 1
    assert not (dest / "by-person").exists()  # nothing written
    assert stats.planned  # actions recorded


def test_copy_mode_copies_bytes(tmp_path, db, clock):
    src = _make_photo(tmp_path / "src", "a.jpg")
    pid = db.create_person(when=clock())
    _add(db, clock, src, pid)
    dest = tmp_path / "out"
    apply(db, dest, mode="copy")
    copied = list((dest / "by-person" / f"person_{pid:03d}").iterdir())[0]
    assert not copied.is_symlink()
    assert copied.read_bytes() == Path(src).read_bytes()


def test_prune_works_with_unresolved_src_root(tmp_path, db, clock):
    """src_root passed unresolved (e.g. /tmp vs /private/tmp) must still prune.

    Links store realpath targets; the prune safety check must canonicalize both
    sides so a rename actually removes the old person's directory.
    """
    srcdir = tmp_path / "src"
    src = _make_photo(srcdir, "a.jpg")
    pid = db.create_person(when=clock())
    _add(db, clock, src, pid)
    dest = tmp_path / "out"

    # Build a sibling symlink to srcdir with a different (unresolved) path.
    alt_root = tmp_path / "src_alias"
    os.symlink(srcdir, alt_root)

    apply(db, dest, mode="symlink", src_root=alt_root)
    old_dir = dest / "by-person" / f"person_{pid:03d}"
    assert old_dir.exists()

    db.rename_person(pid, "Alice", when=clock())
    apply(db, dest, mode="symlink", src_root=alt_root)
    assert (dest / "by-person" / "Alice").exists()
    assert not old_dir.exists()  # pruned despite src_root path mismatch


def test_prune_leaves_foreign_files(tmp_path, db, clock):
    """Pruning must not delete unrelated files a user placed in by-person/."""
    src = _make_photo(tmp_path / "src", "a.jpg")
    pid = db.create_person(when=clock())
    _add(db, clock, src, pid)
    dest = tmp_path / "out"
    apply(db, dest, mode="symlink", src_root=tmp_path / "src")

    # user drops a real file in the tree
    foreign = dest / "by-person" / "notes.txt"
    foreign.write_text("keep me")
    apply(db, dest, mode="symlink", src_root=tmp_path / "src")
    assert foreign.exists()  # not a symlink into src_root -> not pruned
