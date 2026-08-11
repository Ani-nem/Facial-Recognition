"""Build the output ``by-person/`` tree of symlinks (or copies).

Non-destructive: originals are only ever read. The tree is a reconciled
projection of the DB — running ``apply`` repeatedly converges to the same
state (create missing links, fix wrong ones, prune stale ones), so it is safe
to re-run after renames, merges, or new scans.

Layout:  <dest>/by-person/<name-or-personNN>/<stem>__<hash>.<ext>

Each person's photos are linked into their folder. A photo containing two
people is linked once under each. Filenames get a short deterministic hash of
the source path appended, so files with the same basename from different
occasion folders never collide, and the same source always maps to the same
link name (idempotency).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from faceorg.db import Database

BY_PERSON = "by-person"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class ApplyStats:
    created: int = 0
    replaced: int = 0
    skipped: int = 0
    pruned: int = 0
    people: int = 0
    planned: list[str] = field(default_factory=list)  # dry-run descriptions


def sanitize_name(name: str) -> str:
    """Make a person name safe for use as a directory name."""
    cleaned = _UNSAFE.sub("_", name.strip()).strip("._")
    return cleaned or "unnamed"


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def link_name(src_path: str) -> str:
    """Deterministic, collision-resistant filename for a source path."""
    p = Path(src_path)
    stem = sanitize_name(p.stem)
    suffix = p.suffix.lower()
    return f"{stem}__{_short_hash(src_path)}{suffix}"


def _person_dirname(person, person_id: int) -> str:
    if person and person.name:
        return sanitize_name(person.name)
    return f"person_{person_id:03d}"


def apply(
    db: Database,
    dest: Path,
    *,
    mode: str = "symlink",
    min_photos: int = 1,
    prune: bool = True,
    dry_run: bool = False,
    src_root: Path | None = None,
) -> ApplyStats:
    """Reconcile the ``by-person/`` tree under ``dest`` from the DB.

    :param mode: "symlink" (default) or "copy".
    :param min_photos: only emit people with >= this many distinct photos
        (named people are always emitted regardless).
    :param prune: remove managed links no longer wanted (renames/merges).
    :param dry_run: compute and record actions without touching the filesystem.
    :param src_root: if set, pruning only removes symlinks whose target is
        inside this root (extra safety against touching unrelated files).
    """
    if mode not in ("symlink", "copy"):
        raise ValueError(f"mode must be 'symlink' or 'copy', got {mode!r}")

    dest = Path(dest).expanduser()
    root = dest / BY_PERSON
    stats = ApplyStats()

    person_paths = db.iter_person_image_paths()  # person_id -> [distinct paths]

    # Build the desired set: {abs_link_path: abs_src_path}
    desired: dict[Path, str] = {}
    for person_id, paths in person_paths.items():
        person = db.get_person(person_id)
        if person and person.is_ignored:
            continue
        # named people bypass the min_photos filter
        is_named = bool(person and person.name)
        if not is_named and len(paths) < min_photos:
            continue

        stats.people += 1
        person_dir = root / _person_dirname(person, person_id)
        for src in paths:
            link_path = person_dir / link_name(src)
            desired[link_path] = os.path.abspath(src)

    _reconcile(desired, mode, dry_run, stats)

    if prune:
        _prune(root, desired, mode, dry_run, stats, src_root)

    return stats


def _reconcile(
    desired: dict[Path, str], mode: str, dry_run: bool, stats: ApplyStats
) -> None:
    for link_path, src in desired.items():
        state = _link_state(link_path, src, mode)
        if state == "ok":
            stats.skipped += 1
            continue
        if state == "missing":
            stats.created += 1
            stats.planned.append(f"create {link_path} -> {src}")
        else:  # "wrong"
            stats.replaced += 1
            stats.planned.append(f"replace {link_path} -> {src}")

        if dry_run:
            continue

        link_path.parent.mkdir(parents=True, exist_ok=True)
        if state == "wrong" and (link_path.exists() or link_path.is_symlink()):
            _remove(link_path)
        if mode == "symlink":
            os.symlink(src, link_path)
        else:
            shutil.copy2(src, link_path)


def _link_state(link_path: Path, src: str, mode: str) -> str:
    """Return 'ok', 'missing', or 'wrong' for an intended link."""
    if mode == "symlink":
        if link_path.is_symlink():
            try:
                return "ok" if os.readlink(link_path) == src else "wrong"
            except OSError:
                return "wrong"
        return "wrong" if link_path.exists() else "missing"
    # copy mode: match by size (cheap, good enough for a projection)
    if link_path.exists():
        try:
            return "ok" if link_path.stat().st_size == Path(src).stat().st_size else "wrong"
        except OSError:
            return "wrong"
    return "missing"


def _prune(
    root: Path,
    desired: dict[Path, str],
    mode: str,
    dry_run: bool,
    stats: ApplyStats,
    src_root: Path | None,
) -> None:
    """Remove managed entries under ``root`` not in ``desired``; tidy empty dirs."""
    if not root.exists():
        return

    # Canonicalize so /tmp vs /private/tmp (macOS) style symlinks compare equal.
    src_root_abs = os.path.realpath(src_root) if src_root else None

    for entry in root.rglob("*"):
        if entry.is_dir():
            continue
        if entry in desired:
            continue
        # Safety: in symlink mode only prune symlinks that point into src_root.
        if mode == "symlink":
            if not entry.is_symlink():
                continue
            if src_root_abs is not None:
                try:
                    target = os.path.realpath(os.readlink(entry))
                except OSError:
                    continue
                if not target.startswith(src_root_abs + os.sep):
                    continue
        stats.pruned += 1
        stats.planned.append(f"prune {entry}")
        if not dry_run:
            _remove(entry)

    if not dry_run:
        _remove_empty_dirs(root)


def _remove(path: Path) -> None:
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    except FileNotFoundError:
        pass


def _remove_empty_dirs(root: Path) -> None:
    for d in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            try:
                d.rmdir()
            except OSError:
                pass
