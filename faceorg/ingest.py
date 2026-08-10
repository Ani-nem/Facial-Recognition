"""Scan orchestration: walk a source tree, incrementally process images.

For each image: skip if already processed and unchanged (by size+mtime, or
content hash), otherwise detect faces, embed them, and greedily assign each to
a person via the Matcher. This is the "pick up where it left off" behavior.

Detection/embedding are injected (``detector_fn``/``encoder_fn``) so tests can
supply deterministic fakes without dlib. Defaults wire the real functions.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from faceorg.config import Config
from faceorg.db import Database
from faceorg.detect import BBox, detect_faces
from faceorg.embed import encode_faces, load_rgb
from faceorg.match import Matcher

# Injectable seams -----------------------------------------------------------
# loader:   path -> rgb ndarray
# detector: (rgb, model, upsample) -> list[BBox]
# encoder:  (rgb, locations, jitters) -> list[ndarray]
LoaderFn = Callable[[str], np.ndarray]
DetectorFn = Callable[[np.ndarray, str, int], list[BBox]]
EncoderFn = Callable[[np.ndarray, list[BBox], int], list[np.ndarray]]


@dataclass
class ScanStats:
    images_seen: int = 0
    images_processed: int = 0
    images_skipped: int = 0
    images_error: int = 0
    faces_found: int = 0
    new_persons: int = 0
    errors: list[str] = field(default_factory=list)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_image_paths(src: Path, extensions: tuple[str, ...]):
    """Yield absolute paths of image files under ``src`` (recursive, sorted)."""
    exts = {e.lower() for e in extensions}
    for path in sorted(src.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            yield path.resolve()


def scan(
    config: Config,
    db: Database,
    *,
    force: bool = False,
    rehash: bool = False,
    limit: int | None = None,
    clock: Callable[[], float] = time.time,
    loader_fn: LoaderFn = load_rgb,
    detector_fn: DetectorFn = detect_faces,
    encoder_fn: EncoderFn = encode_faces,
    progress: Callable[[str], None] | None = None,
) -> ScanStats:
    """Incrementally scan ``config.src`` into ``db``.

    :param force: reprocess every image regardless of prior status.
    :param rehash: verify content hash even when size+mtime are unchanged.
    :param limit: process at most this many *pending* images (for testing).
    """
    if config.src is None:
        raise ValueError("scan requires a source path (config.src)")
    src = Path(config.src).expanduser()
    if not src.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {src}")

    stats = ScanStats()
    matcher = Matcher(db, tolerance=config.tolerance, clock=clock)

    processed_count = 0
    for path in iter_image_paths(src, config.extensions):
        stats.images_seen += 1
        spath = str(path)
        st = path.stat()
        existing = db.get_image_by_path(spath)

        image_id, should_process = _reconcile_image(
            db, existing, spath, st.st_size, st.st_mtime, force, rehash, clock
        )
        if not should_process:
            stats.images_skipped += 1
            continue

        if limit is not None and processed_count >= limit:
            # leave remaining as pending; they'll be handled next run
            break
        processed_count += 1

        try:
            rgb = loader_fn(spath)
            locations = detector_fn(rgb, config.model, config.upsample)
            encodings = encoder_fn(rgb, locations, config.jitters)
        except Exception as exc:  # noqa: BLE001 - record and continue scanning
            db.mark_image_error(image_id, str(exc), clock())
            stats.images_error += 1
            stats.errors.append(f"{spath}: {exc}")
            if progress:
                progress(f"error {spath}: {exc}")
            continue

        n_faces = 0
        for bbox, enc in zip(locations, encodings):
            _pid, created_new, _dist = matcher.assign(
                image_id, np.asarray(enc, dtype=np.float64), bbox, config.model
            )
            n_faces += 1
            stats.faces_found += 1
            if created_new:
                stats.new_persons += 1

        content_hash = existing.content_hash if existing else None
        if content_hash is None:
            content_hash = _sha256(spath)
        db.mark_image_processed(image_id, n_faces, clock(), content_hash=content_hash)
        stats.images_processed += 1
        if progress:
            progress(f"processed {spath} ({n_faces} faces)")

    return stats


def _reconcile_image(
    db: Database,
    existing,
    path: str,
    size: int,
    mtime: float,
    force: bool,
    rehash: bool,
    clock: Callable[[], float],
) -> tuple[int, bool]:
    """Decide whether an image needs processing; return (image_id, should_process).

    New file            -> insert pending, process.
    Unchanged+processed -> skip (unless force/rehash).
    Changed metadata    -> hash; reprocess only if content differs.
    """
    now = clock()
    if existing is None:
        image_id = db.insert_image(path, size, mtime, created_at=now)
        return image_id, True

    if force:
        db.requeue_image(existing.id, size, mtime, None)
        return existing.id, True

    unchanged_meta = existing.size == size and existing.mtime == mtime
    already_done = existing.status in ("processed", "no_face")

    if unchanged_meta and already_done and not rehash:
        return existing.id, False

    if unchanged_meta and already_done and rehash:
        # metadata says same, but verify content
        new_hash = _sha256(path)
        if existing.content_hash == new_hash:
            return existing.id, False
        db.requeue_image(existing.id, size, mtime, new_hash)
        return existing.id, True

    if not unchanged_meta and already_done:
        # size/mtime changed: was it a real edit or just a touch?
        new_hash = _sha256(path)
        if existing.content_hash == new_hash:
            db.update_image_stat(existing.id, size, mtime)  # touch only
            return existing.id, False
        db.requeue_image(existing.id, size, mtime, new_hash)
        return existing.id, True

    # pending or error -> (re)process
    return existing.id, True
