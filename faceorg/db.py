"""SQLite persistence layer — the persistent "brain".

Stores one row per image (for incremental re-scan), one row per detected face
(with its 128-d embedding), and one row per person (an auto-discovered cluster
the user can name). Face embeddings are stored as fixed-dtype float64 blobs.

All access goes through the ``Database`` class. Timestamps are supplied by the
caller (``time.time()``) so this module has no hidden clock dependency.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from faceorg.models import FaceRow, ImageRow, Person, PersonSummary

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS images (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    path          TEXT NOT NULL UNIQUE,
    size          INTEGER NOT NULL,
    mtime         REAL    NOT NULL,
    content_hash  TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',
    n_faces       INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    processed_at  REAL,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);

CREATE TABLE IF NOT EXISTS persons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    is_ignored  INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_persons_name
    ON persons(name) WHERE name IS NOT NULL;

CREATE TABLE IF NOT EXISTS faces (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id       INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    person_id      INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    embedding      BLOB NOT NULL,
    top            INTEGER NOT NULL,
    right          INTEGER NOT NULL,
    bottom         INTEGER NOT NULL,
    left           INTEGER NOT NULL,
    match_distance REAL,
    detector       TEXT NOT NULL,
    created_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
CREATE INDEX IF NOT EXISTS idx_faces_image  ON faces(image_id);
"""


# --------------------------------------------------------------------------- #
# Embedding serialization
# --------------------------------------------------------------------------- #
def enc_to_blob(v: np.ndarray) -> bytes:
    """Serialize a 128-d encoding to bytes (fixed float64 dtype)."""
    return np.ascontiguousarray(v, dtype=np.float64).tobytes()


def blob_to_enc(b: bytes) -> np.ndarray:
    """Deserialize a blob back into a (128,) float64 array."""
    return np.frombuffer(b, dtype=np.float64)


class Database:
    """Thin data-access wrapper over a SQLite connection."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self.path = str(Path(self.path).expanduser())
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(_SCHEMA)
        cur = self.conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        )
        row = cur.fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Images
    # ------------------------------------------------------------------ #
    def get_image_by_path(self, path: str) -> ImageRow | None:
        cur = self.conn.execute("SELECT * FROM images WHERE path = ?", (path,))
        row = cur.fetchone()
        return _to_image(row) if row else None

    def insert_image(
        self, path: str, size: int, mtime: float, created_at: float
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO images (path, size, mtime, status, created_at)
               VALUES (?, ?, ?, 'pending', ?)""",
            (path, size, mtime, created_at),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_image_stat(self, image_id: int, size: int, mtime: float) -> None:
        self.conn.execute(
            "UPDATE images SET size = ?, mtime = ? WHERE id = ?",
            (size, mtime, image_id),
        )
        self.conn.commit()

    def requeue_image(
        self, image_id: int, size: int, mtime: float, content_hash: str | None
    ) -> None:
        """Reset an image to 'pending' and delete its old faces (content changed)."""
        self.conn.execute("DELETE FROM faces WHERE image_id = ?", (image_id,))
        self.conn.execute(
            """UPDATE images
               SET size = ?, mtime = ?, content_hash = ?, status = 'pending',
                   n_faces = 0, error = NULL, processed_at = NULL
               WHERE id = ?""",
            (size, mtime, content_hash, image_id),
        )
        self.conn.commit()

    def mark_image_processed(
        self,
        image_id: int,
        n_faces: int,
        processed_at: float,
        content_hash: str | None = None,
    ) -> None:
        status = "processed" if n_faces > 0 else "no_face"
        self.conn.execute(
            """UPDATE images
               SET status = ?, n_faces = ?, processed_at = ?,
                   content_hash = COALESCE(?, content_hash), error = NULL
               WHERE id = ?""",
            (status, n_faces, processed_at, content_hash, image_id),
        )
        self.conn.commit()

    def mark_image_error(self, image_id: int, error: str, when: float) -> None:
        self.conn.execute(
            "UPDATE images SET status = 'error', error = ?, processed_at = ? WHERE id = ?",
            (error, when, image_id),
        )
        self.conn.commit()

    def count_images_by_status(self) -> dict[str, int]:
        cur = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM images GROUP BY status"
        )
        return {row["status"]: row["n"] for row in cur.fetchall()}

    # ------------------------------------------------------------------ #
    # Persons
    # ------------------------------------------------------------------ #
    def create_person(self, when: float, name: str | None = None) -> int:
        cur = self.conn.execute(
            """INSERT INTO persons (name, is_ignored, created_at, updated_at)
               VALUES (?, 0, ?, ?)""",
            (name, when, when),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_person(self, person_id: int) -> Person | None:
        cur = self.conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,))
        row = cur.fetchone()
        return _to_person(row) if row else None

    def get_person_by_name(self, name: str) -> Person | None:
        cur = self.conn.execute("SELECT * FROM persons WHERE name = ?", (name,))
        row = cur.fetchone()
        return _to_person(row) if row else None

    def rename_person(self, person_id: int, name: str | None, when: float) -> None:
        self.conn.execute(
            "UPDATE persons SET name = ?, updated_at = ? WHERE id = ?",
            (name, when, person_id),
        )
        self.conn.commit()

    def set_ignored(self, person_id: int, ignored: bool, when: float) -> None:
        self.conn.execute(
            "UPDATE persons SET is_ignored = ?, updated_at = ? WHERE id = ?",
            (1 if ignored else 0, when, person_id),
        )
        self.conn.commit()

    def merge_persons(self, src_id: int, dst_id: int) -> None:
        """Repoint all of src's faces to dst, then delete src."""
        self.conn.execute(
            "UPDATE faces SET person_id = ? WHERE person_id = ?", (dst_id, src_id)
        )
        self.conn.execute("DELETE FROM persons WHERE id = ?", (src_id,))
        self.conn.commit()

    def list_person_summaries(self) -> list[PersonSummary]:
        """All persons with face/photo counts and a sample image path."""
        cur = self.conn.execute(
            """
            SELECT p.id, p.name, p.is_ignored, p.created_at, p.updated_at,
                   COUNT(f.id)                    AS n_faces,
                   COUNT(DISTINCT f.image_id)     AS n_photos,
                   MIN(i.path)                    AS sample_path
            FROM persons p
            LEFT JOIN faces f  ON f.person_id = p.id
            LEFT JOIN images i ON i.id = f.image_id
            GROUP BY p.id
            ORDER BY p.id
            """
        )
        out: list[PersonSummary] = []
        for row in cur.fetchall():
            person = Person(
                id=row["id"],
                name=row["name"],
                is_ignored=bool(row["is_ignored"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            out.append(
                PersonSummary(
                    person=person,
                    n_faces=row["n_faces"],
                    n_photos=row["n_photos"],
                    sample_path=row["sample_path"],
                )
            )
        return out

    # ------------------------------------------------------------------ #
    # Faces
    # ------------------------------------------------------------------ #
    def add_face(
        self,
        image_id: int,
        person_id: int | None,
        embedding: np.ndarray,
        bbox: tuple[int, int, int, int],  # (top, right, bottom, left)
        detector: str,
        created_at: float,
        match_distance: float | None = None,
    ) -> int:
        top, right, bottom, left = bbox
        cur = self.conn.execute(
            """INSERT INTO faces
               (image_id, person_id, embedding, top, right, bottom, left,
                match_distance, detector, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                image_id,
                person_id,
                enc_to_blob(embedding),
                top,
                right,
                bottom,
                left,
                match_distance,
                detector,
                created_at,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def set_face_person(self, face_id: int, person_id: int) -> None:
        self.conn.execute(
            "UPDATE faces SET person_id = ? WHERE id = ?", (person_id, face_id)
        )
        self.conn.commit()

    def load_all_embeddings(self) -> tuple[np.ndarray, list[int]]:
        """Return (matrix [N,128], person_ids[N]) for all faces with a person.

        Faces not yet assigned to a person (person_id IS NULL) are excluded —
        matching is only ever against known people. Returns an empty (0,128)
        matrix when there are none.
        """
        cur = self.conn.execute(
            "SELECT embedding, person_id FROM faces WHERE person_id IS NOT NULL"
        )
        embeddings: list[np.ndarray] = []
        person_ids: list[int] = []
        for row in cur.fetchall():
            embeddings.append(blob_to_enc(row["embedding"]))
            person_ids.append(row["person_id"])
        if not embeddings:
            return np.empty((0, 128), dtype=np.float64), []
        return np.vstack(embeddings), person_ids

    def get_faces_for_person(self, person_id: int) -> list[FaceRow]:
        cur = self.conn.execute(
            "SELECT * FROM faces WHERE person_id = ? ORDER BY id", (person_id,)
        )
        return [_to_face(row) for row in cur.fetchall()]

    def iter_person_image_paths(self) -> dict[int, list[str]]:
        """Map person_id -> sorted list of distinct source image paths.

        Used by `apply` to build the output tree.
        """
        cur = self.conn.execute(
            """
            SELECT DISTINCT f.person_id AS pid, i.path AS path
            FROM faces f
            JOIN images i ON i.id = f.image_id
            WHERE f.person_id IS NOT NULL
            ORDER BY f.person_id, i.path
            """
        )
        out: dict[int, list[str]] = {}
        for row in cur.fetchall():
            out.setdefault(row["pid"], []).append(row["path"])
        return out


# --------------------------------------------------------------------------- #
# Row -> dataclass helpers
# --------------------------------------------------------------------------- #
def _to_image(row: sqlite3.Row) -> ImageRow:
    return ImageRow(
        id=row["id"],
        path=row["path"],
        size=row["size"],
        mtime=row["mtime"],
        content_hash=row["content_hash"],
        status=row["status"],
        n_faces=row["n_faces"],
        error=row["error"],
        processed_at=row["processed_at"],
        created_at=row["created_at"],
    )


def _to_person(row: sqlite3.Row) -> Person:
    return Person(
        id=row["id"],
        name=row["name"],
        is_ignored=bool(row["is_ignored"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _to_face(row: sqlite3.Row) -> FaceRow:
    return FaceRow(
        id=row["id"],
        image_id=row["image_id"],
        person_id=row["person_id"],
        embedding=blob_to_enc(row["embedding"]),
        top=row["top"],
        right=row["right"],
        bottom=row["bottom"],
        left=row["left"],
        match_distance=row["match_distance"],
        detector=row["detector"],
        created_at=row["created_at"],
    )
