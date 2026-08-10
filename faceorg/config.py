"""Configuration: defaults, TOML file loading, and CLI-override precedence.

Precedence (highest wins): CLI flag > config file > built-in default.

A single frozen ``Config`` object is built once per command via
``Config.from_sources(cli_overrides, config_path)`` and threaded through the
rest of the app so every command resolves settings identically.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

# Default location of the global "brain". One DB across all libraries so a
# person accumulates more reference embeddings over time (better recognition)
# and is recognized regardless of which folder their photos are scanned from.
DEFAULT_DB_PATH = Path.home() / ".faceorg" / "faces.db"

# Image extensions considered during a scan.
DEFAULT_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".heic",
)

# face_recognition uses Euclidean distance between 128-d encodings; the
# library's documented default match tolerance is 0.6 (lower = same person).
DEFAULT_TOLERANCE = 0.6


@dataclass(frozen=True)
class Config:
    """Resolved runtime configuration for a single command invocation."""

    # paths
    src: Path | None = None
    dest: Path | None = None
    db: Path = DEFAULT_DB_PATH

    # detect
    model: str = "hog"  # "hog" | "cnn"
    upsample: int = 1
    jitters: int = 1

    # match
    tolerance: float = DEFAULT_TOLERANCE

    # apply
    mode: str = "symlink"  # "symlink" | "copy"
    min_photos: int = 1
    prune: bool = True

    # scan
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_sources(
        cls,
        cli_overrides: dict | None = None,
        config_path: Path | None = None,
    ) -> "Config":
        """Build a Config by layering: defaults -> file -> CLI overrides.

        ``cli_overrides`` values that are ``None`` are ignored (i.e. the flag
        was not supplied), so they never clobber a file/default value.
        """
        cfg = cls()  # built-in defaults

        file_values = _load_config_file(config_path)
        if file_values:
            cfg = cfg._merged(file_values)

        if cli_overrides:
            cli_clean = {k: v for k, v in cli_overrides.items() if v is not None}
            if cli_clean:
                cfg = cfg._merged(cli_clean)

        return cfg._normalized()

    def _merged(self, values: dict) -> "Config":
        """Return a copy with recognized keys overridden by ``values``."""
        known = {f for f in self.__dataclass_fields__}
        clean = {k: v for k, v in values.items() if k in known}
        return replace(self, **clean)

    def _normalized(self) -> "Config":
        """Coerce path-like fields to Path and validate enum-ish fields."""
        changes: dict = {}
        if self.src is not None and not isinstance(self.src, Path):
            changes["src"] = Path(self.src).expanduser()
        elif isinstance(self.src, Path):
            changes["src"] = self.src.expanduser()

        if self.dest is not None and not isinstance(self.dest, Path):
            changes["dest"] = Path(self.dest).expanduser()
        elif isinstance(self.dest, Path):
            changes["dest"] = self.dest.expanduser()

        changes["db"] = Path(self.db).expanduser()

        if isinstance(self.extensions, list):
            changes["extensions"] = tuple(self.extensions)

        if self.model not in ("hog", "cnn"):
            raise ValueError(f"model must be 'hog' or 'cnn', got {self.model!r}")
        if self.mode not in ("symlink", "copy"):
            raise ValueError(f"mode must be 'symlink' or 'copy', got {self.mode!r}")

        return replace(self, **changes)


def _load_config_file(config_path: Path | None) -> dict:
    """Load and flatten a faceorg TOML config into a flat key->value dict.

    If ``config_path`` is None, look for ``faceorg.toml`` in the cwd. A missing
    file is not an error (returns an empty dict); an explicitly-passed missing
    path raises.
    """
    if config_path is None:
        candidate = Path.cwd() / "faceorg.toml"
        if not candidate.exists():
            return {}
        config_path = candidate
    else:
        config_path = Path(config_path).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as fh:
        raw = tomllib.load(fh)

    return _flatten_toml(raw)


# Map TOML [section] keys onto the flat Config field names.
_SECTION_KEYS = {
    "paths": {"src", "dest", "db"},
    "detect": {"model", "upsample", "jitters"},
    "match": {"tolerance"},
    "apply": {"mode", "min_photos", "prune"},
    "scan": {"extensions"},
}


def _flatten_toml(raw: dict) -> dict:
    """Flatten known [section].key TOML structure into flat Config keys.

    Also accepts already-flat top-level keys for convenience.
    """
    flat: dict = {}
    for section, keys in _SECTION_KEYS.items():
        table = raw.get(section)
        if isinstance(table, dict):
            for key in keys:
                if key in table:
                    flat[key] = table[key]
    # allow flat top-level keys too (e.g. db = "...")
    for key, value in raw.items():
        if not isinstance(value, dict):
            flat[key] = value
    return flat
