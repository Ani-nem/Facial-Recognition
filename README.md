# faceorg

Organize a photo library **by person** using facial recognition — without
duplicating your files.

You already sort photos by occasion. `faceorg` scans those folders, recognizes
the people in them, and builds a **parallel tree of symlinks** grouped by
person, so you can grab "all photos of Alice" to share, while your originals
stay exactly where they are.

- **Non-destructive** — output is symlinks by default (or copies, opt-in). Your
  originals are never moved, renamed, or duplicated.
- **Incremental & self-learning** — a global SQLite "brain" remembers every
  face and the names you assign. Re-running only processes new photos and
  auto-matches them to people it already knows. Recognition improves as each
  person accumulates more reference photos.
- **Configurable** — source/dest paths, symlink vs copy, detector model,
  match tolerance, minimum photos per person, and more.

> Status: **Phase 1** (scan / list / rename / merge / status). The `apply`
> command that builds the output tree lands in Phase 2.

## Install

`faceorg` depends on `dlib`, which needs a C++ toolchain. The supported path is
conda-forge:

```bash
conda create -n faceorg python=3.11 -c conda-forge dlib face_recognition
conda activate faceorg
pip install -e .
```

Pip-only (best effort; needs CMake + a compiler):

```bash
brew install cmake          # macOS
pip install -e .
```

## Quickstart

```bash
# 1. Scan your library (recursive). First run auto-buckets everyone.
faceorg scan --src ~/Photos

# 2. See who was found.
faceorg list --sort count

# 3. Figure out who a bucket is, then name them.
faceorg show 7
faceorg rename 7 Alice

# 4. Fix an over-split (person 12 is also Alice).
faceorg merge 12 7

# 5. (Phase 2) Build the by-person/ symlink tree.
# faceorg apply --dest ~/Photos-by-person

# Later, after adding new photos — only new files are processed,
# and Alice's new photos attach to her automatically.
faceorg scan --src ~/Photos
```

## How incremental scanning works

Each image is tracked by path, size, and modification time. On re-scan:

- **Unchanged** and already processed → skipped (no decode).
- **Touched** (mtime changed, content identical) → skipped after a hash check.
- **Edited** (content changed) → old faces dropped, re-processed.
- `--force` reprocesses everything; `--rehash` verifies content hashes even when
  size + mtime look unchanged.

## Configuration

Settings resolve as **CLI flag > `faceorg.toml` > built-in default**. Copy
`faceorg.toml.example` to `faceorg.toml` and edit. Key options:

| Key | Default | Meaning |
|-----|---------|---------|
| `db` | `~/.faceorg/faces.db` | Global brain shared across libraries |
| `detect.model` | `hog` | `hog` (fast, CPU) or `cnn` (accurate, GPU) |
| `match.tolerance` | `0.50` | Euclidean match distance; lower = stricter |
| `apply.mode` | `symlink` | `symlink` or `copy` |
| `apply.min_photos` | `1` | Only output people in ≥ N photos |

## Commands

| Command | Description |
|---------|-------------|
| `scan` | Detect, embed, and cluster faces into people |
| `list` | List people with photo/face counts |
| `show <id>` | List a person's photos |
| `rename <id> <name>` | Name a person (persists across scans) |
| `merge <src> <dst>` | Merge two people (fix over-split) |
| `ignore` / `unignore <id>` | Exclude/include a person in output |
| `status` | DB counts and location |
| `tune --dataset <dir>` | Similarity stats to guide `--tolerance` |

## License

MIT
