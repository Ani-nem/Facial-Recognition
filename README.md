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

> Status: **usable end-to-end** — `scan` → `recluster` → name people → `apply`
> builds the symlink tree. Interactive TUI is still to come.

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

# 2. Re-cluster for best accuracy after a big first scan (recommended).
faceorg recluster

# 3. See who was found, figure out who a bucket is, then name them.
faceorg list --sort count
faceorg show 7
faceorg rename 7 Alice

# 4. Fix an over-split (person 12 is also Alice).
faceorg merge 12 7

# 5. Build the by-person/ symlink tree (preview first with --dry-run).
faceorg apply --dest ~/Photos-by-person --dry-run
faceorg apply --dest ~/Photos-by-person

# Later, after adding new photos — only new files are processed,
# Alice's new photos attach to her, then re-apply to update the tree.
faceorg scan --src ~/Photos
faceorg apply --dest ~/Photos-by-person
```

## How incremental scanning works

Each image is tracked by path, size, and modification time. On re-scan:

- **Unchanged** and already processed → skipped (no decode).
- **Touched** (mtime changed, content identical) → skipped after a hash check.
- **Edited** (content changed) → old faces dropped, re-processed.
- `--force` reprocesses everything; `--rehash` verifies content hashes even when
  size + mtime look unchanged.

## Performance

Detection time scales with image **resolution**, not much else. By default
faceorg downscales each image to a 1400px longest side before detecting faces
(originals are never touched), which keeps things fast on any hardware:

| Working resolution | Time per 24MP photo | 10k photos |
|--------------------|---------------------|------------|
| Full 24MP (`--max-dimension 0`) | ~5 s | hours |
| 1400px (default) | ~0.2 s | ~35 min |

Lower `max_dimension` for more speed; raise it (or set `0`) if you need to
catch very small/distant faces in group shots. RAM use is minimal — images are
processed one at a time and each stored face is ~1 KB, so even 100k faces is
~100 MB. `recluster` is pure vector math (seconds for thousands of faces).

## Configuration

Settings resolve as **CLI flag > `faceorg.toml` > built-in default**. Copy
`faceorg.toml.example` to `faceorg.toml` and edit. Key options:

| Key | Default | Meaning |
|-----|---------|---------|
| `db` | `~/.faceorg/faces.db` | Global brain shared across libraries |
| `detect.model` | `hog` | `hog` (fast, CPU) or `cnn` (accurate, GPU) |
| `detect.max_dimension` | `1400` | Downscale longest side to N px for detection (0 = full res) |
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
