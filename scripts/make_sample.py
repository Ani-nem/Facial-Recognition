#!/usr/bin/env python
"""Build a small labeled face sample for live smoke-testing faceorg.

Downloads a subset of LFW (Labeled Faces in the Wild) via scikit-learn and
writes a few images per person, spread across two "occasion" folders — so
clustering has to group the same person across different events, mirroring the
real photographer workflow.

Usage:
    python scripts/make_sample.py [--dest DIR] [--people N] [--per-person N]

Then:
    faceorg --db /tmp/faceorg_test.db scan --src <DIR>
    faceorg --db /tmp/faceorg_test.db list --sort id

Requires the optional deps: scikit-learn and Pillow (installed in the conda
env alongside dlib/face_recognition).
"""

from __future__ import annotations

import argparse
import collections
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="/tmp/faceorg_sample", type=Path)
    parser.add_argument("--people", type=int, default=4, help="How many people.")
    parser.add_argument(
        "--per-person", type=int, default=6, help="Max images per person."
    )
    parser.add_argument(
        "--min-faces",
        type=int,
        default=20,
        help="LFW min_faces_per_person filter (more = fewer, better-covered people).",
    )
    args = parser.parse_args()

    from PIL import Image
    from sklearn.datasets import fetch_lfw_people

    print("Fetching LFW subset (first run downloads ~200MB, cached after)...")
    lfw = fetch_lfw_people(
        min_faces_per_person=args.min_faces, resize=1.0, color=True, funneled=True
    )
    names = lfw.target_names

    by_person: dict[int, list[np.ndarray]] = collections.defaultdict(list)
    for img, tgt in zip(lfw.images, lfw.target):
        by_person[int(tgt)].append(img)

    chosen = list(by_person.keys())[: args.people]
    dest = Path(args.dest)
    shutil.rmtree(dest, ignore_errors=True)
    events = [dest / "event_A", dest / "event_B"]
    for e in events:
        e.mkdir(parents=True)

    count = 0
    for tgt in chosen:
        safe = names[tgt].replace(" ", "_")
        for i, im in enumerate(by_person[tgt][: args.per_person]):
            # LFW images are float32 in 0..1 — scale to 0..255 before saving.
            arr = np.clip(im * 255.0, 0, 255).astype("uint8")
            Image.fromarray(arr).save(events[i % 2] / f"{safe}_{i}.jpg")
            count += 1

    print(f"Wrote {count} images of {len(chosen)} people across 2 event folders:")
    for e in events:
        print(f"  {e} -> {len(list(e.glob('*.jpg')))} images")
    print("Ground truth:", [names[t] for t in chosen])


if __name__ == "__main__":
    main()
