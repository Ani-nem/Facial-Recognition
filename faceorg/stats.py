"""Threshold-tuning statistics (read-only helper for `faceorg tune`).

Ported from the original project's util.py. Given a labeled dataset where each
subfolder holds images of one person, it computes intra-person cosine
similarities and suggests strict/balanced/lenient thresholds.

Note: this reports *cosine similarity* for human interpretation. The matcher
itself uses Euclidean distance (see faceorg.embed.face_distance); use these
stats as a qualitative guide to how separable your photos are, not as a direct
tolerance value.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")


def calculate_similarity_statistics(dataset_path: str) -> dict:
    """Analyze a dataset (one subfolder per person) of same-person images.

    :param dataset_path: root dir containing person subfolders.
    :return: dict with 'per_person', 'overall', and 'suggested_thresholds'.
    """
    import face_recognition

    person_embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
    person_similarities: dict[str, list[float]] = defaultdict(list)

    for person_folder in sorted(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, person_folder)
        if not os.path.isdir(person_path):
            continue

        for img_file in sorted(os.listdir(person_path)):
            if not img_file.lower().endswith(_IMG_EXTS):
                continue
            img_path = os.path.join(person_path, img_file)
            try:
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    person_embeddings[person_folder].append(encodings[0])
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"Error processing {img_file}: {exc}")

    overall: list[float] = []
    for person, embeddings in person_embeddings.items():
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = float(
                    np.dot(embeddings[i], embeddings[j])
                    / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                )
                person_similarities[person].append(sim)
                overall.append(sim)

    stats: dict = {"per_person": {}, "overall": {}}
    for person, sims in person_similarities.items():
        if sims:
            stats["per_person"][person] = _summarize(sims)

    if overall:
        stats["overall"] = _summarize(overall)
        mean, std = stats["overall"]["mean"], stats["overall"]["std"]
        stats["suggested_thresholds"] = {
            "lenient": mean - std,
            "balanced": mean,
            "strict": mean + std,
        }

    return stats


def _summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "count": int(arr.size),
    }


def format_similarity_stats(stats: dict) -> str:
    """Render the stats dict as a human-readable report string."""
    if not stats.get("overall"):
        return "No same-person comparisons found (need >=2 faces per person folder)."

    lines: list[str] = []
    o = stats["overall"]
    lines.append("=== Overall (cosine similarity) ===")
    lines.append(f"Comparisons: {o['count']}")
    lines.append(f"Mean: {o['mean']:.4f}   Std: {o['std']:.4f}")
    lines.append(f"Range: {o['min']:.4f} to {o['max']:.4f}")

    thr = stats.get("suggested_thresholds", {})
    if thr:
        lines.append("")
        lines.append("=== Suggested similarity thresholds ===")
        lines.append(f"Strict:   {thr['strict']:.4f}")
        lines.append(f"Balanced: {thr['balanced']:.4f}")
        lines.append(f"Lenient:  {thr['lenient']:.4f}")

    per = stats.get("per_person", {})
    if per:
        lines.append("")
        lines.append("=== Per-person ===")
        for person, s in per.items():
            lines.append(
                f"{person}: n={s['count']} mean={s['mean']:.4f} "
                f"std={s['std']:.4f} range={s['min']:.4f}-{s['max']:.4f}"
            )
    return "\n".join(lines)
