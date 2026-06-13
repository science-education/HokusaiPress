"""Learn page-kind decisions from the review log.

Once enough humans/AI have corrected page kinds, the hand-tuned auto
thresholds can be replaced by a model fit to those decisions. To stay
dependency-free this uses a normalized nearest-centroid classifier (numpy
only): each decided class gets a centroid in feature space, prediction picks
the nearest, and confidence is the normalized margin to the runner-up.

Feature vector per page (also used by the review UI):
  deskew_angle, deskew_conf, margin_conf, n_text, n_photo, photo_area_frac.

The model is a plain JSON blob, so it ships in the repo / db trivially and is
inspectable. When the model is absent or unsure, the pipeline keeps the
rule-based AUTO behavior and flags the page — never a silent wrong guess.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

FEATURES = ["deskew_angle", "deskew_conf", "margin_conf",
            "n_text", "n_photo", "photo_area_frac"]
MIN_PER_CLASS = 3          # need at least this many examples of a class to learn it
CONFIDENT_MARGIN = 0.15    # normalized centroid-distance margin to auto-apply


def page_features(params) -> dict:
    """Feature vector for a page (shared by the learner and the review UI)."""
    n_photo = sum(1 for r in params.regions if r.kind.value == "photo")
    photo_area = 0.0
    if params.margin and params.margin.content:
        page_area = max(params.margin.content.width * params.margin.content.height, 1)
        photo_area = sum(
            r.box.width * r.box.height for r in params.regions
            if r.kind.value == "photo"
        ) / page_area
    return {
        "deskew_angle": float(params.deskew.angle_deg),
        "deskew_conf": float(params.deskew.confidence),
        "margin_conf": float(params.margin.confidence) if params.margin else 0.0,
        "n_text": float(sum(1 for r in params.regions if r.kind.value == "text")),
        "n_photo": float(n_photo),
        "photo_area_frac": float(photo_area),
    }


def _vec(features: dict) -> np.ndarray:
    return np.array([float(features.get(f, 0.0)) for f in FEATURES], dtype=np.float64)


def train(decisions: list[dict]) -> Optional[dict]:
    """decisions: rows with 'features' (JSON) and 'new_value' (JSON page kind).

    Returns a model dict or None if there isn't enough labeled data.
    """
    by_class: dict[str, list[np.ndarray]] = {}
    for d in decisions:
        try:
            feats = json.loads(d["features"]) if isinstance(d["features"], str) \
                else d["features"]
            label = json.loads(d["new_value"]) if isinstance(d["new_value"], str) \
                else d["new_value"]
        except (json.JSONDecodeError, TypeError):
            continue
        if label in ("bw", "gray", "color"):
            by_class.setdefault(label, []).append(_vec(feats))

    classes = {c: np.array(v) for c, v in by_class.items() if len(v) >= MIN_PER_CLASS}
    if len(classes) < 2:
        return None

    allv = np.vstack(list(classes.values()))
    mean = allv.mean(axis=0)
    std = allv.std(axis=0)
    std[std < 1e-9] = 1.0
    centroids = {c: ((v - mean) / std).mean(axis=0).tolist() for c, v in classes.items()}
    return {
        "features": FEATURES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "centroids": centroids,
        "counts": {c: int(len(v)) for c, v in classes.items()},
    }


def predict(model: dict, features: dict) -> tuple[Optional[str], float]:
    """Return (page_kind, confidence). confidence is the normalized margin
    between the nearest and second-nearest centroid (0 if only one class)."""
    if not model:
        return None, 0.0
    mean = np.array(model["mean"])
    std = np.array(model["std"])
    x = (_vec(features) - mean) / std
    dists = sorted(
        (float(np.linalg.norm(x - np.array(c))), label)
        for label, c in model["centroids"].items()
    )
    if not dists:
        return None, 0.0
    best_d, best_label = dists[0]
    if len(dists) == 1:
        return best_label, 1.0
    second_d = dists[1][0]
    total = best_d + second_d
    conf = (second_d - best_d) / total if total > 0 else 0.0
    return best_label, float(conf)


def is_confident(confidence: float) -> bool:
    return confidence >= CONFIDENT_MARGIN


def save(model: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)


def load(path: str) -> Optional[dict]:
    import os

    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
