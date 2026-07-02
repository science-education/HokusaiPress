"""Page parameters are compared as normalized structural feature vectors."""

from __future__ import annotations

from typing import TypeVar

import numpy as np

from . import learn
from .model import PageParams

K = TypeVar("K")


def page_vector(params: PageParams) -> np.ndarray:
    """Return the page features in the learner's fixed feature order."""
    features = learn.page_features(params)
    vector = np.array([float(features[name]) for name in learn.FEATURES], dtype=float)
    return np.nan_to_num(vector, nan=0.0)


def rank_similar(
    target: PageParams,
    candidates: list[tuple[K, PageParams]],
    k: int | None = None,
) -> list[tuple[K, float]]:
    """Rank candidate pages by normalized Euclidean feature distance."""
    if not candidates:
        return []

    vectors = np.vstack(
        [page_vector(target), *(page_vector(params) for _, params in candidates)]
    )
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    normalized = np.zeros_like(vectors)
    np.divide(vectors - mean, std, out=normalized, where=std != 0)

    distances = np.linalg.norm(normalized[1:] - normalized[0], axis=1)
    ranked = sorted(
        ((key, float(distance)) for (key, _), distance in zip(candidates, distances)),
        key=lambda item: item[1],
    )
    return ranked if k is None else ranked[:k]
