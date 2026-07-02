"""Rank pending pages by the value of resolving their review."""

from __future__ import annotations

from . import similarity
from .model import ReviewStatus


def rank_pending(
    store,
    doc_id,
    *,
    similar_threshold,
    k=None,
) -> list[tuple[int, float]]:
    """Return pending pages ranked by the number of similar pending peers."""
    pending = [
        row
        for row in store.list_pages(doc_id)
        if row.review_status == ReviewStatus.NEEDS_REVIEW.value
    ]

    ranked = []
    for page in pending:
        candidates = [
            (other.page_index, other.params)
            for other in pending
            if other.page_index != page.page_index
        ]
        score = float(
            sum(
                distance <= similar_threshold
                for _, distance in similarity.rank_similar(page.params, candidates)
            )
        )
        ranked.append((page.page_index, score))

    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked if k is None else ranked[:k]
