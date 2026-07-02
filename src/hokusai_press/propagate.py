"""Propagate a page-kind correction to structurally similar pages."""

from __future__ import annotations

from . import learn, similarity
from .model import DecidedBy, ReviewStatus


def propagate(
    store,
    doc_id,
    source_index,
    page_kind,
    *,
    auto_threshold,
    queue_threshold,
    decided_by="propagation",
) -> dict:
    """Apply or queue a correction according to structural distance."""
    pages = store.list_pages(doc_id)
    source = next(row for row in pages if row.page_index == source_index)
    candidates = [
        (row.page_index, row.params)
        for row in pages
        if row.page_index != source_index
        and row.params.decided_by != DecidedBy.HUMAN
    ]
    params_by_index = {index: params for index, params in candidates}

    applied = []
    queued = []
    for page_index, distance in similarity.rank_similar(source.params, candidates):
        params = params_by_index[page_index]
        old_kind = params.page_kind
        if distance <= auto_threshold:
            params.page_kind = page_kind
            store.upsert_page(doc_id, page_index, params)
            store.log_decision(
                doc_id,
                page_index,
                decided_by,
                field="page_kind",
                old_value=old_kind,
                new_value=page_kind,
                features=learn.page_features(params),
            )
            applied.append(page_index)
        elif distance <= queue_threshold:
            params.review_status = ReviewStatus.NEEDS_REVIEW
            store.upsert_page(doc_id, page_index, params)
            store.log_decision(
                doc_id,
                page_index,
                decided_by,
                field="page_kind",
                old_value=old_kind,
                new_value=old_kind,
                features=learn.page_features(params),
            )
            queued.append(page_index)

    return {"applied": applied, "queued": queued}
