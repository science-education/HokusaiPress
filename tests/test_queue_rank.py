"""Contract for WP-31: active-learning queue ranking (docs/VIEWER_LIBRARY_PLAN.md).

The review queue should surface the page whose decision resolves the MOST other
pending pages first — a page similar to many others in the pending set is worth
one human answer that propagation (WP-22) can then spread. Ranking is over the
doc's needs_review pages, by descending cluster size (ties -> lower page_index).
Pure logic on top of WP-21. Skip until implemented.
"""

import pytest

from hokusai_press.model import (
    Box,
    Deskew,
    PageParams,
    Region,
    RegionKind,
    ReviewStatus,
    SourceRef,
)
from hokusai_press.store import Store

try:
    from hokusai_press.queue_rank import rank_pending
except ImportError:
    pytest.skip(
        "WP-31 unimplemented: hokusai_press.queue_rank missing "
        "(see docs/tasks/WP-31-codex.md)",
        allow_module_level=True,
    )


def _page(n_text=1, n_photo=0, angle=0.0, status=ReviewStatus.NEEDS_REVIEW):
    regions = [Region(kind=RegionKind.TEXT, box=Box(0, 0, 50, 10))
               for _ in range(n_text)]
    regions += [Region(kind=RegionKind.PHOTO, box=Box(0, 20, 80, 90))
                for _ in range(n_photo)]
    return PageParams(
        source=SourceRef(path="s.pdf", page_index=0),
        deskew=Deskew(angle_deg=angle, confidence=1.0),
        regions=regions,
        review_status=status,
    )


def test_only_pending_pages_are_ranked(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.upsert_page("doc", 0, _page(status=ReviewStatus.AUTO))       # not pending
    store.upsert_page("doc", 1, _page(n_photo=4))                      # pending
    store.upsert_page("doc", 2, _page(n_photo=4))                      # pending

    ranked = rank_pending(store, "doc", similar_threshold=0.5)
    idxs = [i for i, _ in ranked]
    assert set(idxs) == {1, 2}
    assert 0 not in idxs


def test_page_similar_to_most_others_ranks_first(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    # A cluster of 3 near-identical pending pages ...
    store.upsert_page("doc", 1, _page(n_photo=4))
    store.upsert_page("doc", 2, _page(n_photo=4))
    store.upsert_page("doc", 3, _page(n_photo=4))
    # ... and one isolated, very different pending page.
    store.upsert_page("doc", 9, _page(n_text=9, n_photo=0, angle=9.0))

    ranked = rank_pending(store, "doc", similar_threshold=0.5)

    # The isolated page must rank last; a cluster member ranks first.
    assert ranked[0][0] in {1, 2, 3}
    assert ranked[-1][0] == 9
    # Scores are descending.
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_k_limits_results(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    for i in range(5):
        store.upsert_page("doc", i, _page(n_photo=4))
    ranked = rank_pending(store, "doc", similar_threshold=0.5, k=2)
    assert len(ranked) == 2


def test_no_pending_returns_empty(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.upsert_page("doc", 0, _page(status=ReviewStatus.AUTO))
    assert rank_pending(store, "doc", similar_threshold=0.5) == []
