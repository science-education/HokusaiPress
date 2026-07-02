"""Contract for WP-22: correction propagation (docs/VIEWER_LIBRARY_PLAN.md).

A single page_kind correction is propagated to structurally similar pages in the
same doc: near ones (<= auto_threshold) get the fix applied automatically and
logged; medium ones (<= queue_threshold) are flagged for review; far ones are
left alone. Human decisions are never overwritten. Every automatic change is
logged as a decision so it can be audited and undone. Skip until implemented.
"""

import pytest

from hokusai_press.model import (
    Box,
    DecidedBy,
    Deskew,
    PageKind,
    PageParams,
    Region,
    RegionKind,
    ReviewStatus,
    SourceRef,
)
from hokusai_press.store import Store

try:
    from hokusai_press.propagate import propagate
except ImportError:
    pytest.skip(
        "WP-22 unimplemented: hokusai_press.propagate missing "
        "(see docs/tasks/WP-22-codex.md)",
        allow_module_level=True,
    )


def _page(n_text=1, n_photo=0, angle=0.0, decided_by=None,
          kind=PageKind.AUTO):
    regions = [Region(kind=RegionKind.TEXT, box=Box(0, 0, 50, 10))
               for _ in range(n_text)]
    regions += [Region(kind=RegionKind.PHOTO, box=Box(0, 20, 80, 90))
                for _ in range(n_photo)]
    return PageParams(
        source=SourceRef(path="s.pdf", page_index=0),
        deskew=Deskew(angle_deg=angle, confidence=1.0),
        regions=regions,
        page_kind=kind,
        decided_by=decided_by,
    )


def _seed(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    # page 0: the corrected source (many photos -> should be GRAY)
    store.upsert_page("doc", 0, _page(n_text=1, n_photo=4, kind=PageKind.GRAY))
    # page 1, 2: near-identical to source -> auto-apply target
    store.upsert_page("doc", 1, _page(n_text=1, n_photo=4))
    store.upsert_page("doc", 2, _page(n_text=1, n_photo=4))
    # page 3: very different (pure text, skewed) -> untouched
    store.upsert_page("doc", 3, _page(n_text=8, n_photo=0, angle=9.0))
    return store


def test_near_pages_get_fix_applied_and_logged(tmp_path):
    store = _seed(tmp_path)
    result = propagate(store, "doc", 0, PageKind.GRAY,
                       auto_threshold=0.5, queue_threshold=3.0)

    assert set(result["applied"]) == {1, 2}
    for i in (1, 2):
        assert store.get_page("doc", i).params.page_kind == PageKind.GRAY
    # each auto-apply is logged as a propagation decision
    logged = store.decisions_for_training("page_kind")
    assert len(logged) >= 2


def test_far_page_untouched(tmp_path):
    store = _seed(tmp_path)
    propagate(store, "doc", 0, PageKind.GRAY,
              auto_threshold=0.5, queue_threshold=3.0)
    assert store.get_page("doc", 3).params.page_kind == PageKind.AUTO


def test_medium_page_is_queued_not_applied(tmp_path):
    store = _seed(tmp_path)
    # A page moderately similar: some photos, mild skew.
    store.upsert_page("doc", 4, _page(n_text=3, n_photo=2, angle=2.0))
    result = propagate(store, "doc", 0, PageKind.GRAY,
                       auto_threshold=0.5, queue_threshold=3.0)

    assert 4 not in result["applied"]
    if 4 in result["queued"]:
        row = store.get_page("doc", 4)
        assert row.params.page_kind == PageKind.AUTO  # not changed
        assert row.review_status == ReviewStatus.NEEDS_REVIEW.value


def test_human_decision_never_overwritten(tmp_path):
    store = _seed(tmp_path)
    # page 1 is near the source but was decided by a human -> must be left alone.
    store.upsert_page("doc", 1,
                      _page(n_text=1, n_photo=4, decided_by=DecidedBy.HUMAN,
                            kind=PageKind.COLOR))
    result = propagate(store, "doc", 0, PageKind.GRAY,
                       auto_threshold=0.5, queue_threshold=3.0)

    assert 1 not in result["applied"]
    assert store.get_page("doc", 1).params.page_kind == PageKind.COLOR


def test_source_page_not_in_results(tmp_path):
    store = _seed(tmp_path)
    result = propagate(store, "doc", 0, PageKind.GRAY,
                       auto_threshold=0.5, queue_threshold=3.0)
    assert 0 not in result["applied"]
    assert 0 not in result.get("queued", [])
