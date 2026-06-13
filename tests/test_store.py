from hokusai_press.model import (
    Box,
    Deskew,
    Flag,
    Margin,
    PageParams,
    ReviewStatus,
    SourceRef,
)
from hokusai_press.store import Store


def _page(flagged: bool) -> PageParams:
    return PageParams(
        source=SourceRef(path="scan.pdf", page_index=0),
        deskew=Deskew(angle_deg=0.3, confidence=4.0),
        margin=Margin(content=Box(0, 0, 100, 100), confidence=0.7),
        flags=[Flag.MARGIN_INCONSISTENT] if flagged else [],
        review_status=ReviewStatus.NEEDS_REVIEW if flagged else ReviewStatus.AUTO,
    )


def test_upsert_and_get(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.upsert_page("doc", 0, _page(True))
    row = store.get_page("doc", 0)
    assert row.params.deskew.confidence == 4.0
    assert row.flags == ["margin_inconsistent_with_neighbors"]


def test_review_queue_filters(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.upsert_page("doc", 0, _page(False))
    store.upsert_page("doc", 1, _page(True))
    q = store.review_queue("doc")
    assert [r.page_index for r in q] == [1]


def test_decision_log(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.upsert_page("doc", 0, _page(True))
    store.log_decision("doc", 0, "human", "page_kind", "auto", "bw",
                       {"deskew_conf": 4.0, "n_text": 12})
    rows = store.decisions_for_training("page_kind")
    assert len(rows) == 1
    assert rows[0]["new_value"] == '"bw"'


def test_concurrent_access_is_consistent(tmp_path):
    # A page load fires several requests at once, all sharing one connection
    # from a threadpool. Without serialization, interleaved cursors return
    # half-populated rows (NULL columns). Hammer it from many threads and
    # assert every read is well-formed.
    import threading

    store = Store(str(tmp_path / "t.db"))
    for i in range(8):
        store.upsert_page("doc", i, _page(True))

    errors: list[Exception] = []

    def worker():
        try:
            for _ in range(60):
                for i in range(8):
                    row = store.get_page("doc", i)
                    assert row is not None
                    assert isinstance(row.flags, list)  # would be None on a race
                store.review_queue("doc")
                store.upsert_page("doc", 0, _page(True))
        except Exception as e:  # pragma: no cover - only on regression
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[0]
