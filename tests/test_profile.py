from hokusai_press.model import (
    Box, Deskew, Margin, PageParams, Region, RegionKind, SourceRef,
)
from hokusai_press import profile
from hokusai_press.profile import ScopeKey, BookProfile


def _page(idx, *, page_number=None, content=(100, 100, 900, 900),
          nombre=None, regions=None, deskew=0.0, blank=False):
    m = Margin(content=Box(*content))
    if nombre:
        m.nombre_box = Box(*nombre)
    return PageParams(
        source=SourceRef(path="x.pdf", page_index=idx),
        deskew=Deskew(angle_deg=deskew, confidence=5.0),
        margin=m,
        regions=regions or [],
        page_number=page_number,
        blank=blank,
    )


def test_robust_stat_handles_none_and_empty():
    assert profile.robust_stat([]) is None
    assert profile.robust_stat([None, None]) is None
    st = profile.robust_stat([1.0, 2.0, 3.0, None])
    assert st.median == 2.0 and st.n == 3


def test_extract_features_normalizes_and_drops_text():
    reg = Region(kind=RegionKind.TEXT, box=Box(100, 200, 300, 400),
                 ocr_text="secret body text", source="dbnet")
    p = _page(0, page_number=1, nombre=(450, 50, 550, 90), regions=[reg])
    pf, rf = profile.extract_features([p], [1000.0], [1000.0])
    assert pf[0].is_ocr is True and pf[0].nombre_value == 1
    assert abs(pf[0].nombre_cx - 0.5) < 1e-9 and abs(pf[0].nombre_cy - 0.07) < 1e-9
    assert abs(pf[0].content_w - 0.8) < 1e-9
    # no text content anywhere in the stored region feature
    assert rf[0].cls == "text" and not hasattr(rf[0], "ocr_text")
    assert (rf[0].x0, rf[0].y0) == (0.1, 0.2)


def test_aggregate_offset_and_class_density():
    pages = [_page(i, page_number=i + 5,
                   regions=[Region(kind=RegionKind.TEXT, box=Box(0, 0, 10, 10),
                                   ocr_text="t")]) for i in range(6)]
    pf, rf = profile.extract_features(pages, [1000.0] * 6, [1400.0] * 6)
    bp = profile.aggregate(pf, rf, [1000.0] * 6, [1400.0] * 6)
    assert bp.dominant_offset == 5
    assert bp.ocr_pages == 6 and bp.page_count == 6
    assert abs(bp.class_density["text"] - 1.0) < 1e-9
    assert bp.size_h.median == 1400.0


def test_backoff_keys_full_partial_global():
    full = profile.backoff_keys(ScopeKey("ADF1", "A5", "novel"))
    assert full[0] == ScopeKey("ADF1", "A5", "novel")
    assert full[-1] == ScopeKey()           # global always last
    # genre missing -> no key that keeps genre, but global still present
    partial = profile.backoff_keys(ScopeKey("ADF1", "A5", None))
    assert all(k.genre is None for k in partial)
    assert ScopeKey("ADF1", "A5") in partial and ScopeKey() in partial


def _bp(n):  # minimal profile carrying a sample count via dominant_offset slot
    return BookProfile(n, n, None, None, None, None, None, None, {})


def test_resolve_profile_picks_finest_then_backs_off():
    buckets = {
        ScopeKey("ADF1", "A5", "novel"): (_bp(2), 2),
        ScopeKey("ADF1", "A5"): (_bp(50), 50),
        ScopeKey(): (_bp(999), 999),
    }
    lookup = lambda k: buckets.get(k)
    # finest has only n=2 (< min 10) -> back off to scanner×fmt (n=50)
    prof, used = profile.resolve_profile(ScopeKey("ADF1", "A5", "novel"), lookup, 10)
    assert used == ScopeKey("ADF1", "A5") and prof.page_count == 50
    # raise the bar above everything but global -> global fallback
    prof2, used2 = profile.resolve_profile(ScopeKey("ADF1", "A5", "novel"), lookup, 100)
    assert used2 == ScopeKey() and prof2.page_count == 999


def test_resolve_profile_none_when_empty():
    assert profile.resolve_profile(ScopeKey("X"), lambda k: None, 1) is None


def test_detect_writing_direction_and_binding():
    from hokusai_press.profile import (
        PageFeature, RegionFeature, detect_binding, detect_writing_direction,
    )
    tall = [RegionFeature(0, "text", 0.1, 0.1, 0.2, 0.9, None, "dbnet")
            for _ in range(6)]                       # h/w = 8 -> vertical
    wide = [RegionFeature(0, "text", 0.1, 0.1, 0.9, 0.15, None, "dbnet")
            for _ in range(6)]                       # h/w = 0.06 -> horizontal
    assert detect_writing_direction(tall) == "vertical"
    assert detect_writing_direction(wide) == "horizontal"
    assert detect_writing_direction([]) == "unknown"

    def pages(even_left: bool):
        out = []
        for i in range(8):
            left = (i % 2 == 0) == even_left
            cx = 0.08 if left else 0.9
            out.append(PageFeature(i, True, 1, 0.0, 5.0, 0.8, 0.8, i + 1,
                                   cx, 0.95, False))
        return out

    assert detect_binding(pages(even_left=True), "vertical") == "right"
    assert detect_binding(pages(even_left=False), "horizontal") == "left"
