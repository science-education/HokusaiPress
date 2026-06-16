import dataclasses
import json

from hokusai_press import profile
from hokusai_press.model import (
    Box, Deskew, Margin, PageParams, Region, RegionKind, SourceRef,
)
from hokusai_press.profile import ScopeKey
from hokusai_press.store import Store


def _page(idx, page_number=None):
    return PageParams(
        source=SourceRef(path="x.pdf", page_index=idx),
        deskew=Deskew(angle_deg=0.0, confidence=5.0),
        margin=Margin(content=Box(100, 100, 900, 900),
                      nombre_box=Box(450, 50, 550, 90)),
        regions=[Region(kind=RegionKind.TEXT, box=Box(100, 200, 300, 400),
                        ocr_text="t", source="dbnet")],
        page_number=page_number,
    )


def test_store_roundtrip_features():
    s = Store(":memory:")
    pages = [_page(i, page_number=i + 1) for i in range(5)]
    pf, rf = profile.extract_features(pages, [1000.0] * 5, [1000.0] * 5)
    s.save_book("bk1", title="T", fmt="A5", source="ocr")
    s.save_page_features("bk1", pf)
    s.save_region_features("bk1", rf)
    bp = profile.aggregate(pf, rf, [1000.0] * 5, [1000.0] * 5)
    s.save_scan_profile("bk1", "ADF1", 0, 4, bp.page_count, bp.ocr_pages,
                        json.dumps(dataclasses.asdict(bp)))

    got = s.page_features("bk1")
    assert len(got) == 5
    assert got[0]["nombre_value"] == 1 and got[0]["is_ocr"] == 1
    # region features stored, no text column exists at all
    assert "ocr_text" not in got[0]
    s.close()


def test_store_bucket_backoff():
    s = Store(":memory:")
    s.upsert_bucket_param("ADF1", "A5", "", "size_h", 1400.0, 2.0, 50)
    s.upsert_bucket_param("", "", "", "size_h", 1390.0, 9.0, 999)

    def lookup(k):
        r = s.get_bucket_param(k.scanner, k.fmt, k.genre, "size_h")
        return (r, r[2]) if r else None

    # genre 'novel' bucket absent -> back off to scanner x fmt (n=50 >= 10)
    prof, used = profile.resolve_profile(ScopeKey("ADF1", "A5", "novel"), lookup, 10)
    assert used == ScopeKey("ADF1", "A5") and prof[0] == 1400.0
    # demand more support than any specific bucket -> global fallback
    prof2, used2 = profile.resolve_profile(ScopeKey("ADF1", "A5", "novel"), lookup, 100)
    assert used2 == ScopeKey() and prof2[0] == 1390.0
    s.close()
