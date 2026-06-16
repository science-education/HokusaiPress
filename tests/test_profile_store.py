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


def test_run_populates_profile(tmp_path):
    import cv2
    import numpy as np

    from hokusai_press import pipeline

    img = np.full((1000, 700, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 120), (600, 880), (0, 0, 0), 2)
    for y in range(160, 840, 40):
        cv2.rectangle(img, (120, y), (580, y + 16), (0, 0, 0), -1)
    src = str(tmp_path / "pg.png")
    cv2.imwrite(src, img)

    db = str(tmp_path / "j.db")
    pipeline.run(src, str(tmp_path / "o.pdf"), db_path=db, use_ocr=False)

    s = Store(db)
    try:
        feats = s.page_features("pg.png")
        assert len(feats) == 1
        assert feats[0]["is_ocr"] == 0          # no OCR -> OCR tier empty
        assert feats[0]["deskew_angle"] is not None  # geometry tier present
    finally:
        s.close()


def test_store_bucket_backoff():
    s = Store(":memory:")
    # same regime (horizontal/left); soft dims back off
    s.upsert_bucket_param("ADF1", "A5", "", "horizontal", "left",
                          "size_h", 1400.0, 2.0, 50)
    s.upsert_bucket_param("", "", "", "horizontal", "left",
                          "size_h", 1390.0, 9.0, 999)

    def lookup(k):
        r = s.get_bucket_param(k.scanner, k.fmt, k.genre, k.writing,
                               k.binding, "size_h")
        return (r, r[2]) if r else None

    key = ScopeKey("ADF1", "A5", "novel", "horizontal", "left")
    # genre 'novel' bucket absent -> back off to scanner x fmt (n=50 >= 10)
    prof, used = profile.resolve_profile(key, lookup, 10)
    assert used == ScopeKey("ADF1", "A5", None, "horizontal", "left")
    assert prof[0] == 1400.0
    # demand more support than any specific bucket -> regime-coarsest fallback
    prof2, used2 = profile.resolve_profile(key, lookup, 100)
    assert used2 == ScopeKey(None, None, None, "horizontal", "left")
    assert prof2[0] == 1390.0
    s.close()
