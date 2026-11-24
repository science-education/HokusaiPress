from hokusai_press.model import (
    Box,
    Deskew,
    Document,
    Flag,
    Margin,
    PageKind,
    PageParams,
    Region,
    RegionKind,
    ReviewStatus,
    SourceRef,
)


def _sample_doc() -> Document:
    page = PageParams(
        source=SourceRef(path="scan.pdf", page_index=2, ocr_scale=0.5),
        dpi=600,
        deskew=Deskew(angle_deg=1.25, confidence=3.1),
        margin=Margin(content=Box(10, 20, 110, 220), confidence=0.8,
                      nombre_box=Box(50, 5, 70, 18)),
        regions=[
            Region(kind=RegionKind.TEXT, box=Box(10, 20, 110, 40),
                   ocr_text="序章", ocr_conf=0.9),
            Region(kind=RegionKind.PHOTO, box=Box(10, 120, 110, 220),
                   source="residual"),
        ],
        page_kind=PageKind.AUTO,
        flags=[Flag.DESKEW_LOW_CONF],
        review_status=ReviewStatus.NEEDS_REVIEW,
    )
    return Document(source_path="scan.pdf", pages=[page])


def test_round_trip_json():
    doc = _sample_doc()
    restored = Document.from_json(doc.to_json())
    assert restored.source_path == "scan.pdf"
    p = restored.pages[0]
    assert p.deskew.angle_deg == 1.25
    assert p.deskew.confidence == 3.1
    assert p.margin.content.width == 100
    assert p.margin.nombre_box.height == 13
    assert p.regions[0].kind == RegionKind.TEXT
    assert p.regions[0].ocr_text == "序章"
    assert p.regions[1].kind == RegionKind.PHOTO
    assert p.flags == [Flag.DESKEW_LOW_CONF]
    assert p.review_status == ReviewStatus.NEEDS_REVIEW
    assert p.needs_review()


def test_reading_text():
    doc = _sample_doc()
    assert doc.pages[0].reading_text() == "序章"


def test_box_dims():
    b = Box(5, 10, 25, 60)
    assert b.width == 20 and b.height == 50


def test_margin_nombre_source_round_trip_and_legacy_decode():
    import json

    doc = _sample_doc()
    doc.pages[0].margin.nombre_source = "folio"
    doc.pages[0].margin.glyph_box = Box(20, 30, 40, 50)
    restored = Document.from_json(doc.to_json())
    assert restored.pages[0].margin.glyph_box == Box(20, 30, 40, 50)
    assert restored.pages[0].margin.nombre_source == "folio"

    # Legacy DB JSON without nombre_source must decode cleanly to None
    raw = json.loads(doc.to_json())
    raw["pages"][0]["margin"].pop("nombre_source", None)
    raw["pages"][0]["margin"].pop("glyph_box")
    legacy = Document.from_json(json.dumps(raw))
    assert legacy.pages[0].margin.glyph_box is None
    assert legacy.pages[0].margin.nombre_source is None
    assert legacy.pages[0].margin.nombre_box is not None
