import json

import numpy as np
import pikepdf

from hokusai_press.cli import main
from hokusai_press.export_md import (
    document_to_data,
    export_html,
    export_json,
    export_markdown,
)
from hokusai_press.model import (
    Box,
    Document,
    Margin,
    PageParams,
    Region,
    RegionKind,
    SourceRef,
)
from hokusai_press.render import _page_label_segments, _set_physical_page_size
from hokusai_press.store import Store


def _page(index, number=None, nombre=None, nombre_left=True, regions=None):
    nombre_x = 5 if nombre_left else 85
    return PageParams(
        source=SourceRef("book.pdf", page_index=index),
        margin=Margin(
            content=Box(10, 10, 90, 90),
            nombre_box=Box(nombre_x, 2, nombre_x + 5, 7),
            page_w=100,
            page_h=100,
        ),
        page_number=number,
        nombre_text=nombre,
        regions=regions or [],
    )


def test_page_label_segments_support_frontmatter_and_full_roman_alphabet():
    document = Document("book.pdf", pages=[
        _page(0),
        _page(1, 48, "xlviii"),
        _page(2, 49, "xlix"),
        _page(3, 50, "L"),
        _page(4, 1, "1"),
        _page(5, 2, "2"),
        _page(6),
        _page(7, 10, "10"),
    ])

    assert _page_label_segments(document) == [
        (0, "/D", 1, "scan-"),
        (1, "/r", 48, None),
        (3, "/R", 50, None),
        (4, "/D", 1, None),
        (6, "/D", 7, "scan-"),
        (7, "/D", 10, None),
    ]


def test_pdf_navigation_uses_page_labels_and_right_facing_layout(tmp_path):
    pages = []
    originals = []
    for index in range(8):
        # Tall text regions establish vertical writing. Alternating folios put
        # even source pages on the left, which means right binding.
        regions = [
            Region(RegionKind.TEXT, Box(70 - n * 10, 10, 74 - n * 10, 90),
                   ocr_text=str(n))
            for n in range(5)
        ]
        pages.append(_page(
            index,
            number=index if index >= 1 else None,
            nombre=str(index) if index >= 1 else None,
            nombre_left=index % 2 == 0,
            regions=regions,
        ))
        originals.append(np.full((100, 100, 3), 255, dtype=np.uint8))
    document = Document("book.pdf", pages=pages)
    path = tmp_path / "navigation.pdf"
    pdf = pikepdf.new()
    for _ in pages:
        pdf.add_blank_page(page_size=(100, 100))
    pdf.save(path)

    _set_physical_page_size(
        str(path), 100, document=document, originals=originals
    )

    with pikepdf.open(path) as result:
        nums = list(result.Root.PageLabels.Nums)
        assert nums[0::2] == [0, 1]
        assert nums[1]["/P"] == "scan-"
        assert nums[3]["/S"] == "/D"
        assert nums[3]["/St"] == 1
        assert result.Root.ViewerPreferences.Direction == "/R2L"
        assert result.Root.PageLayout == "/TwoPageRight"
        assert tuple(int(part) for part in result.pdf_version.split(".")) >= (1, 5)


def test_structured_exports_preserve_region_order_and_escape_html():
    regions = [
        Region(RegionKind.TEXT, Box(50, 10, 90, 90), ocr_text="右<&>"),
        Region(RegionKind.TEXT, Box(10, 10, 40, 90), ocr_text="左"),
        Region(RegionKind.FIGURE, Box(10, 100, 90, 180), source="layout",
               layout_label="chart"),
    ]
    document = Document('book & "notes".pdf', pages=[
        _page(4, 1, "1", regions=regions),
    ])

    data = document_to_data(document)
    assert [item.get("text") for item in data["pages"][0]["elements"][:2]] == [
        "右<&>", "左",
    ]
    markdown = export_markdown(document)
    assert markdown.index("右<&>") < markdown.index("左")
    assert "image_layer_placeholder" not in markdown
    assert "> [chart] bbox=" in markdown
    output_html = export_html(document)
    assert "右&lt;&amp;&gt;" in output_html
    assert 'data-source="book &amp; &quot;notes&quot;.pdf"' in output_html
    assert json.loads(export_json(document)) == data


def test_export_cli_reads_stored_pages_and_infers_format(tmp_path, capsys):
    db = tmp_path / "book.db"
    store = Store(str(db))
    try:
        store.upsert_page("book.pdf", 0, _page(
            0, 1, "1", regions=[
                Region(RegionKind.TEXT, Box(0, 0, 10, 10), ocr_text="本文")
            ],
        ))
    finally:
        store.close()
    out = tmp_path / "book.md"

    assert main([
        "export", "--db", str(db), "--doc", "book.pdf", "--out", str(out)
    ]) == 0
    assert "本文" in out.read_text(encoding="utf-8")
    assert "exported 1 pages" in capsys.readouterr().out
