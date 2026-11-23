"""Text layer: per-character fonts, overlay on existing PDFs, stripping, and
the `ocr` re-OCR path (with a fake engine, no models needed)."""

import hashlib
from io import BytesIO

import numpy as np
import pytest

img2pdf = pytest.importorskip("img2pdf")
pikepdf = pytest.importorskip("pikepdf")
pdfium = pytest.importorskip("pypdfium2")
from PIL import Image  # noqa: E402

from hokusai_press import textlayer  # noqa: E402

W, H, DPI = 1200, 1800, 300


def _scan_pdf(path):
    img = Image.new("1", (W, H), 1)
    a = np.ones((H, W), bool)
    a[200:1600, 900:960] = False  # a fake vertical text column
    img = Image.fromarray(a)
    buf = BytesIO()
    img.save(buf, format="TIFF", compression="group4", tiffinfo={278: H})
    path.write_bytes(img2pdf.convert(
        buf.getvalue(), layout_fun=img2pdf.get_fixed_dpi_layout_fun((DPI, DPI))))


def _image_hashes(path):
    with pikepdf.open(path) as pdf:
        return [hashlib.md5(xo.read_raw_bytes()).hexdigest()
                for page in pdf.pages for _, xo in page.images.items()]


def _text(path, i=0):
    doc = pdfium.PdfDocument(str(path))
    try:
        tp = doc[i].get_textpage()
        return tp.get_text_bounded().replace("\r", "").replace("\n", "")
    finally:
        doc.close()


def test_overlay_keeps_images_and_adds_text(tmp_path):
    src, out = tmp_path / "scan.pdf", tmp_path / "out.pdf"
    _scan_pdf(src)
    textlayer.overlay_text(str(src), [(W, H, [
        {"box": (900, 200, 960, 1600), "text": "縦書きの本文"},
        {"box": (100, 1700, 700, 1760), "text": "横書き Header"},
    ])], str(out))
    assert _image_hashes(src) == _image_hashes(out)
    text = _text(out)
    assert "縦書きの本文" in text
    assert "横書きHeader" in text.replace(" ", "")


def test_vertical_characters_sit_in_their_column(tmp_path):
    src, out = tmp_path / "scan.pdf", tmp_path / "out.pdf"
    _scan_pdf(src)
    textlayer.overlay_text(str(src), [(W, H, [
        {"box": (900, 200, 960, 1600), "text": "あいうえお"}])], str(out))
    doc = pdfium.PdfDocument(str(out))
    tp = doc[0].get_textpage()
    s = 72 / DPI
    for i in range(tp.count_chars()):
        l, b, r, t = tp.get_charbox(i)
        cx, cy = (l + r) / 2, H * s - (b + t) / 2
        assert 900 * s - 2 <= cx <= 960 * s + 2
        assert 200 * s - 2 <= cy <= 1600 * s + 2
    doc.close()


def test_strip_and_image_only_copy(tmp_path):
    src, first, second, bare = (tmp_path / n for n in ("s.pdf", "1.pdf", "2.pdf", "b.pdf"))
    _scan_pdf(src)
    textlayer.overlay_text(str(src), [(W, H, [
        {"box": (900, 200, 960, 1600), "text": "古い層"}])], str(first))
    textlayer.overlay_text(str(first), [(W, H, [
        {"box": (900, 200, 960, 1600), "text": "新しい層"}])], str(second))
    assert "新しい層" in _text(second) and "古い層" not in _text(second)
    textlayer.image_only_copy(str(second), str(bare))
    assert _text(bare) == ""
    assert _image_hashes(src) == _image_hashes(bare)


def test_compat_ideograph_is_folded_and_searchable(tmp_path):
    out = textlayer.draw_text_pages([(600, 900, [
        {"box": (100, 100, 400, 150), "text": "研究者", "direction": "h"}])])
    p = tmp_path / "t.pdf"
    p.write_bytes(out)
    assert "研究者" in _text(p)


def test_draw_text_pages_without_text_has_blank_pages():
    out = textlayer.draw_text_pages([(600, 900, []), (600, 900, [{"box": (0, 0, 1, 1), "text": " "}])])
    with pikepdf.open(BytesIO(out)) as pdf:
        assert len(pdf.pages) == 2


def test_refuses_to_overwrite_input(tmp_path):
    src = tmp_path / "scan.pdf"
    _scan_pdf(src)
    with pytest.raises(ValueError):
        textlayer.overlay_text(str(src), [(W, H, [])], str(src))


def test_reocr_uses_engine_boxes_at_render_resolution(tmp_path):
    from hokusai_press.reocr import ocr_pdf

    src, out = tmp_path / "scan.pdf", tmp_path / "out.pdf"
    _scan_pdf(src)
    seen = []

    def fake_engine(img):
        seen.append(img.shape)
        h, w = img.shape[:2]
        # the column sits at x 900-960 of a 1200 px wide page
        return {"lines": [{"box": (900 * w / W, 200 * h / H, 960 * w / W, 1600 * h / H),
                           "text": "再OCRの本文"}]}

    r = ocr_pdf(str(src), str(out), fake_engine, dpi=150, text_out=str(tmp_path / "o.txt"))
    assert r["pages"] == 1 and r["chars"] == 7
    assert abs(seen[0][0] - 900) <= 1 and abs(seen[0][1] - 600) <= 1  # 300 -> 150 dpi
    assert "再OCRの本文" in _text(out)
    assert _image_hashes(src) == _image_hashes(out)
    assert (tmp_path / "o.txt").read_text(encoding="utf-8") == "再OCRの本文"


def test_margin_mode_none_keeps_the_whole_page():
    import numpy as np
    from hokusai_press.model import Box, Deskew, Margin, PageParams, RenderSettings, SourceRef
    from hokusai_press.render import compose_transform, render_page_image

    orig = np.full((1000, 700, 3), 255, np.uint8)
    orig[20:40, 20:200] = 0          # print near the corner, outside the content box
    orig[400:600, 200:500] = 0       # the "content"
    p = PageParams(source=SourceRef(path="x", page_index=0), dpi=300,
                   deskew=Deskew(0.0, 1.0),
                   margin=Margin(content=Box(200, 400, 500, 600), confidence=1.0))
    s = RenderSettings(target_dpi=300, margin_mode="none")
    _, size, _ = compose_transform(orig.shape, p, s)
    assert size == (700, 1000)
    out, _, _, _ = render_page_image(orig, p, s)
    assert out[30, 100].max() < 50   # corner print survives: no crop, no margin fill
    s2 = RenderSettings(target_dpi=300)
    out2, _, _, _ = render_page_image(orig, p, s2)
    assert out2.shape[:2] != (1000, 700) or out2[30, 100].min() > 200


@pytest.mark.parametrize("inherited", [False, True])
def test_reocr_rejects_rotated_pdf_before_running_ocr(tmp_path, inherited):
    """A9: ocr_pdf must detect /Rotate (including inherited /Pages /Rotate)
    before running OCR on any page."""
    from hokusai_press.reocr import ocr_pdf

    src, rotated, out = tmp_path / "scan.pdf", tmp_path / "rot.pdf", tmp_path / "out.pdf"
    _scan_pdf(src)
    with pikepdf.open(src) as pdf:
        if inherited:
            pdf.Root.Pages["/Rotate"] = 90
        else:
            pdf.pages[0].obj["/Rotate"] = 90
        pdf.save(rotated)

    calls = 0

    def counting_engine(_img):
        nonlocal calls
        calls += 1
        return {"lines": []}

    with pytest.raises(NotImplementedError, match="/Rotate 90"):
        ocr_pdf(str(rotated), str(out), counting_engine)
    assert calls == 0


def test_overlay_text_resolves_inherited_cropbox(tmp_path):
    """A9: overlay_text uses pikepdf's page.cropbox so inherited /CropBox works."""
    src, cropped, out = tmp_path / "scan.pdf", tmp_path / "crop.pdf", tmp_path / "out.pdf"
    _scan_pdf(src)
    with pikepdf.open(src) as pdf:
        mb = [float(v) for v in pdf.pages[0].mediabox]
        # Inherit a half-height CropBox from /Pages
        pdf.Root.Pages["/CropBox"] = pikepdf.Array([mb[0], mb[1], mb[2], mb[3] / 2])
        pdf.save(cropped)

    textlayer.overlay_text(
        str(cropped),
        [(W, H // 2, [{"box": (100, 50, 500, 120), "text": "継承CropBox", "direction": "h"}])],
        str(out),
    )
    assert "継承CropBox" in _text(out)

