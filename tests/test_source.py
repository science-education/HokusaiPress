import numpy as np
import pytest

pytest.importorskip("pypdfium2")
pytest.importorskip("PIL")


def _make_pdf(path, w=300, h=400):
    from PIL import Image

    arr = np.full((h, w, 3), 240, dtype=np.uint8)
    arr[40:80, 30:270] = 0  # a black bar so the page isn't blank
    Image.fromarray(arr).save(str(path), "PDF", resolution=150.0)
    return str(path)


def test_load_single_extracts_embedded_image(tmp_path):
    from hokusai_press.source import load_single

    pdf = _make_pdf(tmp_path / "scan.pdf")
    original, dpi = load_single(pdf, 0)
    assert original.ndim == 3 and original.shape[2] == 3
    assert original.shape[0] > 0 and original.shape[1] > 0


def test_load_single_is_thread_safe(tmp_path):
    # pdfium is not thread-safe; without the module lock, concurrent load_single
    # calls raise "Data format error" / "Failed to load page" or crash the
    # process. Hammer it and assert every call returns a well-formed image.
    import threading

    from hokusai_press.source import load_single

    pdf = _make_pdf(tmp_path / "scan.pdf")
    errors: list[Exception] = []
    shapes: list[tuple] = []

    def worker():
        try:
            for _ in range(15):
                original, _ = load_single(pdf, 0)
                assert original.ndim == 3
                shapes.append(original.shape)
        except Exception as e:  # pragma: no cover - only on regression
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[0]
    assert len(set(shapes)) == 1  # every extraction identical


def test_render_and_embedded_paths_return_same_bgr_colors(tmp_path):
    """C10: PDFium rasterization and direct embedded-image extraction must
    both return true BGR (red = [0, 0, 255], blue = [255, 0, 0])."""
    import cv2
    from PIL import Image
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas

    from hokusai_press.source import load_page

    # 1) Embedded full-page image: left half pure red, right half pure blue (in RGB)
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    rgb[:, :100] = [255, 0, 0]  # Red
    rgb[:, 100:] = [0, 0, 255]  # Blue
    emb_pdf = tmp_path / "embedded_colors.pdf"
    c_emb = Canvas(str(emb_pdf), pagesize=(72, 72))
    c_emb.drawImage(ImageReader(Image.fromarray(rgb)), 0, 0, 72, 72)
    c_emb.showPage()
    c_emb.save()

    # 2) Vector-drawn page (forces PDFium render path): left half red, right half blue
    vec_pdf = tmp_path / "vector_colors.pdf"
    c_vec = Canvas(str(vec_pdf), pagesize=(72, 72))
    c_vec.setFillColorRGB(1, 0, 0)
    c_vec.rect(0, 0, 36, 72, fill=1, stroke=0)
    c_vec.setFillColorRGB(0, 0, 1)
    c_vec.rect(36, 0, 36, 72, fill=1, stroke=0)
    c_vec.showPage()
    c_vec.save()

    (emb_ref, emb_bgr, _, _) = next(load_page(str(emb_pdf)))
    (vec_ref, vec_bgr, _, _) = next(load_page(str(vec_pdf)))

    assert emb_ref.embedded_image_id == "img0"
    assert vec_ref.embedded_image_id is None

    eh, ew = emb_bgr.shape[:2]
    vh, vw = vec_bgr.shape[:2]
    emb_red = emb_bgr[eh // 2, ew // 4].tolist()
    emb_blue = emb_bgr[eh // 2, (3 * ew) // 4].tolist()
    vec_red = vec_bgr[vh // 2, vw // 4].tolist()
    vec_blue = vec_bgr[vh // 2, (3 * vw) // 4].tolist()

    assert emb_red == [0, 0, 255]
    assert emb_blue == [255, 0, 0]
    assert vec_red == emb_red
    assert vec_blue == emb_blue
    # Grayscale conversion luminance must also match (red ~76, blue ~29)
    assert int(cv2.cvtColor(vec_bgr, cv2.COLOR_BGR2GRAY)[vh // 2, vw // 4]) == 76
    assert int(cv2.cvtColor(vec_bgr, cv2.COLOR_BGR2GRAY)[vh // 2, (3 * vw) // 4]) == 29


def test_c5_vector_overlay_and_partial_image_fall_back_to_render(tmp_path):
    """C5 (a) & (b): (a) a full-page image with a visible vector shape on top,
    and (b) an image covering only part of the page must both be rasterized as
    full pages so the vector overlay and surrounding margins are preserved."""
    from PIL import Image
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas
    import pypdfium2 as pdfium

    from hokusai_press.source import OCR_DPI, load_page

    pdf_path = tmp_path / "mixed.pdf"
    c = Canvas(str(pdf_path), pagesize=(72, 72))

    # Page 0 (a): full-page white image + black vector square in center
    white_im = np.full((20, 20, 3), 255, dtype=np.uint8)
    c.drawImage(ImageReader(Image.fromarray(white_im)), 0, 0, 72, 72)
    c.setFillColorRGB(0, 0, 0)
    c.rect(27, 27, 18, 18, stroke=0, fill=1)
    c.showPage()

    # Page 1 (b): small black image placed only in center (27..45 pt on 72x72 pt page)
    black_im = np.zeros((20, 20, 3), dtype=np.uint8)
    c.drawImage(ImageReader(Image.fromarray(black_im)), 27, 27, 18, 18)
    c.showPage()
    c.save()

    loaded = list(load_page(str(pdf_path)))
    assert len(loaded) == 2

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        for idx, (ref, img_bgr, _, _) in enumerate(loaded):
            assert ref.embedded_image_id is None
            expected = np.ascontiguousarray(
                doc[idx].render(scale=OCR_DPI / 72, rev_byteorder=False).to_numpy()[..., :3]
            )
            assert img_bgr.shape == expected.shape
            assert np.array_equal(img_bgr, expected)
            # Both pages have a black center square and white outer margins
            h, w = img_bgr.shape[:2]
            assert np.all(img_bgr[h // 2, w // 2] == 0)
            assert np.all(img_bgr[5, 5] == 255)
    finally:
        doc.close()


def test_c5_full_page_image_with_invisible_ocr_text_extracts_directly(tmp_path):
    """C5: A full-page scan PDF with an invisible OCR text layer (render mode 3)
    must still take the direct embedded-image extraction path."""
    from hokusai_press import textlayer
    from hokusai_press.source import load_page

    raw_pdf = _make_pdf(tmp_path / "raw_scan.pdf", w=300, h=400)
    ocr_pdf = tmp_path / "ocr_scan.pdf"
    textlayer.overlay_text(
        raw_pdf,
        [(300, 400, [{"box": (30, 40, 270, 80), "text": "検索用テキスト"}])],
        str(ocr_pdf),
    )

    ref, orig_bgr, _, _ = next(load_page(str(ocr_pdf)))
    assert ref.embedded_image_id == "img0"
    assert orig_bgr.shape == (400, 300, 3)

