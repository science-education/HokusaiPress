import numpy as np
import pytest

pytest.importorskip("hybrid_ocr")
pytest.importorskip("pikepdf")

import pikepdf

from hokusai_press.mrc import MrcPageBuilder


def _all_filters(pdf) -> str:
    """All image-stream filters in the document (overlays nest images inside
    Form XObjects, so page.images alone misses them)."""
    out = []
    for obj in pdf.objects:
        try:
            if obj.get("/Subtype") == pikepdf.Name("/Image"):
                out.append(str(obj.get("/Filter")))
        except (AttributeError, TypeError):
            continue
    return " ".join(out)


def _page_with_photo():
    img = np.full((600, 400, 3), 255, dtype=np.uint8)
    for y in range(40, 300, 30):           # text bars (top half)
        img[y:y + 12, 40:360] = 0
    # continuous-tone gradient block (bottom half) = a "photo"
    grad = np.tile(np.linspace(0, 255, 320, dtype=np.uint8), (220, 1))
    img[340:560, 40:360] = np.dstack([grad, grad, (grad * 0.6).astype(np.uint8)])
    photo_box = (40, 340, 360, 560)
    lines = [{
        "box": [40, 40, 360, 52],
        "polygon": [[40, 40], [360, 40], [360, 52], [40, 52]],
        "text": "テスト", "direction": "h",
    }]
    return img, lines, [photo_box]


def test_mrc_page_has_bilevel_and_photo_layers(tmp_path):
    img, lines, photo_boxes = _page_with_photo()
    builder = MrcPageBuilder(compress="g4", target_dpi=600, photo_dpi=150)
    builder.add_page(img, lines, photo_boxes, mode="bw")
    out = tmp_path / "mrc.pdf"
    builder.save(str(out))

    with pikepdf.open(out) as pdf:
        assert len(pdf.pages) == 1
        joined = _all_filters(pdf)
        assert "CCITTFaxDecode" in joined   # crisp bilevel text base
        assert "DCTDecode" in joined        # photo overlay as JPEG
        # searchable text layer present
        forms = [x for x in pdf.pages[0].Resources.XObject.values()
                 if x.get("/Subtype") == pikepdf.Name("/Form")]
        assert len(forms) >= 1


def test_mrc_text_only_page_is_pure_bilevel(tmp_path):
    img, lines, _ = _page_with_photo()
    builder = MrcPageBuilder(compress="g4", target_dpi=600)
    builder.add_page(img, lines, [], mode="bw")   # no photo boxes
    out = tmp_path / "txt.pdf"
    builder.save(str(out))
    with pikepdf.open(out) as pdf:
        joined = _all_filters(pdf)
        assert "CCITTFaxDecode" in joined
        assert "DCTDecode" not in joined
