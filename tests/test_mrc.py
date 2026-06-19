import numpy as np
import pytest

pytest.importorskip("hybrid_ocr")
pytest.importorskip("pikepdf")

import pikepdf

from hokusai_press.mrc import (
    MrcPageBuilder,
    _LOCAL_BG_WINDOW_PX,
    _LOCAL_BG_WINDOW_SMALL_PX,
    _TINT_LO,
    _TINT_HI,
    _TINT_TEXT_DELTA,
    _tint_ink_mask,
)
import cv2


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


def _photo_colorspaces(pdf) -> list:
    """ColorSpace of every DCTDecode (JPEG) image in the document."""
    out = []
    for obj in pdf.objects:
        try:
            if (obj.get("/Subtype") == pikepdf.Name("/Image")
                    and "DCTDecode" in str(obj.get("/Filter"))):
                out.append(str(obj.get("/ColorSpace")))
        except (AttributeError, TypeError):
            continue
    return out


def test_mrc_photo_tone_override_forces_gray(tmp_path):
    # crop is genuinely colored (B,R differ) -> auto would pick color;
    # tone="gray" must force a DeviceGray overlay anyway.
    img, lines, photo_boxes = _page_with_photo()
    x0, y0, x1, y1 = photo_boxes[0]
    builder = MrcPageBuilder(compress="g4", target_dpi=600, photo_dpi=150)
    builder.add_page(img, lines, [(x0, y0, x1, y1, "gray")], mode="bw")
    out = tmp_path / "forced_gray.pdf"
    builder.save(str(out))
    with pikepdf.open(out) as pdf:
        spaces = _photo_colorspaces(pdf)
        assert spaces and all("DeviceGray" in s for s in spaces)


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


def test_tint_ink_mask_uses_local_background_for_mixed_tones():
    """A single TintZone can now span areas with genuinely different
    background tone (see tint_zone.py's whole-page redesign).  A single
    whole-zone median would judge the darker region against a threshold
    derived mostly from the lighter one and misclassify its natural scan
    grain as text; the local estimate must not.
    """
    rng = np.random.default_rng(1234)
    h, w = 180, 360
    split = int(w * 0.75)
    crop = np.empty((h, w), dtype=np.int16)
    crop[:, :split] = 190
    crop[:, split:] = 125
    crop += rng.integers(-10, 11, size=(h, w), dtype=np.int16)
    crop_gray = np.clip(crop, 0, 255).astype(np.uint8)

    tint_pix = crop_gray[(crop_gray >= _TINT_LO) & (crop_gray <= _TINT_HI)]
    zone_med = float(np.median(tint_pix))
    global_ink_t = max(_TINT_LO, int(round(zone_med)) - _TINT_TEXT_DELTA)
    old_global_ink = crop_gray < global_ink_t

    # The old single-threshold behavior is driven by the larger light panel and
    # randomly classifies the darker tint background as ink.
    assert old_global_ink[:, split:].mean() > 0.10

    local_ink = _tint_ink_mask(crop_gray)
    assert local_ink[:, :split].mean() == 0.0
    assert local_ink[:, split:].mean() == 0.0


def test_tint_ink_mask_median_avoids_closing_halo_around_small_glyphs():
    """A small standalone zone (e.g. a circular badge with its own small
    caption) must not misclassify the tint background around its own
    letterforms as ink.

    Morphological closing (the earlier approach) is biased toward the
    brighter extreme near any bright feature: dilating spreads a white
    knockout letter's brightness outward, and eroding needs an even wider
    dark margin to pull the estimate back down, so the background estimate
    right around the letter reads brighter than the true tint -- which
    inflates ink_T enough to draw a dark halo hugging every stroke.  A
    median has no such bias, so it shouldn't reproduce the halo even at the
    same (large) window the old closing-based code used.
    """
    crop_gray = np.full((90, 170), 255, dtype=np.uint8)
    y0, y1 = 23, 67
    crop_gray[y0:y1, 20:150] = 180  # the badge's own tint background
    for x in range(35, 135, 20):
        cv2.rectangle(crop_gray, (x, y0 + 8), (x + 3, y1 - 8), 85, -1)
        cv2.rectangle(crop_gray, (x + 8, y0 + 8), (x + 11, y1 - 8), 85, -1)
        cv2.rectangle(crop_gray, (x, y0 + 17), (x + 11, y0 + 21), 85, -1)
    assert max(crop_gray.shape) <= 600  # this is the "small zone" regime

    old_closing_k = 2 * _LOCAL_BG_WINDOW_PX + 1
    old_closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (old_closing_k, old_closing_k)
    )
    old_bg = cv2.morphologyEx(crop_gray, cv2.MORPH_CLOSE, old_closing_kernel)
    old_ink_t = np.maximum(_TINT_LO, old_bg.astype(np.float32) - _TINT_TEXT_DELTA)
    old_closing_ink = crop_gray.astype(np.float32) < old_ink_t

    # Probe only the tint background between/around the glyphs, not the
    # strokes themselves -- closing's halo shows up there as false "ink".
    tint_background = np.zeros_like(crop_gray, dtype=bool)
    tint_background[y0 + 8:y1 - 8, 25:145] = True
    tint_background &= crop_gray == 180
    assert old_closing_ink[tint_background].mean() > 0.90

    median_ink = _tint_ink_mask(crop_gray)
    assert median_ink[tint_background].mean() == 0.0
    assert _LOCAL_BG_WINDOW_SMALL_PX < _LOCAL_BG_WINDOW_PX
