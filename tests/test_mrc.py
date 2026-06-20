from io import BytesIO

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
    _base_edge_fringe_mask,
    _base_whiteout_mask,
    _clean_tint_overlay_edge,
    _rect_edge_dirt_mask,
    _tint_overlay_edge_dirt_mask,
    _tint_ink_mask,
    _try_posterized_tint_overlay_pdf,
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


def _image_streams(pdf):
    out = []
    for obj in pdf.objects:
        try:
            if obj.get("/Subtype") == pikepdf.Name("/Image"):
                out.append(obj)
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


def test_tint_ink_mask_vetoes_knockout_noise_but_keeps_real_strokes():
    """Dark clusters inside white knockout text are scan defects, not tint ink.

    The real 0423 p29 failure had 30-70px black blobs inside/next to white
    knockout glyphs.  A page-level connected-component despeckle cannot remove
    those safely, so the tint mask should reject them while it still preserves
    ordinary black strokes printed directly on the tint.
    """
    crop_gray = np.full((120, 260), 176, dtype=np.uint8)

    # White knockout glyph-like area with three dark source/noise clusters.
    cv2.rectangle(crop_gray, (30, 25), (145, 78), 245, -1)
    cv2.circle(crop_gray, (60, 50), 4, 0, -1)
    cv2.rectangle(crop_gray, (94, 38), (101, 45), 0, -1)
    cv2.ellipse(crop_gray, (126, 60), (5, 3), 0, 0, 360, 0, -1)

    # Genuine black marks printed on the tint, away from knockout white.
    cv2.rectangle(crop_gray, (180, 32), (188, 86), 55, -1)
    cv2.rectangle(crop_gray, (198, 32), (206, 86), 55, -1)
    cv2.rectangle(crop_gray, (180, 55), (206, 63), 55, -1)

    ink = _tint_ink_mask(crop_gray)

    knockout_noise = np.zeros_like(ink, dtype=bool)
    knockout_noise[20:84, 24:151] = True
    real_strokes = np.zeros_like(ink, dtype=bool)
    real_strokes[30:88, 176:210] = True

    assert int((ink & knockout_noise).sum()) == 0
    assert int((ink & real_strokes).sum()) > 250


def test_tint_ink_mask_excludes_ocr_knockout_text_box():
    """Searchable text can be white knockout artwork, not black bilevel ink."""
    crop_gray = np.full((100, 320), 150, dtype=np.uint8)

    # A title-like OCR text box: mostly tint background, visible white glyphs,
    # plus dark defects that would otherwise become black bilevel pixels.
    cv2.rectangle(crop_gray, (25, 28), (285, 72), 245, -1)
    cv2.circle(crop_gray, (80, 50), 6, 0, -1)
    cv2.rectangle(crop_gray, (170, 42), (188, 54), 0, -1)

    # Real black strokes elsewhere on the same tint zone should survive.
    cv2.rectangle(crop_gray, (290, 20), (300, 82), 45, -1)

    knockout_text = np.zeros_like(crop_gray, dtype=bool)
    knockout_text[24:76, 20:290] = True
    ink = _tint_ink_mask(crop_gray, knockout_text)

    assert int(ink[24:76, 20:290].sum()) == 0
    assert int(ink[18:84, 286:304].sum()) > 500


def test_tint_ink_mask_removes_tone_edge_halftone_cluster():
    """Connected halftone dots at an internal tint edge are not text ink."""
    crop_gray = np.full((120, 180), 184, dtype=np.uint8)
    crop_gray[:78, :128] = 156
    crop_gray[78:96, :128] = 205
    crop_gray[:, 128:] = 238

    # The real p29 failure was a compact 100px-ish cluster at the lower-right
    # corner of a dark tint block.  It had no dark text core; its darkest pixel
    # was only around 103.
    blob = np.array(
        [
            [0, 0], [1, 0], [2, 0], [3, 0],
            [-1, 1], [0, 1], [1, 1], [2, 1], [3, 1],
            [-3, 2], [-2, 2], [-1, 2], [0, 2], [1, 2],
            [-5, 3], [-4, 3], [-3, 3], [-2, 3], [-1, 3],
            [-7, 4], [-6, 4], [-5, 4], [-4, 4],
        ],
        dtype=np.int32,
    )
    for dx, dy in blob:
        cv2.circle(crop_gray, (123 + int(dx), 76 + int(dy)), 1, 112, -1)

    # A genuine black stroke nearby still has a dark core and should survive.
    cv2.rectangle(crop_gray, (45, 28), (53, 84), 55, -1)
    cv2.rectangle(crop_gray, (64, 28), (72, 84), 55, -1)
    cv2.rectangle(crop_gray, (45, 52), (72, 60), 55, -1)

    ink = _tint_ink_mask(crop_gray)

    edge_blob = np.zeros_like(ink, dtype=bool)
    edge_blob[70:86, 110:132] = True
    real_strokes = np.zeros_like(ink, dtype=bool)
    real_strokes[24:88, 41:76] = True

    assert int((ink & edge_blob).sum()) == 0
    assert int((ink & real_strokes).sum()) > 700


def test_base_whiteout_mask_expands_outer_edge_but_not_holes():
    """Tint edge cleanup should not become a new hole/body-text cleanup pass."""
    outer = np.zeros((80, 120), dtype=np.uint8)
    hole = np.zeros_like(outer)
    cv2.rectangle(outer, (20, 15), (100, 65), 255, -1)
    cv2.rectangle(hole, (45, 30), (75, 50), 255, -1)

    whiteout = _base_whiteout_mask(outer, hole)

    assert whiteout[14, 60]       # expanded just outside the outer top edge
    assert whiteout[66, 60]       # expanded just outside the outer bottom edge
    assert whiteout[40, 19]       # expanded just outside the outer left edge
    assert whiteout[40, 101]      # expanded just outside the outer right edge
    assert not whiteout[40, 60]   # original hole remains protected
    assert not whiteout[30, 60]   # hole's top edge is not whiteout-expanded
    assert not whiteout[50, 60]   # hole's bottom edge is not whiteout-expanded


def test_base_edge_fringe_mask_reaches_beyond_zone_rect_only_outside():
    cnt = np.array([[[20, 15]], [[100, 15]], [[100, 65]], [[20, 65]]], dtype=np.int32)
    hole = np.array([[[45, 30]], [[75, 30]], [[75, 50]], [[45, 50]]], dtype=np.int32)

    x0, y0, fringe = _base_edge_fringe_mask(cnt, [hole], 80, 120)

    assert fringe[15 - y0 - 1, 60 - x0]      # just above outer top
    assert fringe[65 - y0 + 1, 60 - x0]      # just below outer bottom
    assert fringe[40 - y0, 20 - x0 - 1]      # just left of outer edge
    assert fringe[40 - y0, 100 - x0 + 1]     # just right of outer edge
    assert not fringe[40 - y0, 60 - x0]      # hole stays untouched
    assert not fringe[20 - y0, 60 - x0]      # inside the tint shape is not fringe


def test_clean_tint_overlay_edge_replaces_boundary_dirt_only():
    overlay = np.full((80, 120), 170, dtype=np.uint8)
    outer = np.zeros_like(overlay)
    hole = np.zeros_like(overlay)
    cv2.rectangle(outer, (20, 15), (100, 65), 255, -1)

    overlay[58:64, 98:101] = 0   # dark boundary dirt touching right/bottom edge
    overlay[35:55, 55:60] = 0    # genuine interior dark stroke

    dirt = _tint_overlay_edge_dirt_mask(overlay, outer, hole)
    cleaned = _clean_tint_overlay_edge(overlay, dirt, 170)

    assert int(dirt[58:64, 98:101].sum()) == 18
    assert int(dirt[35:55, 55:60].sum()) == 0
    assert int((cleaned[58:64, 98:101] == 0).sum()) == 0
    assert int((cleaned[35:55, 55:60] == 0).sum()) == 100


def test_rect_edge_dirt_mask_marks_only_boundary_dark_pixels():
    gray = np.full((80, 120), 170, dtype=np.uint8)
    gray[70:75, 115:119] = 0  # boundary dirt
    gray[35:55, 55:60] = 0    # interior stroke

    dirt = _rect_edge_dirt_mask(gray)

    assert int(dirt[70:75, 115:119].sum()) == 20
    assert int(dirt[35:55, 55:60].sum()) == 0


def test_tint_vectorization_decision():
    from hokusai_press.tint_zone import TintZone, try_vectorize_zone

    # 1. 2色の単純なタイント画像 (ベクター化が成功すべき)
    img_simple = np.full((100, 100), 120, dtype=np.uint8)
    img_simple[30:70, 30:70] = 180

    zone_simple = TintZone(
        rect=(0, 0, 100, 100),
        overlay_img=img_simple.copy()
    )
    in_shape = np.ones((100, 100), dtype=bool)
    ink = np.zeros((100, 100), dtype=bool)

    success_simple = try_vectorize_zone(zone_simple, img_simple, in_shape, ink)
    assert success_simple is True
    assert zone_simple.vectorize_success is True
    assert len(zone_simple.vectorized_fills) > 0

    # 2. 多数の色でグラデーションがかかった複雑な画像 (JPEGフォールバックすべき)
    rng = np.random.default_rng(1234)
    img_complex = rng.integers(0, 221, size=(100, 100), dtype=np.uint8)

    zone_complex = TintZone(
        rect=(0, 0, 100, 100),
        overlay_img=img_complex.copy()
    )
    success_complex = try_vectorize_zone(zone_complex, img_complex, in_shape, ink)
    assert success_complex is False

    # 3. 2色でも塩こしょう状に断片化した網点ノイズは、画素誤差だけなら
    #    量子化できるがPDFパスが肥大化するのでJPEGフォールバックすべき。
    speckle = np.where(rng.random((100, 100)) < 0.5, 120, 180).astype(np.uint8)
    zone_speckle = TintZone(
        rect=(0, 0, 100, 100),
        overlay_img=speckle.copy()
    )
    success_speckle = try_vectorize_zone(zone_speckle, speckle, in_shape, ink)
    assert success_speckle is False


def test_tint_vectorization_flat_large_halftone_with_simple_knockout():
    from hokusai_press.tint_zone import TintZone, try_vectorize_zone

    yy, xx = np.indices((900, 900))
    halftone = np.where((xx + yy) % 2 == 0, 156, 184).astype(np.uint8)
    halftone[350:550, 360:540] = 255  # simple white knockout/paper cavity

    zone = TintZone(rect=(0, 0, 900, 900), overlay_img=halftone.copy())
    in_shape = np.ones_like(halftone, dtype=bool)
    ink = np.zeros_like(halftone, dtype=bool)

    success = try_vectorize_zone(zone, halftone, in_shape, ink)

    assert success is True
    assert zone.vectorize_success is True
    assert len(zone.vectorized_fills) <= 4
    assert any(fill["holes"] for fill in zone.vectorized_fills)


def test_tint_vectorization_uses_holes_for_protected_text_pixels():
    from hokusai_press.tint_zone import TintZone, try_vectorize_zone

    yy, xx = np.indices((900, 900))
    halftone = np.where((xx + yy) % 2 == 0, 150, 180).astype(np.uint8)
    protect = np.zeros_like(halftone, dtype=bool)
    protect[360:430, 260:640] = True

    zone = TintZone(rect=(0, 0, 900, 900), overlay_img=halftone.copy())
    in_shape = np.ones_like(halftone, dtype=bool)
    ink = np.zeros_like(halftone, dtype=bool)

    success = try_vectorize_zone(zone, halftone, in_shape, ink, protect)

    assert success is True
    assert not zone.vectorized_patches
    assert any(fill["holes"] for fill in zone.vectorized_fills)


def test_mrc_keeps_tints_as_raster_overlays(tmp_path):
    from hokusai_press.tint_zone import TintZone

    # 2色のタイント画像
    img_bgr = np.full((200, 200, 3), 255, dtype=np.uint8)
    img_bgr[20:180, 20:180] = 150
    img_bgr[50:150, 50:150] = 180

    # 文字を追加
    img_bgr[95:105, 40:160] = 10

    zone = TintZone(
        rect=(20, 20, 180, 180),
        overlay_img=cv2.cvtColor(img_bgr[20:180, 20:180], cv2.COLOR_BGR2GRAY)
    )
    zone.page_contour = np.array([[[20, 20]], [[180, 20]], [[180, 180]], [[20, 180]]], dtype=np.int32)

    builder = MrcPageBuilder(compress="g4", target_dpi=600)
    builder.add_page(img_bgr, [], [], mode="bw", tint_zones=[zone])

    assert zone.vectorize_success is False
    assert len(zone.vectorized_fills) == 0

    out = tmp_path / "raster_tint.pdf"
    builder.save(str(out))

    with pikepdf.open(out) as pdf:
        assert len(pdf.pages) == 1
        joined = _all_filters(pdf)
        assert "DCTDecode" in joined or "FlateDecode" in joined
        assert "CCITTFaxDecode" in joined


def test_tint_posterized_flate_beats_jpeg_on_flat_halftone():
    from hybrid_ocr.pdf_export import encode_page_pdf

    rng = np.random.default_rng(1234)
    h, w = 720, 720
    yy, xx = np.indices((h, w))
    base = np.full((h, w), 176, dtype=np.int16)
    base[:, w // 2:] = 148
    grain = np.where((xx + yy) % 2 == 0, -5, 5)
    noise = rng.integers(-2, 3, size=(h, w), dtype=np.int16)
    overlay = np.clip(base + grain + noise, 0, 255).astype(np.uint8)

    in_shape = np.ones_like(overlay, dtype=bool)
    ink = np.zeros_like(overlay, dtype=bool)
    result = _try_posterized_tint_overlay_pdf(overlay, in_shape, ink)

    assert result is not None
    flate_pdf, stats = result
    jpg_pdf = encode_page_pdf(cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR), "gray", "g4")
    assert stats["codec"] == "k4_flate"
    assert len(flate_pdf) < len(jpg_pdf)

    with pikepdf.open(BytesIO(flate_pdf)) as pdf:
        filters = _all_filters(pdf)
        assert "FlateDecode" in filters
        assert "DCTDecode" not in filters


def test_tint_posterize_skips_photo_like_overlay():
    h, w = 260, 360
    x = np.linspace(40, 225, w, dtype=np.float32)
    y = np.linspace(0, 80, h, dtype=np.float32)[:, None]
    overlay = np.clip(x + y + 25 * np.sin(np.arange(w) / 9.0), 0, 255)
    overlay = np.tile(overlay.astype(np.uint8), (h, 1)) if overlay.ndim == 1 else overlay.astype(np.uint8)

    from hokusai_press.region_class import classify_patch

    assert classify_patch(overlay) == "photo"
    result = _try_posterized_tint_overlay_pdf(
        overlay,
        np.ones_like(overlay, dtype=bool),
        np.zeros_like(overlay, dtype=bool),
    )

    assert result is None


def test_tint_posterize_preserves_ink_and_paper_pixels():
    overlay = np.full((180, 220), 170, dtype=np.uint8)
    overlay[:, 110:] = 145
    overlay[35:65, 30:90] = 255      # paper / knockout area
    overlay[92:106, 25:190] = 20     # black text stroke in source overlay

    in_shape = np.ones_like(overlay, dtype=bool)
    ink = np.zeros_like(overlay, dtype=bool)
    ink[92:106, 25:190] = True

    result = _try_posterized_tint_overlay_pdf(overlay, in_shape, ink)
    assert result is not None
    flate_pdf, _ = result

    with pikepdf.open(BytesIO(flate_pdf)) as pdf:
        images = _image_streams(pdf)
        assert len(images) == 1
        decoded = np.frombuffer(images[0].read_bytes(), dtype=np.uint8).reshape(overlay.shape)

    assert np.array_equal(decoded[92:106, 25:190], overlay[92:106, 25:190])
    assert int((decoded[35:65, 30:90] == 255).sum()) == 30 * 60
