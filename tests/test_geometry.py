import cv2
import numpy as np

from hokusai_press.geometry.deskew import GOOD_CONFIDENCE, find_skew, is_confident
from hokusai_press.geometry.margin import (
    find_content_box, remove_edge_shadows, detect_shadow_bands, apply_shadow_bands,
    region_shadow_mask, remove_region_shadows,
)
from hokusai_press.model import Deskew, Region, RegionKind, Box


def _text_page(rotate_deg=0.0):
    img = np.full((800, 600, 3), 255, dtype=np.uint8)
    for y in range(120, 700, 44):              # horizontal "text lines"
        cv2.rectangle(img, (90, y), (510, y + 16), (0, 0, 0), -1)
    if abs(rotate_deg) > 1e-6:
        m = cv2.getRotationMatrix2D((300, 400), rotate_deg, 1.0)
        img = cv2.warpAffine(img, m, (600, 800), borderValue=(255, 255, 255))
    return img


def test_deskew_recovers_known_angle():
    skewed = _text_page(rotate_deg=2.0)  # CCW rotation by 2 deg
    sk = find_skew(skewed)
    # the correcting angle undoes the introduced rotation
    assert abs(sk.angle_deg + 2.0) < 0.5
    assert sk.confidence >= GOOD_CONFIDENCE
    assert is_confident(sk)


def test_deskew_straight_page_is_near_zero():
    sk = find_skew(_text_page(0.0))
    assert abs(sk.angle_deg) < 0.3


def test_blank_page_not_deskew_flagged():
    # A blank page has no skew to correct (angle 0); its dull projection yields
    # zero confidence, but since nothing is rotated it needs no deskew review --
    # blank leaves are handled by the blank / no-text mechanism, not here.
    blank = np.full((400, 300, 3), 255, dtype=np.uint8)
    sk = find_skew(blank)
    assert sk.confidence == 0.0
    assert sk.angle_deg == 0.0
    assert is_confident(sk)  # upright -> not flagged for deskew review


def test_figure_plus_text_is_confident():
    # the real-world false positive: a big solid figure on top half + text lines
    # on the bottom. The text peaks sharply at 0 deg, so the page must be trusted
    # even though the figure adds a flat baseline to the projection.
    img = np.full((800, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (80, 40), (520, 380), (0, 0, 0), -1)   # figure block
    for y in range(430, 760, 36):                              # text lines below
        cv2.rectangle(img, (90, y), (510, y + 14), (0, 0, 0), -1)
    sk = find_skew(img)
    assert abs(sk.angle_deg) < 0.3
    assert is_confident(sk)            # was flagged low-confidence before the fix


def test_structureless_page_not_confident():
    # random speckle has no line structure -> deskew is unreliable -> flag it
    rng = np.random.default_rng(0)
    noise = np.where(rng.random((1000, 800, 1)) < 0.15, 0, 255).astype(np.uint8)
    noise = np.repeat(noise, 3, axis=2)
    sk = find_skew(noise)
    assert not is_confident(sk)


def test_content_box_excludes_border_noise():
    img = np.full((800, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (100, 150), (500, 650), (0, 0, 0), 3)  # body frame
    img[0:800, 0:4] = 0   # left scan-edge streak (touches border -> ignored)
    mg = find_content_box(img, Deskew())
    assert mg.content.x0 >= 90
    assert mg.content.x1 <= 510
    assert mg.content.y0 >= 140
    assert mg.confidence > 0


def test_content_box_excludes_inset_binding_shadow():
    # a near-solid dark bar a few px INSIDE the right edge (binding/ADF shadow,
    # not touching the border) must not blow the content box out to full width.
    img = np.full((800, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (100, 150), (480, 650), (0, 0, 0), 3)   # real body frame
    img[0:800, 585:590] = 0   # inset vertical shadow bar near the right edge
    mg = find_content_box(img, Deskew())
    assert mg.content.x1 <= 500          # shadow bar excluded, not at ~589
    assert mg.content.width < 600 * 0.85  # not full-bleed


def test_remove_edge_shadows_keeps_text_dots():
    # sparse text near the edge = many small components (kept); a solid bar = one
    # tall component (whitened). Gray image: 0 = ink, 255 = background.
    img = np.full((400, 300), 255, dtype=np.uint8)
    img[::3, 20:40] = 0           # sparse text dots near the left edge
    img[:, 290:294] = 0           # solid shadow bar at the right edge
    out = remove_edge_shadows(img)
    assert (out[::3, 20:40] == 0).any()     # text kept
    assert (out[:, 290:294] == 255).all()   # shadow removed


def _book_pages(n, with_left_bar=True, bar_x=8, running_head=False, jitter=True):
    """n synthetic yokogaki pages (HxW = 300x400). Optional fixed left shadow
    bar (outer band), optional recurring running head, body ink that wanders."""
    rng = np.random.default_rng(0)
    pages = []
    for i in range(n):
        g = np.full((300, 400), 255, dtype=np.uint8)
        bx = 60 + (rng.integers(-20, 20) if jitter else 0)   # body left edge wanders
        for y in range(40, 270, 18):                          # body text lines
            g[y:y+8, bx:bx+260] = 0
        if with_left_bar:
            g[:, bar_x:bar_x+3] = 40                          # fixed thin edge shadow
        if running_head:
            g[18:22, 70:330] = 0                              # wide line at constant y
        pages.append(g)
    return pages


def test_detect_shadow_bands_finds_consistent_edge_line():
    pages = _book_pages(12, with_left_bar=True, bar_x=8)
    bands = detect_shadow_bands(pages)
    vbands = [b for b in bands if b[0] == 0]
    assert vbands, "should detect the fixed left edge bar"
    # the band covers x=8 (=0.02 of W=400)
    assert any(lo <= 8/400 <= hi for _, lo, hi in vbands)
    # applying it whitens the bar but keeps body ink
    out = apply_shadow_bands(pages[0].copy(), bands)
    assert (out[:, 8:11] == 255).all()                # bar removed
    assert (out[:, 60:320] < 128).any()               # body text kept


def test_detect_shadow_bands_ignores_wandering_content():
    # no fixed edge bar; body ink wanders -> no consistent column -> no band
    pages = _book_pages(12, with_left_bar=False, jitter=True)
    bands = detect_shadow_bands(pages)
    assert [b for b in bands if b[0] == 0] == []


def test_detect_shadow_bands_ignores_running_head():
    # a recurring horizontal running head must NOT be detected (vertical-only),
    # else it (and the nombre) would be whitened
    pages = _book_pages(12, with_left_bar=False, running_head=True, jitter=True)
    bands = detect_shadow_bands(pages)
    assert bands == []


def test_detect_shadow_bands_needs_minimum_pages():
    pages = _book_pages(4, with_left_bar=True)
    assert detect_shadow_bands(pages) == []


def test_ink_threshold_book_valley_handles_showthrough():
    from hokusai_press.geometry.margin import ink_threshold, document_ink_valley
    # a normal inked page: clear dark text on white -> Otsu valley in the middle
    normal = np.full((200, 200), 255, dtype=np.uint8)
    normal[40:160, 40:160] = 30
    valley = document_ink_valley([normal] * 10)
    assert valley is not None
    # the same normal page keeps its own Otsu (not the floor)
    assert ink_threshold(normal, valley) == ink_threshold(normal, None)
    # a show-through page: only faint gray bleed, no real dark ink -> its own
    # Otsu shoots high; with the book valley it must fall back to the valley
    show = np.full((200, 200), 245, dtype=np.uint8)
    show[::4, ::4] = 205     # faint bleed pattern, nothing genuinely dark
    with_valley = ink_threshold(show, valley)
    # show-through Otsu is well above the book valley -> clamped to it
    assert with_valley <= valley + 1


def test_flag_skew_outliers_flags_large_book_relative_outlier():
    from hokusai_press.geometry.deskew import flag_skew_outliers
    # a mostly-straight book (angles ~0) with two genuinely skewed pages
    ds = [Deskew(angle_deg=a, confidence=5.0) for a in
          ([0.0, 0.1, -0.1, 0.05, 0.0, 0.1, -0.05, 0.0, 0.1, -0.1] + [0.8, -0.9])]
    flags = flag_skew_outliers(ds)
    assert flags[-1] and flags[-2]              # the 0.8 / -0.9 pages flagged
    assert not any(flags[:10])                  # the straight bulk not flagged


def test_flag_skew_outliers_spares_consistent_small_skew():
    from hokusai_press.geometry.deskew import flag_skew_outliers
    # every page has a small ~0.3deg skew (consistent) -> none is an outlier,
    # and all are below the absolute-review floor anyway
    ds = [Deskew(angle_deg=0.3, confidence=1.0) for _ in range(12)]
    assert not any(flag_skew_outliers(ds))


def test_flag_skew_outliers_small_book_falls_back_to_confidence():
    from hokusai_press.geometry.deskew import flag_skew_outliers, GOOD_CONFIDENCE
    # < SKEW_MIN_PAGES -> per-page rule: a tilted low-confidence page is flagged
    ds = [Deskew(angle_deg=1.0, confidence=GOOD_CONFIDENCE - 0.5)]
    assert flag_skew_outliers(ds) == [True]
    ds2 = [Deskew(angle_deg=0.0, confidence=0.0)]   # upright -> not flagged
    assert flag_skew_outliers(ds2) == [False]


def _txt(x0, y0, x1, y1):
    return Region(kind=RegionKind.TEXT, box=Box(x0, y0, x1, y1))


def test_region_shadow_removed_outside_text_box():
    # A thin near-full-height dark line in the left margin (outside the text box)
    # is a binding/ADF shadow -> removed. The body text is kept.
    g = np.full((400, 300), 255, dtype=np.uint8)
    g[:, 8:10] = 30                       # thin tall shadow line at x=8 (margin)
    for y in range(40, 360, 20):          # body text lines, left edge at x=80
        g[y:y+8, 80:250] = 0
    regions = [_txt(80, 40, 250, 360)]
    out = remove_region_shadows(g, regions)
    assert (out[:, 8:10] == 255).all()    # shadow line removed
    assert (out[:, 80:250] < 128).any()   # body text kept


def test_region_shadow_does_not_touch_figure():
    # A figure that bleeds into the left margin must be protected: a dark line
    # inside the figure's box is NOT whitened, even though it is left of the text.
    g = np.full((400, 300), 255, dtype=np.uint8)
    g[50:350, 5:120] = 60                 # a figure block reaching the left edge
    for y in range(40, 360, 20):
        g[y:y+8, 150:260] = 0             # text to the right of the figure
    regions = [
        _txt(150, 40, 260, 360),
        Region(kind=RegionKind.PHOTO, box=Box(5, 50, 120, 350)),
    ]
    out = remove_region_shadows(g, regions)
    # the figure interior is untouched (not whitened to 255)
    assert (out[60:340, 10:110] < 255).any()


def test_region_shadow_keeps_margin_text():
    # Sparse marginal text (no >=25%-tall consecutive run) in the margin must NOT
    # be removed -- only a solid tall line qualifies as a shadow.
    g = np.full((400, 300), 255, dtype=np.uint8)
    for y in range(40, 360, 40):          # sparse dots in the left margin
        g[y:y+6, 12:20] = 0
    for y in range(40, 360, 20):
        g[y:y+8, 80:250] = 0
    regions = [_txt(80, 40, 250, 360)]
    out = remove_region_shadows(g, regions)
    assert (out[:, 12:20] < 128).any()    # sparse margin marks kept (not a line)


def test_region_shadow_keeps_glyph_crossing_tight_text_boundary():
    # A vertical glyph can extend beyond the OCR box. Looking only outside the
    # box turns that clipped-off sliver into a tall/thin false shadow. Its
    # connection to ink inside the box must protect the whole glyph, while a
    # separate real margin line is still removed.
    g = np.full((400, 300), 255, dtype=np.uint8)
    g[80:320, 235:260] = 0                # glyph crosses right OCR edge x=250
    g[40:360, 285:288] = 30               # disconnected right-margin shadow
    regions = [_txt(80, 40, 250, 360)]

    out = remove_region_shadows(g, regions)

    assert (out[80:320, 235:260] == 0).all()
    assert (out[40:360, 285:288] == 255).all()
