import cv2
import numpy as np

from hokusai_press.geometry.deskew import GOOD_CONFIDENCE, find_skew, is_confident
from hokusai_press.geometry.margin import find_content_box, remove_edge_shadows
from hokusai_press.model import Deskew


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


def test_blank_page_low_confidence():
    blank = np.full((400, 300, 3), 255, dtype=np.uint8)
    sk = find_skew(blank)
    assert sk.confidence == 0.0
    assert not is_confident(sk)


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
