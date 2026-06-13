import cv2
import numpy as np

from hokusai_press.geometry.deskew import GOOD_CONFIDENCE, find_skew, is_confident
from hokusai_press.geometry.margin import find_content_box
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


def test_content_box_excludes_border_noise():
    img = np.full((800, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (100, 150), (500, 650), (0, 0, 0), 3)  # body frame
    img[0:800, 0:4] = 0   # left scan-edge streak (touches border -> ignored)
    mg = find_content_box(img, Deskew())
    assert mg.content.x0 >= 90
    assert mg.content.x1 <= 510
    assert mg.content.y0 >= 140
    assert mg.confidence > 0
