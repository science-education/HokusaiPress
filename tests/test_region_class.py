import cv2
import numpy as np

from hokusai_press.region_class import classify_patch


def test_uniform_gray_fill_is_solid_fill():
    patch = np.full((80, 120), 160, dtype=np.uint8)

    assert classify_patch(patch) == "solid_fill"


def test_thin_black_lines_are_line_art():
    patch = np.full((120, 160), 255, dtype=np.uint8)
    cv2.line(patch, (12, 25), (145, 25), 0, 1)
    cv2.line(patch, (12, 55), (145, 55), 0, 1)
    cv2.rectangle(patch, (30, 75), (120, 105), 0, 1)

    assert classify_patch(patch) == "line_art"


def test_smooth_gradient_is_photo():
    row = np.linspace(30, 225, 160, dtype=np.uint8)
    patch = np.tile(row, (120, 1))

    assert classify_patch(patch) == "photo"


def test_random_continuous_tone_is_photo():
    rng = np.random.default_rng(123)
    patch = rng.normal(128, 35, (120, 160)).clip(20, 235).astype(np.uint8)

    assert classify_patch(patch) == "photo"


def test_near_empty_white_patch_is_other():
    patch = np.full((120, 160), 255, dtype=np.uint8)
    cv2.line(patch, (10, 10), (20, 10), 240, 1)

    assert classify_patch(patch) == "other"
