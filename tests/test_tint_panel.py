import cv2
import numpy as np

from hokusai_press.tint_panel import detect_tint_panels


def test_tp1_detects_horizontal_flat_tint_band():
    img = np.full((600, 900, 3), 255, dtype=np.uint8)
    img[50:250, :] = 160

    fills = detect_tint_panels(img)

    assert fills
    fill = fills[0]
    x0, y0, x1, y1 = fill["rect"]
    assert x0 <= 5
    assert x1 >= 895
    assert y0 <= 70
    assert y1 >= 230
    assert abs(fill["gray"] - 160 / 255.0) < 0.02


def test_tp2_blank_white_page_has_no_panels():
    img = np.full((600, 900, 3), 255, dtype=np.uint8)

    assert detect_tint_panels(img) == []


def test_tp3_text_lines_only_page_has_no_panels():
    img = np.full((600, 900, 3), 255, dtype=np.uint8)
    for y in range(80, 420, 45):
        img[y:y + 8, 120:760] = 0

    assert detect_tint_panels(img) == []


def test_tp4_detects_flat_tint_under_text():
    img = np.full((600, 900, 3), 255, dtype=np.uint8)
    img[80:320, :] = 150
    for i, text in enumerate(["Hokusai", "Press", "Tint"]):
        cv2.putText(
            img,
            text,
            (80 + i * 230, 180 + i * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (0, 0, 0),
            5,
        )

    fills = detect_tint_panels(img)

    assert fills
    x0, y0, x1, y1 = fills[0]["rect"]
    assert x0 <= 5
    assert x1 >= 895
    assert y0 <= 105
    assert y1 >= 295


def test_tp5_photo_like_random_image_is_not_detected():
    rng = np.random.default_rng(42)
    gray = rng.integers(60, 200, size=(600, 900), dtype=np.uint8)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    assert detect_tint_panels(img) == []


def test_tp6_detects_vertical_flat_tint_band():
    img = np.full((600, 900, 3), 255, dtype=np.uint8)
    img[:, 0:100] = 140

    fills = detect_tint_panels(img)

    assert fills
    x0, y0, x1, y1 = fills[0]["rect"]
    assert x0 <= 5
    assert x1 >= 80
    assert y0 <= 5
    assert y1 >= 595
