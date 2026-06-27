"""Small vector-fill PDF content primitives."""

from __future__ import annotations


def px_to_pt(v, dpi) -> float:
    """Convert image pixels to PDF points at ``dpi``."""
    return v * 72.0 / dpi


def fill_rect_ops(x0, y0, x1, y1, gray, page_h_pt, scale) -> bytes:
    """Return PDF operators that fill one image-space rectangle.

    Input coordinates are image pixels with a top-left origin and y increasing
    downward. Output coordinates are PDF user-space points with a bottom-left
    origin and y increasing upward.
    """
    left_px, right_px = sorted((x0, x1))
    top_px, bottom_px = sorted((y0, y1))

    x_pt = left_px * scale
    y_pt = page_h_pt - bottom_px * scale
    w_pt = (right_px - left_px) * scale
    h_pt = (bottom_px - top_px) * scale

    fragment = (
        f"{gray:.2f} g "
        f"{x_pt:.2f} {y_pt:.2f} {w_pt:.2f} {h_pt:.2f} re f\n"
    )
    return fragment.encode("ascii")
