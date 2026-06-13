"""Preview images for the review UI.

Two views per flagged page:
- analysis overlay: the deskewed original with the detected content box, the
  normalized crop, region rectangles colored by class (text/figure/photo) and
  the nombre, so a reviewer can see what the analysis decided and why a page
  was flagged.
- output preview: the actual rendered page (what the final PDF will contain).

Both are returned as PNG bytes, downscaled for the browser.
"""

from __future__ import annotations

import cv2
import numpy as np

from .model import PageParams, RegionKind, RenderSettings

_PREVIEW_MAX = 1100
_COLORS = {  # BGR
    RegionKind.TEXT: (60, 160, 60),
    RegionKind.FIGURE: (40, 140, 230),
    RegionKind.PHOTO: (60, 60, 220),
}
# overlay boxes that are not region classes (BGR)
_CONTENT_COLOR = (200, 120, 0)
_CROP_COLOR = (0, 200, 200)
_NOMBRE_COLOR = (200, 0, 200)


def _bgr_to_hex(bgr) -> str:
    b, g, r = (int(c) for c in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def legend() -> list[tuple[str, str]]:
    """(label, #rrggbb) for every box color the analysis overlay draws.

    Derived from the same constants the overlay uses, so the UI legend can
    never drift from the actual rendered colors.
    """
    return [
        ("text 文字（二値）", _bgr_to_hex(_COLORS[RegionKind.TEXT])),
        ("figure 線画（二値）", _bgr_to_hex(_COLORS[RegionKind.FIGURE])),
        ("photo 写真（トーン維持）", _bgr_to_hex(_COLORS[RegionKind.PHOTO])),
        ("content 内容枠", _bgr_to_hex(_CONTENT_COLOR)),
        ("crop 出力枠", _bgr_to_hex(_CROP_COLOR)),
        ("nombre ノンブル", _bgr_to_hex(_NOMBRE_COLOR)),
    ]


def _downscale(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= _PREVIEW_MAX:
        return img
    s = _PREVIEW_MAX / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _deskew(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-3:
        return img.copy()
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderValue=(255, 255, 255))


def analysis_overlay(original_bgr: np.ndarray, params: PageParams) -> bytes:
    canvas = _deskew(original_bgr, params.deskew.angle_deg)

    def rect(box, color, thick):
        cv2.rectangle(canvas, (int(box.x0), int(box.y0)),
                      (int(box.x1), int(box.y1)), color, thick)

    for r in params.regions:
        rect(r.box, _COLORS.get(r.kind, (128, 128, 128)), 2)
    if params.margin:
        if params.margin.content:
            rect(params.margin.content, _CONTENT_COLOR, 3)   # blue
        if params.margin.crop:
            rect(params.margin.crop, _CROP_COLOR, 2)         # yellow
        if params.margin.nombre_box:
            rect(params.margin.nombre_box, _NOMBRE_COLOR, 3)  # magenta

    ok, buf = cv2.imencode(".png", _downscale(canvas))
    return buf.tobytes()


def output_preview(original_bgr: np.ndarray, params: PageParams,
                   settings: RenderSettings) -> bytes:
    """PNG of the faithful output: binarized base + gray/color photo overlays
    (or whole-page gray/color), so 'output' actually looks like the final PDF."""
    from .render import render_output_preview

    out_bgr = render_output_preview(original_bgr, params, settings)
    ok, buf = cv2.imencode(".png", _downscale(out_bgr))
    return buf.tobytes()
