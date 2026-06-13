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
            rect(params.margin.content, (200, 120, 0), 3)   # content: blue
        if params.margin.crop:
            rect(params.margin.crop, (0, 200, 200), 2)      # crop: yellow
        if params.margin.nombre_box:
            rect(params.margin.nombre_box, (200, 0, 200), 3)  # nombre: magenta

    ok, buf = cv2.imencode(".png", _downscale(canvas))
    return buf.tobytes()


def output_preview(original_bgr: np.ndarray, params: PageParams,
                   settings: RenderSettings) -> bytes:
    from .render import render_page_image

    out_bgr, _, _, _ = render_page_image(original_bgr, params, settings)
    ok, buf = cv2.imencode(".png", _downscale(out_bgr))
    return buf.tobytes()
