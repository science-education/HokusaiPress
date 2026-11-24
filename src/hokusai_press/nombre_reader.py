"""Engine-independent folio (nombre) detection and recognition.

This stage deliberately uses page pixels, not OCR engine output, to find small
isolated text in the top/bottom margin bands.  The text crop is then read by a
small CTC digit recognizer exported to ONNX.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .geometry.margin import ink_threshold, remove_edge_shadows
from .model import Box
from .nombre import parse_numeral, parse_roman

NOMBRE_BAND_FRAC = 0.14
MAX_HEIGHT_FRAC = 0.045
MAX_WIDTH_FRAC = 0.28
MIN_AREA_FRAC = 2.0e-6
PAD_FRAC = 0.45
MAX_CANDIDATES_PER_BAND = 12
BINARIZE_METHODS = {"adaptive_gaussian", "sauvola", "otsu"}

IMG_H = 32
IMG_W = 128
CHARSET = "0123456789ivx一二三四五六七八九十〇"
BLANK_IDX = 0
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARSET)}


@dataclass(frozen=True)
class Cand:
    band: str
    value: int | None
    kind: str
    box: Box
    pos: tuple[float, float, float]
    conf: float
    text: str


def read_folio_candidates(
    deskewed_bgr: np.ndarray,
    content_box: Box,
    page_h: int | float,
    page_w: int | float,
    extra_boxes: list[Box] | None = None,
    model_path: str | os.PathLike[str] | None = None,
) -> list[Cand]:
    """Return readable folio candidates from page pixels.

    `deskewed_bgr` and `content_box` are expected to be in the same coordinate
    space.  `extra_boxes` may contain optional layout engine page_number boxes;
    they are treated only as additional crop proposals.
    """
    boxes = _rank_candidate_boxes(
        _detect_candidate_boxes(deskewed_bgr, content_box, page_h, page_w),
        page_h,
        page_w,
    )
    if extra_boxes:
        boxes.extend(_clamp_box(b, page_w, page_h) for b in extra_boxes)
    boxes = _dedupe_boxes([b for b in boxes if b.width > 0 and b.height > 0])
    if not boxes:
        return []

    recognizer = _Recognizer(model_path)
    methods = _binarize_methods() or [None]
    out: list[Cand] = []
    for box in boxes:
        crop = _crop_with_padding(deskewed_bgr, box, page_w, page_h)
        if crop.size == 0:
            continue
        band = "top" if (box.y0 + box.y1) / 2.0 <= page_h / 2.0 else "bottom"
        cx = (box.x0 + box.x1) / 2.0 / max(1.0, float(page_w))
        cy = (box.y0 + box.y1) / 2.0 / max(1.0, float(page_h))
        content_cx = (content_box.x0 + content_box.x1) / 2.0
        side = -1.0 if (box.x0 + box.x1) / 2.0 < content_cx else 1.0
        # A box is one digit token = one true value, so read it under each active
        # binarization and keep only the single highest-confidence reading rather
        # than emitting one candidate per method: union'd readings inject the
        # weaker method's competing misreading into the cross-page vote (it
        # regressed otsu-strong books). Per-box best-confidence lets sauvola win a
        # faint corner crop and otsu win a clean one without doubling the vote.
        best: tuple[float, int | None, str, str] | None = None
        for method in methods:
            text, conf = recognizer.predict(crop, method)
            value = parse_numeral(text)
            kind = "num"
            if value is None:
                value = parse_roman(text)
                kind = "roman"
            if value is None or value <= 0:
                # Keep unparsed strings for fuzzy matching later, but set value to None
                value = None
                kind = "unparsed"
            if best is None or conf > best[0]:
                best = (conf, value, kind, text)
        if best is None:
            continue
        conf, value, kind, text = best
        out.append(Cand(band=band, value=value, kind=kind, box=box,
                        pos=(cx, cy, side), conf=conf, text=text))
    return out


def _detect_candidate_boxes(
    img_bgr: np.ndarray, content: Box, page_h: int | float, page_w: int | float
) -> list[Box]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    gray = remove_edge_shadows(gray)
    boxes: list[Box] = []
    for method in (_binarize_methods() or [None]):
        boxes.extend(_detect_boxes_for_method(gray, method))
    boxes.extend(_corner_probe_boxes(content, page_h, page_w))
    return boxes


def _detect_boxes_for_method(gray: np.ndarray, method: str | None) -> list[Box]:
    if method:
        binary = np.zeros(gray.shape, dtype=np.uint8)
    else:
        thr = ink_threshold(gray)
        binary = (gray <= thr).astype(np.uint8)

    h, w = binary.shape
    band_h = max(1, int(h * NOMBRE_BAND_FRAC))
    max_h = max(2, int(h * MAX_HEIGHT_FRAC))
    max_w = max(2, int(w * MAX_WIDTH_FRAC))
    min_area = max(6, int(h * w * MIN_AREA_FRAC))

    boxes: list[Box] = []
    for band_name, y0, y1 in (("top", 0, band_h), ("bottom", h - band_h, h)):
        if y1 <= y0:
            continue
        if method:
            band_bin = _binarize(gray[y0:y1, :], method)
            band = (band_bin == 0).astype(np.uint8)
            binary[y0:y1, :] = band
        else:
            band = binary[y0:y1, :]
        # Join adjacent glyph components inside one short folio token while
        # keeping running heads and page body out through size and band gates.
        kx = max(3, int(w * 0.006))
        ky = max(2, int(h * 0.0025))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        joined = cv2.dilate(band, kernel, iterations=1)
        n, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
        for i in range(1, n):
            x, yy, cw, ch, area = stats[i]
            if area < min_area or ch > max_h or cw > max_w:
                continue
            if cw < 3 or ch < 3:
                continue
            bx = Box(float(x), float(y0 + yy), float(x + cw), float(y0 + yy + ch))
            boxes.append(_tighten_to_ink(binary, bx))
    return boxes


def _corner_probe_boxes(content: Box, page_h: int | float, page_w: int | float) -> list[Box]:
    """Fallback proposals for faint corner folios swallowed by page texture.

    Some books print the folio as a very light one- or two-digit token near the
    top outside corner.  On textured scans the thresholded background can join
    the whole top band into one huge component, so connected-components never
    yields the digit.  Small fixed corner probes keep those pages available to
    the recognizer; cross-page position+offset resolution rejects empty/noisy
    probes.
    """
    w = float(page_w)
    h = float(page_h)
    probe_w = max(28.0, min(46.0, w * 0.020))
    probe_h = max(32.0, min(54.0, h * 0.016))
    y0 = max(0.0, min(h * 0.026, content.y0 - probe_h * 0.60))
    y1 = min(h, y0 + probe_h)
    inset = max(92.0, w * 0.055)
    return [
        _clamp_box(Box(inset, y0, inset + probe_w, y1), w, h),
        _clamp_box(Box(w - inset - probe_w, y0, w - inset, y1), w, h),
    ]


def _tighten_to_ink(binary: np.ndarray, box: Box) -> Box:
    h, w = binary.shape
    x0 = max(0, int(np.floor(box.x0)))
    y0 = max(0, int(np.floor(box.y0)))
    x1 = min(w, int(np.ceil(box.x1)))
    y1 = min(h, int(np.ceil(box.y1)))
    ys, xs = np.nonzero(binary[y0:y1, x0:x1])
    if xs.size == 0:
        return box
    return Box(float(x0 + xs.min()), float(y0 + ys.min()),
               float(x0 + xs.max() + 1), float(y0 + ys.max() + 1))


def _clamp_box(box: Box, page_w: int | float, page_h: int | float) -> Box:
    return Box(float(max(0, min(page_w, box.x0))),
               float(max(0, min(page_h, box.y0))),
               float(max(0, min(page_w, box.x1))),
               float(max(0, min(page_h, box.y1))))


def _dedupe_boxes(boxes: list[Box]) -> list[Box]:
    out: list[Box] = []
    for b in sorted(boxes, key=lambda x: (x.y0, x.x0, x.width * x.height)):
        if any(_iou(b, kept) > 0.55 for kept in out):
            continue
        out.append(b)
    return out


def _rank_candidate_boxes(
    boxes: list[Box], page_h: int | float, page_w: int | float
) -> list[Box]:
    ranked: list[Box] = []
    for is_bottom in (True, False):
        band_boxes = [
            b for b in boxes
            if ((b.y0 + b.y1) / 2.0 > page_h / 2.0) == is_bottom
        ]
        if not band_boxes:
            continue
        def score(b: Box):
            cy = (b.y0 + b.y1) / 2.0
            cx = (b.x0 + b.x1) / 2.0
            edge_y = page_h - cy if is_bottom else cy
            x_anchor = min(abs(cx - page_w / 2.0), cx, page_w - cx)
            return (edge_y, abs(b.height - page_h * 0.010),
                    min(x_anchor, page_w * 0.18), b.width * b.height)
        ranked.extend(sorted(band_boxes, key=score)[:MAX_CANDIDATES_PER_BAND])
    return ranked


def _iou(a: Box, b: Box) -> float:
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    area = a.width * a.height + b.width * b.height - inter
    return inter / max(area, 1.0)


def _crop_with_padding(img: np.ndarray, box: Box, page_w, page_h) -> np.ndarray:
    pad = max(4, int(max(box.width, box.height) * PAD_FRAC))
    b = _clamp_box(Box(box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad),
                   page_w, page_h)
    return img[int(b.y0):int(b.y1), int(b.x0):int(b.x1)]


def _default_model_path() -> Path:
    """The shipped digit recognizer, bundled in-package (small, in-house
    trained -- no third-party model license applies, unlike the OCR engine
    models in models/, which stay external/gitignored for that reason).

    HOKUSAI_FOLIO_DIGIT_ONNX overrides this with a path to an alternate
    checkpoint (e.g. one of the training run's other variants) for local
    experiments, without needing to touch this default.
    """
    env = os.environ.get("HOKUSAI_FOLIO_DIGIT_ONNX")
    if env:
        return Path(env)
    return Path(__file__).parent / "data" / "folio_digit.onnx"


class _Recognizer:
    def __init__(self, model_path: str | os.PathLike[str] | None = None):
        self.model_path = Path(model_path) if model_path else _default_model_path()
        self.session = _session(self.model_path)

    def predict(self, image: np.ndarray, method: str | None) -> tuple[str, float]:
        x = _image_to_tensor(image, method)
        logits = self.session.run(None, {"image": x})[0][0]
        probs = _softmax(logits, axis=1)
        idx = np.argmax(probs, axis=1)
        text = _ctc_decode(idx)
        conf = _ctc_confidence(idx, probs)
        return text, conf


@lru_cache(maxsize=4)
def _session(model_path: Path):
    import onnxruntime as ort

    providers = ort.get_available_providers()
    use = []
    if "CoreMLExecutionProvider" in providers:
        use.append("CoreMLExecutionProvider")
    if "CPUExecutionProvider" in providers:
        use.append("CPUExecutionProvider")
    if not use:
        use = providers[:1]
    return ort.InferenceSession(str(model_path), providers=use)


def _image_to_tensor(image: np.ndarray, method: str | None) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if method:
        gray = _binarize(gray, method)
    im = _normalize_image(gray)
    arr = im.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr[None, None, :, :]


def _normalize_image(gray: np.ndarray) -> np.ndarray:
    arr = gray.astype(np.uint8)
    if arr.mean() < 128:
        arr = 255 - arr
    bg = float(np.percentile(arr, 90))
    fg = float(np.percentile(arr, 5))
    delta = max(14.0, min(36.0, (bg - fg) * 0.30))
    ink = arr < max(0.0, bg - delta)
    ys, xs = np.nonzero(ink)
    if xs.size:
        l, r = xs.min(), xs.max() + 1
        t, b = ys.min(), ys.max() + 1
        pad_x = max(1, int((r - l) * 0.18))
        pad_y = max(1, int((b - t) * 0.25))
        arr = arr[max(0, t - pad_y):min(arr.shape[0], b + pad_y),
                  max(0, l - pad_x):min(arr.shape[1], r + pad_x)]
        local_bg = float(np.percentile(arr, 90))
        local_fg = float(np.percentile(arr, 5))
        local_delta = max(14.0, min(36.0, (local_bg - local_fg) * 0.30))
        local_ink = arr < max(0.0, local_bg - local_delta)
        arr = np.where(local_ink, arr, 255).astype(np.uint8)
    h, w = arr.shape[:2]
    scale = min((IMG_W - 8) / max(1, w), (IMG_H - 6) / max(1, h))
    new_w = max(1, min(IMG_W - 2, int(round(w * scale))))
    new_h = max(1, min(IMG_H - 2, int(round(h * scale))))
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
    x = max(0, (IMG_W - new_w) // 2)
    y = max(0, (IMG_H - new_h) // 2)
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def _ctc_decode(indices: np.ndarray | list[int]) -> str:
    out: list[str] = []
    prev = BLANK_IDX
    for raw in indices:
        idx = int(raw)
        if idx != BLANK_IDX and idx != prev:
            out.append(IDX_TO_CHAR.get(idx, ""))
        prev = idx
    return "".join(out)


def _ctc_confidence(indices: np.ndarray, probs: np.ndarray) -> float:
    vals = []
    prev = BLANK_IDX
    for t, raw in enumerate(indices):
        idx = int(raw)
        if idx != BLANK_IDX and idx != prev:
            vals.append(float(probs[t, idx]))
        prev = idx
    return float(np.prod(vals) ** (1.0 / max(1, len(vals)))) if vals else 0.0


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _binarize_methods() -> list[str]:
    """Active binarization method(s) for detection and recognition.

    Returns an empty list when binarization is disabled (the recognizer then
    reads the raw crop), a single-element list for one method, or
    ``["otsu", "sauvola"]`` for fusion. otsu and sauvola are complementary
    (otsu wins on some books, sauvola on faint-gray corner folios); fusion runs
    both, unions the detected boxes and emits a candidate per method so the
    cross-page offset vote in ``nombre.resolve`` keeps the consistent reading.
    """
    flag = os.environ.get("HOKUSAI_NOMBRE_BINARIZE", "1").strip().lower()
    if flag in {"0", "false", "off", "no"}:
        return []
    method = os.environ.get("HOKUSAI_NOMBRE_BINARIZE_METHOD", "otsu").strip().lower()
    if method in {"", "none", "off"}:
        return []
    if method in {"fusion", "both", "multi"}:
        return ["otsu", "sauvola"]
    if method not in BINARIZE_METHODS:
        return ["otsu"]
    return [method]


def _binarize(gray_or_bgr: np.ndarray, method: str) -> np.ndarray:
    gray = _as_white_bg_gray(gray_or_bgr)
    if gray.size == 0:
        return gray.copy()
    if method == "adaptive_gaussian":
        work = cv2.GaussianBlur(gray, (3, 3), 0)
        win = _odd_window(gray.shape, frac=0.62, minimum=15, maximum=39)
        out = cv2.adaptiveThreshold(
            work, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, win, 2
        )
        return _clean_binary(out)
    if method == "sauvola":
        return _sauvola(gray)
    work = cv2.GaussianBlur(gray, (3, 3), 0)
    out = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return _clean_binary(out)


def _as_white_bg_gray(gray_or_bgr: np.ndarray) -> np.ndarray:
    if gray_or_bgr.ndim == 3:
        gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bgr
    gray = gray.astype(np.uint8, copy=False)
    if float(gray.mean()) < 128.0:
        gray = 255 - gray
    return gray


def _odd_window(shape: tuple[int, int], frac: float, minimum: int, maximum: int) -> int:
    h, w = shape[:2]
    win = int(round(min(h, w) * frac))
    win = max(minimum, min(maximum, win))
    if win % 2 == 0:
        win += 1
    return max(3, win)


def _clean_binary(binary: np.ndarray) -> np.ndarray:
    black = (binary == 0).astype(np.uint8)
    if min(binary.shape[:2]) >= 18:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        black = cv2.morphologyEx(black, cv2.MORPH_OPEN, kernel, iterations=1)
    return np.where(black > 0, 0, 255).astype(np.uint8)


def _sauvola(gray: np.ndarray) -> np.ndarray:
    win = _odd_window(gray.shape, frac=0.72, minimum=15, maximum=41)
    src = gray.astype(np.float32)
    mean = cv2.boxFilter(src, ddepth=-1, ksize=(win, win), normalize=True, borderType=cv2.BORDER_REPLICATE)
    sqmean = cv2.boxFilter(src * src, ddepth=-1, ksize=(win, win), normalize=True, borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(np.maximum(sqmean - mean * mean, 0.0))
    threshold = np.clip(mean * (1.0 + 0.10 * (std / 128.0 - 1.0)) + 3.0, 0.0, 255.0)
    return _clean_binary(np.where(src <= threshold, 0, 255).astype(np.uint8))


# =============================================================================
# Pass-2a: Pass-1-guided margin-band folio localization + sequence verification
# =============================================================================
GUIDED_WORK_DPI = 150
GUIDED_PX_PER_MM = GUIDED_WORK_DPI / 25.4
GUIDED_MIN_GROUP_PAGES = 5
GUIDED_SEQ_CONF_FLOOR = 0.45


def _detect_band_tokens(
    gray_band: np.ndarray,
    y_offset: int,
    px_per_mm: float = GUIDED_PX_PER_MM,
) -> list[tuple[float, float, float, float]]:
    """Extract candidate folio token boxes (x0, y0, x1, y1) in WORK_DPI output coords.

    Handles both normal white-margin pages and halftone/screened (網掛け)
    chapter-opening panels: any connected component larger than a folio token
    is re-binarized over its own dark-mask pixels with a darker threshold
    (min(Otsu(blob_pixels), median - 45)) so dark digits sitting on light gray
    screens split cleanly out of the tint panel.
    """
    bh, bw = gray_band.shape
    if bh < 4 or bw < 4:
        return []

    thr = min(ink_threshold(gray_band), 215)
    bin_otsu = (gray_band <= thr).astype(np.uint8)
    bg = float(np.percentile(gray_band, 90))
    if bg > 180.0:
        bin_loc = (gray_band <= max(100.0, bg - 38.0)).astype(np.uint8)
        dark = np.maximum(bin_otsu, bin_loc)
    else:
        dark = bin_otsu

    edge_px = max(2, int(round(1.5 * px_per_mm)))
    dark[:, :edge_px] = 0
    dark[:, max(0, bw - edge_px):] = 0
    if y_offset == 0:
        dark[:edge_px, :] = 0
    else:
        dark[max(0, bh - edge_px):, :] = 0

    kx = max(3, int(round(1.4 * px_per_mm)))
    ky = max(1, int(round(0.4 * px_per_mm)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    joined = cv2.dilate(dark, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)

    min_h = 1.2 * px_per_mm
    max_h = 11.5 * px_per_mm
    min_w = 0.6 * px_per_mm
    max_w = 16.0 * px_per_mm
    min_area = 6

    def _tighten_halftone_box(bx0: float, by0: float, bx1: float, by1: float) -> list[tuple[float, float, float, float]]:
        """Re-threshold a token box when it sits on a halftone/screened panel.

        A halftone screen whose connected blob stays below max_w/max_h skips
        the oversized-blob branch and leaves the box bloated with dots (e.g.
        13 mm wide around a 3.2 mm '88'). Smoothing suppresses the screen dots
        so adaptive background thresholding tightens the box onto the digits.
        """
        ix0 = max(0, int(np.floor(bx0)))
        iy0 = max(0, int(np.floor(by0 - y_offset)))
        ix1 = min(bw, int(np.ceil(bx1)))
        iy1 = min(bh, int(np.ceil(by1 - y_offset)))
        if ix1 - ix0 < 4 or iy1 - iy0 < 4:
            return [(bx0, by0, bx1, by1)]
        roi = gray_band[iy0:iy1, ix0:ix1]
        sm = cv2.GaussianBlur(roi, (3, 3), 0)
        bg = float(np.percentile(sm, 75))
        fg = float(np.percentile(sm, 5))
        if not (140.0 <= bg <= 232.0 and (bg - float(sm.min())) >= 30.0 and (bg - fg) >= 25.0):
            return [(bx0, by0, bx1, by1)]
        sub_thr = bg - max(25.0, (bg - fg) * 0.45)
        sub_mask = (sm <= sub_thr).astype(np.uint8)
        if float(sub_mask.mean()) > 0.65:
            return [(bx0, by0, bx1, by1)]
        j2 = cv2.dilate(sub_mask, kernel)
        n2, _, st2, _ = cv2.connectedComponentsWithStats(j2, connectivity=8)
        tight: list[tuple[float, float, float, float]] = []
        for j in range(1, n2):
            rx, ry, rcw, rch, rarea = (int(v) for v in st2[j])
            if rarea < min_area or rch < min_h or rch > max_h or rcw < min_w or rcw > max_w:
                continue
            sub = sub_mask[ry:ry + rch, rx:rx + rcw]
            ys, xs = np.nonzero(sub)
            if xs.size < 4:
                continue
            tx0 = float(ix0 + rx + int(xs.min()))
            ty0 = float(y_offset + iy0 + ry + int(ys.min()))
            tx1 = float(ix0 + rx + int(xs.max()) + 1)
            ty1 = float(y_offset + iy0 + ry + int(ys.max()) + 1)
            tw, th = tx1 - tx0, ty1 - ty0
            if th < min_h or th > max_h or tw < min_w or tw > max_w:
                continue
            tight.append((tx0, ty0, tx1, ty1))
        return tight if tight else [(bx0, by0, bx1, by1)]

    tokens: list[tuple[float, float, float, float]] = []

    def _collect_from_mask(mask: np.ndarray, ox: int, oy: int) -> None:
        j2 = cv2.dilate(mask, kernel)
        n2, _, st2, _ = cv2.connectedComponentsWithStats(j2, connectivity=8)
        for j in range(1, n2):
            rx, ry, rcw, rch, rarea = (int(v) for v in st2[j])
            if rarea < min_area or rch < min_h or rch > max_h or rcw < min_w or rcw > max_w:
                continue
            sub = mask[ry:ry + rch, rx:rx + rcw]
            ys, xs = np.nonzero(sub)
            if xs.size < 4:
                continue
            tx0 = float(ox + rx + int(xs.min()))
            ty0 = float(oy + ry + int(ys.min()))
            tx1 = float(ox + rx + int(xs.max()) + 1)
            ty1 = float(oy + ry + int(ys.max()) + 1)
            tw, th = tx1 - tx0, ty1 - ty0
            if th < min_h or th > max_h or tw < min_w or tw > max_w:
                continue
            tokens.extend(_tighten_halftone_box(tx0, ty0, tx1, ty1))

    for i in range(1, n):
        x, y, cw, ch, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        if ch > max_h or cw > max_w:
            # Oversized blob: check for dark digits printed over a lighter
            # halftone/screened background (網掛け) by re-binarizing the blob's
            # own foreground pixels with a darker threshold.
            roi = gray_band[y:y + ch, x:x + cw]
            roi_mask = dark[y:y + ch, x:x + cw] > 0
            vals = roi[roi_mask]
            if vals.size >= 16:
                v_med = float(np.median(vals))
                v_min = float(vals.min())
                if v_med - v_min >= 40.0:
                    o_val, _ = cv2.threshold(
                        vals.reshape(-1, 1), 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                    )
                    sub_thr = min(float(o_val), v_med - 45.0)
                    sub_dark = (roi <= sub_thr).astype(np.uint8)
                    if x < edge_px:
                        sub_dark[:, :edge_px - x] = 0
                    if x + cw > bw - edge_px:
                        sub_dark[:, max(0, bw - edge_px - x):] = 0
                    if y_offset == 0 and y < edge_px:
                        sub_dark[:edge_px - y, :] = 0
                    if y_offset > 0 and y + ch > bh - edge_px:
                        sub_dark[max(0, bh - edge_px - y):, :] = 0
                    _collect_from_mask(sub_dark, x, y_offset + y)
            continue
        if ch < min_h or cw < min_w:
            continue
        sub = dark[y:y + ch, x:x + cw]
        ys, xs = np.nonzero(sub)
        if xs.size < 4:
            continue
        tx0 = float(x + int(xs.min()))
        ty0 = float(y_offset + y + int(ys.min()))
        tx1 = float(x + int(xs.max()) + 1)
        ty1 = float(y_offset + y + int(ys.max()) + 1)
        tw, th = tx1 - tx0, ty1 - ty0
        if th < min_h or th > max_h or tw < min_w or tw > max_w:
            continue
        tokens.extend(_tighten_halftone_box(tx0, ty0, tx1, ty1))

    if len(tokens) > 1:
        deduped: list[tuple[float, float, float, float]] = []
        for t in sorted(tokens, key=lambda b: (b[1], b[0], (b[2] - b[0]) * (b[3] - b[1]))):
            tb = Box(*t)
            if any(_iou(tb, Box(*k)) > 0.55 for k in deduped):
                continue
            deduped.append(t)
        return deduped
    return tokens


def _read_token_crop(
    rec: _Recognizer,
    strip: np.ndarray,
    y_off: int,
    box: tuple[float, float, float, float],
    max_v: int,
    *,
    expected_v: int | None = None,
    try_sauvola: bool = True,
) -> tuple[int | None, str, float, bool]:
    """Read a candidate token crop with folio_digit.onnx.

    Uses height-proportional padding so wide 3-digit numbers do not pull in
    vertical margin/halftone clutter. Returns (value, text, conf, matched_expected).
    """
    x0, y0, x1, y1 = box
    th = max(1.0, y1 - y0)
    pad_x = max(3, int(round(th * 0.45)))
    pad_y = max(2, int(round(th * 0.35)))
    cx0 = max(0, int(np.floor(x0)) - pad_x)
    cy0 = max(0, int(np.floor(y0 - y_off)) - pad_y)
    cx1 = min(strip.shape[1], int(np.ceil(x1)) + pad_x)
    cy1 = min(strip.shape[0], int(np.ceil(y1 - y_off)) + pad_y)
    crop = strip[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        return None, "", -1.0, False

    methods: tuple[str | None, ...] = (None, "otsu", "sauvola") if try_sauvola else (None, "otsu")
    best_v: int | None = None
    best_txt = ""
    best_conf = -1.0
    exp_txt = ""
    exp_conf = -1.0

    for meth in methods:
        txt, conf = rec.predict(crop, meth)
        v = parse_numeral(txt)
        if v is None or v <= 0 or v > max_v:
            continue
        if expected_v is not None and v == expected_v and conf >= GUIDED_SEQ_CONF_FLOOR:
            if conf > exp_conf:
                exp_txt, exp_conf = txt, conf
        if conf > best_conf:
            best_v, best_txt, best_conf = v, txt, conf

    if expected_v is not None and exp_conf >= GUIDED_SEQ_CONF_FLOOR:
        return expected_v, exp_txt, exp_conf, True
    return best_v, best_txt, best_conf, False


def _extract_page_bgr_fast(pdf, page_index: int, source_path: str, p) -> np.ndarray:
    """Fast raw JPEG extraction matching source.load_single + orient; falls back safely."""
    from .source import load_single, orient

    if pdf is not None:
        try:
            import pikepdf

            pg = pdf.pages[page_index]
            xobjs = pg.Resources.get("/XObject")
            if xobjs is not None and len(xobjs) == 1:
                obj = next(iter(xobjs.values()))
                if obj.get("/Filter") == pikepdf.Name("/DCTDecode"):
                    raw = bytes(obj.read_raw_bytes())
                    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        rot = int(pg.get("/Rotate", 0)) % 360
                        if rot == 90:
                            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                        elif rot == 180:
                            img = cv2.rotate(img, cv2.ROTATE_180)
                        elif rot == 270:
                            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        img = orient(img, p.source)
                        m = p.margin
                        if m and m.page_w and m.page_h:
                            if abs(img.shape[1] - m.page_w) <= 2 and abs(img.shape[0] - m.page_h) <= 2:
                                return img
        except Exception:
            pass
    img, _ = load_single(source_path, page_index)
    return orient(img, p.source)


def _clear_unlocated_nombre(p) -> None:
    """Drop a stale nombre on a page where the guided pass found no folio.

    Only called for unverified sources (None / "geom"): their page_number /
    nombre_text are leftovers of an earlier heuristic run and showed up as
    one-page outliers in otherwise continuous sequences. Roman front-matter
    numerals that agree with their text are kept.
    """
    p.margin.nombre_box = None
    p.margin.nombre_source = None
    is_roman = bool(
        p.nombre_text and parse_roman(p.nombre_text) == p.page_number
        and parse_numeral(p.nombre_text) is None
    )
    if not is_roman:
        p.page_number = p.page_number_source = None
        p.nombre_text = None


def _spaced_glyphs(strip, y_off, box):
    """Extract separated, similarly tall glyphs from a single baseline."""
    x, y, right, bottom = map(round, box)
    crop = strip[y-y_off:bottom-y_off, x:right]
    if min(crop.shape) < 4:
        return []
    _, mask = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    edges = np.flatnonzero(np.diff(np.r_[False, mask.any(axis=0), False]))
    if not 4 <= len(edges) <= 12:
        return []
    glyphs = []
    for left, end in edges.reshape(-1, 2):
        part = mask[:, left:end]
        rows = np.flatnonzero(part.any(axis=1))
        height = rows[-1] - rows[0] + 1
        if height < 0.75 * len(mask) or end - left > 0.85 * height:
            return []
        # Average grayscale strokes retain details lost in the segmentation mask.
        ink = 255 - crop[rows[0]:rows[-1]+1, left:end]
        glyphs.append(cv2.resize(ink, (10, 24), interpolation=cv2.INTER_AREA))
    return glyphs


def _read_spaced_pages(records, offset, rec):
    """Learn per-digit shapes only from independent, sequence-verified OCR."""
    samples = {}
    padded_widths = []
    for p, strip, y_off, box, confirmed, _ in records:
        if not confirmed or not (p.nombre_text or "").isascii() or not p.nombre_text.isdigit():
            continue
        parts = _spaced_glyphs(strip, y_off, box)
        if len(parts) == len(p.nombre_text):
            if p.nombre_text.startswith("0"):
                padded_widths.append(len(parts))
            for char, glyph in zip(p.nombre_text, parts):
                samples.setdefault(char, []).append(glyph)
    exemplars = {c: np.stack(items).astype(np.float32) / 255
                 for c, items in samples.items() if len(items) >= 2}
    if len(exemplars) < 5:
        return 0
    added = 0
    for p, strip, y_off, box, confirmed, neighbors in records:
        if confirmed or p.page_number is not None:
            continue
        # Join same-height fragments only when their baselines and gaps agree.
        x, y, right, bottom = box
        height = bottom - y
        line = [b for b in neighbors if abs(b[1]-y) <= height*.2
                and abs(b[3]-bottom) <= height*.2]
        group = [box]
        for b in sorted(line, reverse=True):
            if 0 < group[0][0]-b[2] <= height:
                group.insert(0, b)
        for b in sorted(line):
            if 0 < b[0]-group[-1][2] <= height:
                group.append(b)
        merged = (group[0][0], min(b[1] for b in group),
                  group[-1][2], max(b[3] for b in group))
        parts = _spaced_glyphs(strip, y_off, merged)
        digits = []
        for part in parts:
            scores = sorted((float(np.min(np.mean(np.abs(bank - part / 255.0),
                                                   axis=(1, 2)))), c)
                            for c, bank in exemplars.items())
            if scores[0][0] > .19:
                break
            if scores[1][0] - scores[0][0] < .012:
                # In fixed-width padded folios, an independent single-digit
                # OCR read can disambiguate a narrow 8 from a similar 0.
                if padded_widths.count(len(parts)) < 4:
                    break
                glyph = np.pad(255 - cv2.resize(part, (15, 24)), 5, constant_values=255)
                char, conf = rec.predict(glyph, None)
                if char != scores[0][1] or conf < GUIDED_SEQ_CONF_FLOOR:
                    break
            digits.append(scores[0][1])
        text = "".join(digits)
        if text and len(text) == len(parts) and int(text) == p.source.page_index + offset:
            old = p.margin.nombre_box
            sx = old.width / (right-x)
            sy = old.height / height
            p.margin.nombre_box = Box(old.x0+(merged[0]-x)*sx, old.y0+(merged[1]-y)*sy,
                                      old.x1+(merged[2]-right)*sx, old.y1+(merged[3]-bottom)*sy)
            p.page_number, p.nombre_text, p.page_number_source = int(text), text, "folio"
            added += 1
    return added


def _read_inverted_tabs(records, rec, max_v):
    """Read central light digits on a screened tab at the agreed folio position."""
    from collections import defaultdict

    votes = defaultdict(list)
    for p, strip, y_off, box, _, _ in records:
        if p.page_number is not None:
            continue
        x, y, right, bottom = map(round, box)
        tab = strip[y-y_off:bottom-y_off, x:right]
        h, w = tab.shape
        if h < 16 or w < 10 or np.median(tab) > 215:
            continue
        core = 255 - tab[round(.29*h):round(.72*h), round(.08*w):round(.92*w)]
        smooth = cv2.GaussianBlur(core, (3, 3), .75)
        lo, hi = np.percentile(smooth, (3, 75))
        if hi - lo < 12:
            continue
        contrast = np.uint8(np.clip((smooth-lo)*255/(hi-lo), 0, 255))
        text, conf = rec.predict(contrast, "otsu")
        value = parse_numeral(text)
        if value and value <= max_v and conf >= GUIDED_SEQ_CONF_FLOOR:
            votes[value-p.source.page_index].append((p, text, value))
    ranked = sorted(votes.values(), key=len, reverse=True)
    if (not ranked or len(ranked[0]) < 6 or
            (len(ranked) > 1 and len(ranked[0]) < 2*len(ranked[1])) or
            len({p.source.page_index % 2 for p, _, _ in ranked[0]}) < 2):
        return 0
    for p, text, value in ranked[0]:
        p.page_number, p.nombre_text, p.page_number_source = value, text, "inverted_folio"
    return len(ranked[0])


def _to_roman(n: int) -> str:
    """Convert a positive integer (1..3999) to lowercase canonical Roman numeral."""
    if n <= 0 or n >= 4000:
        return ""
    table = (
        (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
        (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
        (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
    )
    out: list[str] = []
    rem = n
    for v, sym in table:
        while rem >= v:
            out.append(sym)
            rem -= v
    return "".join(out)


def _parse_roman_token(text: str) -> tuple[int, str, bool, bool] | None:
    """Parse ONNX output as a lowercase Roman numeral, allowing '1'<->'i' confusion.

    Returns (value, canonical_roman, is_canonical, has_explicit_roman_char) or None.
    """
    if not text:
        return None
    s = "".join(text.split()).lower()
    if not s or len(s) > 8 or any(c not in "ivx1" for c in s):
        return None
    has_roman_char = any(c in "ivx" for c in s)
    norm = s.replace("1", "i")
    val = parse_roman(norm)
    if val is None or val <= 0 or val > 50:
        return None
    canonical = _to_roman(val)
    return val, canonical, (norm == canonical), has_roman_char


def _detect_thin_roman_tokens(
    strip: np.ndarray,
    y_off: int,
    mcx_mm: float,
    mcy_mm: float,
    px_per_mm: float = GUIDED_PX_PER_MM,
) -> list[tuple[float, float, float, float]]:
    """Detect narrow folio tokens (such as single 'i') near the consensus slot."""
    sh, sw = strip.shape
    cx = int(round(mcx_mm * px_per_mm))
    cy = int(round(mcy_mm * px_per_mm)) - y_off
    rx = int(round(14.0 * px_per_mm))
    ry = int(round(8.0 * px_per_mm))
    x0, x1 = max(0, cx - rx), min(sw, cx + rx)
    y0, y1 = max(0, cy - ry), min(sh, cy + ry)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return []

    thr = min(ink_threshold(strip), 215)
    bg = float(np.percentile(strip, 90))
    patch = strip[y0:y1, x0:x1]
    if bg > 180.0:
        dark = (patch <= max(float(thr), bg - 38.0)).astype(np.uint8)
    else:
        dark = (patch <= thr).astype(np.uint8)

    kx = max(2, int(round(0.8 * px_per_mm)))
    ky = max(2, int(round(0.6 * px_per_mm)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    joined = cv2.dilate(dark, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)

    min_h = 1.2 * px_per_mm
    max_h = 11.5 * px_per_mm
    min_w = 0.2 * px_per_mm
    max_w = 16.0 * px_per_mm
    out: list[tuple[float, float, float, float]] = []
    for i in range(1, n):
        bx, by, cw, ch, area = (int(v) for v in stats[i])
        if area < 4 or ch < min_h or ch > max_h or cw < min_w or cw > max_w:
            continue
        sub = dark[by:by + ch, bx:bx + cw]
        ys, xs = np.nonzero(sub)
        if xs.size < 4:
            continue
        tx0 = float(x0 + bx + int(xs.min()))
        ty0 = float(y_off + y0 + by + int(ys.min()))
        tx1 = float(x0 + bx + int(xs.max()) + 1)
        ty1 = float(y_off + y0 + by + int(ys.max()) + 1)
        tw, th = tx1 - tx0, ty1 - ty0
        if min_h <= th <= max_h and min_w <= tw <= max_w:
            out.append((tx0, ty0, tx1, ty1))
    return out


def _expand_to_word(strip, y_off, box):
    """Grow a one-glyph box to the whole numeral on its baseline.

    Wide-set Roman folios ("i i i", "i v") are often tokenized glyph by
    glyph; the outer-edge alignment needs the full word. Extends left/right
    across ink columns while gaps stay below ~0.7 x glyph height, then
    re-fits the vertical extent (keeps the dots of "i").
    """
    x0, y0, x1, y1 = box
    th = max(2.0, y1 - y0)
    r0 = max(0, int(np.floor(y0 - y_off - 0.45 * th)))
    r1 = min(strip.shape[0], int(np.ceil(y1 - y_off + 0.1 * th)))
    c0 = max(0, int(np.floor(x0 - 4 * th)))
    c1 = min(strip.shape[1], int(np.ceil(x1 + 4 * th)))
    win = strip[r0:r1, c0:c1]
    if win.size == 0 or r1 - r0 < 3:
        return box
    _, mask = cv2.threshold(win, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if win.std() < 8:
        return box
    cols = mask.any(axis=0)
    gap_max = max(2, int(round(0.7 * th)))
    left = max(0, int(np.floor(x0)) - c0)
    right = min(mask.shape[1] - 1, int(np.ceil(x1)) - c0)
    gap = 0
    j = left - 1
    while j >= 0 and gap <= gap_max:
        if cols[j]:
            left, gap = j, 0
        else:
            gap += 1
        j -= 1
    gap = 0
    j = right + 1
    while j < mask.shape[1] and gap <= gap_max:
        if cols[j]:
            right, gap = j, 0
        else:
            gap += 1
        j += 1
    if (right - left + 1) > 6 * th:      # swallowed neighbouring text: keep as is
        return box
    rows = np.flatnonzero(mask[:, left:right + 1].any(axis=1))
    if rows.size == 0:
        return box
    return (
        float(min(x0, c0 + left)),
        float(min(y0, r0 + rows[0] + y_off)),
        float(max(x1, c0 + right + 1)),
        float(max(y1, r0 + rows[-1] + 1 + y_off)),
    )


def _read_roman_front_matter(
    pages: list,
    page_tokens: list[dict],
    band_grays_cache: dict[int, tuple[np.ndarray, np.ndarray]],
    band: str,
    centers: dict[int, tuple[float, float]],
    bot_y0: int,
    rec: _Recognizer,
) -> int:
    """Detect and sequence-verify front-matter Roman folios near the body folio slot."""
    import difflib
    from collections import defaultdict

    body_first_idx = min(
        (
            p.source.page_index
            for p in pages
            if p.page_number is not None
            and p.page_number_source == "folio"
            and not (
                p.nombre_text
                and parse_roman(p.nombre_text) == p.page_number
                and parse_numeral(p.nombre_text) is None
            )
        ),
        default=len(pages),
    )
    if body_first_idx <= 0:
        return 0

    cands_by_idx: dict[int, list[tuple[float, float, tuple[float, float, float, float], list[tuple[str, float]]]]] = {}
    pages_by_idx = {p.source.page_index: (p, pt) for p, pt in zip(pages, page_tokens)}

    for idx in range(body_first_idx):
        pair = pages_by_idx.get(idx)
        if pair is None:
            continue
        p, pt = pair
        if p.blank or not p.margin or p.margin.nombre_source == "manual" or pt["box_M"] is None:
            continue
        if idx not in band_grays_cache:
            continue

        par = idx % 2
        mcx, mcy = centers[par]
        strip = band_grays_cache[idx][0 if band == "top" else 1]
        y_off = 0 if band == "top" else pt.get("bot_y0", bot_y0)

        boxes: list[tuple[float, float, float, float]] = []
        for box in pt[band]:
            cx_mm = (box[0] + box[2]) / 2.0 / GUIDED_PX_PER_MM
            cy_mm = (box[1] + box[3]) / 2.0 / GUIDED_PX_PER_MM
            if abs(cx_mm - mcx) <= 14.0 and abs(cy_mm - mcy) <= 8.0:
                boxes.append(box)
        for tbox in _detect_thin_roman_tokens(strip, y_off, mcx, mcy):
            tb = Box(*tbox)
            if not any(_iou(tb, Box(*b)) > 0.55 for b in boxes):
                boxes.append(tbox)
        if not boxes:
            continue

        page_cands = []
        for box in boxes:
            bx0, by0, bx1, by1 = box
            th = max(1.0, by1 - by0)
            pad_x = max(4, int(round(th * 0.45)))
            pad_y = max(2, int(round(th * 0.35)))
            cx0 = max(0, int(np.floor(bx0)) - pad_x)
            cy0 = max(0, int(np.floor(by0 - y_off)) - pad_y)
            cx1 = min(strip.shape[1], int(np.ceil(bx1)) + pad_x)
            cy1 = min(strip.shape[0], int(np.ceil(by1 - y_off)) + pad_y)
            crop = strip[cy0:cy1, cx0:cx1]
            if crop.size == 0:
                continue
            readings = [rec.predict(crop, meth) for meth in (None, "otsu", "sauvola")]
            cx_mm = (bx0 + bx1) / 2.0 / GUIDED_PX_PER_MM
            cy_mm = (by0 + by1) / 2.0 / GUIDED_PX_PER_MM
            page_cands.append((abs(cx_mm - mcx), abs(cy_mm - mcy), box, readings))
        if page_cands:
            cands_by_idx[idx] = page_cands

    if not cands_by_idx:
        return 0

    votes: dict[int, dict[int, tuple[float, bool]]] = defaultdict(dict)
    for idx, page_cands in cands_by_idx.items():
        for dx, dy, _, readings in page_cands:
            if dx > 10.0 or dy > 6.0:
                continue
            for txt, conf in readings:
                if conf < 0.50:
                    continue
                parsed = _parse_roman_token(txt)
                if parsed is None:
                    continue
                val, _, is_canon, has_rom = parsed
                if not is_canon:
                    continue
                off = val - idx
                d = dx + dy
                prev = votes[off].get(idx)
                if prev is None or d < prev[0] or (d == prev[0] and has_rom and not prev[1]):
                    votes[off][idx] = (d, prev[1] or has_rom if prev else has_rom)

    valid_runs = [
        (len(pg_map), sum(1 for _, hr in pg_map.values() if hr), off, pg_map)
        for off, pg_map in votes.items()
        if len(pg_map) >= 3 and sum(1 for _, hr in pg_map.values() if hr) >= 2
    ]
    if not valid_runs:
        return 0
    valid_runs.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best_cnt, _, roman_off, best_pg_map = valid_runs[0]
    if len(valid_runs) > 1 and best_cnt == valid_runs[1][0]:
        return 0

    sup_idxs = sorted(best_pg_map.keys())
    lo_idx = max(0, sup_idxs[0] - 2)
    hi_idx = min(body_first_idx - 1, sup_idxs[-1] + 2)

    added = 0
    for idx in range(lo_idx, hi_idx + 1):
        expected_v = idx + roman_off
        if expected_v <= 0 or expected_v > 50 or idx not in cands_by_idx:
            continue
        exp_str = _to_roman(expected_v)
        p, pt = pages_by_idx[idx]

        exact_hits = []
        fuzzy_hits = []
        for dx, dy, box, readings in cands_by_idx[idx]:
            if dx > 12.0 or dy > 6.0:
                continue
            for txt, conf in readings:
                s = "".join(txt.split()).lower().replace("1", "i")
                parsed = _parse_roman_token(txt)
                if parsed is not None and parsed[2] and parsed[0] == expected_v and conf >= 0.45:
                    exact_hits.append((dx + dy, -conf, box))
                elif dx <= 5.0 and dy <= 3.0 and s and all(c in "ivx" for c in s):
                    sim = difflib.SequenceMatcher(None, exp_str, s).ratio()
                    if (sim >= 0.66 and conf >= 0.45) or (expected_v == 1 and s == "i" and conf >= 0.35):
                        fuzzy_hits.append((dx + dy, -sim, -conf, box))

        chosen_box = None
        if exact_hits:
            exact_hits.sort()
            chosen_box = exact_hits[0][2]
        elif fuzzy_hits:
            fuzzy_hits.sort()
            chosen_box = fuzzy_hits[0][3]

        if chosen_box is None:
            continue
        strip = band_grays_cache[idx][0 if band == "top" else 1]
        y_off = 0 if band == "top" else pt.get("bot_y0", bot_y0)
        chosen_box = _expand_to_word(strip, y_off, chosen_box)

        A = pt["box_M"]
        sx, tx = float(A[0, 0]), float(A[0, 2])
        sy, ty = float(A[1, 1]), float(A[1, 2])
        p.margin.nombre_box = Box(
            (chosen_box[0] - tx) / sx,
            (chosen_box[1] - ty) / sy,
            (chosen_box[2] - tx) / sx,
            (chosen_box[3] - ty) / sy,
        )
        p.margin.nombre_source = "folio"
        p.page_number = expected_v
        p.nombre_text = exp_str
        p.page_number_source = "folio_roman"
        added += 1

    return added


def localize_folios_guided(
    pages: list,
    *,
    originals: list[np.ndarray] | None = None,
    source_path: str | None = None,
    model_path: str | os.PathLike[str] | None = None,
    max_workers: int = 1,
) -> dict:
    """Pass-2a: Re-identify printed folios in Pass-1 top/bottom margin strips.

    Expects Pass-1 margin normalization (`margin.crop`, `margin.target_w/h`) to
    have already run on `pages`. Warps only the top and bottom margin bands of
    each page at 150 dpi, clusters mirrored/centered folio candidates across
    pages, verifies them with `folio_digit.onnx` against the book's slope-1
    page sequence, and maps verified boxes back into deskewed-original pixel
    coordinates (`margin.nombre_box`, `margin.nombre_source="folio"`,
    `page_number`, `nombre_text`).

    Never overwrites pages with `margin.nombre_source == "manual"`. When no
    reliable book-level folio pair exists, leaves `pages` untouched.
    """
    import threading
    import time
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    from .geometry.margin import type_area_bodies
    from .model import RenderSettings
    from .render import _map_box, compose_box_transform, compose_transform

    t0 = time.perf_counter()
    have = [
        p for p in pages
        if p.margin and p.margin.content and p.margin.crop
        and p.margin.target_w and p.margin.target_h
    ]
    if len(have) < 2 * GUIDED_MIN_GROUP_PAGES:
        return {"applied": False, "reason": "too_few_pages", "elapsed_s": 0.0}
    if originals is None and not source_path:
        return {"applied": False, "reason": "no_image_source", "elapsed_s": 0.0}

    mpath = Path(model_path) if model_path else _default_model_path()
    if not mpath.exists():
        return {"applied": False, "reason": "model_not_found", "elapsed_s": 0.0}

    settings = RenderSettings(target_dpi=GUIDED_WORK_DPI)
    bodies = type_area_bodies(have)
    body_ys0: list[float] = []
    body_ys1: list[float] = []
    out_w0 = out_h0 = 0
    for p in have:
        pw = int(round(p.margin.page_w)) if p.margin.page_w else 1500
        ph = int(round(p.margin.page_h)) if p.margin.page_h else 2200
        box_M, (ow, oh), _ = compose_box_transform((ph, pw), p, settings)
        out_w0, out_h0 = ow, oh
        if id(p) in bodies:
            bb = _map_box(bodies[id(p)][1], box_M)
            body_ys0.append(bb.y0)
            body_ys1.append(bb.y1)
    if out_w0 < 20 or out_h0 < 40:
        return {"applied": False, "reason": "tiny_output", "elapsed_s": 0.0}

    pad_body = 1.5 * GUIDED_PX_PER_MM
    top_y1 = (
        int(round(min(out_h0 * 0.16, max(out_h0 * 0.08, float(np.median(body_ys0)) - pad_body))))
        if body_ys0 else int(out_h0 * 0.14)
    )
    bot_y0 = (
        int(round(max(out_h0 * 0.84, min(out_h0 * 0.92, float(np.median(body_ys1)) + pad_body))))
        if body_ys1 else int(out_h0 * 0.86)
    )
    top_y1 = max(top_y1, int(round(20.0 * GUIDED_PX_PER_MM)))
    bot_y0 = min(bot_y0, out_h0 - int(round(20.0 * GUIDED_PX_PER_MM)))
    top_y1 = max(4, min(top_y1, out_h0 // 4))
    bot_y0 = min(out_h0 - 4, max(bot_y0, out_h0 - out_h0 // 4))

    # Pass A: warp top/bottom strips & detect candidate tokens per page
    is_pdf = bool(source_path and source_path.lower().endswith(".pdf"))
    tls = threading.local()
    opened_pdfs: list = []
    pdf_lock = threading.Lock()

    def _get_thread_pdf():
        if not is_pdf:
            return None
        pdf = getattr(tls, "pdf", None)
        if pdf is None:
            try:
                import pikepdf

                pdf = pikepdf.open(source_path)
                tls.pdf = pdf
                with pdf_lock:
                    opened_pdfs.append(pdf)
            except Exception:
                return None
        return pdf

    def _extract_one(item):
        i, p = item
        if p.blank or not p.margin or not p.margin.content or not p.margin.crop:
            return {"top": [], "bottom": [], "box_M": None, "ow": out_w0, "oh": out_h0}, None
        if originals is not None:
            orig = originals[i]
        else:
            orig = _extract_page_bgr_fast(_get_thread_pdf(), p.source.page_index, source_path, p)
        gray_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY) if orig.ndim == 3 else orig
        gray_clean = remove_edge_shadows(gray_orig)
        M, (ow, oh), _ = compose_transform(orig.shape, p, settings)
        box_M, _, _ = compose_box_transform(orig.shape, p, settings)

        t_h = max(4, min(top_y1, oh - 4))
        b_y0 = max(0, min(bot_y0, oh - 4))
        M_top = M.copy()
        top_strip = cv2.warpAffine(gray_clean, M_top, (ow, t_h), flags=cv2.INTER_AREA, borderValue=255)
        M_bot = M.copy()
        M_bot[1, 2] -= float(b_y0)
        bot_strip = cv2.warpAffine(gray_clean, M_bot, (ow, oh - b_y0), flags=cv2.INTER_AREA, borderValue=255)

        t_toks = _detect_band_tokens(top_strip, 0)
        b_toks = _detect_band_tokens(bot_strip, b_y0)
        return (
            {"top": t_toks, "bottom": b_toks, "box_M": box_M, "ow": ow, "oh": oh, "bot_y0": b_y0},
            (top_strip, bot_strip),
        )

    page_tokens: list[dict] = []
    band_grays_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    try:
        items = list(enumerate(pages))
        if max_workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                extracted = list(ex.map(_extract_one, items))
        else:
            extracted = [_extract_one(it) for it in items]
    finally:
        for pdf in opened_pdfs:
            try:
                pdf.close()
            except Exception:
                pass

    for p, (pt, strips) in zip(pages, extracted):
        page_tokens.append(pt)
        if strips is not None:
            band_grays_cache[p.source.page_index] = strips

    # Pass B: cluster candidate positions per (band, parity) in 2D (cx, cy) mm
    def _find_clusters(band: str, parity: int) -> list[dict]:
        pts = []
        for p, pt in zip(pages, page_tokens):
            idx = p.source.page_index
            if idx % 2 != parity:
                continue
            for ti, box in enumerate(pt[band]):
                cx_mm = (box[0] + box[2]) / 2.0 / GUIDED_PX_PER_MM
                cy_mm = (box[1] + box[3]) / 2.0 / GUIDED_PX_PER_MM
                pts.append((idx, ti, cx_mm, cy_mm, box))
        if not pts:
            return []
        bins = Counter((int(round(cx / 3.0)), int(round(cy / 2.0))) for _, _, cx, cy, _ in pts)
        modes = []
        used_centers: list[tuple[float, float]] = []
        min_cluster_n = max(GUIDED_MIN_GROUP_PAGES, int(len(pages) * 0.08))
        for (bx, by), _ in bins.most_common(25):
            cx0, cy0 = bx * 3.0, by * 2.0
            if any(abs(cx0 - ux) < 10.0 and abs(cy0 - uy) < 6.0 for ux, uy in used_centers):
                continue
            near = [q for q in pts if abs(q[2] - cx0) <= 12.0 and abs(q[3] - cy0) <= 7.0]
            if not near:
                continue
            mcx = float(np.median([q[2] for q in near]))
            mcy = float(np.median([q[3] for q in near]))
            by_pg: dict[int, tuple] = {}
            for q in pts:
                dx, dy = abs(q[2] - mcx), abs(q[3] - mcy)
                if dx <= 12.0 and dy <= 7.0:
                    d = dx + dy
                    if q[0] not in by_pg or d < by_pg[q[0]][0]:
                        by_pg[q[0]] = (d, q)
            if len(by_pg) < min_cluster_n:
                continue
            chosen = [v[1] for v in by_pg.values()]
            mcx = float(np.median([q[2] for q in chosen]))
            mcy = float(np.median([q[3] for q in chosen]))
            mad_x = float(np.median([abs(q[2] - mcx) for q in chosen]))
            mad_y = float(np.median([abs(q[3] - mcy) for q in chosen]))
            med_w = float(np.median([(q[4][2] - q[4][0]) / GUIDED_PX_PER_MM for q in chosen]))
            used_centers.append((mcx, mcy))
            modes.append({
                "band": band,
                "parity": parity,
                "mcx": mcx,
                "mcy": mcy,
                "mad_x": mad_x,
                "mad_y": mad_y,
                "med_w": med_w,
                "n": len(chosen),
                "items": chosen,
            })
        return modes

    rec = _Recognizer(mpath)
    max_v = max(len(pages), max((p.source.page_index + 1 for p in pages), default=0)) + 50
    by_pidx_pt = {p.source.page_index: pt for p, pt in zip(pages, page_tokens)}

    def _score_cluster_sequence(cluster: dict) -> tuple[int, int | None, int]:
        items = sorted(cluster["items"], key=lambda q: q[0])
        if len(items) > 60:
            step = len(items) / 60.0
            items = [items[int(k * step)] for k in range(60)]
        band = cluster["band"]
        votes: Counter = Counter()
        for idx, _, _, _, box in items:
            strip = band_grays_cache[idx][0 if band == "top" else 1]
            y_off = 0 if band == "top" else by_pidx_pt[idx].get("bot_y0", bot_y0)
            v, _, _, _ = _read_token_crop(rec, strip, y_off, box, max_v, try_sauvola=True)
            if v is not None:
                votes[v - idx] += 1
        if not votes:
            return 0, None, len(items)
        off, cnt = votes.most_common(1)[0]
        return cnt, off, len(items)

    out_w_mm = out_w0 / GUIDED_PX_PER_MM
    best_pair = None
    best_pair_score = -1e9

    for band in ("bottom", "top"):
        m0s = _find_clusters(band, 0)
        m1s = _find_clusters(band, 1)
        for m in m0s + m1s:
            cnt, off, ntest = _score_cluster_sequence(m)
            m["seq_cnt"] = cnt
            m["seq_off"] = off
            m["seq_frac"] = cnt / max(1, ntest)

        for c0 in m0s:
            for c1 in m1s:
                dy = abs(c0["mcy"] - c1["mcy"])
                if dy > 10.0:
                    continue
                fx0, fx1 = c0["mcx"] / out_w_mm, c1["mcx"] / out_w_mm
                is_mirror = abs((fx0 + fx1) - 1.0) <= 0.14 and abs(fx0 - fx1) >= 0.20
                is_center = abs(fx0 - 0.5) <= 0.12 and abs(fx1 - 0.5) <= 0.12
                if not (is_mirror or is_center):
                    continue
                same_off = (c0["seq_off"] is not None and c0["seq_off"] == c1["seq_off"])
                has_seq = (same_off and (c0["seq_cnt"] + c1["seq_cnt"]) >= 6) or (
                    max(c0["seq_cnt"], c1["seq_cnt"]) >= 5
                )
                if not has_seq:
                    # Without sequence verification (e.g. white-on-black tabs),
                    # require tight geometric consistency so books without real
                    # folios are never falsely matched.
                    min_geom_n = max(6, int(len(have) * 0.15))
                    if c0["n"] < min_geom_n or c1["n"] < min_geom_n:
                        continue
                    if c0["mad_x"] > 3.5 or c1["mad_x"] > 3.5 or c0["mad_y"] > 2.5 or c1["mad_y"] > 2.5:
                        continue
                    if is_center and (c0["mad_x"] > 2.0 or c1["mad_x"] > 2.0 or c0["med_w"] > 10.0 or c1["med_w"] > 10.0):
                        continue
                    if is_mirror and (min(fx0, 1.0 - fx0) > 0.25 or min(fx1, 1.0 - fx1) > 0.25):
                        continue

                seq_score = c0["seq_cnt"] + c1["seq_cnt"] + (25 if same_off else 0)
                outer_bonus = (
                    (1.0 - min(fx0, 1.0 - fx0) - min(fx1, 1.0 - fx1)) * 10.0
                    if is_mirror else 0.0
                )
                total_score = (
                    seq_score * 5.0
                    + (c0["n"] + c1["n"]) * 0.5
                    + outer_bonus
                    - (c0["mad_x"] + c1["mad_x"]) * 4.0
                )
                if total_score > best_pair_score:
                    best_pair_score = total_score
                    best_pair = (band, c0, c1)

    if best_pair is None:
        return {
            "applied": False,
            "reason": "no_consensus_pair",
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }

    # Pass C: Extract final folio box for every page and write back to PageParams
    band, c0, c1 = best_pair
    dom_off: int | None = None
    if c0["seq_off"] is not None and c0["seq_off"] == c1["seq_off"]:
        dom_off = c0["seq_off"]
    elif c0["seq_cnt"] >= c1["seq_cnt"] and c0["seq_cnt"] >= 5:
        dom_off = c0["seq_off"]
    elif c1["seq_cnt"] >= 5:
        dom_off = c1["seq_off"]

    centers = {0: (c0["mcx"], c0["mcy"]), 1: (c1["mcx"], c1["mcy"])}
    seq_verified_n = 0
    pos_only_n = 0
    records = []

    for p, pt in zip(pages, page_tokens):
        if not p.margin or p.margin.nombre_source == "manual":
            continue
        idx = p.source.page_index
        if pt["box_M"] is None:
            if p.margin.nombre_source in (None, "geom"):
                _clear_unlocated_nombre(p)
            continue

        par = idx % 2
        mcx, mcy = centers[par]
        cands = []
        for box in pt[band]:
            cx_mm = (box[0] + box[2]) / 2.0 / GUIDED_PX_PER_MM
            cy_mm = (box[1] + box[3]) / 2.0 / GUIDED_PX_PER_MM
            dx, dy = abs(cx_mm - mcx), abs(cy_mm - mcy)
            if dx <= 20.0 and dy <= 15.0:
                cands.append((dx, dy, box))

        if not cands:
            if p.margin.nombre_source in (None, "geom"):
                _clear_unlocated_nombre(p)
            continue

        strip = band_grays_cache[idx][0 if band == "top" else 1]
        y_off = 0 if band == "top" else pt.get("bot_y0", bot_y0)
        expected_v = (idx + dom_off) if dom_off is not None else None

        scored = []
        for dx, dy, box in cands:
            v_best, t_best, conf_best, is_seq = _read_token_crop(
                rec, strip, y_off, box, max_v,
                expected_v=expected_v, try_sauvola=False,
            )
            if expected_v is not None and not is_seq:
                v2, t2, c2, is_seq2 = _read_token_crop(
                    rec, strip, y_off, box, max_v,
                    expected_v=expected_v, try_sauvola=True,
                )
                if is_seq2:
                    v_best, t_best, conf_best, is_seq = v2, t2, c2, True
            scored.append((is_seq, dx, dy, box, v_best, t_best, conf_best))

        best_box = None
        best_val = None
        best_txt = None
        verified = False

        seq_matches = [s for s in scored if s[0]]
        if seq_matches:
            seq_matches.sort(key=lambda s: s[1] + s[2])
            _, _, _, best_box, best_val, best_txt, _ = seq_matches[0]
            verified = True
            seq_verified_n += 1
        else:
            tight = [s for s in scored if s[1] <= 12.0 and s[2] <= 8.0]
            if tight:
                tight.sort(key=lambda s: s[1] + s[2] * 1.5)
                _, _, _, best_box, best_val, best_txt, _ = tight[0]
                pos_only_n += 1

        if best_box is not None:
            records.append((p, strip, y_off, best_box, verified, pt[band]))
            A = pt["box_M"]
            sx, tx = float(A[0, 0]), float(A[0, 2])
            sy, ty = float(A[1, 1]), float(A[1, 2])
            ox0 = (best_box[0] - tx) / sx
            oy0 = (best_box[1] - ty) / sy
            ox1 = (best_box[2] - tx) / sx
            oy1 = (best_box[3] - ty) / sy
            p.margin.nombre_box = Box(ox0, oy0, ox1, oy1)
            p.margin.nombre_source = "folio"
            if verified:
                p.page_number_source = "folio"
                p.page_number = best_val
                p.nombre_text = best_txt
            else:
                # Avoid trusting a stale mismatched page_number on unverified pages
                # (preserve valid Roman front-matter numerals if present).
                is_roman = bool(
                    p.nombre_text and parse_roman(p.nombre_text) == p.page_number
                    and parse_numeral(p.nombre_text) is None
                )
                if not is_roman:
                    p.page_number = p.page_number_source = None
                    p.nombre_text = None
        elif p.margin.nombre_source in (None, "geom"):
            _clear_unlocated_nombre(p)

    extra = (_read_spaced_pages(records, dom_off, rec) if dom_off is not None
             else _read_inverted_tabs(records, rec, max_v))
    seq_verified_n += extra
    pos_only_n -= extra
    roman_added = _read_roman_front_matter(
        pages, page_tokens, band_grays_cache, band, centers, bot_y0, rec,
    )
    return {
        "applied": True,
        "band": band,
        "dom_off": dom_off,
        "total_detected": seq_verified_n + pos_only_n,
        "seq_verified": seq_verified_n,
        "pos_only": pos_only_n,
        "roman_verified": roman_added,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }
