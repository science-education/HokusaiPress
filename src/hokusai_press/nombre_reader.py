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
    value: int
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
        best: tuple[float, int, str, str] | None = None
        for method in methods:
            text, conf = recognizer.predict(crop, method)
            value = parse_numeral(text)
            kind = "num"
            if value is None:
                value = parse_roman(text)
                kind = "roman"
            if value is None or value <= 0:
                continue
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
    env = os.environ.get("HOKUSAI_FOLIO_DIGIT_ONNX")
    if env:
        return Path(env)
    # Platform-independent default: look relative to the project root
    # (two levels above this source file), then the current working directory.
    pkg_root = Path(__file__).resolve().parents[2]
    candidates = [
        pkg_root / "models" / "nombre",
        Path("models") / "nombre",
    ]
    for root in candidates:
        if not root.is_dir():
            continue
        bin_model = root / "folio_digit_bin.onnx"
        if bin_model.exists():
            return bin_model
        ft = root / "folio_digit_ft.onnx"
        if ft.exists():
            return ft
        base = root / "folio_digit.onnx"
        if base.exists():
            return base
    # No model found — return a descriptive path so the error message is clear.
    return pkg_root / "models" / "nombre" / "folio_digit.onnx"


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
    use = ["CPUExecutionProvider"]
    if "CPUExecutionProvider" not in providers:
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
