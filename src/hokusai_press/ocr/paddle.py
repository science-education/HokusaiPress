"""PaddleOCR adapters.

PaddleOCR 3.x returns rich result objects whose exact keys differ between
PP-OCR and PaddleOCR-VL. This module normalizes the useful parts into the
small HokusaiPress OCR contract: text lines plus optional layout boxes.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .base import LayoutBox, OCRLine, OCRResult

_LAYOUT_IMAGE_LABELS = {
    "image",
    "figure",
    "figure_title",
    "chart",
    "table",
    "formula",
    "algorithm",
    "seal",
}


def _device_for_paddle(device: str) -> str | None:
    if device in ("", "auto", None):
        return None
    if device == "npu":
        return "npu:0"
    if device in ("gpu", "xpu", "mlu", "dcu", "metax_gpu", "iluvatar_gpu"):
        return f"{device}:0"
    return device


def _result_mapping(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj.get("res", obj)
    if hasattr(obj, "res") and isinstance(obj.res, Mapping):
        return obj.res
    if hasattr(obj, "json"):
        js = obj.json
        if isinstance(js, Mapping):
            return js.get("res", js)
    return {}


def _iter_results(output: Any) -> Iterable[Any]:
    if output is None:
        return []
    if isinstance(output, (str, bytes, Mapping)):
        return [output]
    try:
        return list(output)
    except TypeError:
        return [output]


def _to_float(v: Any) -> float:
    if hasattr(v, "item"):
        v = v.item()
    return float(v)


def _poly_to_box(poly: Any) -> tuple[float, float, float, float]:
    arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    return (_to_float(mins[0]), _to_float(mins[1]),
            _to_float(maxs[0]), _to_float(maxs[1]))


def _box_to_poly(box: Any) -> list[list[float]]:
    x0, y0, x1, y1 = [_to_float(v) for v in box]
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _box_rows(boxes: Any) -> list[Any]:
    if boxes is None:
        return []
    arr = np.asarray(boxes)
    if arr.ndim == 1 and arr.size == 4:
        return [arr.tolist()]
    return list(boxes)


def _poly_rows(polys: Any) -> list[Any]:
    if polys is None:
        return []
    arr = np.asarray(polys)
    if arr.ndim == 2 and arr.shape[-1] == 2:
        return [arr.tolist()]
    return list(polys)


def _dedupe_lines(lines: list[OCRLine]) -> list[OCRLine]:
    seen = set()
    out: list[OCRLine] = []
    for line in lines:
        box = line.get("box")
        key = (
            tuple(round(float(v), 1) for v in box) if box else None,
            line.get("text"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def _build(cls, kwargs: dict[str, Any]):
    """Instantiate a Paddle class across minor API drift.

    Current PaddleOCR docs expose `engine` and `device`, while older releases
    used narrower constructor signatures. Retry by dropping optional keywords
    so importing HokusaiPress does not bind us to one exact PaddleOCR release.
    """
    keys_to_drop = [
        (),
        ("engine",),
        ("engine", "device"),
        ("engine", "device", "text_detection_model_dir", "text_recognition_model_dir"),
        ("engine", "device", "layout_detection_model_dir", "vl_rec_model_dir"),
        ("engine", "device", "layout_detection_model_dir", "vl_rec_model_dir",
         "layout_shape_mode"),
    ]
    last_error = None
    for drop in keys_to_drop:
        filtered = {k: v for k, v in kwargs.items() if k not in drop}
        try:
            return cls(**filtered)
        except TypeError as exc:
            last_error = exc
    raise last_error  # type: ignore[misc]


def _normalize_lines(data: Any, source: str) -> list[OCRLine]:
    lines: list[OCRLine] = []

    # PaddleOCR 2.x style fallback: [[[poly], (text, score)], ...]
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            poly, rec = item[0], item[1]
            text, score = (rec[0], rec[1]) if isinstance(rec, (list, tuple)) else (None, None)
            lines.append({
                "polygon": np.asarray(poly, dtype=np.float32).reshape(-1, 2).tolist(),
                "box": _poly_to_box(poly),
                "text": str(text) if text else None,
                "det_score": None if score is None else _to_float(score),
                "source": source,
            })
        return lines
    if not isinstance(data, Mapping):
        return lines

    polys = _poly_rows(data.get("rec_polys", data.get("dt_polys")))
    boxes = _box_rows(data.get("rec_boxes"))
    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", data.get("dt_scores", []))
    if polys or boxes:
        n = max(len(texts), len(scores), len(polys), len(boxes))
        for i in range(n):
            poly = None if i >= len(polys) else polys[i]
            box = None if i >= len(boxes) else boxes[i]
            if poly is None and box is None:
                continue
            if box is None:
                norm_box = _poly_to_box(poly)
                norm_poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2).tolist()
            else:
                norm_box = tuple(_to_float(v) for v in box)
                norm_poly = (_box_to_poly(norm_box) if poly is None else
                             np.asarray(poly, dtype=np.float32).reshape(-1, 2).tolist())
            lines.append({
                "polygon": norm_poly,
                "box": norm_box,  # type: ignore[typeddict-item]
                "text": None if i >= len(texts) else str(texts[i]) or None,
                "det_score": None if i >= len(scores) else _to_float(scores[i]),
                "source": source,
            })
        return lines

    raw_lines = data.get("lines", [])
    if isinstance(raw_lines, list):
        for item in raw_lines:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            poly, rec = item[0], item[1]
            text, score = (rec[0], rec[1]) if isinstance(rec, (list, tuple)) else (None, None)
            lines.append({
                "polygon": np.asarray(poly, dtype=np.float32).reshape(-1, 2).tolist(),
                "box": _poly_to_box(poly),
                "text": str(text) if text else None,
                "det_score": None if score is None else _to_float(score),
                "source": source,
            })
    return lines


def _normalize_layout_boxes(data: Mapping[str, Any], source: str) -> list[LayoutBox]:
    layout = data.get("layout_det_res") or data.get("layout_res") or {}
    if isinstance(layout, Mapping):
        boxes = layout.get("boxes", [])
    else:
        boxes = []
    out: list[LayoutBox] = []
    for item in boxes:
        if not isinstance(item, Mapping):
            continue
        label = str(item.get("label", "")).lower()
        coord = item.get("coordinate") or item.get("box") or item.get("bbox")
        if not coord or len(coord) < 4:
            continue
        if label not in _LAYOUT_IMAGE_LABELS:
            continue
        out.append({
            "box": tuple(_to_float(v) for v in coord[:4]),  # type: ignore[typeddict-item]
            "label": label,
            "score": None if item.get("score") is None else _to_float(item["score"]),
            "source": source,
        })
    return out


def normalize_paddle_ocr_result(output: Any, source: str = "ppocr-v6") -> OCRResult:
    lines: list[OCRLine] = []
    for res in _iter_results(output):
        data = _result_mapping(res)
        if data:
            lines.extend(_normalize_lines(data, source))
        elif isinstance(res, list):
            lines.extend(_normalize_lines(res, source))  # type: ignore[arg-type]
    return {"lines": _dedupe_lines(lines), "layout_boxes": [], "raw": output}


def normalize_paddle_vl_result(output: Any, source: str = "paddle-vl") -> OCRResult:
    lines: list[OCRLine] = []
    layout_boxes: list[LayoutBox] = []
    for res in _iter_results(output):
        data = _result_mapping(res)
        if not data:
            continue
        top_lines = _normalize_lines(data, source)
        lines.extend(top_lines)
        layout_boxes.extend(_normalize_layout_boxes(data, source))
        # Some PaddleOCR-VL builds nest OCR output under named sub-results.
        if not top_lines:
            for key in ("overall_ocr_res", "ocr_res", "text_det_res"):
                sub = data.get(key)
                if isinstance(sub, Mapping):
                    lines.extend(_normalize_lines(sub, source))
    return {"lines": _dedupe_lines(lines), "layout_boxes": layout_boxes, "raw": output}


class PPOCRv6Engine:
    def __init__(self, model_dir: str | None = None, device: str = "auto",
                 engine: str | None = "paddle"):
        from paddleocr import PaddleOCR  # lazy, optional dependency

        kwargs: dict[str, Any] = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "engine": engine,
        }
        paddle_device = _device_for_paddle(device)
        if paddle_device:
            kwargs["device"] = paddle_device
        if model_dir:
            kwargs["text_detection_model_dir"] = model_dir
            kwargs["text_recognition_model_dir"] = model_dir
        self._ocr = _build(PaddleOCR, {k: v for k, v in kwargs.items() if v is not None})
        self._lock = threading.Lock()

    def __call__(self, image_bgr) -> OCRResult:
        with self._lock:
            return normalize_paddle_ocr_result(self._ocr.predict(image_bgr))


class PaddleOCRVLEngine:
    def __init__(self, model_dir: str | None = None, device: str = "auto",
                 engine: str | None = "paddle"):
        from paddleocr import PaddleOCRVL  # lazy, optional dependency

        kwargs: dict[str, Any] = {
            "pipeline_version": "v1.6",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_layout_detection": True,
            "layout_shape_mode": "rect",
            "engine": engine,
        }
        paddle_device = _device_for_paddle(device)
        if paddle_device:
            kwargs["device"] = paddle_device
        if model_dir:
            kwargs["layout_detection_model_dir"] = model_dir
            kwargs["vl_rec_model_dir"] = model_dir
        self._pipeline = _build(
            PaddleOCRVL,
            {k: v for k, v in kwargs.items() if v is not None},
        )
        self._lock = threading.Lock()

    def __call__(self, image_bgr) -> OCRResult:
        with self._lock:
            return normalize_paddle_vl_result(self._pipeline.predict(input=image_bgr))
