"""Adapters for the native Yomitoku and NDL-OCR Lite pipelines."""

from __future__ import annotations

import argparse
import importlib
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .base import LayoutBox, OCRLine, OCRResult
from .crop_policy import expand_quad, get_crop_padding


def _box_from_polygon(points: Any) -> tuple[float, float, float, float]:
    polygon = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def _yomitoku_device(device: str) -> str:
    if device not in ("", "auto"):
        return device
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _configure_yomitoku_runtime(engine, device: str) -> None:
    # Yomitoku defaults to four DataLoader processes. Console entry points can
    # support that, but embedded/web callers and macOS spawn cannot reliably do
    # so. OCR inference is already batched, so keep it in-process.
    engine.recognizer._cfg.data.num_workers = 0
    if device != "mps":
        return

    # Yomitoku 0.4.1's device setter recognizes CUDA only and otherwise falls
    # back to CPU. Its DBNet and PARSeq models work on MPS, so move both models
    # explicitly until upstream accepts the native device name.
    import torch

    mps = torch.device("mps")
    for module in (engine.detector, engine.recognizer):
        module._device = mps
        module.model.to(mps)


class YomitokuOCREngine:
    """Yomitoku's native DBNet detector plus PARSeq recognizer."""

    def __init__(self, device: str = "auto"):
        from yomitoku.ocr import OCR

        configs = {"text_detector": {}, "text_recognizer": {}}
        runtime_device = _yomitoku_device(device)
        self._engine = OCR(
            configs=configs,
            device=runtime_device,
            visualize=False,
        )
        _configure_yomitoku_runtime(self._engine, runtime_device)
        self._engine.detector = _ExpandingYomitokuDetector(
            self._engine.detector, get_crop_padding("yomitoku")
        )
        self._lock = threading.Lock()

    def __call__(self, image_bgr) -> OCRResult:
        with self._lock:
            result, _ = self._engine(image_bgr)
        lines: list[OCRLine] = []
        for word in result.words:
            polygon = [[float(x), float(y)] for x, y in word.points]
            lines.append({
                "polygon": polygon,
                "box": _box_from_polygon(polygon),
                "text": word.content or None,
                "det_score": float(word.det_score),
                "source": "yomitoku",
            })
        return {"lines": lines, "layout_boxes": [], "raw": result}

    def recognize_text(self, image_bgr) -> list[OCRLine]:
        return self(image_bgr).get("lines", [])


_NDL_LAYOUT_LABELS = {
    2: "caption",
    3: "advertisement",
    4: "note",
    5: "note",
    6: "figure",
    7: "advertisement",
    8: "running_head",
    9: "page_number",
    10: "rubi",
    11: "chart",
    12: "equation",
    13: "colophon",
    15: "table",
    16: "title",
}


def _clean_folio_text(text: str) -> str:
    import re

    text = (text or "").strip()
    match = re.search(r"[0-9０-９ivxlcdmIVXLCDM一二三四五六七八九十〇]{1,8}", text)
    return match.group(0) if match else ""


def _recognize_ndl_folios(image_bgr, detections, recognizers) -> list[OCRLine]:
    """Re-read DEIM folio/pillar crops omitted by ndlocr-lite's JSON lines."""
    height, width = image_bgr.shape[:2]
    lines = []
    for item in detections:
        class_id = int(item["class_index"])
        if class_id not in {8, 9}:  # block_pillar / block_folio
            continue
        x0, y0, x1, y1 = (int(round(float(v))) for v in item["box"])
        cy = (y0 + y1) / 2.0
        if height * 0.14 < cy < height * 0.86:
            continue
        if class_id == 8:
            # The folio is at the outside end of a running-head block.
            if (x0 + x1) / 2.0 < width / 2.0:
                x1 = min(x1, x0 + 50)
            else:
                x0 = max(x0, x1 - 50)
        pad = 5
        bx0, by0 = max(0, x0 - pad), max(0, y0 - pad)
        bx1, by1 = min(width, x1 + pad), min(height, y1 + pad)
        crop = image_bgr[by0:by1, bx0:bx1]
        readings = []
        for priority, recognizer in enumerate(recognizers):
            if not hasattr(recognizer, "read"):
                continue
            try:
                text = _clean_folio_text(recognizer.read(crop))
            except Exception:
                continue
            if text:
                readings.append((len(text), -priority, text))
        if not readings:
            continue
        text = max(readings)[2]
        polygon = [[float(bx0), float(by0)], [float(bx1), float(by0)],
                   [float(bx1), float(by1)], [float(bx0), float(by1)]]
        lines.append({
            "polygon": polygon,
            "box": (float(bx0), float(by0), float(bx1), float(by1)),
            "text": text,
            "det_score": float(item.get("confidence", 0.0)),
            "source": "ndlocr-folio",
            "layout_label": "page_number",
        })
    return lines


class _ExpandingYomitokuDetector:
    def __init__(self, detector, padding):
        self._detector = detector
        self._padding = padding

    def __call__(self, image, *args, **kwargs):
        output = self._detector(image, *args, **kwargs)
        result = output[0] if isinstance(output, tuple) else output
        result.points = [
            expand_quad(points, image.shape, self._padding).tolist()
            for points in result.points
        ]
        return output

    def __getattr__(self, name):
        return getattr(self._detector, name)


_NDL_TEXT_CLASSES = {0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 14, 16}


class _RecordingDetector:
    def __init__(self, detector, padding=None):
        self._detector = detector
        self._padding = padding or get_crop_padding("ndlocr")
        self.detections = []

    def detect(self, image):
        self.detections = []
        for original in self._detector.detect(image):
            item = dict(original)
            if int(item["class_index"]) in _NDL_TEXT_CLASSES:
                x0, y0, x1, y1 = item["box"]
                quad = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                padded = expand_quad(quad, image.shape, self._padding)
                low, high = padded.min(axis=0), padded.max(axis=0)
                item["box"] = [float(low[0]), float(low[1]),
                               float(high[0]), float(high[1])]
            self.detections.append(item)
        return self.detections

    def __getattr__(self, name):
        return getattr(self._detector, name)


def _ndl_args(base_dir: Path, device: str) -> argparse.Namespace:
    model_dir = base_dir / "model"
    config_dir = base_dir / "config"
    return argparse.Namespace(
        det_weights=str(model_dir / "deim-s-1024x1024.onnx"),
        det_classes=str(config_dir / "ndl.yaml"),
        det_score_threshold=0.2,
        det_conf_threshold=0.25,
        det_iou_threshold=0.2,
        rec_weights30=str(
            model_dir / "parseq-ndl-24x256-30-tiny-189epoch-tegaki3-r8data-202604.onnx"
        ),
        rec_weights50=str(
            model_dir / "parseq-ndl-24x384-50-tiny-300epoch-tegaki3-r8data-202604.onnx"
        ),
        rec_weights=str(
            model_dir / "parseq-ndl-24x768-100-tiny-153epoch-tegaki3-r8data-202604.onnx"
        ),
        rec_classes=str(config_dir / "NDLmoji.yaml"),
        device="cuda" if device == "cuda" else "cpu",
        enable_tcy=False,
    )


class NdlocrLiteEngine:
    """Official NDL-OCR Lite DEIMv2 detector plus PARSeq cascade."""

    def __init__(self, device: str = "auto"):
        try:
            ndl = importlib.import_module("ocr")
        except ImportError as exc:
            raise ImportError(
                "NDL-OCR Lite is not installed; install the 'ocr' or 'mac-ocr' extra"
            ) from exc
        if not hasattr(ndl, "_run_ocr_on_image_array"):
            raise ImportError("the installed 'ocr' module is not NDL-OCR Lite")

        args = _ndl_args(Path(ndl.__file__).resolve().parent, device)
        detector = ndl.get_detector(args)
        self._detector = _RecordingDetector(detector)
        self._recognizer30 = ndl.get_recognizer(args, args.rec_weights30)
        self._recognizer50 = ndl.get_recognizer(args, args.rec_weights50)
        self._recognizer100 = ndl.get_recognizer(args, args.rec_weights)
        self._run = ndl._run_ocr_on_image_array
        self._lock = threading.Lock()

    def __call__(self, image_bgr) -> OCRResult:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            result = self._run(
                detector=self._detector,
                recognizer30=self._recognizer30,
                recognizer50=self._recognizer50,
                recognizer100=self._recognizer100,
                inputname="page.png",
                img=image_rgb,
                outputpath=os.devnull,
                save_viz=False,
            )

        lines: list[OCRLine] = []
        for item in result.get("json_lines", []):
            x0, y0, x1, y1 = _box_from_polygon(item["boundingBox"])
            polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            lines.append({
                "polygon": polygon,
                "box": (x0, y0, x1, y1),
                "text": item.get("text") or None,
                "det_score": float(item.get("confidence", 0.0)),
                "source": "ndlocr-lite",
            })
        lines.extend(_recognize_ndl_folios(
            image_bgr, self._detector.detections,
            (self._recognizer30, self._recognizer50, self._recognizer100),
        ))

        layout_boxes: list[LayoutBox] = []
        for item in self._detector.detections:
            class_id = int(item["class_index"])
            label = _NDL_LAYOUT_LABELS.get(class_id)
            if label is None:
                continue
            x0, y0, x1, y1 = (float(v) for v in item["box"])
            layout_boxes.append({
                "box": (x0, y0, x1, y1),
                "label": label,
                "raw_label": str(item.get("class_name", "")),
                "class_id": class_id,
                "score": float(item.get("confidence", 0.0)),
                "source": "ndlocr-lite",
            })
        return {"lines": lines, "layout_boxes": layout_boxes, "raw": result}

    def recognize_text(self, image_bgr) -> list[OCRLine]:
        return self(image_bgr).get("lines", [])

    def analyze_layout(self, image_bgr) -> list[LayoutBox]:
        return self(image_bgr).get("layout_boxes", [])
