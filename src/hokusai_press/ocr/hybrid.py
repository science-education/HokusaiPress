"""Adapter for the existing Yomitoku/NDL hybrid OCR engine."""

from __future__ import annotations

import os


class HybridOCREngine:
    def __init__(self, model_dir: str, device: str, openvino_cache_dir=None):
        from hybrid_ocr.pipeline import HybridOCR  # lazy, optional dependency

        if not os.path.isdir(model_dir):
            try:
                from hybrid_ocr.cli import resolve_model_dir

                model_dir = resolve_model_dir(None)
            except Exception:
                pass
        self._engine = HybridOCR(
            model_dir=model_dir,
            device=device,
            openvino_cache_dir=openvino_cache_dir,
        )

    def __call__(self, image_bgr):
        result = self._engine(image_bgr)
        for line in result.get("lines", []):
            line.setdefault("source", "ndlocr")
        return result


class NdlocrTextEngine(HybridOCREngine):
    def recognize_text(self, image_bgr):
        return self(image_bgr).get("lines", [])


class YomitokuLayoutEngine:
    def __init__(self, device: str = "cpu"):
        from hokusai_press.layout import YomitokuLayoutProvider

        self._provider = YomitokuLayoutProvider(device=device)

    def analyze_layout(self, image_bgr):
        boxes = []
        for box in self._provider.figures(image_bgr):
            boxes.append({
                "box": (box.x0, box.y0, box.x1, box.y1),
                "label": "figure",
                "score": None,
                "source": "yomitoku",
            })
        return boxes
