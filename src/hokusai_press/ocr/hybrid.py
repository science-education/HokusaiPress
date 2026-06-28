"""Adapter for the existing Yomitoku/NDL hybrid OCR engine."""

from __future__ import annotations

import os


class _OnnxOutputNameCompatSession:
    """Translate hybrid-ocr's legacy output name for single-output models.

    YomiToku's current dynamic DBNet export calls its sole output ``preds``,
    while hybrid-ocr 0.1.0 requests the older fixed-export name ``output``.
    Delegate everything except that one unambiguous name mismatch.
    """

    def __init__(self, session, output_name: str):
        self._session = session
        self._output_name = output_name

    def run(self, output_names, input_feed, *args, **kwargs):
        if output_names == ["output"]:
            output_names = [self._output_name]
        return self._session.run(output_names, input_feed, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._session, name)


def _fix_detector_output_name(engine) -> None:
    """Make hybrid-ocr compatible with the current YomiToku DBNet export."""

    detector = getattr(engine, "detector", None)
    session = getattr(detector, "session", None)
    if session is None:
        return
    output_names = [output.name for output in session.get_outputs()]
    if "output" not in output_names and len(output_names) == 1:
        detector.session = _OnnxOutputNameCompatSession(session, output_names[0])


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
        _fix_detector_output_name(self._engine)

    def __call__(self, image_bgr):
        result = self._engine(image_bgr)
        for line in result.get("lines", []):
            line.setdefault("source", "ndlocr")
        return result

    def recognize_text(self, image_bgr):
        return self(image_bgr).get("lines", [])


class NdlocrTextEngine(HybridOCREngine):
    def recognize_text(self, image_bgr):
        return self(image_bgr).get("lines", [])


class YomitokuLayoutEngine:
    def __init__(self, device: str = "cpu"):
        from hokusai_press.layout import YomitokuLayoutProvider

        self._provider = YomitokuLayoutProvider(device=device)

    def analyze_layout(self, image_bgr):
        boxes = []
        try:
            figures = self._provider.figures(image_bgr)
        except ImportError:
            return boxes
        for box in figures:
            boxes.append({
                "box": (box.x0, box.y0, box.x1, box.y1),
                "label": "figure",
                "score": None,
                "source": "yomitoku",
            })
        return boxes
