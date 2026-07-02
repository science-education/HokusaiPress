import sys
import types

import numpy as np

from hokusai_press.ocr.native import (
    NdlocrLiteEngine,
    YomitokuOCREngine,
    _RecordingDetector,
)
from hokusai_press.ocr.crop_policy import CropPadding, expand_quad, parse_overrides


class _Word:
    points = [[1, 2], [11, 2], [11, 12], [1, 12]]
    content = "科学"
    det_score = 0.9
    rec_score = 0.8


class _YomitokuResult:
    words = [_Word()]


class _FakeYomitokuOCR:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.recognizer = types.SimpleNamespace(
            _cfg=types.SimpleNamespace(data=types.SimpleNamespace(num_workers=4))
        )
        self.detector = object()

    def __call__(self, image):
        return _YomitokuResult(), None


class _FakeDetector:
    def detect(self, image):
        return [{
            "box": [20, 30, 80, 90],
            "class_index": 6,
            "class_name": "block_fig",
            "confidence": 0.95,
        }]


class _FakeNdlModule(types.ModuleType):
    __file__ = "/tmp/ndlocr-lite/ocr.py"

    @staticmethod
    def get_detector(args):
        return _FakeDetector()

    @staticmethod
    def get_recognizer(args, weights_path=None):
        return object()

    @staticmethod
    def _run_ocr_on_image_array(**kwargs):
        kwargs["detector"].detect(kwargs["img"])
        return {
            "json_lines": [{
                "boundingBox": [[2, 3], [2, 13], [12, 3], [12, 13]],
                "text": "研究",
                "confidence": 0.85,
            }],
        }


def test_yomitoku_native_engine_normalizes_words(monkeypatch):
    module = types.ModuleType("yomitoku.ocr")
    module.OCR = _FakeYomitokuOCR
    monkeypatch.setitem(sys.modules, "yomitoku.ocr", module)

    engine = YomitokuOCREngine(device="cpu")
    result = engine(np.zeros((20, 20, 3), dtype=np.uint8))

    assert result["lines"] == [{
        "polygon": [[1.0, 2.0], [11.0, 2.0], [11.0, 12.0], [1.0, 12.0]],
        "box": (1.0, 2.0, 11.0, 12.0),
        "text": "科学",
        "det_score": 0.9,
        "source": "yomitoku",
    }]
    assert engine._engine.recognizer._cfg.data.num_workers == 0


def test_ndlocr_native_engine_normalizes_lines_and_layout(monkeypatch):
    monkeypatch.setitem(sys.modules, "ocr", _FakeNdlModule("ocr"))

    engine = NdlocrLiteEngine(device="cpu")
    result = engine(np.zeros((100, 100, 3), dtype=np.uint8))

    assert result["lines"][0]["text"] == "研究"
    assert result["lines"][0]["box"] == (2.0, 3.0, 12.0, 13.0)
    assert result["layout_boxes"] == [{
        "box": (20.0, 30.0, 80.0, 90.0),
        "label": "figure",
        "raw_label": "block_fig",
        "class_id": 6,
        "score": 0.95,
        "source": "ndlocr-lite",
    }]


def test_crop_padding_parses_pixels_and_percent():
    values = parse_overrides(["ndlocr=4px", "yomitoku=2%"])
    assert values["ndlocr"] == CropPadding(pixels=4.0)
    assert values["yomitoku"] == CropPadding(ratio=0.02)


def test_expand_quad_uses_short_side_and_clamps():
    result = expand_quad(
        [[1, 2], [21, 2], [21, 12], [1, 12]],
        (20, 30, 3), CropPadding(ratio=0.2),
    )
    assert result.tolist() == [[0.0, 0.0], [23.0, 0.0], [23.0, 14.0], [0.0, 14.0]]


def test_ndl_padding_expands_text_but_not_figure():
    class Detector:
        def detect(self, image):
            return [
                {"box": [10, 20, 40, 30], "class_index": 1},
                {"box": [10, 40, 40, 60], "class_index": 6},
            ]

    detector = _RecordingDetector(Detector(), CropPadding(pixels=2))
    detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert detections[0]["box"] == [8.0, 18.0, 42.0, 32.0]
    assert detections[1]["box"] == [10, 40, 40, 60]
