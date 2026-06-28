import sys
import types

import numpy as np

from hokusai_press.ocr.native import (
    NdlocrLiteEngine,
    YomitokuOCREngine,
)


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
