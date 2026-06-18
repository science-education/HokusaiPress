from hokusai_press.ocr.base import CompositeOCREngine
from hokusai_press.ocr.factory import resolve_engine_selection


class FakeText:
    def recognize_text(self, image_bgr):
        return [{
            "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "box": (0, 0, 10, 10),
            "text": "hello",
            "det_score": 0.9,
            "source": "fake-text",
        }]


class FakeLayout:
    def analyze_layout(self, image_bgr):
        return [{
            "box": (20, 20, 60, 60),
            "label": "figure",
            "score": 0.8,
            "source": "fake-layout",
        }]


class BrokenLayout:
    def analyze_layout(self, image_bgr):
        raise RuntimeError("layout backend failed")


def test_legacy_hybrid_maps_to_yomitoku_and_ndlocr():
    assert resolve_engine_selection("hybrid", None, None) == ("yomitoku", "ndlocr")


def test_split_engines_override_legacy_selector():
    assert resolve_engine_selection(
        "hybrid",
        "pp-structure",
        "ppocr-v6",
    ) == ("pp-structurev3", "ppocr-v6")


def test_paddle_vl_legacy_maps_to_both_components():
    assert resolve_engine_selection("paddle-vl", None, None) == (
        "paddle-vl",
        "paddle-vl",
    )


def test_composite_engine_merges_text_and_layout_results():
    result = CompositeOCREngine(FakeText(), FakeLayout())(None)

    assert result["lines"][0]["source"] == "fake-text"
    assert result["layout_boxes"][0]["source"] == "fake-layout"


def test_composite_engine_keeps_text_when_layout_fails():
    result = CompositeOCREngine(FakeText(), BrokenLayout())(None)

    assert result["lines"][0]["source"] == "fake-text"
    assert result["layout_boxes"] == []
