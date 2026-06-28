from hokusai_press.ocr.base import CompositeOCREngine
from hokusai_press.ocr.factory import resolve_engine_selection
from hokusai_press.ocr.hybrid import (
    _OnnxOutputNameCompatSession,
    _fix_detector_output_name,
)


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


class FakeOutput:
    def __init__(self, name):
        self.name = name


class FakeSession:
    def __init__(self, output_names):
        self.output_names = output_names
        self.calls = []

    def get_outputs(self):
        return [FakeOutput(name) for name in self.output_names]

    def run(self, output_names, input_feed, *args, **kwargs):
        self.calls.append((output_names, input_feed))
        return ["prediction"]


class FakeDetector:
    def __init__(self, session):
        self.session = session


class FakeHybridEngine:
    def __init__(self, session):
        self.detector = FakeDetector(session)


def test_legacy_hybrid_maps_to_native_hybrid_pipeline():
    assert resolve_engine_selection("hybrid", None, None) == ("hybrid", "hybrid")


def test_native_ocr_engines_keep_their_own_detector_and_recognizer():
    assert resolve_engine_selection("yomitoku", None, None) == (
        "yomitoku", "yomitoku",
    )
    assert resolve_engine_selection("ndlocr", None, None) == (
        "ndlocr", "ndlocr",
    )


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


def test_paddle_vl_16_alias_maps_to_both_components():
    assert resolve_engine_selection("paddle-vl-1.6", None, None) == (
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


def test_hybrid_detector_uses_actual_single_output_name():
    session = FakeSession(["preds"])
    engine = FakeHybridEngine(session)

    _fix_detector_output_name(engine)
    result = engine.detector.session.run(["output"], {"input": "tensor"})

    assert isinstance(engine.detector.session, _OnnxOutputNameCompatSession)
    assert session.calls == [(["preds"], {"input": "tensor"})]
    assert result == ["prediction"]


def test_hybrid_detector_keeps_matching_output_session_unchanged():
    session = FakeSession(["output"])
    engine = FakeHybridEngine(session)

    _fix_detector_output_name(engine)

    assert engine.detector.session is session


def test_hybrid_detector_does_not_guess_among_multiple_outputs():
    session = FakeSession(["preds", "aux"])
    engine = FakeHybridEngine(session)

    _fix_detector_output_name(engine)

    assert engine.detector.session is session
