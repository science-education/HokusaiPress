import numpy as np

from hokusai_press import pipeline
from hokusai_press.model import Box, Deskew, Margin, SourceRef


def test_finish_page_forwards_ocr_engine_options(monkeypatch):
    received = {}

    def fake_analyze(*args, **kwargs):
        received.update(kwargs)
        return [], []

    margin = Margin(Box(0, 0, 20, 20), confidence=1.0)
    monkeypatch.setattr(pipeline.content_mod, "analyze", fake_analyze)
    monkeypatch.setattr(
        pipeline,
        "compute_margin",
        lambda original, skew, regions: (original, margin),
    )

    image = np.full((20, 20, 3), 255, np.uint8)
    source = SourceRef("page.png", page_index=0)
    args = (
        source,
        image,
        image,
        300,
        set(),
        "models",
        "npu",
        "paddle-vl",
        "deim",
        "paddle-text",
        True,
        None,
        "cache",
        "openvino",
    )

    pipeline._finish_page(args, Deskew(), image)

    assert received["ocr_engine"] == "paddle-vl"
    assert received["layout_engine"] == "deim"
    assert received["text_engine"] == "paddle-text"
    assert received["runtime"] == "openvino"

