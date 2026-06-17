"""Shared OCR adapter types."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict

import numpy as np


class OCRLine(TypedDict, total=False):
    polygon: list[list[float]]
    box: tuple[float, float, float, float]
    text: str | None
    det_score: float | None
    source: str


class LayoutBox(TypedDict, total=False):
    box: tuple[float, float, float, float]
    label: str
    score: float | None
    source: str


class OCRResult(TypedDict, total=False):
    lines: list[OCRLine]
    layout_boxes: list[LayoutBox]
    raw: Any


class OCREngine(Protocol):
    def __call__(self, image_bgr: np.ndarray) -> OCRResult:
        ...


class TextEngine(Protocol):
    def recognize_text(self, image_bgr: np.ndarray) -> list[OCRLine]:
        ...


class LayoutEngine(Protocol):
    def analyze_layout(self, image_bgr: np.ndarray) -> list[LayoutBox]:
        ...


class CompositeOCREngine:
    """Compose independent layout and text engines behind the old result shape."""

    def __init__(
        self,
        text_engine: TextEngine | None = None,
        layout_engine: LayoutEngine | None = None,
    ):
        self._text_engine = text_engine
        self._layout_engine = layout_engine

    def __call__(self, image_bgr: np.ndarray) -> OCRResult:
        lines = (
            self._text_engine.recognize_text(image_bgr)
            if self._text_engine is not None else []
        )
        layout_boxes = (
            self._layout_engine.analyze_layout(image_bgr)
            if self._layout_engine is not None else []
        )
        return {"lines": lines, "layout_boxes": layout_boxes}
