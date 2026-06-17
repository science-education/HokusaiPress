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
