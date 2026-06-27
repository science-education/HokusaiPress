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


# Canonical layout vocabulary shared across OCR engines. Each engine's worker
# normalizes its native classes into one of these labels; `raw_label` keeps the
# engine-native name (e.g. "block_folio") for traceability and future use.
CANONICAL_LAYOUT_LABELS = frozenset({
    "text", "title", "caption", "note", "rubi",
    "figure", "chart", "table", "equation",
    "page_number", "running_head", "advertisement", "colophon",
})
# Labels a consumer should treat as a figure/photo (image) region (MRC).
FIGURE_LIKE_LABELS = frozenset({"figure", "chart", "table", "photo", "advertisement"})


class LayoutBox(TypedDict, total=False):
    box: tuple[float, float, float, float]
    label: str            # canonical label (see CANONICAL_LAYOUT_LABELS)
    raw_label: str        # engine-native label, e.g. "block_folio"
    class_id: int         # engine-native class id (optional)
    text: str             # recognized text for furniture (folio / running head)
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
        layout_boxes = []
        if self._layout_engine is not None:
            try:
                layout_boxes = self._layout_engine.analyze_layout(image_bgr)
            except Exception:
                # Layout is an enrichment signal. Keep text OCR usable when an
                # optional layout backend is absent or fails on a page.
                layout_boxes = []
        return {"lines": lines, "layout_boxes": layout_boxes}
