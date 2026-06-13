"""Optional layout provider (figure/photo regions).

content.analyze() takes any object with `figures(img_bgr) -> list[Box]`
returning figure/photo regions in image pixels. This decouples the pipeline
from a specific layout model: pass YomitokuLayoutProvider for RT-DETRv2 (the
strongest figure detector available here, GO on Intel NPU at ~4.8x in the
sibling project), a custom provider, or None to fall back to the residual-ink
photo heuristic in content.py.

The yomitoku adapter is lazy and optional; importing this module never
requires yomitoku.
"""

from __future__ import annotations

from typing import Protocol

from .model import Box


class LayoutProvider(Protocol):
    def figures(self, img_bgr) -> list[Box]:
        ...


class YomitokuLayoutProvider:
    """RT-DETRv2 layout via yomitoku's LayoutParser. Returns figure regions
    (figures + photos). Loaded lazily on first call."""

    def __init__(self, device: str = "cpu", visualize: bool = False):
        self._device = device
        self._parser = None

    def _ensure(self):
        if self._parser is None:
            from yomitoku.layout_parser import LayoutParser  # optional dep

            self._parser = LayoutParser(device=self._device, visualize=False)
        return self._parser

    def figures(self, img_bgr) -> list[Box]:
        parser = self._ensure()
        results, _ = parser(img_bgr)
        boxes: list[Box] = []
        for el in getattr(results, "figures", []):
            x0, y0, x1, y1 = el.box
            boxes.append(Box(float(x0), float(y0), float(x1), float(y1)))
        return boxes
