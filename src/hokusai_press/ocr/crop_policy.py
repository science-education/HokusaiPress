"""Model-specific expansion of detector boxes before text recognition."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CropPadding:
    pixels: float = 0.0
    ratio: float = 0.0


DEFAULT_CROP_PADDING = {
    "ndlocr": CropPadding(ratio=0.03),
    "yomitoku": CropPadding(ratio=0.01),
    "hybrid": CropPadding(ratio=0.03),
}


def parse_padding(value: str) -> CropPadding:
    value = value.strip().lower()
    try:
        if value.endswith("px"):
            return CropPadding(pixels=float(value[:-2]))
        if value.endswith("%"):
            return CropPadding(ratio=float(value[:-1]) / 100.0)
    except ValueError as exc:
        raise ValueError(f"invalid crop padding: {value!r}") from exc
    raise ValueError(f"crop padding must end in px or %: {value!r}")


def parse_overrides(specs: list[str]) -> dict[str, CropPadding]:
    result = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"crop padding must be MODEL=VALUE: {spec!r}")
        model, value = (part.strip().lower() for part in spec.split("=", 1))
        if model not in DEFAULT_CROP_PADDING:
            raise ValueError(
                f"crop padding model must be one of {', '.join(DEFAULT_CROP_PADDING)}"
            )
        result[model] = parse_padding(value)
    return result


def get_crop_padding(model: str) -> CropPadding:
    padding = DEFAULT_CROP_PADDING.get(model, CropPadding())
    raw = os.environ.get("HOKUSAI_OCR_CROP_PADDING", "")
    if raw:
        padding = parse_overrides([s for s in raw.split(",") if s.strip()]).get(
            model, padding
        )
    return padding


def expand_quad(points, image_shape, padding: CropPadding):
    """Expand an OCR quadrilateral while retaining its approximate perspective."""
    quad = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    low = quad.min(axis=0)
    high = quad.max(axis=0)
    short_side = float(min(high - low))
    amount = padding.pixels + padding.ratio * short_side
    if amount <= 0:
        return quad
    centre = (low + high) / 2.0
    quad[:, 0] += np.where(quad[:, 0] < centre[0], -amount, amount)
    quad[:, 1] += np.where(quad[:, 1] < centre[1], -amount, amount)
    height, width = image_shape[:2]
    quad[:, 0] = np.clip(quad[:, 0], 0, max(0, width - 1))
    quad[:, 1] = np.clip(quad[:, 1], 0, max(0, height - 1))
    return quad

