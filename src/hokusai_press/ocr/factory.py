"""OCR engine factory with one process-wide composite engine instance.

NPU-backed OCR engines are expensive and fragile when several contexts are
opened in one process. Keep a single active layout+text combination and rebuild
it only when the requested configuration changes.
"""

from __future__ import annotations

import gc
import importlib.util
import threading
from dataclasses import dataclass

from .base import CompositeOCREngine, LayoutEngine, OCREngine, TextEngine


@dataclass(frozen=True)
class OCRConfig:
    layout_engine: str
    text_engine: str
    model_dir: str
    device: str
    runtime: str | None
    openvino_cache_dir: str | None


_engine: OCREngine | None = None
_config: OCRConfig | None = None
_engine_lock = threading.Lock()


def get_ocr_engine(
    name: str | None = None,
    model_dir: str = "models",
    device: str = "auto",
    openvino_cache_dir: str | None = None,
    runtime: str | None = "paddle",
    layout_engine: str | None = None,
    text_engine: str | None = None,
) -> OCREngine:
    global _engine, _config

    layout_name, text_name = resolve_engine_selection(name, layout_engine, text_engine)
    runtime = None if runtime in ("", "auto") else runtime
    cfg = OCRConfig(
        layout_engine=layout_name,
        text_engine=text_name,
        model_dir=model_dir,
        device=device,
        runtime=runtime,
        openvino_cache_dir=openvino_cache_dir,
    )
    if _engine is not None and _config == cfg:
        return _engine

    with _engine_lock:
        if _engine is not None and _config == cfg:
            return _engine
        _engine = None
        _config = None
        gc.collect()

        # Avoid loading a heavy VLM or out-of-process worker twice when the user
        # intentionally delegates both layout and text to one backend.
        if layout_name == "paddle-vl" and text_name == "paddle-vl":
            from .paddle import PaddleOCRVLEngine

            _engine = PaddleOCRVLEngine(model_dir, device, runtime)
        elif layout_name == "deim-oop" and text_name == "deim-oop":
            from .oop import OutOfProcessEngine

            _engine = OutOfProcessEngine(device="cpu" if device == "auto" else device)
        else:
            _engine = CompositeOCREngine(
                text_engine=_build_text_engine(text_name, model_dir, device,
                                               openvino_cache_dir, runtime),
                layout_engine=_build_layout_engine(layout_name, model_dir, device,
                                                   runtime),
            )
        _config = cfg
        return _engine


def resolve_engine_selection(
    legacy_engine: str | None,
    layout_engine: str | None,
    text_engine: str | None,
) -> tuple[str, str]:
    """Map legacy single-engine choices into layout/text components."""

    legacy = _norm(legacy_engine or "hybrid")
    legacy_map = {
        "auto": ("yomitoku", "ndlocr"),
        "hybrid": ("yomitoku", "ndlocr"),
        "yomitoku": ("yomitoku", "ndlocr"),
        "ndlocr": ("yomitoku", "ndlocr"),
        "ppocr-v6": ("none", "ppocr-v6"),
        "paddleocr-v6": ("none", "ppocr-v6"),
        "pp-structurev3": ("pp-structurev3", "none"),
        "pp-structure": ("pp-structurev3", "none"),
        "paddle-vl": ("paddle-vl", "paddle-vl"),
        "paddleocr-vl": ("paddle-vl", "paddle-vl"),
        "deim-oop": ("deim-oop", "deim-oop"),
        "none": ("none", "none"),
    }
    base_layout, base_text = legacy_map.get(legacy, (legacy, legacy))
    return (
        _norm(layout_engine) if layout_engine else base_layout,
        _norm(text_engine) if text_engine else base_text,
    )


def _norm(name: str | None) -> str:
    value = (name or "none").lower().replace("_", "-")
    aliases = {
        "paddleocr-v6": "ppocr-v6",
        "paddleocr-vl": "paddle-vl",
        "pp-structure": "pp-structurev3",
        "ppstructurev3": "pp-structurev3",
        "off": "none",
        "false": "none",
    }
    return aliases.get(value, value)


def _build_text_engine(
    name: str,
    model_dir: str,
    device: str,
    openvino_cache_dir: str | None,
    runtime: str | None,
) -> TextEngine | None:
    if name == "none":
        return None
    if name == "ndlocr":
        from .hybrid import NdlocrTextEngine

        return NdlocrTextEngine(model_dir, device, openvino_cache_dir)
    if name == "ppocr-v6":
        from .paddle import PPOCRv6Engine

        return PPOCRv6Engine(model_dir, device, runtime)
    if name == "paddle-vl":
        from .paddle import PaddleOCRVLEngine

        return PaddleOCRVLEngine(model_dir, device, runtime)
    if name == "deim-oop":
        from .oop import OutOfProcessEngine

        return OutOfProcessEngine(device="cpu" if device == "auto" else device)
    raise ValueError(f"unknown text engine: {name}")


def _build_layout_engine(
    name: str,
    model_dir: str,
    device: str,
    runtime: str | None,
) -> LayoutEngine | None:
    if name == "none":
        return None
    if name == "yomitoku":
        if importlib.util.find_spec("yomitoku") is None:
            return None
        try:
            from .hybrid import YomitokuLayoutEngine

            return YomitokuLayoutEngine(device=device)
        except ImportError:
            # Yomitoku layout is optional; keep the existing text OCR path alive
            # when the RT-DETR layout dependency is not installed.
            return None
    if name == "pp-structurev3":
        from .paddle import PPStructureV3LayoutEngine

        return PPStructureV3LayoutEngine(model_dir, device, runtime)
    if name == "paddle-vl":
        from .paddle import PaddleOCRVLEngine

        return PaddleOCRVLEngine(model_dir, device, runtime)
    if name == "deim-oop":
        from .oop import OutOfProcessEngine

        return OutOfProcessEngine(device="cpu" if device == "auto" else device)
    raise ValueError(f"unknown layout engine: {name}")
