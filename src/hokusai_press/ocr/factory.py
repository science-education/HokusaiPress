"""OCR engine factory with one process-wide engine instance.

NPU-backed OCR engines are expensive and fragile when several contexts are
opened in one process. Keep a single active adapter and rebuild it only when
the requested configuration changes.
"""

from __future__ import annotations

import gc
import threading
from dataclasses import dataclass

from .base import OCREngine


@dataclass(frozen=True)
class OCRConfig:
    name: str
    model_dir: str
    device: str
    openvino_cache_dir: str | None
    paddle_engine: str | None


_engine: OCREngine | None = None
_config: OCRConfig | None = None
_engine_lock = threading.Lock()


def get_ocr_engine(
    name: str = "hybrid",
    model_dir: str = "models",
    device: str = "auto",
    openvino_cache_dir: str | None = None,
    paddle_engine: str | None = "paddle",
) -> OCREngine:
    global _engine, _config

    normalized = {
        "auto": "hybrid",
        "hybrid": "hybrid",
        "yomitoku": "hybrid",
        "ndlocr": "hybrid",
        "ppocr-v6": "ppocr-v6",
        "paddleocr-v6": "ppocr-v6",
        "paddle-vl": "paddle-vl",
        "paddleocr-vl": "paddle-vl",
    }.get(name, name)
    cfg = OCRConfig(normalized, model_dir, device, openvino_cache_dir, paddle_engine)
    if _engine is not None and _config == cfg:
        return _engine

    with _engine_lock:
        if _engine is not None and _config == cfg:
            return _engine
        _engine = None
        _config = None
        gc.collect()

        if normalized == "hybrid":
            from .hybrid import HybridOCREngine

            _engine = HybridOCREngine(model_dir, device, openvino_cache_dir)
        elif normalized == "ppocr-v6":
            from .paddle import PPOCRv6Engine

            _engine = PPOCRv6Engine(model_dir, device, paddle_engine)
        elif normalized == "paddle-vl":
            from .paddle import PaddleOCRVLEngine

            _engine = PaddleOCRVLEngine(model_dir, device, paddle_engine)
        else:
            raise ValueError(f"unknown OCR engine: {name}")
        _config = cfg
        return _engine
