"""OCR engine adapters used by HokusaiPress.

The public contract is intentionally small: every engine returns text lines in
the shape content.analyze() already consumed from hybrid-ocr, with optional
layout boxes for engines that can parse document structure.
"""

from .factory import get_ocr_engine

__all__ = ["get_ocr_engine"]
