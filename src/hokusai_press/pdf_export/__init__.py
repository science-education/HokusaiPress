from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .exceptions import (
    PDFExportError,
    UnsupportedCodecError,
    MissingFontError,
    BackendUnavailableError,
)
from .backend import (
    decide_page_mode as _internal_decide_page_mode,
    encode_page_pdf as _internal_encode_page_pdf,
    build_text_overlay as _internal_build_text_overlay,
    _encode_gray_flate_page_pdf,
)

__all__ = [
    "PDFExportError",
    "UnsupportedCodecError",
    "MissingFontError",
    "BackendUnavailableError",
    "SearchablePdfBuilder",
    "encode_page_pdf",
    "build_text_overlay",
    "decide_page_mode",
    "_encode_gray_flate_page_pdf",
]


def decide_page_mode(img_bgr: np.ndarray, lines: list, backend: str = "auto") -> str:
    """auto mode decision for page image mode."""
    if backend not in ("auto", "internal", "hybrid_ocr"):
        raise ValueError(f"backend must be one of {('auto', 'internal', 'hybrid_ocr')}")
    if not isinstance(img_bgr, np.ndarray):
        raise TypeError("img_bgr must be a numpy.ndarray")
    if img_bgr.size == 0 or img_bgr.ndim < 2:
        raise ValueError("img_bgr must be a non-empty 2D or 3D array")

    chosen = backend
    if chosen == "auto":
        chosen = "internal"

    if chosen == "hybrid_ocr":
        try:
            from hybrid_ocr.pdf_export import decide_page_mode as hybrid_decide
            return hybrid_decide(img_bgr, lines)
        except (ImportError, AttributeError) as e:
            raise BackendUnavailableError("hybrid_ocr backend is not available") from e
    else:
        return _internal_decide_page_mode(img_bgr, lines)


def encode_page_pdf(
    img_bgr: np.ndarray,
    mode: str,
    compress: str,
    backend: str = "auto",
    jpeg_quality: int = 85,
) -> bytes:
    """Encode one page image as a single-page PDF (no text layer)."""
    if backend not in ("auto", "internal", "hybrid_ocr"):
        raise ValueError(f"backend must be one of {('auto', 'internal', 'hybrid_ocr')}")
    if not isinstance(img_bgr, np.ndarray):
        raise TypeError("img_bgr must be a numpy.ndarray")
    if img_bgr.size == 0 or img_bgr.ndim < 2:
        raise ValueError("img_bgr must be a non-empty 2D or 3D array")

    chosen = backend
    if chosen == "auto":
        chosen = "internal"

    if chosen == "hybrid_ocr":
        try:
            from hybrid_ocr.pdf_export import encode_page_pdf as hybrid_encode
            return hybrid_encode(img_bgr, mode, compress)
        except (ImportError, AttributeError) as e:
            raise BackendUnavailableError("hybrid_ocr backend is not available") from e
    else:
        return _internal_encode_page_pdf(img_bgr, mode, compress, jpeg_quality)


def build_text_overlay(
    pages: list,
    font_path: str | None = None,
    backend: str = "auto",
) -> bytes:
    """Invisible-text-only PDF: one page per (width, height, lines) tuple."""
    if backend not in ("auto", "internal", "hybrid_ocr"):
        raise ValueError(f"backend must be one of {('auto', 'internal', 'hybrid_ocr')}")

    chosen = backend
    if chosen == "auto":
        chosen = "internal"

    if chosen == "hybrid_ocr":
        try:
            from hybrid_ocr.pdf_export import build_text_overlay as hybrid_build
            return hybrid_build(pages, font_path)
        except (ImportError, AttributeError) as e:
            raise BackendUnavailableError("hybrid_ocr backend is not available") from e
    else:
        return _internal_build_text_overlay(pages, font_path)


@dataclass
class InternalSearchablePdfBuilder:
    """Internal implementation of SearchablePdfBuilder."""

    mode: str = "auto"
    compress: str = "g4"
    font_path: str | None = None
    _pages: list = field(default_factory=list)

    def add_page(self, img_bgr: np.ndarray, lines: list, mode: str | None = None) -> str:
        mode = mode or self.mode
        if mode == "auto":
            mode = _internal_decide_page_mode(img_bgr, lines)
        pdf_bytes = _internal_encode_page_pdf(img_bgr, mode, self.compress)
        h, w = img_bgr.shape[:2]
        self._pages.append({"pdf": pdf_bytes, "w": w, "h": h, "lines": lines})
        return mode

    def save(self, output_path: str):
        import pikepdf
        from io import BytesIO

        if not self._pages:
            raise ValueError("no pages added")
        overlay_bytes = _internal_build_text_overlay(
            [(p["w"], p["h"], p["lines"]) for p in self._pages], self.font_path
        )
        out = pikepdf.new()
        sources = [pikepdf.open(BytesIO(p["pdf"])) for p in self._pages]
        try:
            for src in sources:
                out.pages.extend(src.pages)
            with pikepdf.open(BytesIO(overlay_bytes)) as text_pdf:
                for page, tpage in zip(out.pages, text_pdf.pages):
                    page.add_overlay(tpage)
                out.save(output_path)
        finally:
            for src in sources:
                src.close()


@dataclass
class SearchablePdfBuilder:
    """Accumulates pages (already OCR'd) and writes one searchable PDF.

    Supports facade/backend selection: auto (default), internal, hybrid_ocr.
    """

    mode: str = "auto"
    compress: str = "g4"
    font_path: str | None = None
    backend: str = "auto"
    _builder: Any = field(init=False, default=None)

    def __post_init__(self):
        if self.mode not in ("auto", "bw", "gray", "color"):
            raise ValueError(f"mode must be one of {('auto', 'bw', 'gray', 'color')}")
        if self.compress not in ("g4", "jbig2"):
            raise ValueError(f"compress must be one of {('g4', 'jbig2')}")
        if self.backend not in ("auto", "internal", "hybrid_ocr"):
            raise ValueError(f"backend must be one of {('auto', 'internal', 'hybrid_ocr')}")

        chosen = self.backend
        if chosen == "auto":
            chosen = "internal"

        if chosen == "hybrid_ocr":
            try:
                from hybrid_ocr.pdf_export import SearchablePdfBuilder as HybridBuilder
                self._builder = HybridBuilder(
                    mode=self.mode,
                    compress=self.compress,
                    font_path=self.font_path,
                )
            except (ImportError, AttributeError) as e:
                raise BackendUnavailableError("hybrid_ocr backend is not available") from e
        else:
            if self.compress == "jbig2":
                try:
                    import pyjbig2  # noqa
                except ImportError as e:
                    raise UnsupportedCodecError(
                        "jbig2 compression requested but 'pyjbig2' package is not available."
                    ) from e
            self._builder = InternalSearchablePdfBuilder(
                mode=self.mode,
                compress=self.compress,
                font_path=self.font_path,
            )

    def add_page(self, img_bgr: np.ndarray, lines: list, mode: str | None = None) -> str:
        """Add a page image and its OCR line bounding boxes."""
        return self._builder.add_page(img_bgr, lines, mode)

    def save(self, output_path: str):
        """Build and write the final searchable PDF to output_path."""
        self._builder.save(output_path)
