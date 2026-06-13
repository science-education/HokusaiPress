"""Parameter-driven document model — the backbone of HokusaiPress.

Design principle (inherited from ScanTailor): the original scan is never
destructively rewritten. Every processing stage records *parameters*
(angles, boxes, region classes, decisions) here as metadata; only the final
renderer reads the original image and produces output pixels, in a single
resampling pass. A human/AI correction in the review UI is an edit to these
parameters, never a re-edit of an image — so any page can be regenerated
from scratch at any time, losslessly.

All geometric coordinates are in ORIGINAL-image pixel space unless a field
says otherwise. The 300 dpi raster used for OCR is a work artifact; OCR
boxes are mapped back here via SourceRef.ocr_scale.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


class PageKind(str, Enum):
    """How the image layer of a page should be encoded in the final PDF."""

    AUTO = "auto"      # decide from content regions at render time
    BW = "bw"          # bilevel everywhere (G4 / JBIG2)
    GRAY = "gray"      # grayscale JPEG
    COLOR = "color"    # color JPEG
    MRC = "mrc"        # mixed: bilevel text layer + image layer for figures


class RegionKind(str, Enum):
    TEXT = "text"        # characters / ruled lines -> bilevel
    FIGURE = "figure"    # line art / diagram -> bilevel-friendly, may stay gray
    PHOTO = "photo"      # halftone / continuous tone -> gray or color


class ReviewStatus(str, Enum):
    AUTO = "auto"                # passed batch, not flagged
    NEEDS_REVIEW = "needs_review"  # flagged, awaiting human/AI
    APPROVED = "approved"        # reviewer accepted the auto result
    CORRECTED = "corrected"      # reviewer overrode parameters


class DecidedBy(str, Enum):
    AUTO = "auto"
    HUMAN = "human"
    AI = "ai"


# Flag reasons that send a page to the review queue. Keep these stable; they
# are logged and later feed the decision-learning model.
class Flag(str, Enum):
    DESKEW_LOW_CONF = "deskew_low_confidence"
    MARGIN_NOT_FOUND = "margin_not_found"
    MARGIN_INCONSISTENT = "margin_inconsistent_with_neighbors"
    KIND_BORDERLINE = "page_kind_borderline"
    OCR_LOW_COVERAGE = "ocr_low_coverage"
    OCR_DROPOUT_RETRY = "ocr_dropout_retry_fired"
    DETECTION_DENSITY = "detection_density_anomaly"
    NO_TEXT = "no_text_detected"


@dataclass
class SourceRef:
    """Where a page's original pixels come from."""

    path: str                       # source file (PDF or image)
    page_index: int = 0             # 0-based page within a PDF
    embedded_image_id: Optional[str] = None  # pikepdf XObject key, if extracted
    ocr_scale: float = 1.0          # original_px = ocr_px / ocr_scale


@dataclass
class Deskew:
    angle_deg: float = 0.0          # positive = clockwise (ScanTailor convention)
    confidence: float = 0.0         # >= GOOD_CONFIDENCE means trustworthy


@dataclass
class Box:
    """Axis-aligned box in original-image pixels."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class Margin:
    """Content box plus the per-side output margin to normalize against."""

    content: Box
    confidence: float = 0.0
    # nombre (page-number) anchor used to align margins across pages, if found
    nombre_box: Optional[Box] = None
    # normalized output crop (deskewed coords), set by margin normalization so
    # every page in a parity group renders to the same size with the body in a
    # consistent position. Already includes the output margin. When present the
    # renderer crops to this instead of content+margin.
    crop: Optional[Box] = None


@dataclass
class Region:
    kind: RegionKind
    box: Box
    source: str = "dbnet"           # dbnet | rtdetr | residual | manual
    ocr_text: Optional[str] = None
    ocr_conf: Optional[float] = None


@dataclass
class PageParams:
    source: SourceRef
    dpi: Optional[float] = None     # original image dpi if known
    deskew: Deskew = field(default_factory=Deskew)
    margin: Optional[Margin] = None
    regions: list[Region] = field(default_factory=list)
    page_kind: PageKind = PageKind.AUTO
    flags: list[Flag] = field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.AUTO
    decided_by: Optional[DecidedBy] = None

    def needs_review(self) -> bool:
        return self.review_status == ReviewStatus.NEEDS_REVIEW

    def reading_text(self) -> str:
        return "".join(r.ocr_text or "" for r in self.regions if r.ocr_text)


@dataclass
class RenderSettings:
    """Global, applied once at render time from the original image."""

    target_dpi: int = 600           # output resolution for the final image layer
    output_margin_mm: float = 5.0   # normalized margin added around content
    bilevel_codec: str = "g4"       # g4 | jbig2
    jpeg_quality: int = 85
    despeckle: bool = True          # applied to the bilevel layer only


@dataclass
class Document:
    source_path: str
    pages: list[PageParams] = field(default_factory=list)
    render: RenderSettings = field(default_factory=RenderSettings)

    # --- serialization: the whole document state is a single JSON blob ---
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, default=_enc)

    @staticmethod
    def from_json(text: str) -> "Document":
        return _decode_document(json.loads(text))


def _enc(obj):
    if isinstance(obj, Enum):
        return obj.value
    # numpy scalars (from OpenCV stats) -> native python numbers
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not serializable: {type(obj)}")


# --- explicit decoders (dataclasses + enums are not auto-rehydrated) ---
def _box(d) -> Optional[Box]:
    return None if d is None else Box(**d)


def _decode_region(d) -> Region:
    return Region(
        kind=RegionKind(d["kind"]),
        box=_box(d["box"]),
        source=d.get("source", "dbnet"),
        ocr_text=d.get("ocr_text"),
        ocr_conf=d.get("ocr_conf"),
    )


def _decode_page(d) -> PageParams:
    margin = d.get("margin")
    return PageParams(
        source=SourceRef(**d["source"]),
        dpi=d.get("dpi"),
        deskew=Deskew(**d.get("deskew", {})),
        margin=None
        if margin is None
        else Margin(
            content=_box(margin["content"]),
            confidence=margin.get("confidence", 0.0),
            nombre_box=_box(margin.get("nombre_box")),
            crop=_box(margin.get("crop")),
        ),
        regions=[_decode_region(r) for r in d.get("regions", [])],
        page_kind=PageKind(d.get("page_kind", "auto")),
        flags=[Flag(f) for f in d.get("flags", [])],
        review_status=ReviewStatus(d.get("review_status", "auto")),
        decided_by=None if d.get("decided_by") is None else DecidedBy(d["decided_by"]),
    )


def _decode_document(d) -> Document:
    return Document(
        source_path=d["source_path"],
        pages=[_decode_page(p) for p in d.get("pages", [])],
        render=RenderSettings(**d.get("render", {})),
    )
