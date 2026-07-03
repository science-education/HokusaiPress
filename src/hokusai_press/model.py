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


class NumberingSystem(str, Enum):
    ARABIC = "arabic"
    ROMAN = "roman"
    NONE = "none"


class PaginationRole(str, Enum):
    PRINTED = "printed"
    COUNTED_UNPRINTED = "counted_unprinted"
    UNCOUNTED = "uncounted"


class PaginationMethod(str, Enum):
    OBSERVED = "observed"
    INTERPOLATED = "interpolated"
    SEQUENCE_MODEL = "sequence_model"
    MANUAL = "manual"


@dataclass
class PrintedFolio:
    """Physical ink observed on the page, independent of sequence inference."""

    text: str
    parsed_value: Optional[int]
    numbering_system: NumberingSystem
    box: Box
    confidence: float = 0.0
    source: str = ""


@dataclass
class PaginationDecision:
    """Book-level logical pagination; may exist without a printed folio."""

    logical_number: Optional[int] = None
    numbering_system: NumberingSystem = NumberingSystem.NONE
    role: PaginationRole = PaginationRole.UNCOUNTED
    method: PaginationMethod = PaginationMethod.SEQUENCE_MODEL
    confidence: float = 0.0
    supporting_pages: list[int] = field(default_factory=list)
    conflict: bool = False
    predicted_number: Optional[int] = None
    pdf_label: Optional[str] = None


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
    OCR_FAILED = "ocr_failed_on_page"            # recognition error on one page
    NOMBRE_UNREADABLE = "nombre_unreadable"      # no readable page number found
    PAGE_NUMBER_GAP = "page_number_gap"          # sequence break (possible miss)
    PAGE_REORIENTED = "page_reoriented_to_match_book"  # forced portrait<->landscape
    CONTENT_SCALED_DOWN = "content_scaled_down_to_match_book"  # outlier-large page


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
    # original (deskewed) page size in pixels, so normalize_margins can place
    # a real-but-small content box (e.g. a single part-title line) at its own
    # proportional position on the uniform-size page instead of dead-centering
    # it -- a deliberately off-center layout shouldn't be dragged to the
    # middle just because there's no nombre to anchor on.
    page_w: Optional[float] = None
    page_h: Optional[float] = None
    # Document-wide UNIFORM output size, in this page's own original-pixel
    # space (same space as `crop`). `crop` is allowed to be LARGER than this
    # for an outlier page (so its real content is never clipped -- see
    # normalize_margins' crop_size), but the final rendered page size must
    # still be exactly this value for every page in the book (page-format
    # uniformity is non-negotiable; a fold-out is the only real exception,
    # and even that should be an explicit, flagged decision -- not a silently
    # bigger page). compose_transform shrinks (never crops) an oversized
    # page's content down to fit this target instead of letting it inflate
    # the output canvas.
    target_w: Optional[float] = None
    target_h: Optional[float] = None


@dataclass
class Region:
    kind: RegionKind
    box: Box
    source: str = "dbnet"           # dbnet | rtdetr | residual | manual
    ocr_text: Optional[str] = None
    ocr_conf: Optional[float] = None
    # For PHOTO regions only: force the overlay's color treatment. None = auto
    # (decide gray vs color from the crop's chroma). This is how a grayscale
    # picture inside an otherwise bilevel ("bw") page is handled — the page
    # stays bw and just this region renders as a gray JPEG overlay.
    tone: Optional[str] = None       # None (auto) | "gray" | "color"
    layout_label: Optional[str] = None  # canonical layout hint from DEIM/OOP


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
    # page number read from the nombre region (OCR), and its parsed integer
    nombre_text: Optional[str] = None
    page_number: Optional[int] = None
    # Auditable folio pipeline: raw candidates stay separated by recognizer;
    # the final decision records which channel(s) and sequence rules supported it.
    nombre_evidence: dict = field(default_factory=dict)
    nombre_decision: dict = field(default_factory=dict)
    # Canonical pagination model. Legacy nombre_text/page_number remain mirrored
    # for stored-project and API compatibility.
    printed_folio: Optional[PrintedFolio] = None
    pagination: PaginationDecision = field(default_factory=PaginationDecision)
    # OCR found no content and the page has no real ink -> render as blank white
    # (rejects show-through). Only set when OCR ran, so it never erases text.
    blank: bool = False
    # Human orientation correction: clockwise degrees (0/90/180/270) applied to
    # the loaded original before any other processing. All stored boxes are in
    # the rotation-APPLIED frame (rotate_page_params keeps them in sync), so
    # every loader of this page's original must apply this rotation first.
    rotation: int = 0

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
    # Tint (halftone-background) overlays: k4-posterized Multiply layers that
    # preserve a gray screened background behind text. Disabled for now by
    # user direction (2026-06-21): defer tuning until deskew/margin/shadow are
    # settled. While off, a tint background simply goes through the bilevel
    # base layer (light screens threshold to white). The k4 code path is kept
    # and unit-tested directly; only its automatic per-page extraction is
    # gated here. Re-enable by setting this True.
    tint_overlay: bool = False
    # Document-level binding/ADF shadow bands, detected once across all pages
    # (margin.detect_shadow_bands) and re-applied at render. Each entry is
    # (axis, lo_frac, hi_frac); axis 0 = a vertical band at x in [lo,hi]*W.
    shadow_bands: list = field(default_factory=list)
    # Document-level binarization valley (median per-page Otsu). When set, the
    # output binarization uses it so a show-through page falls back to the
    # book's stable valley instead of a fixed floor (2-pass binarization).
    ink_valley: "int | None" = None


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
        tone=d.get("tone"),
        layout_label=d.get("layout_label"),
    )


def _decode_page(d) -> PageParams:
    margin = d.get("margin")
    printed = d.get("printed_folio")
    pagination = d.get("pagination") or {}
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
            page_w=margin.get("page_w"),
            page_h=margin.get("page_h"),
            target_w=margin.get("target_w"),
            target_h=margin.get("target_h"),
        ),
        regions=[_decode_region(r) for r in d.get("regions", [])],
        page_kind=PageKind(d.get("page_kind", "auto")),
        flags=[Flag(f) for f in d.get("flags", [])],
        review_status=ReviewStatus(d.get("review_status", "auto")),
        decided_by=None if d.get("decided_by") is None else DecidedBy(d["decided_by"]),
        nombre_text=d.get("nombre_text"),
        page_number=d.get("page_number"),
        rotation=int(d.get("rotation", 0)),
        nombre_evidence=d.get("nombre_evidence", {}),
        nombre_decision=d.get("nombre_decision", {}),
        printed_folio=(None if printed is None else PrintedFolio(
            text=printed.get("text", ""),
            parsed_value=printed.get("parsed_value"),
            numbering_system=NumberingSystem(
                printed.get("numbering_system", "none")),
            box=_box(printed["box"]),
            confidence=printed.get("confidence", 0.0),
            source=printed.get("source", ""),
        )),
        pagination=PaginationDecision(
            logical_number=pagination.get("logical_number", d.get("page_number")),
            numbering_system=NumberingSystem(
                pagination.get("numbering_system", "none")),
            role=PaginationRole(pagination.get("role", "uncounted")),
            method=PaginationMethod(
                pagination.get("method", "sequence_model")),
            confidence=pagination.get("confidence", 0.0),
            supporting_pages=list(pagination.get("supporting_pages", [])),
            conflict=pagination.get("conflict", False),
            predicted_number=pagination.get("predicted_number"),
            pdf_label=pagination.get("pdf_label"),
        ),
        blank=d.get("blank", False),
    )


def _decode_document(d) -> Document:
    return Document(
        source_path=d["source_path"],
        pages=[_decode_page(p) for p in d.get("pages", [])],
        render=RenderSettings(**d.get("render", {})),
    )


# --- human-readable flag descriptions (review UI) ---------------------------
# label: short name shown in lists; desc: what the batch detected;
# check: what the reviewer should look at on the page.
FLAG_INFO: dict = {
    Flag.DESKEW_LOW_CONF.value: {
        "label": "傾き検出が不確か",
        "desc": "ページの傾き角の自動測定に自信がありません。",
        "check": "行が水平か確認し、傾いていれば角度を調整してください。",
    },
    Flag.MARGIN_NOT_FOUND.value: {
        "label": "本文領域が見つからない",
        "desc": "本文の範囲(余白との境界)を自動検出できませんでした。",
        "check": "本文を囲む枠をドラッグで指定してください。",
    },
    Flag.MARGIN_INCONSISTENT.value: {
        "label": "本文領域が前後とずれ",
        "desc": "検出した本文範囲が前後のページと大きく異なります。",
        "check": "本文枠が正しいか確認し、必要なら修正してください。",
    },
    Flag.KIND_BORDERLINE.value: {
        "label": "ページ種別が判定困難",
        "desc": "白黒/グレー/カラーのどれで出力すべきか自信がありません。",
        "check": "写真や色の有無を見て、ページ種別を選んでください。",
    },
    Flag.OCR_LOW_COVERAGE.value: {
        "label": "文字認識が部分的",
        "desc": "ページの一部しか文字を認識できていない可能性があります。",
        "check": "本文の量に対して認識領域が少なくないか確認してください。",
    },
    Flag.OCR_DROPOUT_RETRY.value: {
        "label": "文字認識を再試行した",
        "desc": "最初の認識で欠落があり、再試行が行われました。",
        "check": "文字領域の枠が本文を覆えているか確認してください。",
    },
    Flag.DETECTION_DENSITY.value: {
        "label": "検出数が異常",
        "desc": "検出された領域の数が前後のページと比べて不自然です。",
        "check": "誤検出の枠や、見落とされた本文がないか確認してください。",
    },
    Flag.NO_TEXT.value: {
        "label": "文字が見つからない",
        "desc": "このページから文字が検出されませんでした。",
        "check": "白紙・図版ページなら問題ありません。本文があるなら要修正です。",
    },
    Flag.OCR_FAILED.value: {
        "label": "文字認識エラー",
        "desc": "このページの文字認識が途中で失敗しました。",
        "check": "画像の乱れがないか確認してください。再処理が必要な場合があります。",
    },
    Flag.NOMBRE_UNREADABLE.value: {
        "label": "ページ番号が読めない",
        "desc": "ページ番号(ノンブル)を読み取れませんでした。",
        "check": "番号が印刷されていれば、その位置を枠で指定してください。",
    },
    Flag.PAGE_NUMBER_GAP.value: {
        "label": "ページ番号が飛んでいる",
        "desc": "前後のページ番号がつながっていません(読み違いの可能性)。",
        "check": "実際の印刷番号と照らして、誤読があれば修正してください。",
    },
    Flag.PAGE_REORIENTED.value: {
        "label": "向きを自動補正した",
        "desc": "他のページに合わせて縦横の向きを自動で変更しました。",
        "check": "向きが正しいか確認してください。誤っていれば回転で戻せます。",
    },
    Flag.CONTENT_SCALED_DOWN.value: {
        "label": "サイズを自動縮小した",
        "desc": "他より極端に大きいページを、本の判型に合わせて縮小しました。",
        "check": "縮小後も内容が読めるか確認してください。",
    },
}


def _rotate_box(b: Optional[Box], delta: int, w: float, h: float) -> Optional[Box]:
    """Exact box coordinates after rotating a w x h image `delta` deg clockwise."""
    if b is None:
        return None
    if delta == 90:      # (x,y) -> (h - y, x)
        return Box(h - b.y1, b.x0, h - b.y0, b.x1)
    if delta == 180:     # (x,y) -> (w - x, h - y)
        return Box(w - b.x1, h - b.y1, w - b.x0, h - b.y0)
    if delta == 270:     # (x,y) -> (y, w - x)
        return Box(b.y0, w - b.x1, b.y1, w - b.x0)
    return Box(b.x0, b.y0, b.x1, b.y1)


def rotate_page_params(params: PageParams, delta: int, width: float, height: float) -> None:
    """Apply a clockwise 90-deg-step rotation to a page IN PLACE.

    `width`/`height` are the dimensions of the CURRENT original frame (i.e.
    with params.rotation already applied). Every stored box is permuted
    exactly (no resampling, no information loss), params.rotation accumulates,
    and page_w/page_h swap for odd quarter turns. The deskew angle was
    measured against the old axes, so it is reset for 90/270 (meaningless
    there) and kept for 180 (a baseline at theta stays at theta).
    """
    delta = delta % 360
    if delta not in (90, 180, 270):
        return
    for r in params.regions:
        r.box = _rotate_box(r.box, delta, width, height)
    if params.printed_folio is not None:
        params.printed_folio.box = _rotate_box(
            params.printed_folio.box, delta, width, height)
    m = params.margin
    if m is not None:
        m.content = _rotate_box(m.content, delta, width, height)
        m.crop = _rotate_box(m.crop, delta, width, height)
        m.nombre_box = _rotate_box(m.nombre_box, delta, width, height)
        if delta in (90, 270):
            m.page_w, m.page_h = m.page_h, m.page_w
            m.target_w, m.target_h = m.target_h, m.target_w
    if delta in (90, 270):
        params.deskew.angle_deg = 0.0
    params.rotation = (params.rotation + delta) % 360
