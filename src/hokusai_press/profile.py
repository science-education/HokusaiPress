"""Copyright-safe per-book feature extraction and parameter profiles.

Pure logic (no DB, no I/O). Turns a book's PageParams into geometric/structural
features -- positions, classes, counts, page-number values -- but never the OCR
text content, so the resulting profiles are safe to store and later share. The
aggregate is a robust per-book posterior; ScopeKey/backoff implement the
hierarchical prior lookup (finest bucket with enough support, else back off).
See docs/PARAM_PROFILE_PLAN.md.
"""

from __future__ import annotations

import dataclasses
import statistics
from collections import Counter
from typing import Callable, Optional

from .model import Box, PageParams, Region, RegionKind  # noqa: F401


@dataclasses.dataclass(frozen=True)
class RobustStat:
    median: float
    mad: float
    n: int


def robust_stat(values) -> Optional[RobustStat]:
    clean = [v for v in values if v is not None]
    n = len(clean)
    if n == 0:
        return None
    med = statistics.median(clean)
    mad = 1.4826 * statistics.median([abs(x - med) for x in clean])
    return RobustStat(median=float(med), mad=float(mad), n=n)


@dataclasses.dataclass(frozen=True)
class PageFeature:
    page_index: int
    is_ocr: bool
    region_count: int
    deskew_angle: float
    deskew_conf: float
    content_w: Optional[float]   # normalized by page width
    content_h: Optional[float]   # normalized by page height
    nombre_value: Optional[int]
    nombre_cx: Optional[float]   # normalized center
    nombre_cy: Optional[float]
    blank: bool


@dataclasses.dataclass(frozen=True)
class RegionFeature:
    page_index: int
    cls: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: Optional[float]
    source: str


def extract_features(
    pages: list[PageParams], widths: list[float], heights: list[float]
) -> tuple[list[PageFeature], list[RegionFeature]]:
    page_features: list[PageFeature] = []
    region_features: list[RegionFeature] = []
    for p, w, h in zip(pages, widths, heights):
        idx = p.source.page_index
        is_ocr = any(bool(getattr(r, "ocr_text", None)) for r in p.regions)
        content_w = content_h = nombre_cx = nombre_cy = None
        if p.margin:
            content_w = p.margin.content.width / w if w > 0 else 0.0
            content_h = p.margin.content.height / h if h > 0 else 0.0
            if p.margin.nombre_box:
                nb = p.margin.nombre_box
                nombre_cx = ((nb.x0 + nb.x1) / 2) / w if w > 0 else 0.0
                nombre_cy = ((nb.y0 + nb.y1) / 2) / h if h > 0 else 0.0
        page_features.append(PageFeature(
            page_index=idx,
            is_ocr=is_ocr,
            region_count=len(p.regions),
            deskew_angle=p.deskew.angle_deg,
            deskew_conf=p.deskew.confidence,
            content_w=content_w,
            content_h=content_h,
            nombre_value=getattr(p, "page_number", None),
            nombre_cx=nombre_cx,
            nombre_cy=nombre_cy,
            blank=p.blank,
        ))
        for r in p.regions:
            region_features.append(RegionFeature(
                page_index=idx,
                cls=r.kind.value,
                x0=r.box.x0 / w if w > 0 else 0.0,
                y0=r.box.y0 / h if h > 0 else 0.0,
                x1=r.box.x1 / w if w > 0 else 0.0,
                y1=r.box.y1 / h if h > 0 else 0.0,
                conf=getattr(r, "ocr_conf", None),
                source=r.source,
            ))
    return page_features, region_features


@dataclasses.dataclass(frozen=True)
class BookProfile:
    page_count: int
    ocr_pages: int
    size_w: Optional[RobustStat]
    size_h: Optional[RobustStat]
    deskew_angle: Optional[RobustStat]
    content_w: Optional[RobustStat]
    content_h: Optional[RobustStat]
    dominant_offset: Optional[int]
    class_density: dict
    writing_dir: str = "unknown"   # vertical(縦書き) | horizontal(横書き) | unknown
    binding: str = "unknown"       # right(右綴じ) | left(左綴じ) | top(上綴じ) | unknown


def detect_writing_direction(region_features: list[RegionFeature]) -> str:
    """vertical(縦書き)/horizontal(横書き) from text-line shape: a tategaki line
    is a tall narrow column (h>w), a yokogaki line is wide (w>h)."""
    ar = [(r.y1 - r.y0) / (r.x1 - r.x0)
          for r in region_features
          if r.cls == "text" and r.x1 > r.x0 and r.y1 > r.y0]
    if len(ar) < 5:
        return "unknown"
    med = statistics.median(ar)
    if med > 1.2:
        return "vertical"
    if med < 0.8:
        return "horizontal"
    return "unknown"


def detect_binding(page_features: list[PageFeature], writing_dir: str) -> str:
    """right(右綴じ)/left(左綴じ) binding. The page-number's outer corner
    alternates by leaf, so the even-index side reveals the turning direction:
    even nombre on the LEFT => right binding (right-to-left, typical 縦書き).
    Falls back to the writing-direction default when nombre data is sparse."""
    even = [pf.nombre_cx for pf in page_features
            if pf.nombre_cx is not None and pf.page_index % 2 == 0]
    odd = [pf.nombre_cx for pf in page_features
           if pf.nombre_cx is not None and pf.page_index % 2 == 1]
    if len(even) >= 3 and len(odd) >= 3:
        e, o = statistics.median(even), statistics.median(odd)
        if abs(e - o) > 0.2:
            return "right" if e < o else "left"
    if writing_dir == "vertical":
        return "right"
    if writing_dir == "horizontal":
        return "left"
    return "unknown"


def aggregate(
    page_features: list[PageFeature],
    region_features: list[RegionFeature],
    widths: list[float],
    heights: list[float],
) -> BookProfile:
    page_count = len(page_features)
    offsets = [pf.nombre_value - pf.page_index
               for pf in page_features if pf.nombre_value is not None]
    counts = Counter(rf.cls for rf in region_features)
    density = ({cls: c / page_count for cls, c in counts.items()}
               if page_count > 0 else {})
    writing_dir = detect_writing_direction(region_features)
    return BookProfile(
        page_count=page_count,
        ocr_pages=sum(1 for pf in page_features if pf.is_ocr),
        size_w=robust_stat(widths),
        size_h=robust_stat(heights),
        deskew_angle=robust_stat([pf.deskew_angle for pf in page_features]),
        content_w=robust_stat([pf.content_w for pf in page_features]),
        content_h=robust_stat([pf.content_h for pf in page_features]),
        dominant_offset=(Counter(offsets).most_common(1)[0][0]
                         if offsets else None),
        class_density=density,
        writing_dir=writing_dir,
        binding=detect_binding(page_features, writing_dir),
    )


@dataclasses.dataclass(frozen=True)
class ScopeKey:
    # writing/binding are a HARD regime partition -- never pooled across, so a
    # vertical/right-bound book's stats never contaminate a horizontal/left one
    # (and within a bucket the binding is fixed, so page-number left/right is
    # already consistent: no outer/inner remapping needed). scanner/fmt/genre are
    # the soft dimensions that back off when a bucket is under-supported.
    scanner: Optional[str] = None
    fmt: Optional[str] = None
    genre: Optional[str] = None
    writing: Optional[str] = None
    binding: Optional[str] = None


# soft dimensions, most-specific to coarsest; writing+binding always kept
_SOFT_CONFIGS = [
    ("scanner", "fmt", "genre"),
    ("scanner", "fmt"),
    ("scanner", "genre"),
    ("fmt", "genre"),
    ("scanner",),
    ("fmt",),
    ("genre",),
    (),
]


def backoff_keys(key: ScopeKey) -> list[ScopeKey]:
    """Finest to coarsest within the (writing, binding) regime. Soft dimensions
    are dropped progressively; the regime is fixed in every key. The coarsest is
    the regime itself (soft dims all None)."""
    res: list[ScopeKey] = []
    seen: set = set()
    for config in _SOFT_CONFIGS:
        if all(getattr(key, dim) is not None for dim in config):
            k = ScopeKey(
                scanner=key.scanner if "scanner" in config else None,
                fmt=key.fmt if "fmt" in config else None,
                genre=key.genre if "genre" in config else None,
                writing=key.writing,
                binding=key.binding,
            )
            if k not in seen:
                res.append(k)
                seen.add(k)
    return res


def resolve_profile(
    key: ScopeKey,
    lookup: Callable[[ScopeKey], Optional[tuple]],
    min_n: int,
) -> Optional[tuple]:
    """Walk the backoff chain; return (profile, used_key) for the finest bucket
    with n >= min_n. Fall back to the coarsest (regime) bucket regardless of n;
    None if nothing exists in the regime at all (cold start -> code defaults)."""
    chain = backoff_keys(key)
    coarsest = chain[-1] if chain else key
    coarsest_found = None
    for k in chain:
        found = lookup(k)
        if not found:
            continue
        profile, n = found
        if k == coarsest:
            coarsest_found = profile
        if n >= min_n:
            return profile, k
    if coarsest_found is not None:
        return coarsest_found, coarsest
    return None
