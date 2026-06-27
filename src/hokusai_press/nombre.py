"""Page-number (nombre) resolution from OCR — robust to misreads.

The geometric heuristic scatters the box and the OCR text in a margin band is
noisy: running headers (柱), chapter/figure numbers, and stray "0"s all look
like page numbers. Picking greedily produces wrong numbers and false
"missing page" warnings.

Key idea: a *real* page number is linear in the physical page index with slope
~1 (number = index + offset) over a contiguous numbering run. So we:

- collect numeric tokens in the consistent margin band, parsing Arabic, kanji
  AND roman numerals (front matter is often i, ii, iii ...);
- vote for (kind, offset = value - index): the true run forms a strong cluster,
  while headers/chapter numbers/misreads scatter and are rejected;
- assign each page only the candidate that matches a *supported* cluster, so one
  misread can never pollute a page or fire a false warning;
- pages with no number (a title page, an unnumbered table of contents, a plate)
  simply stay None — that is normal and not flagged;
- a genuine missing page shows up as an offset shift between two supported runs,
  which is what the gap warning reports.

Runs only when OCR produced text; with --no-ocr the geometric box stands.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Optional

from .model import Box, Flag, PageParams, Region  # noqa: F401

BAND_FRAC = 0.14          # top/bottom 14% of the page is the nombre band
MAX_NOMBRE_LEN = 6        # a page-number token is short
MIN_SUPPORT = 3           # a numbering run must span >= this many pages to trust
GAP_TOLERANCE = 1         # skipped numbers beyond this (with no blank pages) warn
POS_K = 4.0               # robust spread multiplier for the position model
POS_FLOOR_X = 0.03        # minimum horizontal acceptance window (page fraction)
POS_FLOOR_Y = 0.02        # minimum vertical acceptance window (page fraction)
POS_SEP_THRESH = 0.25     # parity x separation: alternating corners vs pooled
POS_MIN_ANCHORS = 8       # below this, keep the historical vote-only behavior
POS_MAX_SPREAD = 0.15     # too broad means the document has no stable nombre pos
POS_DOM_FRAC = 0.3        # a cluster is "primary" only if it reaches this fraction
                         # of its own numbering system's strongest offset (so a
                         # minority misread offset within a kind is rejected, while
                         # a legitimate secondary system like roman front matter,
                         # which is dominant within its own kind, is kept)

_KANJI_DIGIT = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
                "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_KANJI_UNIT = {"十": 10, "百": 100, "千": 1000}
_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def parse_numeral(text: str) -> Optional[int]:
    """Parse Arabic (half/full-width) or kanji numerals to int (else None)."""
    if not text:
        return None
    s = "".join(text.split())
    if not s or len(s) > MAX_NOMBRE_LEN:
        return None
    trans = {ord("０") + i: ord("0") + i for i in range(10)}
    a = s.translate(trans)
    if a.isdigit():
        try:
            return int(a)
        except ValueError:
            return None
    return _parse_kanji(s)


def _parse_kanji(s: str) -> Optional[int]:
    if any(c not in _KANJI_DIGIT and c not in _KANJI_UNIT for c in s):
        return None
    if all(c in _KANJI_DIGIT for c in s):           # positional 一二三 -> 123
        return int("".join(str(_KANJI_DIGIT[c]) for c in s))
    total, section, last_digit = 0, 0, 0             # structured 十二 -> 12
    for c in s:
        if c in _KANJI_DIGIT:
            last_digit = _KANJI_DIGIT[c]
            section += last_digit
        else:
            unit = _KANJI_UNIT[c]
            section = section - last_digit + max(last_digit, 1) * unit
            total += section
            section, last_digit = 0, 0
    return total + section


def parse_roman(text: str) -> Optional[int]:
    """Parse a lower/upper-case roman numeral (i, ii, iv, xii ...) to int."""
    if not text:
        return None
    s = "".join(text.split()).lower()
    if not s or len(s) > 8 or any(c not in _ROMAN for c in s):
        return None
    total, prev = 0, 0
    for c in reversed(s):
        val = _ROMAN[c]
        total += -val if val < prev else val
        prev = max(prev, val)
    return total if total > 0 else None


def _band(box: Box, page_h: float) -> Optional[str]:
    cy = (box.y0 + box.y1) / 2.0
    if cy <= page_h * BAND_FRAC:
        return "top"
    if cy >= page_h * (1.0 - BAND_FRAC):
        return "bottom"
    return None


_LEAD_DIGITS = re.compile(r"^\s*([0-9０-９]{1,6})")
_TRAIL_DIGITS = re.compile(r"([0-9０-９]{1,6})\s*$")


def _fused_digit_candidates(r: Region, page_h: float):
    """A page number is often OCR'd fused with the running header into one region
    ('112第3部'). Recover a leading and/or trailing digit run as its own
    candidate, with a sub-box at the matching end so the position model still
    sees a corner-anchored nombre. The downstream position + primary-cluster gate
    rejects the chapter/figure digits this also picks up, so being liberal here
    is safe."""
    text = r.ocr_text or ""
    w = r.box.width
    n = len(text)
    if n == 0 or w <= 0:
        return []
    out = []
    m = _LEAD_DIGITS.match(text)
    if m:
        v = parse_numeral(m.group(1))
        if v and v > 0:
            frac = len(m.group(0)) / n
            sub = Box(r.box.x0, r.box.y0, r.box.x0 + w * frac, r.box.y1)
            out.append((v, sub))
    m = _TRAIL_DIGITS.search(text)
    if m:
        v = parse_numeral(m.group(1))
        if v and v > 0:
            frac = len(m.group(1)) / n
            sub = Box(r.box.x1 - w * frac, r.box.y0, r.box.x1, r.box.y1)
            out.append((v, sub))
    return out


def _candidates(params: PageParams, page_h: float):
    """(band, kind, value, region) for numeric tokens in a margin band.

    kind is 'num' (Arabic/kanji) or 'roman' — separate numbering systems get
    separate offset clusters, so roman front matter and the Arabic body don't
    interfere.
    """
    out = []
    for r in params.regions:
        if not r.ocr_text:
            continue
        b = _band(r.box, page_h)
        if not b:
            continue
        v = parse_numeral(r.ocr_text)
        if v is not None and v > 0:
            out.append((b, "num", v, r))
            continue
        rv = parse_roman(r.ocr_text)
        if rv is not None and rv > 0:
            out.append((b, "roman", rv, r))
            continue
        # number fused with the running header ('112第3部'): recover the digits
        for v, sub in _fused_digit_candidates(r, page_h):
            out.append((b, "num", v, Region(kind=r.kind, box=sub,
                                            ocr_text=str(v), source=r.source)))
    return out


@dataclass
class _PosStats:
    mx: float
    my: float
    sx: float
    sy: float
    n: int


@dataclass
class _PositionModel:
    x_mode: str
    x_stats: dict[int, _PosStats] | _PosStats
    y_stats: dict[int, _PosStats]


def resolve(
    pages: list[PageParams],
    page_heights: list[float],
    page_widths: list[float],
) -> list[str]:
    """Assign page_number/nombre_text robustly and return gap warnings.

    Mutates each page's margin.nombre_box / page_number / nombre_text / flags.
    """
    cands = [_candidates(p, h) for p, h in zip(pages, page_heights)]

    # choose the band (top/bottom) that carries numbers on the most pages
    pages_with = {"top": set(), "bottom": set()}
    for i, cl in enumerate(cands):
        for band, _, _, _ in cl:
            pages_with[band].add(i)
    band = ("bottom" if len(pages_with["bottom"]) >= len(pages_with["top"])
            else "top")
    if not pages_with[band]:
        return []

    # vote for (kind, offset) with slope 1; a real run clusters, noise scatters
    max_v = len(pages) + 50
    votes: Counter = Counter()
    for i, cl in enumerate(cands):
        for b, kind, v, _ in cl:
            if b == band and v <= max_v:
                votes[(kind, v - i)] += 1
    supported = {k for k, c in votes.items() if c >= MIN_SUPPORT}
    if not supported:
        return []

    dominant = max(supported, key=lambda k: votes[k])

    # Assign once by the historical vote-only method. If the position model is
    # under-supported or unstable, these assignments are the fallback behavior.
    fallback_warnings = _assign_supported(
        pages, cands, band, votes, supported, max_v)

    anchors = []
    for i, cl in enumerate(cands):
        for b, kind, v, r in cl:
            if b != band or v > max_v or (kind, v - i) != dominant:
                continue
            fx, fy = _box_center_frac(r.box, page_widths[i], page_heights[i])
            anchors.append((pages[i].source.page_index, fx, fy))
            break
    model = _build_position_model(anchors)
    if model is None:
        return fallback_warnings

    return _assign_with_position_model(
        pages, cands, page_heights, page_widths, band, votes, max_v, model)


def _assign_supported(pages, cands, band, votes, supported, max_v) -> list[str]:
    # assign each page the candidate matching the strongest supported cluster
    assigns = []  # (page_index, kind, value)
    for i, (p, cl) in enumerate(zip(pages, cands)):
        choice, best_sup = None, -1
        for b, kind, v, r in cl:
            if b != band or v > max_v:
                continue
            key = (kind, v - i)
            if key in supported and votes[key] > best_sup:
                choice, best_sup = (kind, v, r), votes[key]
        if choice is None:
            p.page_number, p.nombre_text = None, None
            if p.margin:           # drop the unreliable geometric box (often in
                p.margin.nombre_box = None   # the wrong band) for unread pages
            continue
        kind, v, r = choice
        p.page_number, p.nombre_text = v, r.ocr_text
        if p.margin:
            p.margin.nombre_box = r.box
        assigns.append((p.source.page_index, kind, v))
    _clear_gap_flags(pages)
    return _gap_warnings(assigns, {p.source.page_index: p for p in pages})


def _box_center_frac(box: Box, page_w: float, page_h: float) -> tuple[float, float]:
    return ((box.x0 + box.x1) / 2.0 / page_w,
            (box.y0 + box.y1) / 2.0 / page_h)


def _robust_stats(points: list[tuple[float, float]]) -> Optional[_PosStats]:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx = median(xs)
    my = median(ys)
    sx = 1.4826 * median([abs(x - mx) for x in xs])
    sy = 1.4826 * median([abs(y - my) for y in ys])
    return _PosStats(mx, my, sx, sy, len(points))


def _inside_stats(fx: float, fy: float, st: _PosStats) -> bool:
    return (abs(fx - st.mx) <= max(POS_K * st.sx, POS_FLOOR_X)
            and abs(fy - st.my) <= max(POS_K * st.sy, POS_FLOOR_Y))


def _stats_after_rejection(points: list[tuple[float, float]]) -> Optional[_PosStats]:
    st = _robust_stats(points)
    if st is None:
        return None
    kept = [p for p in points if _inside_stats(p[0], p[1], st)]
    return _robust_stats(kept)


def _too_broad(stats) -> bool:
    if isinstance(stats, dict):
        return any(_too_broad(v) for v in stats.values())
    return stats.sx > POS_MAX_SPREAD or stats.sy > POS_MAX_SPREAD


def _build_position_model(
    anchors: list[tuple[int, float, float]],
) -> Optional[_PositionModel]:
    if len(anchors) < POS_MIN_ANCHORS:
        return None

    by_parity = {
        0: [(fx, fy) for idx, fx, fy in anchors if idx % 2 == 0],
        1: [(fx, fy) for idx, fx, fy in anchors if idx % 2 == 1],
    }
    use_merged = any(len(points) < POS_MIN_ANCHORS
                     for points in by_parity.values())

    if use_merged:
        all_points = [(fx, fy) for _, fx, fy in anchors]
        st = _stats_after_rejection(all_points)
        if st is None or st.n < POS_MIN_ANCHORS or _too_broad(st):
            return None
        return _PositionModel(
            x_mode="pooled",
            x_stats=st,
            y_stats={0: st, 1: st},
        )

    y_stats = {}
    filtered_by_parity = {}
    for parity, points in by_parity.items():
        st = _stats_after_rejection(points)
        if st is None or st.n < POS_MIN_ANCHORS:
            return None
        y_stats[parity] = st
        filtered_by_parity[parity] = [
            p for p in points if _inside_stats(p[0], p[1], st)
        ]
    if _too_broad(y_stats):
        return None

    sep = abs(y_stats[0].mx - y_stats[1].mx)
    if sep > POS_SEP_THRESH:
        x_mode = "parity"
        x_stats: dict[int, _PosStats] | _PosStats = y_stats
    else:
        x_mode = "pooled"
        pooled = [p for points in filtered_by_parity.values() for p in points]
        pooled_stats = _stats_after_rejection(pooled)
        if (pooled_stats is None or pooled_stats.n < POS_MIN_ANCHORS
                or _too_broad(pooled_stats)):
            return None
        x_stats = pooled_stats

    return _PositionModel(x_mode, x_stats, y_stats)


def _inside_position_model(
    idx: int, box: Box, page_h: float, page_w: float, model: _PositionModel,
) -> bool:
    fx, fy = _box_center_frac(box, page_w, page_h)
    parity = idx % 2
    if model.x_mode == "parity":
        x_stats = model.x_stats[parity]  # type: ignore[index]
    else:
        x_stats = model.x_stats          # type: ignore[assignment]
    y_stats = model.y_stats[parity]
    return (abs(fx - x_stats.mx) <= max(POS_K * x_stats.sx, POS_FLOOR_X)
            and abs(fy - y_stats.my) <= max(POS_K * y_stats.sy, POS_FLOOR_Y))


def _primary_clusters(votes: Counter) -> set:
    """(kind, offset) clusters that are dominant *within their own numbering
    system*. A kind's strongest offset sets the bar; an offset reaching
    POS_DOM_FRAC of it is primary. This keeps a legitimate secondary system
    (roman front matter is the only roman cluster, so it qualifies) while
    rejecting a minority misread offset inside the body's kind (e.g. a few
    single digits mis-OCR'd into a +N cluster)."""
    kind_top: dict = {}
    for (kind, _off), c in votes.items():
        kind_top[kind] = max(kind_top.get(kind, 0), c)
    return {
        (kind, off) for (kind, off), c in votes.items()
        if c >= max(MIN_SUPPORT, POS_DOM_FRAC * kind_top[kind])
    }


def _in_position_candidate(p, cl, i, band, max_v, page_heights, page_widths,
                           model, predicate):
    """Best in-position candidate satisfying predicate(kind, offset, votes-key),
    as (kind, v, r), or None."""
    best = None
    for b, kind, v, r in cl:
        if b != band or v > max_v:
            continue
        if not predicate(kind, v - i):
            continue
        if _inside_position_model(
                p.source.page_index, r.box, page_heights[i], page_widths[i],
                model):
            if best is None:
                best = (kind, v, r)
    return best


def _assign_with_position_model(
    pages, cands, page_heights, page_widths, band, votes, max_v, model,
) -> list[str]:
    # Position is the gate (kind-agnostic). Pass 1: among in-position candidates
    # keep only those whose (kind, offset) is primary, and trust their own value
    # -- no snapping, so a real missing-page gap (its own primary cluster) is
    # never hidden. Pass 2: a page with no primary candidate but an in-position
    # candidate of the dominant kind, bracketed on both sides by the dominant
    # run, is an OCR misread -> recover it as index+dominant_offset.
    primary = _primary_clusters(votes)
    dom_kind, dom_offset = max(votes, key=lambda k: votes[k])
    chosen: list = [None] * len(pages)
    dom_idx: list[int] = []

    for i, (p, cl) in enumerate(zip(pages, cands)):
        in_model = []
        for b, kind, v, r in cl:
            if b != band or v > max_v or (kind, v - i) not in primary:
                continue
            if _inside_position_model(
                    p.source.page_index, r.box, page_heights[i],
                    page_widths[i], model):
                in_model.append((votes[(kind, v - i)], kind, v, r))
        if in_model:
            _, kind, v, r = max(in_model, key=lambda item: item[0])
            chosen[i] = (kind, v, r)
            if kind == dom_kind and v - i == dom_offset:
                dom_idx.append(i)

    has_before = [False] * len(pages)
    seen = False
    for i in range(len(pages)):
        has_before[i] = seen
        if i in set(dom_idx):
            seen = True
    dom_set = set(dom_idx)
    seen = False
    has_after = [False] * len(pages)
    for i in range(len(pages) - 1, -1, -1):
        has_after[i] = seen
        if i in dom_set:
            seen = True

    for i, (p, cl) in enumerate(zip(pages, cands)):
        if chosen[i] is not None or not (has_before[i] and has_after[i]):
            continue
        cand = _in_position_candidate(
            p, cl, i, band, max_v, page_heights, page_widths, model,
            lambda kind, off: kind == dom_kind)
        if cand is not None:
            chosen[i] = (dom_kind, i + dom_offset, cand[2])  # misread -> expected

    # Pass 3 (anchor-only, no page_number): a page with no in-position primary
    # or bracketed-misread candidate -- e.g. front matter that has its own real
    # per-page numbering, just too few pages to ever form a "primary" cluster
    # (_primary_clusters), and before the dominant run even starts so pass 2's
    # bracketing never reaches it either. Its OCR'd digit VALUE isn't trusted
    # (too few corroborating pages, easy 1-vs-misread-digit confusion), but the
    # box's PHYSICAL POSITION is real: every page prints its nombre in the same
    # slot, so a candidate sitting in the position model's expected spot is
    # almost certainly the real nombre regardless of what digit it reads as.
    # Anchor margins on it without touching page_number/gap tracking.
    # Use cl's candidates first (their value at least parsed as a positive
    # numeral); if none sits in position, fall back to ANY OCR'd text region
    # in the band, value unparsed/zero/garbled and all (e.g. "00" -- a single
    # real digit so badly misread parse_numeral rejected it as non-positive).
    # Position is the only thing being trusted here, so a parse failure is no
    # reason to give up on it.
    geo_box: list = [None] * len(pages)
    for i, (p, cl) in enumerate(zip(pages, cands)):
        if chosen[i] is not None:
            continue
        cand = _in_position_candidate(
            p, cl, i, band, max_v, page_heights, page_widths, model,
            lambda kind, off: True)
        if cand is not None:
            geo_box[i] = cand[2].box
            continue
        for r in p.regions:
            if not r.ocr_text or _band(r.box, page_heights[i]) != band:
                continue
            if _inside_position_model(p.source.page_index, r.box,
                                      page_heights[i], page_widths[i], model):
                geo_box[i] = r.box
                break

    assigns = []
    for i, p in enumerate(pages):
        c = chosen[i]
        if c is None:
            p.page_number, p.nombre_text = None, None
            if p.margin:
                # Store the TRUE detected box, never a y-snapped one: margin
                # normalization anchors the crop on this box to land the real
                # nombre INK at one shared output height. Snapping the box to a
                # common band (the old _aligned_box) moves the box but NOT the
                # ink, so the render would still show each page's real nombre
                # offset -- the very recto/verso height mismatch we're aligning.
                p.margin.nombre_box = geo_box[i]
            continue
        kind, v, r = c
        p.page_number, p.nombre_text = v, r.ocr_text
        if p.margin:
            p.margin.nombre_box = r.box
        assigns.append((p.source.page_index, kind, v))

    _clear_gap_flags(pages)
    return _gap_warnings(assigns, {p.source.page_index: p for p in pages})


def _clear_gap_flags(pages: list[PageParams]) -> None:
    for p in pages:
        p.flags = [f for f in p.flags if f != Flag.PAGE_NUMBER_GAP]


def _gap_warnings(assigns, idx2page) -> list[str]:
    """Within each numbering system, warn when consecutive assigned pages jump
    by more printed numbers than physical pages between them can explain. Misread
    tokens were never assigned, so they cannot trigger this; only a real
    offset shift (a missing page) does. Decreases (section restarts) are ignored.
    Flags the later page PAGE_NUMBER_GAP so it also reaches the review queue.
    """
    warnings: list[str] = []
    by_kind: dict[str, list] = {}
    for idx, kind, v in assigns:
        by_kind.setdefault(kind, []).append((idx, v))
    for lst in by_kind.values():
        lst.sort()
        for (i1, v1), (i2, v2) in zip(lst, lst[1:]):
            if v2 <= v1:
                continue
            span = i2 - i1
            if v2 - v1 > span + GAP_TOLERANCE:
                missing = (v2 - v1) - span
                warnings.append(
                    f"p{i2}: page number {v1} -> {v2} "
                    f"(~{missing} page(s) may be missing)")
                p = idx2page.get(i2)
                if p is not None and Flag.PAGE_NUMBER_GAP not in p.flags:
                    p.flags.append(Flag.PAGE_NUMBER_GAP)
    return warnings
