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

import difflib
import os
import re
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Optional

from .model import (
    Box, Flag, NumberingSystem, PageParams, PaginationDecision,
    PaginationMethod, PaginationRole, PrintedFolio, Region,
)

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
POS_CLUSTER_X = 0.055     # pre-vote page-crossing position cluster windows
POS_CLUSTER_Y = 0.030
POS_CLUSTER_MIN_SUPPORT = 3
POS_CLUSTER_MIN_OFFSET_SUPPORT = 2

# Human-estimated book-design priors. They are ranking evidence, not hard
# exclusions: unusual books remain possible when OCR/geometry strongly agrees.
PRIOR_COVER_UNNUMBERED = 0.99
PRIOR_PRE_TOC_UNNUMBERED = 0.90
PRIOR_ALTERNATING_SIDES = 0.99
PRIOR_POSITION_CONSISTENT = 0.95
PRIOR_HEIGHT_ALIGNED = 0.99
PRIOR_OUTSIDE_BODY = 0.95

_KANJI_DIGIT = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
                "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_KANJI_UNIT = {"十": 10, "百": 100, "千": 1000}
_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


@dataclass(frozen=True)
class _SequenceCandidate:
    page: int
    band: str
    kind: str
    value: int
    region: Region
    fx: float
    fy: float
    variant: str


def _sequence_candidates(pages, page_heights, page_widths):
    """Candidates from the main OCR only, including vertical digit reversal.

    The small folio CTC model is useful as a fallback crop recognizer, but its
    broad corner probes produce too many false positives to establish a number
    sequence. Main OCR regions provide much stronger geometry. For vertical
    Japanese folios OCR engines may return 12 as 21, so both orders participate
    in sequence voting; geometry and neighbouring pages decide which one wins.
    """
    out: list[_SequenceCandidate] = []
    for i, (page, h, w) in enumerate(zip(pages, page_heights, page_widths)):
        for region in page.regions:
            text = (region.ocr_text or "").strip()
            band = _band(region.box, h)
            if not text or band is None or len(text) > MAX_NOMBRE_LEN:
                continue
            fx, fy = _box_center_frac(region.box, w, h)
            value = parse_numeral(text)
            if value is not None and 0 < value <= len(pages) + 200:
                out.append(_SequenceCandidate(
                    i, band, "num", value, region, _cluster_x(fx), fy, "direct"))
                normalized = text.translate(
                    {ord("０") + n: ord("0") + n for n in range(10)})
                if normalized.isdigit() and len(normalized) > 1:
                    reversed_value = int(normalized[::-1])
                    if reversed_value > 0 and reversed_value != value:
                        out.append(_SequenceCandidate(
                            i, band, "num", reversed_value, region,
                            _cluster_x(fx), fy, "reversed"))
                continue
            value = parse_roman(text)
            if value is not None and value > 0:
                out.append(_SequenceCandidate(
                    i, band, "roman", value, region,
                    _cluster_x(fx), fy, "direct"))
                continue
            # OCR often joins a small folio to a running head or punctuation.
            # These partial tokens may support a sequence but never stand alone.
            for value, box in _fused_digit_candidates(region, h):
                sub_fx, sub_fy = _box_center_frac(box, w, h)
                sub = Region(kind=region.kind, box=box, source=region.source,
                             ocr_text=str(value), ocr_conf=region.ocr_conf)
                out.append(_SequenceCandidate(
                    i, band, "num", value, sub,
                    _cluster_x(sub_fx), sub_fy, "fused"))
            for value, box in _fused_roman_candidates(region):
                sub_fx, sub_fy = _box_center_frac(box, w, h)
                sub = Region(kind=region.kind, box=box, source=region.source,
                             ocr_text=text, ocr_conf=region.ocr_conf)
                out.append(_SequenceCandidate(
                    i, band, "roman", value, sub,
                    _cluster_x(sub_fx), sub_fy, "fused"))
        for band, kind, value, region in getattr(page, "_nombre_candidates", []) or []:
            if value is None or value <= 0:
                continue
            fx, fy = _box_center_frac(region.box, w, h)
            out.append(_SequenceCandidate(
                i, band, kind, value, region, _cluster_x(fx), fy, "reader"))
    return out


def _candidate_tracks(candidates: list[_SequenceCandidate]):
    tracks: list[list[_SequenceCandidate]] = []
    for cand in sorted(candidates, key=lambda c: (c.band, c.fy, c.fx)):
        best = None
        best_distance = float("inf")
        for track in tracks:
            if track[0].band != cand.band:
                continue
            mx = median([c.fx for c in track])
            my = median([c.fy for c in track])
            dx, dy = abs(cand.fx - mx), abs(cand.fy - my)
            if dx <= POS_CLUSTER_X and dy <= POS_CLUSTER_Y:
                distance = dx / POS_CLUSTER_X + dy / POS_CLUSTER_Y
                if distance < best_distance:
                    best, best_distance = track, distance
        if best is None:
            tracks.append([cand])
        else:
            best.append(cand)
    return tracks


def _resolve_sequence_tracks(pages, page_heights, page_widths) -> bool:
    """Resolve supported position-consistent sequences before noisy fallback.

    Returns True when at least one trustworthy run was found. A run is founded
    only by exact sequence votes (three Arabic pages, or two Roman pages). OCR
    correction/inference is then limited to pages inside the run and at most two
    following pages, and still requires ink at the same folio position.
    """
    tracks = _candidate_tracks(
        _sequence_candidates(pages, page_heights, page_widths))
    hypotheses = []
    for track_id, track in enumerate(tracks):
        by_key: dict[tuple[str, int], dict[int, list[_SequenceCandidate]]] = {}
        for cand in track:
            by_key.setdefault((cand.kind, cand.value - cand.page), {}).setdefault(
                cand.page, []).append(cand)
        for (kind, offset), by_page in by_key.items():
            minimum = 2 if kind == "roman" else MIN_SUPPORT
            # The compact CTC reader is noisy per crop, but five consecutive
            # position-consistent readings are stronger evidence than a main
            # OCR engine repeatedly inventing a leading stroke/digit.
            reader_only = all(
                all(c.variant == "reader" for c in choices)
                for choices in by_page.values()
            )
            if reader_only and kind == "num":
                minimum = 5
            indices = sorted(by_page)
            if len(indices) < minimum:
                continue
            # Split unrelated sections sharing an accidental offset.
            runs, current = [], [indices[0]]
            for index in indices[1:]:
                if index - current[-1] <= 2:
                    current.append(index)
                else:
                    runs.append(current)
                    current = [index]
            runs.append(current)
            for run in runs:
                if len(run) >= minimum:
                    hypotheses.append({
                        "track": track_id, "kind": kind, "offset": offset,
                        "exact": run, "by_page": by_page,
                        "edge": (median([c.fy for c in track])
                                 if track[0].band == "bottom"
                                 else 1.0 - median([c.fy for c in track])),
                    })
    if not hypotheses:
        return False

    # Stronger/longer runs claim their exact pages first. This prevents a weak
    # alternative reading of the same glyphs from overwriting the main series.
    def prior_score(h):
        track = tracks[h["track"]]
        exact = set(h["exact"])
        chosen = []
        for i in sorted(exact):
            expected = i + h["offset"]
            options = [c for c in track if c.page == i and c.kind == h["kind"]
                       and c.value == expected]
            if options:
                chosen.append(options[0])
        if not chosen:
            return 0.0
        # Outside-corner folios normally alternate with recto/verso. Use raw x
        # only for this prior; track clustering itself intentionally mirrors x.
        sides = [
            ((c.region.box.x0 + c.region.box.x1) / 2.0) >= page_widths[c.page] / 2.0
            for c in chosen
        ]
        alternating = sum(a != b for a, b in zip(sides, sides[1:]))
        alt_rate = alternating / max(1, len(sides) - 1)
        fx_spread = max(c.fx for c in chosen) - min(c.fx for c in chosen)
        fy_spread = max(c.fy for c in chosen) - min(c.fy for c in chosen)
        position = max(0.0, 1.0 - fx_spread / POS_CLUSTER_X)
        height = max(0.0, 1.0 - fy_spread / POS_CLUSTER_Y)
        # All candidates already lie in a top/bottom margin band. Closer to the
        # physical edge is stronger evidence that the token is outside body text.
        outside = sum(
            (1.0 - c.fy if c.band == "top" else c.fy) for c in chosen
        ) / len(chosen)
        return (
            PRIOR_ALTERNATING_SIDES * alt_rate
            + PRIOR_POSITION_CONSISTENT * position
            + PRIOR_HEIGHT_ALIGNED * height
            + PRIOR_OUTSIDE_BODY * outside
        )

    for h in hypotheses:
        h["prior"] = prior_score(h)
    hypotheses.sort(key=lambda h: (
        -len(h["exact"]), -h["prior"], -h["edge"], h["exact"][0]))
    exact_owner = {}
    accepted = []
    for h in hypotheses:
        available = [i for i in h["exact"] if i not in exact_owner]
        minimum = 2 if h["kind"] == "roman" else MIN_SUPPORT
        if len(available) < minimum:
            continue
        h = dict(h, exact=available)
        accepted.append(h)
        for i in available:
            exact_owner[i] = h

    assigned: dict[int, tuple[str, int, Region]] = {}
    for h in accepted:
        kind, offset = h["kind"], h["offset"]
        for i in h["exact"]:
            expected = i + offset
            choices = [c for c in h["by_page"][i] if c.value == expected]
            # Prefer a direct reading when both direct and reversed variants fit.
            cand = min(choices, key=lambda c: c.variant != "direct")
            assigned[i] = (kind, expected, cand.region)

    # A front-matter Roman run ending in iii/iv establishes the preceding i/ii
    # even when those tiny glyphs were unreadable. This inference is restricted
    # to the first two physical pages and the canonical offset=1 sequence.
    for h in accepted:
        if h["kind"] != "roman" or h["offset"] != 1:
            continue
        start = min(h["exact"])
        if start > 2:
            continue
        # Fill unread i/ii before the first anchor and holes bracketed by the
        # same front-matter run. No physical nombre box is invented.
        for i in range(0, max(h["exact"]) + 1):
            if i not in assigned:
                assigned[i] = ("roman", i + 1, None)

    protected = set(exact_owner)
    for h in accepted:
        track = tracks[h["track"]]
        kind, offset = h["kind"], h["offset"]
        start, end = min(h["exact"]), max(h["exact"])
        candidates_by_page: dict[int, list[_SequenceCandidate]] = {}
        for cand in track:
            if cand.kind == kind:
                candidates_by_page.setdefault(cand.page, []).append(cand)
        for i in range(start, min(len(pages), end + 3)):
            if i in assigned or (i in protected and exact_owner[i] is not h):
                continue
            choices = candidates_by_page.get(i, [])
            if not choices:
                continue
            expected = i + offset
            if expected <= 0:
                continue
            # Geometry proves this is the folio position; keep the OCR text for
            # auditability while the sequence supplies the corrected value.
            cand = max(choices, key=lambda c: c.region.ocr_conf or 0.0)
            assigned[i] = (kind, expected, cand.region)

    if not assigned:
        return False
    for i, page in enumerate(pages):
        choice = assigned.get(i)
        if choice is None:
            page.page_number, page.nombre_text = None, None
            if page.margin:
                page.margin.nombre_box = None
            continue
        kind, value, region = choice
        page.page_number = value
        page.nombre_text = (
            region.ocr_text if region is not None
            else _format_roman(value) if kind == "roman" else str(value)
        )
        if page.margin and region is not None:
            page.margin.nombre_box = region.box

    # Printed-folio geometry and logical labels are separate facts. A broad
    # fallback probe can support the i/ii/... sequence without proving visible
    # ink on a cover or pre-TOC leaf. Keep the logical label, but suppress that
    # weak physical box according to the strong book-design priors.
    toc_index = next((
        i for i, page in enumerate(pages)
        if any("目次" in (region.ocr_text or "") for region in page.regions)
    ), None)
    for i, page in enumerate(pages):
        choice = assigned.get(i)
        region = choice[2] if choice is not None else None
        weak_box = region is None or region.source == "nombre_reader"
        reader_value = None
        if region is not None and region.source == "nombre_reader":
            reader_value = parse_numeral(region.ocr_text or "")
            if reader_value is None:
                reader_value = parse_roman(region.ocr_text or "")
        strong_reader_box = (
            region is not None
            and region.source == "nombre_reader"
            and (region.ocr_conf or 0.0) >= 0.80
            and choice is not None
            and reader_value == choice[1]
        )
        cover_prior = i == 0 and PRIOR_COVER_UNNUMBERED >= 0.99
        pre_toc_prior = (
            toc_index is not None and 0 < i < toc_index
            and PRIOR_PRE_TOC_UNNUMBERED >= 0.90
        )
        # Priors break ambiguity; they must never erase strong observed ink.
        # The cover remains the sole near-hard exception (99% prior).
        suppress = cover_prior or (pre_toc_prior and not strong_reader_box)
        if page.margin and weak_box and suppress:
            page.margin.nombre_box = None
    return True


def _format_roman(value: int) -> str:
    parts = []
    for number, token in ((1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
                          (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
                          (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while value >= number:
            parts.append(token)
            value -= number
    return "".join(parts)


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
_LEAD_ROMAN = re.compile(r"^\s*([ivxlcdmIVXLCDM]{1,8})(?=\s|[^A-Za-z]|$)")
_TRAIL_ROMAN = re.compile(r"(?<![A-Za-z])([ivxlcdmIVXLCDM]{1,8})\s*$")


def _fused_roman_candidates(r: Region):
    """Recover a Roman folio fused with a running head, e.g. ``iii 目次``."""
    text = r.ocr_text or ""
    width, length = r.box.width, len(text)
    if not text or width <= 0:
        return []
    out = []
    match = _LEAD_ROMAN.match(text)
    if match:
        value = parse_roman(match.group(1))
        if value:
            frac = len(match.group(0)) / length
            out.append((value, Box(r.box.x0, r.box.y0,
                                   r.box.x0 + width * frac, r.box.y1)))
    match = _TRAIL_ROMAN.search(text)
    if match and (not out or match.start(1) > 0):
        value = parse_roman(match.group(1))
        if value:
            frac = len(match.group(1)) / length
            out.append((value, Box(r.box.x1 - width * frac, r.box.y0,
                                   r.box.x1, r.box.y1)))
    return out


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
    for item in getattr(params, "_nombre_candidates", []) or []:
        try:
            band, kind, value, region = item
        except ValueError:
            continue
        if value is not None and value > 0:
            out.append((band, kind, value, region))
        else:
            out.append((band, "unparsed", None, region))
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
        fused = _fused_digit_candidates(r, page_h)
        if fused:
            for v, sub in fused:
                out.append((b, "num", v, Region(kind=r.kind, box=sub,
                                                ocr_text=str(v), source=r.source)))
            continue
        
        # Keep as unparsed for fuzzy matching fallback
        out.append((b, "unparsed", None, r))
    return out


def main_ocr_evidence(params: PageParams, page_h: float, page_w: float) -> list[dict]:
    """Serializable raw folio-like candidates from the selected main OCR."""
    evidence = []
    for region in params.regions:
        text = region.ocr_text or ""
        band = _band(region.box, page_h)
        if not text or band is None:
            continue
        values = []
        value = parse_numeral(text)
        kind = "num"
        if value is None:
            value = parse_roman(text)
            kind = "roman"
        if value is not None and value > 0:
            values.append({"kind": kind, "value": value, "variant": "direct"})
        normalized = text.strip().translate(
            {ord("０") + n: ord("0") + n for n in range(10)})
        if normalized.isdigit() and len(normalized) > 1:
            reversed_value = int(normalized[::-1])
            if reversed_value > 0 and reversed_value != value:
                values.append({"kind": "num", "value": reversed_value,
                               "variant": "reversed"})
        for fused_value, _ in _fused_digit_candidates(region, page_h):
            values.append({"kind": "num", "value": fused_value,
                           "variant": "fused"})
        for fused_value, _ in _fused_roman_candidates(region):
            values.append({"kind": "roman", "value": fused_value,
                           "variant": "fused"})
        if not values:
            continue
        evidence.append({
            "band": band, "text": text, "confidence": region.ocr_conf,
            "source": region.source, "values": values,
            "box": {"x0": region.box.x0, "y0": region.box.y0,
                    "x1": region.box.x1, "y1": region.box.y1},
            "position": [
                (region.box.x0 + region.box.x1) / 2.0 / page_w,
                (region.box.y0 + region.box.y1) / 2.0 / page_h,
            ],
        })
    return evidence


def _box_iou_dict(box: Box, raw: dict) -> float:
    other = Box(**raw)
    x0, y0 = max(box.x0, other.x0), max(box.y0, other.y0)
    x1, y1 = min(box.x1, other.x1), min(box.y1, other.y1)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = box.width * box.height + other.width * other.height - inter
    return inter / max(1.0, union)


def _box_containment_dict(box: Box, raw: dict) -> float:
    """Intersection over smaller box, for folios fused into a running head."""
    other = Box(**raw)
    x0, y0 = max(box.x0, other.x0), max(box.y0, other.y0)
    x1, y1 = min(box.x1, other.x1), min(box.y1, other.y1)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    smaller = min(box.width * box.height, other.width * other.height)
    return inter / max(1.0, smaller)


def record_decisions(pages: list[PageParams]) -> None:
    """Persist final folio choice separately from both OCR evidence streams."""
    for page in pages:
        box = page.margin.nombre_box if page.margin else None
        channels = []
        if box is not None:
            for channel in ("main_ocr", "dedicated_ocr"):
                if any(_box_iou_dict(box, item["box"]) >= 0.45
                       for item in page.nombre_evidence.get(channel, [])):
                    channels.append(channel)
        inferred = page.page_number is not None and not channels
        page.nombre_decision = {
            "text": page.nombre_text,
            "value": page.page_number,
            "box": (None if box is None else {
                "x0": box.x0, "y0": box.y0, "x1": box.x1, "y1": box.y1,
            }),
            "support_channels": channels,
            "series_inferred": inferred,
            "printed_folio_detected": box is not None,
            "pagination": {
                "logical_number": page.pagination.logical_number,
                "numbering_system": page.pagination.numbering_system.value,
                "role": page.pagination.role.value,
                "method": page.pagination.method.value,
                "confidence": page.pagination.confidence,
                "supporting_pages": page.pagination.supporting_pages,
                "conflict": page.pagination.conflict,
                "predicted_number": page.pagination.predicted_number,
                "pdf_label": page.pagination.pdf_label,
            },
        }


def _evidence_folio(page: PageParams) -> Optional[PrintedFolio]:
    """Recover the observed glyph value at the selected physical folio box.

    Sequence resolution is deliberately ignored here: evidence says what ink
    was read, while pagination says what number the book sequence assigns.
    """
    box = page.margin.nombre_box if page.margin else None
    if box is None:
        return None
    grouped: dict[tuple[str, int], dict] = {}
    for channel in ("main_ocr", "dedicated_ocr"):
        for item in page.nombre_evidence.get(channel, []):
            raw_box = item.get("box")
            if (not raw_box
                    or (_box_iou_dict(box, raw_box) < 0.35
                        and _box_containment_dict(box, raw_box) < 0.70)):
                continue
            values = item.get("values")
            if values is None:
                values = [{"kind": item.get("kind"), "value": item.get("value")}]
            for parsed in values:
                value = parsed.get("value")
                kind = parsed.get("kind")
                if value is None or kind not in {"num", "roman"}:
                    continue
                key = (kind, int(value))
                entry = grouped.setdefault(key, {
                    "score": 0.0, "channels": set(), "text": item.get("text", ""),
                    "confidence": 0.0, "source": channel,
                })
                # Main OCR is less probe-heavy; independent channel agreement is
                # stronger than either confidence score by itself.
                if channel not in entry["channels"]:
                    entry["score"] += (1.15 if channel == "main_ocr" else 1.0)
                    entry["channels"].add(channel)
                conf = float(item.get("confidence") or 0.0)
                entry["score"] += 0.15 * conf
                if conf >= entry["confidence"]:
                    entry.update(text=item.get("text", ""), confidence=conf,
                                 source=channel)
    if not grouped:
        return None
    predicted = page.page_number
    key, best = max(grouped.items(), key=lambda pair: (
        pair[1]["score"] + (0.08 if pair[0][1] == predicted else 0.0),
        len(pair[1]["channels"]), pair[1]["confidence"],
    ))
    kind, value = key
    return PrintedFolio(
        text=best["text"], parsed_value=value,
        numbering_system=(NumberingSystem.ROMAN if kind == "roman"
                          else NumberingSystem.ARABIC),
        box=box, confidence=min(1.0, best["score"] / 2.2),
        source="+".join(sorted(best["channels"])),
    )


def finalize_pagination(pages: list[PageParams]) -> None:
    """Separate printed folios from logical pagination and PDF navigation.

    Unprinted pages are counted only when bracketed by two observed folios in
    one linear sequence. This fills chapter-title holes but will not bridge a
    colophon into a restarted appendix. Every page receives a unique PDF label.
    """
    predicted = [page.page_number for page in pages]
    anchors: list[int] = []
    for i, page in enumerate(pages):
        page.printed_folio = _evidence_folio(page)
        folio = page.printed_folio
        if folio is None or folio.parsed_value is None:
            page.pagination = PaginationDecision(
                predicted_number=predicted[i], pdf_label=f"scan-{i + 1}")
            continue
        value = folio.parsed_value
        conflict = predicted[i] is not None and predicted[i] != value
        independently_supported = "+" in folio.source
        logical = (predicted[i] if conflict and not independently_supported
                   else value)
        method = (PaginationMethod.SEQUENCE_MODEL
                  if conflict and not independently_supported
                  else PaginationMethod.OBSERVED)
        page.pagination = PaginationDecision(
            logical_number=logical, numbering_system=folio.numbering_system,
            role=PaginationRole.PRINTED, method=method,
            confidence=folio.confidence, supporting_pages=[i + 1],
            conflict=conflict, predicted_number=predicted[i],
            pdf_label=(_format_roman(logical) if folio.numbering_system
                       == NumberingSystem.ROMAN else str(logical)),
        )
        if method == PaginationMethod.OBSERVED:
            anchors.append(i)

    # Only interpolate inside an observed, slope-1 sequence. Adjacent observed
    # anchors define the boundary, so a restart (239 -> 26) remains a boundary.
    for left, right in zip(anchors, anchors[1:]):
        a, b = pages[left].printed_folio, pages[right].printed_folio
        if (a is None or b is None
                or a.numbering_system != b.numbering_system
                or b.parsed_value - a.parsed_value != right - left):
            continue
        for i in range(left + 1, right):
            value = a.parsed_value + i - left
            label = (_format_roman(value) if a.numbering_system
                     == NumberingSystem.ROMAN else str(value))
            pages[i].pagination = PaginationDecision(
                logical_number=value, numbering_system=a.numbering_system,
                role=PaginationRole.COUNTED_UNPRINTED,
                method=PaginationMethod.INTERPOLATED,
                confidence=min(a.confidence, b.confidence) * 0.95,
                supporting_pages=[left + 1, right + 1],
                predicted_number=predicted[i], pdf_label=label,
            )

    # Legacy fields mirror the logical decision, while nombre_text remains a
    # literal observation and is never replaced with a sequence expectation.
    for i, page in enumerate(pages):
        page.page_number = page.pagination.logical_number
        page.nombre_text = page.printed_folio.text if page.printed_folio else None


def _cluster_x(fx: float) -> float:
    """Parity-tolerant x coordinate: outside corners mirror to one position."""
    return min(fx, 1.0 - fx)


def _cluster_feature(box: Box, page_w: float, page_h: float) -> tuple[float, float]:
    fx, fy = _box_center_frac(box, page_w, page_h)
    return _cluster_x(fx), fy


def _position_clustered_candidates(cands, pages, page_heights, page_widths, max_v):
    """Prefer candidates from the strongest cross-page position+offset cluster.

    The historical resolver first voted by number offset and only then learned a
    position model.  When body fragments intrude into the nombre band, that order
    can let repeated non-footer fragments dominate.  This lightweight pre-pass
    groups candidates by mirrored x/y position, then scores each position by its
    slope-1 offset support.  If no stable position exists, callers keep the old
    candidate set unchanged.
    """
    clusters: list[dict] = []
    for i, cl in enumerate(cands):
        for cand_i, (band, kind, v, r) in enumerate(cl):
            if v is None or v > max_v:
                continue
            x, y = _cluster_feature(r.box, page_widths[i], page_heights[i])
            best = None
            best_d = None
            for c in clusters:
                if c["band"] != band:
                    continue
                dx = abs(x - c["mx"])
                dy = abs(y - c["my"])
                if dx <= POS_CLUSTER_X and dy <= POS_CLUSTER_Y:
                    d = dx / POS_CLUSTER_X + dy / POS_CLUSTER_Y
                    if best is None or d < best_d:
                        best, best_d = c, d
            if best is None:
                best = {"band": band, "items": [], "mx": x, "my": y}
                clusters.append(best)
            best["items"].append((i, cand_i, kind, v, x, y))
            n = len(best["items"])
            best["mx"] += (x - best["mx"]) / n
            best["my"] += (y - best["my"]) / n

    best_cluster = None
    best_score = 0.0
    for c in clusters:
        pages_seen = {i for i, *_ in c["items"]}
        if len(pages_seen) < POS_CLUSTER_MIN_SUPPORT:
            continue
        vote_pages: dict[tuple[str, int], set[int]] = {}
        for i, _, kind, v, _, _ in c["items"]:
            vote_pages.setdefault((kind, v - i), set()).add(i)
        support = max((len(s) for s in vote_pages.values()), default=0)
        if support < POS_CLUSTER_MIN_OFFSET_SUPPORT:
            continue
        xs = [x for *_, x, _ in c["items"]]
        ys = [y for *_, y in c["items"]]
        spread = (max(xs) - min(xs) if xs else 1.0) + (max(ys) - min(ys) if ys else 1.0)
        score = support * 4.0 + len(pages_seen) - spread * 20.0
        if score > best_score:
            best_cluster, best_score = c, score

    if best_cluster is None:
        return None

    selected = [set() for _ in cands]
    best_pages = {i for i, *_ in best_cluster["items"]}
    for i, cand_i, *_ in best_cluster["items"]:
        selected[i].add(cand_i)

    analysis = []
    assignment = []
    for i, cl in enumerate(cands):
        picked = [cand for j, cand in enumerate(cl) if j in selected[i]]
        analysis.append(picked)
        assignment.append(picked if picked else cl)
    return analysis, assignment, len(best_pages)


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
    if _resolve_sequence_tracks(pages, page_heights, page_widths):
        assigns = [
            (p.source.page_index,
             "roman" if parse_roman(p.nombre_text or "") is not None else "num",
             p.page_number)
            for p in pages if p.page_number is not None
        ]
        _clear_gap_flags(pages)
        return _gap_warnings(assigns, {p.source.page_index: p for p in pages})

    raw_cands = [_candidates(p, h) for p, h in zip(pages, page_heights)]
    max_v = len(pages) + 50
    # Position clustering rejects body-text fragments but currently interacts with
    # the position model so the "missing pages" gap warning can be lost. Keep it
    # opt-in until that is made gap-warning-safe (default OFF = proven behavior).
    if os.environ.get("HOKUSAI_NOMBRE_POSITION_CLUSTER") == "1":
        clustered = _position_clustered_candidates(
            raw_cands, pages, page_heights, page_widths, max_v)
    else:
        clustered = None
    if clustered is None:
        cands = raw_cands
        assign_cands = raw_cands
        min_support = MIN_SUPPORT
        min_anchors = POS_MIN_ANCHORS
        strict_short_cluster = False
    else:
        cands, assign_cands, _cluster_pages = clustered
        min_support = MIN_SUPPORT
        strict_short_cluster = False
        min_anchors = POS_MIN_ANCHORS

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
    votes: Counter = Counter()
    for i, cl in enumerate(cands):
        for b, kind, v, _ in cl:
            if b == band and v is not None and v <= max_v:
                votes[(kind, v - i)] += 1
    supported = {k for k, c in votes.items() if c >= min_support}
    if not supported and clustered is not None:
        min_support = POS_CLUSTER_MIN_OFFSET_SUPPORT
        supported = {k for k, c in votes.items() if c >= min_support}
    if not supported:
        return []

    dominant = max(supported, key=lambda k: (votes[k], -abs(k[1])))
    if clustered is not None and votes[dominant] < POS_MIN_ANCHORS:
        strict_short_cluster = True
        min_anchors = POS_CLUSTER_MIN_OFFSET_SUPPORT

    # Assign once by the historical vote-only method. If the position model is
    # under-supported or unstable, these assignments are the fallback behavior.
    fallback_warnings = _assign_supported(
        pages, assign_cands, band, votes, supported, max_v)

    anchors = []
    for i, cl in enumerate(assign_cands):
        for b, kind, v, r in cl:
            if b != band or v is None or v > max_v or (kind, v - i) != dominant:
                continue
            fx, fy = _box_center_frac(r.box, page_widths[i], page_heights[i])
            anchors.append((pages[i].source.page_index, fx, fy))
            break
    model = _build_position_model(anchors, min_anchors)
    if model is None:
        return fallback_warnings

    return _assign_with_position_model(
        pages, assign_cands, page_heights, page_widths, band, votes, max_v,
        model, min_support, strict_short_cluster)


def _assign_supported(pages, cands, band, votes, supported, max_v) -> list[str]:
    # assign each page the candidate matching the strongest supported cluster
    assigns = []  # (page_index, kind, value)
    for i, (p, cl) in enumerate(zip(pages, cands)):
        choice, best_sup = None, -1
        for b, kind, v, r in cl:
            if b != band or v is None or v > max_v:
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
    min_anchors: int = POS_MIN_ANCHORS,
) -> Optional[_PositionModel]:
    if len(anchors) < min_anchors:
        return None

    by_parity = {
        0: [(fx, fy) for idx, fx, fy in anchors if idx % 2 == 0],
        1: [(fx, fy) for idx, fx, fy in anchors if idx % 2 == 1],
    }
    use_merged = any(len(points) < min_anchors
                     for points in by_parity.values())

    if use_merged:
        all_points = [(fx, fy) for _, fx, fy in anchors]
        st = _stats_after_rejection(all_points)
        if st is None or st.n < min_anchors or _too_broad(st):
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
        if st is None or st.n < min_anchors:
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
        if (pooled_stats is None or pooled_stats.n < min_anchors
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


def _primary_clusters(votes: Counter, min_support: int = MIN_SUPPORT) -> set:
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
        if c >= max(min_support, POS_DOM_FRAC * kind_top[kind])
    }


def _in_position_candidate(p, cl, i, band, max_v, page_heights, page_widths,
                           model, predicate):
    """Best in-position candidate satisfying predicate(kind, offset, votes-key),
    as (kind, v, r), or None."""
    best = None
    for b, kind, v, r in cl:
        if b != band:
            continue
        if v is not None and v > max_v:
            continue
        if not predicate(kind, v if v is not None else -1 - i):
            continue
        if _inside_position_model(
                p.source.page_index, r.box, page_heights[i], page_widths[i],
                model):
            if best is None:
                best = (kind, v, r)
    return best


def _assign_with_position_model(
    pages, cands, page_heights, page_widths, band, votes, max_v, model,
    min_support=MIN_SUPPORT, strict_dominant=False,
) -> list[str]:
    # Position is the gate (kind-agnostic). Pass 1: among in-position candidates
    # keep only those whose (kind, offset) is primary, and trust their own value
    # -- no snapping, so a real missing-page gap (its own primary cluster) is
    # never hidden. Pass 2: a page with no primary candidate but an in-position
    # candidate of the dominant kind, bracketed on both sides by the dominant
    # run, is an OCR misread -> recover it as index+dominant_offset.
    primary = _primary_clusters(votes, min_support)
    dom_kind, dom_offset = max(votes, key=lambda k: (votes[k], -abs(k[1])))
    chosen: list = [None] * len(pages)
    dom_idx: list[int] = []
    if strict_dominant:
        primary = {(dom_kind, dom_offset)}

    for i, (p, cl) in enumerate(zip(pages, cands)):
        in_model = []
        for b, kind, v, r in cl:
            if b != band or v is None or v > max_v or (kind, v - i) not in primary:
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
        near_short_cluster = (
            strict_dominant and (has_before[i] or has_after[i])
            and dom_idx and min(dom_idx) - 1 <= i <= max(dom_idx) + 1
        )
        if (chosen[i] is not None
                or not ((has_before[i] and has_after[i]) or near_short_cluster)):
            continue
        cand = _in_position_candidate(
            p, cl, i, band, max_v, page_heights, page_widths, model,
            lambda kind, off: kind == dom_kind)
        if cand is not None:
            chosen[i] = (dom_kind, i + dom_offset, cand[2])  # misread -> expected

    # Pass 2.5: Similarity Rescue
    # For pages that still have no valid candidate, check unparsed candidates.
    # If an unparsed candidate sits in the correct geometric position, compute
    # similarity to the expected number (i + dom_offset).
    for i, (p, cl) in enumerate(zip(pages, cands)):
        if chosen[i] is not None:
            continue
        expected_val = i + dom_offset
        if expected_val <= 0:
            continue
        expected_str = str(expected_val)
        best_sim_cand = None
        best_sim = 0.0
        for b, kind, v, r in cl:
            if b != band or v is not None:
                continue
            if not _inside_position_model(p.source.page_index, r.box, page_heights[i], page_widths[i], model):
                continue
            if not r.ocr_text:
                continue
            sim = difflib.SequenceMatcher(None, expected_str, r.ocr_text).ratio()
            if sim >= 0.7 and sim > best_sim:
                best_sim = sim
                best_sim_cand = r
        if best_sim_cand is not None:
            chosen[i] = (dom_kind, expected_val, best_sim_cand)

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
