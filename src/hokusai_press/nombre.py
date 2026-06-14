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

from collections import Counter
from typing import Optional

from .model import Box, Flag, PageParams, Region  # noqa: F401

BAND_FRAC = 0.14          # top/bottom 14% of the page is the nombre band
MAX_NOMBRE_LEN = 6        # a page-number token is short
MIN_SUPPORT = 3           # a numbering run must span >= this many pages to trust
GAP_TOLERANCE = 1         # skipped numbers beyond this (with no blank pages) warn

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
    return out


def resolve(pages: list[PageParams], page_heights: list[float]) -> list[str]:
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
            continue
        kind, v, r = choice
        p.page_number, p.nombre_text = v, r.ocr_text
        if p.margin:
            p.margin.nombre_box = r.box
        assigns.append((p.source.page_index, kind, v))
    return _gap_warnings(assigns, {p.source.page_index: p for p in pages})


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
