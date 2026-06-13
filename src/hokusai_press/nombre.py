"""Page-number (nombre) resolution from OCR.

The geometric nombre heuristic alone picks up stray small blocks, scattering the
box to different edges and even splitting it between top and bottom across pages.
But the OCR pass already recognized every text line, so we can do much better:

- a real nombre is a short token in the top/bottom margin band that *reads as a
  number* (Arabic 1/２/... or kanji 一/二/十二/...);
- across a document the nombre lives in ONE consistent band (almost never split
  top vs bottom), so we pick the band that explains the most pages and take the
  numeric candidate there;
- with numbers in hand we can flag pages whose number is unreadable, and warn on
  clear sequence breaks (a likely missing page).

This runs only when OCR produced text; with --no-ocr the geometric box stands.
"""

from __future__ import annotations

from typing import Optional

from .model import Box, Flag, PageParams, Region

BAND_FRAC = 0.14          # top/bottom 14% of the page is the nombre band
MAX_NOMBRE_LEN = 6        # a page number token is short
GAP_TOLERANCE = 1         # skipped numbers beyond this (with no blank pages) warn

_KANJI_DIGIT = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
                "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_KANJI_UNIT = {"十": 10, "百": 100, "千": 1000}


def parse_numeral(text: str) -> Optional[int]:
    """Parse a page-number token to int. Handles half/full-width Arabic digits
    and kanji numerals (both positional like 一二 and structured like 十二)."""
    if not text:
        return None
    s = "".join(text.split())
    if not s or len(s) > MAX_NOMBRE_LEN:
        return None
    # Arabic (normalize full-width to half-width)
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
    # positional (no unit chars), e.g. 一二三 -> 123
    if all(c in _KANJI_DIGIT for c in s):
        return int("".join(str(_KANJI_DIGIT[c]) for c in s))
    # structured, e.g. 十二 -> 12, 二百五 -> 205
    total, section, last_digit = 0, 0, 0
    for c in s:
        if c in _KANJI_DIGIT:
            last_digit = _KANJI_DIGIT[c]
            section += last_digit
        else:  # unit
            unit = _KANJI_UNIT[c]
            section = section - last_digit + max(last_digit, 1) * unit
            total += section
            section, last_digit = 0, 0
    return total + section


def _band(box: Box, page_h: float) -> Optional[str]:
    cy = (box.y0 + box.y1) / 2.0
    if cy <= page_h * BAND_FRAC:
        return "top"
    if cy >= page_h * (1.0 - BAND_FRAC):
        return "bottom"
    return None


def _candidates(params: PageParams, page_h: float) -> list[tuple[str, int, Region]]:
    """(band, value, region) for every short numeric text line in a margin band."""
    out = []
    for r in params.regions:
        if not r.ocr_text:
            continue
        v = parse_numeral(r.ocr_text)
        if v is None:
            continue
        b = _band(r.box, page_h)
        if b:
            out.append((b, v, r))
    return out


def resolve(pages: list[PageParams], page_heights: list[float]) -> list[str]:
    """Assign page_number/nombre_text from OCR, choosing one consistent band,
    flag unreadable pages, and return human-readable gap warnings.

    Mutates each page's margin.nombre_box / page_number / nombre_text / flags.
    """
    cands = [_candidates(p, h) for p, h in zip(pages, page_heights)]

    # pick the band (top/bottom) that yields numbers on the most pages
    counts = {"top": 0, "bottom": 0}
    for cl in cands:
        for b in {b for b, _, _ in cl}:
            counts[b] += 1
    band = "bottom" if counts["bottom"] >= counts["top"] else "top"
    if counts[band] == 0:
        return []   # no numeric nombre anywhere; leave geometric boxes as-is

    # assign per page from the chosen band, preferring continuity with prev
    prev: Optional[int] = None
    for p, cl in zip(pages, cands):
        inband = [(v, r) for b, v, r in cl if b == band]
        if not inband:
            p.page_number, p.nombre_text = None, None
            continue
        if prev is not None:
            inband.sort(key=lambda vr: abs(vr[0] - (prev + 1)))
        else:
            inband.sort(key=lambda vr: vr[0])
        value, region = inband[0]
        p.page_number = value
        p.nombre_text = region.ocr_text
        if p.margin:
            p.margin.nombre_box = region.box   # snap nombre to the real number
        prev = value

    # Only warn on *clear* sequence breaks (likely missing pages), and only when
    # the document is clearly numbered. We deliberately do NOT flag every page
    # that lacks a number: unnumbered front matter / plates are normal, and the
    # goal is to reduce review volume, not flood it.
    numbered = sum(1 for p in pages if p.page_number is not None)
    if numbered < max(3, len(pages) // 2):
        return []
    return _gap_warnings(pages)


def _gap_warnings(pages: list[PageParams]) -> list[str]:
    """Flag PAGE_NUMBER_GAP and describe likely missing pages. Conservative: we
    only warn on a forward jump larger than the unnumbered (blank/plate) pages
    between two numbered pages can explain, plus a tolerance. Decreases are NOT
    warned — section restarts (chapters renumbering from 1) are normal."""
    warnings: list[str] = []
    last_val: Optional[int] = None
    gap_since = 0
    for p in pages:
        if p.page_number is None:
            gap_since += 1
            continue
        if last_val is not None and p.page_number > last_val:
            delta = p.page_number - last_val
            expected = 1 + gap_since   # +1 per physical page, incl. unnumbered
            if delta > expected + GAP_TOLERANCE:
                if Flag.PAGE_NUMBER_GAP not in p.flags:
                    p.flags.append(Flag.PAGE_NUMBER_GAP)
                missing = delta - expected
                warnings.append(
                    f"p{p.source.page_index}: page number {last_val} -> "
                    f"{p.page_number} (~{missing} page(s) may be missing)")
        last_val = p.page_number
        gap_since = 0
    return warnings
