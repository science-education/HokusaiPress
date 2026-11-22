"""Which way up a page is, decided from its content.

A book scan has one page orientation; a page saved sideways is a mistake to
undo, not a layout to keep. The image's aspect ratio only says THAT a page
differs from the book -- not which way to turn it, and not even whether it
is turned at all: a black chapter page whose scan was saved with the gray
scanner bed around it is wide but upright (img20261121_0010 p15). So:

1. crop the scanner bed (source.find_paper); if the sheet then has the
   book's orientation, nothing is turned;
2. otherwise OCR the page turned each way and keep the upright reading.
   Engines differ in what confidence they report (NDL reports none for
   recognition), so the test is the lines themselves: Japanese read the
   right way up starts every line flush and ends paragraphs short; upside
   down the ragged side is reversed (layout_score). Kana counts only break
   ties: NDL reads upside-down tategaki into plausible kana too.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

_KANA = re.compile(r"[ぁ-ゖァ-ヺー]")
_WORD = re.compile(r"[A-Za-z]{3,}")


def text_score(lines) -> float:
    """Kana plus Latin words in the OCR output (a weak signal: NDL reads
    upside-down tategaki into plausible kana too -- used to break ties)."""
    text = "".join((ln.get("text") or "") for ln in lines)
    return float(len(_KANA.findall(text)) + sum(len(w) for w in _WORD.findall(text)))


def layout_score(lines) -> float:
    """How upright the text lines sit: in print every line STARTS flush with
    the block (top in tategaki, left in yokogaki) and the last line of a
    paragraph ENDS short, while a paragraph's first line is indented one
    character at the start. Each such line votes +1; the mirror image (an
    upside-down page) votes -1."""
    boxes = [ln["box"] for ln in lines if ln.get("box")]
    boxes = [(float(x0), float(y0), float(x1), float(y1)) for x0, y0, x1, y1 in boxes
             if x1 > x0 and y1 > y0]
    if len(boxes) < 4:
        return 0.0
    vert = sum((y1 - y0) > (x1 - x0) for x0, y0, x1, y1 in boxes) * 2 > len(boxes)
    if vert:
        segs = [(y0, y1, x1 - x0) for x0, y0, x1, y1 in boxes if (y1 - y0) >= 3 * (x1 - x0)]
    else:
        segs = [(x0, x1, y1 - y0) for x0, y0, x1, y1 in boxes if (x1 - x0) >= 3 * (y1 - y0)]
    if len(segs) < 4:
        return 0.0
    lengths = np.array([b - a for a, b, _ in segs])
    tol = float(np.median([t for _, _, t in segs]))
    start = float(np.percentile([a for a, _, _ in segs], 10))
    end = float(np.percentile([b for _, b, _ in segs], 90))
    short = lengths < 0.7 * np.median(lengths)
    votes = 0
    for (a, b, _), sh in zip(segs, short):
        at_start, at_end = abs(a - start) < 0.5 * tol, abs(b - end) < 0.5 * tol
        if sh:
            votes += int(at_start and not at_end) - int(at_end and not at_start)
        else:
            # a paragraph's first line is indented one character at the START
            ind_start = 0.5 * tol <= a - start <= 1.8 * tol
            ind_end = 0.5 * tol <= end - b <= 1.8 * tol
            votes += int(ind_start and at_end) - int(ind_end and at_start)
    return float(votes)


_CV2 = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def choose_rotation(img: np.ndarray, engine, candidates=(90, 270)) -> tuple[int, dict]:
    """Clockwise rotation among candidates under which the page reads upright:
    line layout first (layout_score), kana count to break a tie.
    Returns (rotation, {rotation: (layout, text)})."""
    scores = {}
    for r in candidates:
        im = cv2.rotate(img, _CV2[r]) if r else img
        try:
            lines = engine(np.ascontiguousarray(im)).get("lines", [])
            scores[r] = (layout_score(lines), text_score(lines))
        except Exception:
            scores[r] = (-1e9, -1.0)
    best = max(scores, key=lambda r: (scores[r], -candidates.index(r)))
    return best, scores
