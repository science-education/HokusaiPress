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


def _crop_rect(paper: list[float] | None, raw_w: float, raw_h: float) -> tuple[float, float, float, float]:
    if paper:
        x0, y0, x1, y1 = (float(v) for v in paper)
        return max(0.0, x0), max(0.0, y0), min(raw_w, x1), min(raw_h, y1)
    return 0.0, 0.0, float(raw_w), float(raw_h)


def oriented_size(paper: list[float] | None, rotate: int,
                  raw_w: float, raw_h: float) -> tuple[float, float]:
    x0, y0, x1, y1 = _crop_rect(paper, raw_w, raw_h)
    cw, ch = max(1.0, x1 - x0), max(1.0, y1 - y0)
    if int(rotate or 0) % 360 in (90, 270):
        return ch, cw
    return cw, ch


def _orient_to_raw_matrix(paper: list[float] | None, rotate: int,
                          raw_w: float, raw_h: float) -> np.ndarray:
    """3x3 affine matrix mapping oriented-crop coords -> raw-image coords."""
    x0, y0, x1, y1 = _crop_rect(paper, raw_w, raw_h)
    cw, ch = max(1.0, x1 - x0), max(1.0, y1 - y0)
    r = int(rotate or 0) % 360
    if r == 0:
        M = np.array([[1.0, 0.0, x0], [0.0, 1.0, y0], [0.0, 0.0, 1.0]])
    elif r == 90:
        # x_or = ch - (y_raw - y0), y_or = x_raw - x0
        M = np.array([[0.0, 1.0, x0], [-1.0, 0.0, y0 + ch], [0.0, 0.0, 1.0]])
    elif r == 180:
        # x_or = cw - (x_raw - x0), y_or = ch - (y_raw - y0)
        M = np.array([[-1.0, 0.0, x0 + cw], [0.0, -1.0, y0 + ch], [0.0, 0.0, 1.0]])
    else:  # 270
        # x_or = y_raw - y0, y_or = cw - (x_raw - x0)
        M = np.array([[0.0, -1.0, x0 + cw], [1.0, 0.0, y0], [0.0, 0.0, 1.0]])
    return M


def _raw_to_orient_matrix(paper: list[float] | None, rotate: int,
                          raw_w: float, raw_h: float) -> np.ndarray:
    """3x3 affine matrix mapping raw-image coords -> oriented-crop coords."""
    return np.linalg.inv(_orient_to_raw_matrix(paper, rotate, raw_w, raw_h))


def box_transform_matrix(
    raw_size: tuple[float, float],
    old_paper: list[float] | None, old_rotate: int, old_deskew_deg: float,
    new_paper: list[float] | None, new_rotate: int, new_deskew_deg: float,
) -> np.ndarray:
    """2x3 affine matrix mapping boxes in old deskewed-oriented coords
    to new deskewed-oriented coords."""
    raw_w, raw_h = raw_size
    ow, oh = oriented_size(old_paper, old_rotate, raw_w, raw_h)
    nw, nh = oriented_size(new_paper, new_rotate, raw_w, raw_h)

    R_old = cv2.getRotationMatrix2D((ow / 2.0, oh / 2.0), float(old_deskew_deg), 1.0)
    R_old_inv = np.vstack([cv2.invertAffineTransform(R_old), [0.0, 0.0, 1.0]])

    M_to_raw = _orient_to_raw_matrix(old_paper, old_rotate, raw_w, raw_h)
    M_from_raw = _raw_to_orient_matrix(new_paper, new_rotate, raw_w, raw_h)

    R_new = cv2.getRotationMatrix2D((nw / 2.0, nh / 2.0), float(new_deskew_deg), 1.0)
    R_new_h = np.vstack([R_new, [0.0, 0.0, 1.0]])

    return (R_new_h @ M_from_raw @ M_to_raw @ R_old_inv)[:2, :]


def map_box_coords(box: tuple[float, float, float, float], M: np.ndarray) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0, 1.0], [x1, y0, 1.0], [x0, y1, 1.0], [x1, y1, 1.0]], dtype=np.float64).T
    out = M @ pts
    return float(out[0].min()), float(out[1].min()), float(out[0].max()), float(out[1].max())


def choose_rotation_from_regions(
    regions,
    raw_size: tuple[float, float],
    old_paper: list[float] | None,
    old_rotate: int,
    old_deskew_deg: float,
    new_paper: list[float] | None,
    candidates: tuple[int, ...] = (0, 180),
) -> tuple[int, dict]:
    """Pick upright rotation among `candidates` using already-stored OCR regions
    (no OCR re-run)."""
    scores = {}
    for r in candidates:
        M = box_transform_matrix(
            raw_size, old_paper, old_rotate, old_deskew_deg,
            new_paper, r, 0.0,
        )
        lines = []
        for reg in regions:
            b = reg.box
            nb = map_box_coords((b.x0, b.y0, b.x1, b.y1), M)
            lines.append({"box": nb, "text": reg.ocr_text or ""})
        scores[r] = (layout_score(lines), text_score(lines))
    best = max(scores, key=lambda r: (scores[r], -candidates.index(r)))
    return best, scores
