"""Duplex sheet-pair utilities (A: pairing, B: translation, C: skew, D: blank insertion).

Follows the base-then-exception principle: standard per-page processing runs
first; these routines only intervene on pages that fail a duplex-pair check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from .model import Box, Deskew, Flag, PageParams

TRUSTED_FOLIO_SOURCES = frozenset({"folio", "inverted_folio", "folio_roman"})
ARABIC_FOLIO_SOURCES = frozenset({"folio", "inverted_folio"})
MIN_DUPLEX_PAIRS = 10
MIN_DUPLEX_CORR_X = -0.5
SKEW_PAIR_DIFF_DEG = 0.5
SKEW_INDEP_MIN_DIFF_DEG = 0.3
SKEW_INDEP_MAX_RESIDUAL_DEG = 0.6


@dataclass
class DuplexPairing:
    """Book-level duplex sheet pairing determined from folio parity and shift correlation."""

    known: bool
    parity: int  # 0: sheets are (2k, 2k+1); 1: shifted (2k+1, 2k+2)
    status: str  # "通常" | "ずれ" | "不明"
    pair_count: int
    corr_x: float
    med: Optional[dict[int, tuple[float, float]]]
    trusted: list[bool]
    n_pages: int

    def partner_of(self, index: int) -> Optional[int]:
        """Return the partner page index on the same physical sheet, or None."""
        j = index + 1 if (index % 2 == self.parity) else index - 1
        return j if 0 <= j < self.n_pages else None


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _crop_center_offset_mm(p: PageParams) -> Optional[tuple[float, float]]:
    m = p.margin
    if not (m and m.crop and m.page_w and m.page_h):
        return None
    dpi = p.dpi if p.dpi and p.dpi > 0 else 300.0
    mm = 25.4 / dpi
    return (
        ((m.crop.x0 + m.crop.x1) / 2.0 - m.page_w / 2.0) * mm,
        ((m.crop.y0 + m.crop.y1) / 2.0 - m.page_h / 2.0) * mm,
    )


def _is_trusted_folio(p: PageParams) -> bool:
    m = p.margin
    return bool(
        m and m.crop and m.page_w and m.page_h and not p.blank
        # A tinted (full-sheet ground) page is cropped as the whole sheet,
        # not by its type area, so its offset is no reference for its partner.
        and not getattr(m, "tinted", False)
        and p.page_number is not None
        and p.page_number_source in TRUSTED_FOLIO_SOURCES
        and m.nombre_box is not None
        and m.nombre_source != "geom"
    )


def determine_duplex_pairing(
    pages: list[PageParams],
    min_pairs: int = MIN_DUPLEX_PAIRS,
    min_corr_x: float = MIN_DUPLEX_CORR_X,
) -> DuplexPairing:
    """Part A: determine physical sheet pairing for a book.

    Uses Arabic folio parity vs page_index to detect normal (2k, 2k+1) vs
    shifted (2k+1, 2k+2) pairing, then verifies that trusted pairs show
    mirrored horizontal crop-center residuals (Spearman corr <= min_corr_x).
    """
    n = len(pages)
    trusted = [_is_trusted_folio(p) for p in pages]

    c_norm = 0
    c_shift = 0
    for i, p in enumerate(pages):
        if p.blank or p.page_number is None or p.page_number_source not in ARABIC_FOLIO_SOURCES:
            continue
        if (p.page_number + i) % 2 == 1:
            c_norm += 1
        else:
            c_shift += 1
    parity = 0 if c_norm >= c_shift else 1

    offsets = [_crop_center_offset_mm(p) for p in pages]
    med: dict[int, tuple[float, float]] = {}
    for par in (0, 1):
        pts = [offsets[i] for i in range(n) if trusted[i] and i % 2 == par and offsets[i] is not None]
        if len(pts) < 5:
            return DuplexPairing(
                known=False, parity=parity, status="不明",
                pair_count=0, corr_x=0.0, med=None, trusted=trusted, n_pages=n,
            )
        med[par] = (
            float(np.median([pt[0] for pt in pts])),
            float(np.median([pt[1] for pt in pts])),
        )

    pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for i in range(n):
        j = i + 1 if (i % 2 == parity) else i - 1
        if 0 <= j < n and i < j and trusted[i] and trusted[j]:
            oi, oj = offsets[i], offsets[j]
            if oi is not None and oj is not None:
                ri = (oi[0] - med[i % 2][0], oi[1] - med[i % 2][1])
                rj = (oj[0] - med[j % 2][0], oj[1] - med[j % 2][1])
                pairs.append((ri, rj))

    if len(pairs) < min_pairs:
        return DuplexPairing(
            known=False, parity=parity, status="不明",
            pair_count=len(pairs), corr_x=0.0, med=med, trusted=trusted, n_pages=n,
        )

    ax = np.array([q[0][0] for q in pairs], dtype=np.float64)
    bx = np.array([q[1][0] for q in pairs], dtype=np.float64)
    corr_x = _spearman(ax, bx)
    if corr_x > min_corr_x:
        return DuplexPairing(
            known=False, parity=parity, status="不明",
            pair_count=len(pairs), corr_x=corr_x, med=med, trusted=trusted, n_pages=n,
        )

    return DuplexPairing(
        known=True,
        parity=parity,
        status="通常" if parity == 0 else "ずれ",
        pair_count=len(pairs),
        corr_x=corr_x,
        med=med,
        trusted=trusted,
        n_pages=n,
    )


def apply_duplex_translation_exceptions(
    pages: list[PageParams],
    pairing: Optional[DuplexPairing] = None,
) -> dict:
    """Part B: 3-stage translation exception for body pages with unreliable folio.

    Stages:
      1. Copy partner's residual (negate X, keep Y) around same-parity median.
      2. If that clips interior OCR text, fall back to same-parity median.
      3. If that also clips, revert to base crop (uncorrected) and record warning.
    """
    from .geometry.margin import _clips_interior_ocr_text, folio_not_found

    if pairing is None:
        pairing = determine_duplex_pairing(pages)
    result = {"applied": False, "partner": [], "median": [], "uncorrected": []}
    if not pairing.known or pairing.med is None:
        return result

    folio_indices = [i for i, p in enumerate(pages) if p.page_number is not None]
    if not folio_indices:
        return result
    lo, hi = min(folio_indices), max(folio_indices)
    result["applied"] = True

    trusted_cws = [
        pages[k].margin.content.width
        for k in range(len(pages))
        if pairing.trusted[k] and pages[k].margin and pages[k].margin.content
    ]
    trusted_chs = [
        pages[k].margin.content.height
        for k in range(len(pages))
        if pairing.trusted[k] and pages[k].margin and pages[k].margin.content
    ]
    med_cw = float(np.median(trusted_cws)) if trusted_cws else 0.0
    med_ch = float(np.median(trusted_chs)) if trusted_chs else 0.0

    for i in range(lo, hi + 1):
        p = pages[i]
        m = p.margin
        if p.blank or not (m and m.crop and m.page_w and m.page_h):
            continue
        if m.fill or getattr(m, "tinted", False) or Flag.CONTENT_SCALED_DOWN in p.flags:
            continue
        if getattr(m, "bg_tint_frac", 0.0) >= 0.45:
            continue
        if not folio_not_found(p):
            continue
        j = pairing.partner_of(i)
        if j is None or not pairing.trusted[j] or folio_not_found(pages[j]):
            continue
        oj = _crop_center_offset_mm(pages[j])
        if oj is None:
            continue

        dpi = p.dpi if p.dpi and p.dpi > 0 else 300.0
        px_per_mm = dpi / 25.4
        cw, ch = m.crop.width, m.crop.height
        base_crop = Box(m.crop.x0, m.crop.y0, m.crop.x1, m.crop.y1)

        rj_x = oj[0] - pairing.med[j % 2][0]
        rj_y = oj[1] - pairing.med[j % 2][1]
        ox_p = pairing.med[i % 2][0] - rj_x
        oy_p = pairing.med[i % 2][1] + rj_y
        x0 = m.page_w / 2.0 + ox_p * px_per_mm - cw / 2.0
        y0 = m.page_h / 2.0 + oy_p * px_per_mm - ch / 2.0
        cand_partner = Box(x0, y0, x0 + cw, y0 + ch)

        # Skip non-body layout pages (chapter title / plate pages with small
        # off-type-area content where Pass 1 placed them independently).
        if (
            med_cw > 0 and med_ch > 0 and m.content is not None
            and m.content.width < 0.35 * med_cw
            and m.content.height < 0.75 * med_ch
            and sum(1 for r in (p.regions or []) if r.ocr_text) <= 5
            and max(abs(cand_partner.x0 - base_crop.x0), abs(cand_partner.y0 - base_crop.y0)) > 20.0 * px_per_mm
        ):
            continue
        if not _clips_interior_ocr_text(p, cand_partner, dpi):
            m.crop = cand_partner
            result["partner"].append(i)
            continue

        ox_m, oy_m = pairing.med[i % 2]
        x0_m = m.page_w / 2.0 + ox_m * px_per_mm - cw / 2.0
        y0_m = m.page_h / 2.0 + oy_m * px_per_mm - ch / 2.0
        cand_median = Box(x0_m, y0_m, x0_m + cw, y0_m + ch)
        if not _clips_interior_ocr_text(p, cand_median, dpi):
            m.crop = cand_median
            result["median"].append(i)
            continue

        m.crop = base_crop
        result["uncorrected"].append(i)
        if Flag.MARGIN_INCONSISTENT not in p.flags:
            p.flags.append(Flag.MARGIN_INCONSISTENT)

    return result


def _binary_without_streaks(img_bgr: np.ndarray, rotate: int = 0) -> tuple[np.ndarray, list]:
    """Binarize for deskew scoring and zero out detected scanner streaks."""
    from .geometry.deskew import _binary
    from .streak import page_streaks

    b = _binary(img_bgr)
    bh, bw = b.shape
    ih, iw = img_bgr.shape[:2]
    scale = bw / max(1, iw)
    streaks = page_streaks(img_bgr, rotate)
    if streaks:
        axis = 1 if int(rotate or 0) % 180 else 0
        for pos_frac, width_px in streaks:
            pad = max(2, int(np.ceil(width_px * scale / 2.0)) + 2)
            if axis == 0:
                cx = int(round(pos_frac * bw))
                b[:, max(0, cx - pad):min(bw, cx + pad + 1)] = 0.0
            else:
                cy = int(round(pos_frac * bh))
                b[max(0, cy - pad):min(bh, cy + pad + 1), :] = 0.0
    return b, streaks


def score_skew_pair(
    img_a: np.ndarray,
    rot_a: int,
    ang_a: float,
    img_b: np.ndarray,
    rot_b: int,
    ang_b: float,
) -> tuple[float, float, tuple[float, float], tuple[float, float]]:
    """Score (ang_a, -ang_a) vs (-ang_b, ang_b) with scanner streaks removed."""
    from .geometry.deskew import _profile_sharpness, _rotate

    ba, _ = _binary_without_streaks(img_a, rot_a)
    bb, _ = _binary_without_streaks(img_b, rot_b)
    sa_1 = _profile_sharpness(_rotate(ba, ang_a))
    sa_2 = _profile_sharpness(_rotate(ba, -ang_b))
    sb_1 = _profile_sharpness(_rotate(bb, -ang_a))
    sb_2 = _profile_sharpness(_rotate(bb, ang_b))

    na_1 = sa_1 / max(sa_1, sa_2, 1e-9)
    na_2 = sa_2 / max(sa_1, sa_2, 1e-9)
    nb_1 = sb_1 / max(sb_1, sb_2, 1e-9)
    nb_2 = sb_2 / max(sb_1, sb_2, 1e-9)
    return (na_1 + nb_1, na_2 + nb_2, (sa_1, sa_2), (sb_1, sb_2))


def _rotate_box(box: Box, delta_deg: float, cx: float, cy: float) -> Box:
    """Rotate an axis-aligned Box by delta_deg (OpenCV CCW) around (cx, cy)."""
    M = cv2.getRotationMatrix2D((cx, cy), delta_deg, 1.0)
    pts = np.array([
        [box.x0, box.y0, 1.0],
        [box.x1, box.y0, 1.0],
        [box.x0, box.y1, 1.0],
        [box.x1, box.y1, 1.0],
    ], dtype=np.float64)
    rot = pts @ M.T
    return Box(
        float(rot[:, 0].min()),
        float(rot[:, 1].min()),
        float(rot[:, 0].max()),
        float(rot[:, 1].max()),
    )


def rotate_page_geometry(p: PageParams, delta_deg: float, w: int, h: int) -> None:
    """Transform stored regions and margin boxes when deskew angle changes by delta_deg."""
    if abs(delta_deg) < 1e-4:
        return
    cx, cy = w / 2.0, h / 2.0
    for r in p.regions or []:
        r.box = _rotate_box(r.box, delta_deg, cx, cy)
    if p.margin:
        if p.margin.nombre_box is not None:
            p.margin.nombre_box = _rotate_box(p.margin.nombre_box, delta_deg, cx, cy)
        if p.margin.glyph_box is not None:
            p.margin.glyph_box = _rotate_box(p.margin.glyph_box, delta_deg, cx, cy)


def _fit_line_angle_robust(xs: np.ndarray, ys: np.ndarray, vertical: bool) -> Optional[float]:
    """Fit a Huber line to boundary points and return residual skew angle in degrees."""
    if len(xs) < 6:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01)
    vx, vy = float(vx[0]), float(vy[0])
    if vertical:
        if vy < 0:
            vx, vy = -vx, -vy
        ang = float(np.degrees(np.arctan2(-vx, vy)))
    else:
        if vx < 0:
            vx, vy = -vx, -vy
        ang = float(np.degrees(np.arctan2(vy, vx)))
    return ang if abs(ang) <= 12.0 else None


def measure_residual_skew_independent(
    img_bgr: np.ndarray,
    angle_deg: float,
    rotate: int = 0,
) -> Optional[float]:
    """Measure residual skew (deg) after rotating by ``angle_deg``, independent of projection sharpness.

    Combines boundary line fits of smeared text columns (tategaki) and rows (yokogaki)
    with Hough line segments on thin ruled lines/borders (after masking out thick
    solid/photo regions and scanner streaks). Returns ``None`` if too few lines exist.
    """
    from .geometry.deskew import _rotate

    b, _ = _binary_without_streaks(img_bgr, rotate)
    if float(b.sum()) < 150.0:
        return None

    b_rot = _rotate(b, angle_deg)
    bh, bw = b_rot.shape
    u8 = (b_rot > 0).astype(np.uint8) * 255

    # 1. Ruled lines / borders via HoughLinesP on thin components (masking out thick photo/solid blobs)
    thick = cv2.erode(u8, np.ones((9, 9), np.uint8))
    thick_mask = cv2.dilate(thick, np.ones((17, 17), np.uint8))
    thin_u8 = cv2.bitwise_and(u8, cv2.bitwise_not(thick_mask))
    min_len = max(120, int(min(bh, bw) * 0.14))
    lines = cv2.HoughLinesP(
        thin_u8, 1, np.pi / 1800.0, threshold=90, minLineLength=min_len, maxLineGap=3
    )
    hough_angs: list[float] = []
    hough_weights: list[float] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = float(x2 - x1), float(y2 - y1)
            length = float(np.hypot(dx, dy))
            if length < min_len:
                continue
            if abs(dx) >= abs(dy):
                if dx < 0:
                    dx, dy = -dx, -dy
                ang = float(np.degrees(np.arctan2(dy, dx)))
            else:
                if dy < 0:
                    dx, dy = -dx, -dy
                ang = float(np.degrees(np.arctan2(-dx, dy)))
            if abs(ang) <= 10.0:
                hough_angs.append(ang)
                hough_weights.append(length)

    # 2. Text column (tategaki) and row (yokogaki) boundary fits
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(u8)
    cw_arr = stats[:, cv2.CC_STAT_WIDTH]
    ch_arr = stats[:, cv2.CC_STAT_HEIGHT]
    area_arr = stats[:, cv2.CC_STAT_AREA]
    valid = (
        (area_arr >= 6)
        & (cw_arr <= bw * 0.25)
        & (ch_arr <= bh * 0.25)
        & ((cw_arr >= 2) | (ch_arr >= 2))
    )
    valid[0] = False
    char_count = int(valid.sum())
    keep_lut = np.zeros(n_lbl, dtype=np.uint8)
    keep_lut[valid] = 255
    clean = keep_lut[labels]

    col_angs: list[float] = []
    col_weights: list[float] = []
    row_angs: list[float] = []
    row_weights: list[float] = []
    if char_count >= 10:
        k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
        v_smear = cv2.dilate(clean, k_v)
        nv, lv, sv, _ = cv2.connectedComponentsWithStats(v_smear)
        min_col_h = max(120, int(bh * 0.10))
        for k in range(1, nv):
            cw, ch = sv[k, cv2.CC_STAT_WIDTH], sv[k, cv2.CC_STAT_HEIGHT]
            if ch < min_col_h or cw > 90 or cw < 5:
                continue
            x0, y0 = sv[k, cv2.CC_STAT_LEFT], sv[k, cv2.CC_STAT_TOP]
            sub = np.where(lv[y0:y0 + ch, x0:x0 + cw] == k, clean[y0:y0 + ch, x0:x0 + cw], 0)
            lx, ly, rx, ry = [], [], [], []
            for yy in range(0, ch, 8):
                band = sub[yy:min(ch, yy + 8), :]
                xs_on = np.flatnonzero(band.any(axis=0))
                if xs_on.size > 0:
                    ymid = float(yy + band.shape[0] / 2.0)
                    lx.append(float(xs_on[0]))
                    ly.append(ymid)
                    rx.append(float(xs_on[-1]))
                    ry.append(ymid)
            a_l = _fit_line_angle_robust(np.array(lx), np.array(ly), vertical=True)
            a_r = _fit_line_angle_robust(np.array(rx), np.array(ry), vertical=True)
            if a_l is not None and a_r is not None and abs(a_l - a_r) <= 2.2:
                col_angs.append((a_l + a_r) / 2.0)
                col_weights.append(float(ch))

        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        h_smear = cv2.dilate(clean, k_h)
        nh, lh, sh, _ = cv2.connectedComponentsWithStats(h_smear)
        min_row_w = max(120, int(bw * 0.10))
        for k in range(1, nh):
            cw, ch = sh[k, cv2.CC_STAT_WIDTH], sh[k, cv2.CC_STAT_HEIGHT]
            if cw < min_row_w or ch > 90 or ch < 5:
                continue
            x0, y0 = sh[k, cv2.CC_STAT_LEFT], sh[k, cv2.CC_STAT_TOP]
            sub = np.where(lh[y0:y0 + ch, x0:x0 + cw] == k, clean[y0:y0 + ch, x0:x0 + cw], 0)
            tx, ty, bx, by = [], [], [], []
            for xx in range(0, cw, 8):
                band = sub[:, xx:min(cw, xx + 8)]
                ys_on = np.flatnonzero(band.any(axis=1))
                if ys_on.size > 0:
                    xmid = float(xx + band.shape[1] / 2.0)
                    tx.append(xmid)
                    ty.append(float(ys_on[0]))
                    bx.append(xmid)
                    by.append(float(ys_on[-1]))
            a_t = _fit_line_angle_robust(np.array(tx), np.array(ty), vertical=False)
            a_b = _fit_line_angle_robust(np.array(bx), np.array(by), vertical=False)
            if a_t is not None and a_b is not None and abs(a_t - a_b) <= 2.2:
                row_angs.append((a_t + a_b) / 2.0)
                row_weights.append(float(cw))

    all_angs: list[float] = []
    all_weights: list[float] = []
    col_len = sum(col_weights)
    row_len = sum(row_weights)
    if col_angs and col_len >= row_len * 0.8:
        all_angs.extend(col_angs)
        all_weights.extend(col_weights)
    elif row_angs and row_len > col_len * 0.8:
        all_angs.extend(row_angs)
        all_weights.extend(row_weights)

    if len(hough_angs) >= 2:
        if len(hough_angs) > 80:
            top_idx = np.argsort(hough_weights)[-80:]
            hough_angs = [hough_angs[idx] for idx in top_idx]
            hough_weights = [hough_weights[idx] for idx in top_idx]
        all_angs.extend(hough_angs)
        all_weights.extend([w * 1.2 for w in hough_weights])

    strong_single = len(all_angs) == 1 and all_weights[0] >= 180.0
    if len(all_angs) < 2 and not strong_single:
        return None

    order = np.argsort(all_angs)
    s_ang = np.array(all_angs)[order]
    s_w = np.array(all_weights)[order]
    cdf = np.cumsum(s_w) / s_w.sum()
    return round(float(s_ang[np.searchsorted(cdf, 0.5)]), 3)


def apply_duplex_skew_exceptions(
    pages: list[PageParams],
    pairing: DuplexPairing,
    load_image_fn: Callable[[int], np.ndarray],
    threshold_deg: float = SKEW_PAIR_DIFF_DEG,
    min_diff_deg: float = SKEW_INDEP_MIN_DIFF_DEG,
    max_residual_deg: float = SKEW_INDEP_MAX_RESIDUAL_DEG,
) -> list[dict]:
    """Part C: evaluate discrepant pairs (|theta_a + theta_b| > threshold_deg) using
    the independent residual skew metric.

    Each page is updated independently only when both its current angle and the
    partner-mirrored angle are measurable, the partner-mirrored angle reduces
    |residual| by at least ``min_diff_deg``, and the resulting |residual| is
    within ``max_residual_deg``.
    """
    if not pairing.known:
        return []
    changes: list[dict] = []
    n = len(pages)
    for i in range(n):
        j = pairing.partner_of(i)
        if j is None or i >= j:
            continue
        pa, pb = pages[i], pages[j]
        if pa.blank or pb.blank:
            continue
        ang_a = pa.deskew.angle_deg if pa.deskew else 0.0
        ang_b = pb.deskew.angle_deg if pb.deskew else 0.0
        if abs(ang_a + ang_b) <= threshold_deg:
            continue

        img_a = load_image_fn(pa.source.page_index)
        img_b = load_image_fn(pb.source.page_index)
        rot_a = pa.source.rotate if pa.source else 0
        rot_b = pb.source.rotate if pb.source else 0

        for p, partner_p, img, rot, old_ang, cand_ang in (
            (pa, pb, img_a, rot_a, ang_a, -ang_b),
            (pb, pa, img_b, rot_b, ang_b, -ang_a),
        ):
            res_curr = measure_residual_skew_independent(img, old_ang, rot)
            res_cand = measure_residual_skew_independent(img, cand_ang, rot)
            if res_curr is None or res_cand is None:
                continue
            diff = abs(res_curr) - abs(res_cand)
            if diff >= min_diff_deg and abs(res_cand) <= max_residual_deg:
                new_ang = cand_ang
                if abs(new_ang - old_ang) > 1e-3:
                    h, w = img.shape[:2]
                    conf = p.deskew.confidence if p.deskew else 0.0
                    p.deskew = Deskew(angle_deg=round(float(new_ang), 3), confidence=conf)
                    rotate_page_geometry(p, new_ang - old_ang, w, h)
                    changes.append({
                        "page_index": p.source.page_index,
                        "partner_index": partner_p.source.page_index,
                        "old_angle": old_ang,
                        "new_angle": round(float(new_ang), 3),
                        "res_curr": res_curr,
                        "res_cand": res_cand,
                        "res_diff": round(float(diff), 3),
                    })
    return changes


def detect_blank_insertions(
    pages: list[PageParams],
    pairing: DuplexPairing,
) -> dict:
    """Part D: detect unambiguous blank-skip insertion position for shifted books.

    Condition (a): pairing.known and pairing.parity == 1 (odd folios on shifted side).
    Condition (b): unique transition index s in [1, first_folio_idx] (s odd) where
    pairing switches from (2k, 2k+1) to (2k+1, 2k+2). Only inserts when unique.
    """
    if not pairing.known or pairing.parity != 1:
        return {"insert_before": [], "reason": "not_shifted", "candidates": []}

    folio_indices = [i for i, p in enumerate(pages) if p.page_number is not None]
    if not folio_indices:
        return {"insert_before": [], "reason": "no_folios", "candidates": []}
    first_fn = min(folio_indices)
    candidates = list(range(1, first_fn + 1, 2))
    if len(candidates) == 1:
        return {
            "insert_before": candidates,
            "reason": "unique_before_first_folio",
            "candidates": candidates,
        }
    return {
        "insert_before": [],
        "reason": "ambiguous_interval",
        "candidates": candidates,
    }
