"""Automatic tone restoration for diagram regions degraded by binarization.

Compares the pre-binarization rendered image (``out_bgr``) against the
actual composited output (bilevel + photo overlays + vector fills + tint
zones) in the exact same output coordinate frame, detects continuous-tone
diagram components lost to black crush or whiteout, groups related parts
using OCR body-line boundaries and paper whitespace, and produces an alpha
mask for single-pass grayscale/color overlay compositing.

Thresholds and grouping rules are derived from the validated ``release``
prototype (``hp-diagram-trial/audit-20261127/auto.py`` and
``auto_group.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import cv2
import numpy as np


# Analysis runs at ~144 dpi equivalent so thresholds are invariant to target_dpi.
_ANALYSIS_DPI: float = 144.0

# --- Candidate detection thresholds (prototype release mode) ---
_INTERIOR_Y0: float = 0.035
_INTERIOR_Y1: float = 0.965
_INTERIOR_X0: float = 0.045
_INTERIOR_X1: float = 0.955
_PAPER_PERCENTILE: float = 94.0

_MID_MIN_LOCAL: float = 45.0
_MID_PAPER_MARGIN: float = 11.0
_MID_MAX_STD: float = 16.0

_LOSS_WHITE_MARGIN: float = 8.0
_LOSS_WHITE_DELTA: float = 15.0
_LOSS_BLACK_MAX: float = 42.0
_LOSS_BLACK_DELTA: float = 26.0

_CORE_PAPER_MARGIN: float = 13.0

_MIN_AREA_PX: int = 120
_MIN_AREA_FRAC: float = 0.00014
_MIN_SIDE_PX: int = 5
_TINY_W_FRAC: float = 0.046
_TINY_H_FRAC: float = 0.041
_FULLPAGE_W_FRAC: float = 0.91
_FULLPAGE_H_FRAC: float = 0.84
_BORDER_X0_FRAC: float = 0.02
_BORDER_Y0_FRAC: float = 0.027
_BORDER_X1_FRAC: float = 0.98
_BORDER_Y1_FRAC: float = 0.976

_MIN_CHANGED_PX: int = 64
_MIN_CHANGED_FRAC: float = 0.000055
_MIN_STABLE_PX: int = 105
_MIN_STABLE_AREA_FRAC: float = 0.23
_LONG_LOSS_FRAC: float = 0.10
_MIN_CHANGED_STABLE_FRAC: float = 0.17

_MIN_SUPPORT_PX: int = 45
_MIN_SUPPORT_AREA_FRAC: float = 0.10
_MIN_LONG_SIDE_FRAC: float = 0.068

# Release shape filters (reject vertical text columns, long rules, body rows)
_TALL_ASPECT_RATIO: float = 6.0
_TALL_MAX_W_FRAC: float = 0.057
_WIDE_ASPECT_RATIO: float = 8.0
_WIDE_MAX_H_FRAC: float = 0.017
_RELEASE_TALL_ASPECT: float = 2.75
_RELEASE_TALL_MAX_W_FRAC: float = 0.038
_BODY_ROW_ASPECT: float = 6.0
_BODY_ROW_MAX_H_FRAC: float = 0.032
_BODY_ROW_OVERLAP_FRAC: float = 0.28

_SILHOUETTE_AREA_MULT: int = 5
_SILHOUETTE_PAGE_FRAC: float = 0.085
_SILHOUETTE_BBOX_FRAC: float = 0.20

_NO_BODY_MIN_W_FRAC: float = 0.32
_NO_BODY_MIN_H_FRAC: float = 0.28

# --- OCR body-line detection thresholds (auto_group.py) ---
_BODY_MIN_CHARS: int = 15
_BODY_VERT_MIN_H_FRAC: float = 0.22
_BODY_HORIZ_MIN_W_FRAC: float = 0.25
_BODY_LINE_ASPECT: float = 7.0
_BODY_CORR_GAP_MIN: float = 0.005
_BODY_CORR_GAP_MAX: float = 0.045
_BODY_CORR_OVERLAP_FRAC: float = 0.68

# --- Diagram grouping thresholds (auto_group.py) ---
_GROUP_REACH_X_FRAC: float = 0.125
_GROUP_REACH_Y_FRAC: float = 0.10
_GROUP_MIN_W_FRAC: float = 0.26
_GROUP_MIN_H_FRAC: float = 0.085
_GROUP_MAX_AREA_FRAC: float = 0.44
_GROUP_MAX_SINGLE_FRAC: float = 0.83
_GROW_PAD_X_FRAC: float = 0.065
_GROW_PAD_Y_FRAC: float = 0.195
_GROW_CLAMP_X0_FRAC: float = 0.022
_GROW_CLAMP_X1_FRAC: float = 0.978
_GROW_CLAMP_Y0_FRAC: float = 0.03
_GROW_CLAMP_Y1_FRAC: float = 0.966
_GROW_MAX_EXPANDED_FRAC: float = 0.40
_GROW_BODY_OVERLAP_FRAC: float = 0.30
_INK_PAPER_MARGIN: float = 16.0
_INK_DENSITY_FRAC: float = 0.009
_SAFE_ROW_FRAC: float = 0.028
_CUT_ROW_MAX_FRAC: float = 0.055


@dataclass
class ToneRestoreResult:
    """Result of automatic tone-restoration analysis for one page."""

    alpha_mask: np.ndarray | None = None  # float32 (H, W) in [0, 1], or None
    zones: list[tuple[int, int, int, int]] = field(default_factory=list)
    expanded_regions: list[tuple[int, int, int, int]] = field(default_factory=list)
    loss_before: int = 0
    loss_after: int = 0
    adopted: bool = False
    reason: str = "no_candidates"


def _ellipse(radius: int) -> np.ndarray:
    side = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))


def _box_overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    u = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    v = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return u * v


def extract_body_lines(
    lines: list[dict],
    scale_x: float,
    scale_y: float,
    width: int,
    height: int,
) -> list[tuple[tuple[int, int, int, int], bool]]:
    """Identify corroborated horizontal/vertical body-text lines in analysis space."""
    if not lines:
        return []
    candidates: list[tuple[tuple[float, float, float, float], bool]] = []
    for ln in lines:
        text = (ln.get("text") or "").strip()
        if len(text) < _BODY_MIN_CHARS:
            continue
        bx0, by0, bx1, by1 = ln["box"]
        rect = (bx0 * scale_x, by0 * scale_y, bx1 * scale_x, by1 * scale_y)
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        if w <= 0 or h <= 0:
            continue
        vertical = ln.get("direction") == "v" or h > w
        if (vertical and h > _BODY_VERT_MIN_H_FRAC * height and h > _BODY_LINE_ASPECT * w) or (
            not vertical and w > _BODY_HORIZ_MIN_W_FRAC * width and w > _BODY_LINE_ASPECT * h
        ):
            candidates.append((rect, vertical))

    def _corroborated(box: tuple[float, float, float, float], vertical: bool) -> bool:
        along = (1, 3) if vertical else (0, 2)
        across = 0 if vertical else 1
        dim = width if vertical else height
        u, v = along
        for other, same_dir in candidates:
            if vertical != same_dir:
                continue
            gap = abs(box[across] - other[across])
            ov = min(box[v], other[v]) - max(box[u], other[u])
            required = _BODY_CORR_OVERLAP_FRAC * min(box[v] - box[u], other[v] - other[u])
            if _BODY_CORR_GAP_MIN * dim < gap < _BODY_CORR_GAP_MAX * dim and ov > required:
                return True
        return False

    return [(tuple(int(round(c)) for c in b), v) for b, v in candidates if _corroborated(b, v)]


def _find_tone_candidates(
    original_gray: np.ndarray,
    current_gray: np.ndarray,
    body: list[tuple[tuple[int, int, int, int], bool]],
) -> tuple[np.ndarray, list[tuple[int, int, int, int, int, int, int, int]], np.ndarray]:
    """Detect connected diagram components where continuous tone was lost."""
    height, width = original_gray.shape
    page_area = height * width
    interior = np.s_[
        int(_INTERIOR_Y0 * height):int(_INTERIOR_Y1 * height),
        int(_INTERIOR_X0 * width):int(_INTERIOR_X1 * width),
    ]
    paper = float(np.percentile(original_gray[interior], _PAPER_PERCENTILE)) if interior[0].stop > interior[0].start and interior[1].stop > interior[1].start else float(np.percentile(original_gray, _PAPER_PERCENTILE))
    source = original_gray.astype(np.float32)
    local = cv2.boxFilter(source, -1, (5, 5))
    std = np.sqrt(np.maximum(0.0, cv2.boxFilter(source * source, -1, (5, 5)) - local * local))
    final = cv2.boxFilter(current_gray.astype(np.float32), -1, (3, 3))

    mid = (local >= _MID_MIN_LOCAL) & (local <= paper - _MID_PAPER_MARGIN) & (std <= _MID_MAX_STD)
    loss = mid & (
        ((final >= paper - _LOSS_WHITE_MARGIN) & (final - local >= _LOSS_WHITE_DELTA))
        | ((final <= _LOSS_BLACK_MAX) & (local - final >= _LOSS_BLACK_DELTA))
    )

    core = cv2.GaussianBlur(source, (3, 3), 0.65) < (paper - _CORE_PAPER_MARGIN)
    core8 = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_CLOSE, _ellipse(1))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core8)

    chosen = np.zeros(original_gray.shape, np.uint8)
    rows: list[tuple[int, int, int, int, int, int, int, int]] = []
    min_dim = min(width, height)

    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if (
            area < max(_MIN_AREA_PX, page_area * _MIN_AREA_FRAC)
            or min(w, h) < _MIN_SIDE_PX
            or (w < width * _TINY_W_FRAC and h < height * _TINY_H_FRAC)
            or (w >= _FULLPAGE_W_FRAC * width and h >= _FULLPAGE_H_FRAC * height)
            or x < _BORDER_X0_FRAC * width
            or y < _BORDER_Y0_FRAC * height
            or x + w > _BORDER_X1_FRAC * width
            or y + h > _BORDER_Y1_FRAC * height
        ):
            continue

        patch = np.s_[y:y + h, x:x + w]
        label = labels[patch] == i
        core_loss = loss[patch] & label
        core_mid = mid[patch] & label
        changed = int(core_loss.sum())
        stable = int(core_mid.sum())
        if changed < max(_MIN_CHANGED_PX, page_area * _MIN_CHANGED_FRAC) or stable < max(_MIN_STABLE_PX, _MIN_STABLE_AREA_FRAC * area):
            continue

        strips = max(int(core_loss.sum(axis=0).max()), int(core_loss.sum(axis=1).max()))
        long_loss = strips >= min_dim * _LONG_LOSS_FRAC
        if changed < stable * _MIN_CHANGED_STABLE_FRAC and not long_loss:
            continue

        solid_core = cv2.erode(core_mid.astype(np.uint8), _ellipse(2))
        support = int(solid_core.sum())
        long_side = max(w / width, h / height)
        if support < max(_MIN_SUPPORT_PX, _MIN_SUPPORT_AREA_FRAC * area) or long_side < _MIN_LONG_SIDE_FRAC:
            continue

        bbox = (x, y, x + w, y + h)
        if (h > _TALL_ASPECT_RATIO * w and w < width * _TALL_MAX_W_FRAC) or (
            w > _WIDE_ASPECT_RATIO * h and h < height * _WIDE_MAX_H_FRAC
        ):
            continue
        if h > _RELEASE_TALL_ASPECT * w and w < width * _RELEASE_TALL_MAX_W_FRAC:
            continue
        if w > _BODY_ROW_ASPECT * h and h < height * _BODY_ROW_MAX_H_FRAC:
            if any(not vertical and _box_overlap_area(bbox, b) > _BODY_ROW_OVERLAP_FRAC * w * h for b, vertical in body):
                continue

        contours, _ = cv2.findContours(label.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        silhouette = cv2.drawContours(np.zeros(label.shape, np.uint8), contours, -1, 1, cv2.FILLED)
        if int(silhouette.sum()) > max(area * _SILHOUETTE_AREA_MULT, page_area * _SILHOUETTE_PAGE_FRAC) and (
            w * h > page_area * _SILHOUETTE_BBOX_FRAC
        ):
            silhouette = label.astype(np.uint8)

        chosen[patch] |= silhouette
        rows.append((x, y, w, h, area, stable, changed, strips))

    if not body and rows:
        if not any(w >= width * _NO_BODY_MIN_W_FRAC or h >= height * _NO_BODY_MIN_H_FRAC for _, _, w, h, *_ in rows):
            return np.zeros_like(chosen), [], loss

    return chosen, rows, loss


def _grow_diagrams(
    mask: np.ndarray,
    zones: list[tuple[int, int, int, int, int, int, int, int]],
    source_gray: np.ndarray,
    body: list[tuple[tuple[int, int, int, int], bool]],
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Group nearby diagram components bounded by OCR body lines and paper gaps."""
    h, w = source_gray.shape
    if len(zones) < 2:
        return mask, []

    rectangles = [(x, y, x + ww, y + hh) for x, y, ww, hh, *_ in zones]
    regions: list[tuple[int, int, int, int]] = []
    pending = list(rectangles)

    while pending:
        group = [pending.pop(0)]
        for other in group:
            ax, ay, bx, by = other
            reachable = []
            for box in pending:
                x, y, x1, y1 = box
                dx = max(ax - x1, x - bx, 0)
                dy = max(ay - y1, y - by, 0)
                if (dx / (_GROUP_REACH_X_FRAC * w)) ** 2 + (dy / (_GROUP_REACH_Y_FRAC * h)) ** 2 < 1.0:
                    reachable.append(box)
            for item in reachable:
                pending.remove(item)
                group.append(item)

        if len(group) < 2:
            continue

        x = min(b[0] for b in group)
        y = min(b[1] for b in group)
        x1 = max(b[2] for b in group)
        y1 = max(b[3] for b in group)
        zone_area = (x1 - x) * (y1 - y)
        greatest = max((b[2] - b[0]) * (b[3] - b[1]) for b in group)
        if (
            x1 - x < _GROUP_MIN_W_FRAC * w
            or y1 - y < _GROUP_MIN_H_FRAC * h
            or zone_area > _GROUP_MAX_AREA_FRAC * w * h
            or greatest > _GROUP_MAX_SINGLE_FRAC * zone_area
        ):
            continue

        xlow = max(round(_GROW_CLAMP_X0_FRAC * w), x - round(_GROW_PAD_X_FRAC * w))
        xhigh = min(round(_GROW_CLAMP_X1_FRAC * w), x1 + round(_GROW_PAD_X_FRAC * w))
        lower = max(round(_GROW_CLAMP_Y0_FRAC * h), y - round(_GROW_PAD_Y_FRAC * h))
        upper = min(round(_GROW_CLAMP_Y1_FRAC * h), y1 + round(_GROW_PAD_Y_FRAC * h))

        for clip, vertical in body:
            bx, by, bx1, by1 = clip
            if vertical:
                if by >= y1 or by1 <= y:
                    continue
                if bx1 <= x:
                    xlow = max(xlow, bx1 + 3)
                if bx >= x1:
                    xhigh = min(xhigh, bx - 3)
                continue
            if min(x1, bx1) - max(x, bx) < 0.23 * (x1 - x):
                continue
            if by1 <= y:
                lower = max(lower, by1 + 3)
            if by >= y1:
                upper = min(upper, by - 3)

        for other in rectangles:
            if other in group:
                continue
            bx, by, bx1, by1 = other
            if min(x1, bx1) - max(x, bx) <= 0.12 * (x1 - x):
                continue
            if by1 < y:
                lower = max(lower, (by1 + y) // 2)
            elif by > y1:
                upper = min(upper, (y1 + by) // 2)

        if lower > y or upper < y1 or xlow > x or xhigh < x1:
            continue
        proposed = (xlow, lower, xhigh, upper)
        if (xhigh - xlow) * (upper - lower) > _GROW_MAX_EXPANDED_FRAC * w * h:
            continue
        if any(_box_overlap_area(proposed, b) > _GROW_BODY_OVERLAP_FRAC * (b[2] - b[0]) * (b[3] - b[1]) for b, _ in body):
            continue

        paper = float(np.percentile(source_gray[round(h * 0.08):round(h * 0.92), round(w * 0.08):round(w * 0.92)], _PAPER_PERCENTILE))
        dark = source_gray < (paper - _INK_PAPER_MARGIN)
        pr = dark[lower:upper, xlow:xhigh].sum(axis=1)
        ink = np.flatnonzero(pr >= max(2, round(_INK_DENSITY_FRAC * (xhigh - xlow))))
        if not len(ink):
            continue
        low = max(lower, min(y - 2, lower + int(ink[0]) - 2))
        high = min(upper, max(y1 + 2, lower + int(ink[-1]) + 3))
        safe_rows = pr <= _SAFE_ROW_FRAC * (xhigh - xlow)
        first = np.flatnonzero(safe_rows[low - lower:y - lower])
        last = np.flatnonzero(safe_rows[y1 - lower:high - lower])
        if not len(first) or not len(last):
            continue
        low += int(first[0])
        high = y1 + int(last[-1]) + 1
        cols = dark[low:high, xlow:xhigh].sum(axis=0)
        active = np.flatnonzero(cols >= max(2, round(_INK_DENSITY_FRAC * (high - low))))
        if not len(active):
            continue
        left = max(xlow, min(x - 2, xlow + int(active[0]) - 2))
        right = min(xhigh, max(x1 + 2, xlow + int(active[-1]) + 3))
        if pr[low - lower] > _CUT_ROW_MAX_FRAC * (xhigh - xlow) or pr[high - lower - 1] > _CUT_ROW_MAX_FRAC * (xhigh - xlow):
            continue

        mask[low:high, left:right] = 1
        regions.append((left, low, right, high))

    return mask, regions


def _build_soft_alpha(mask: np.ndarray) -> np.ndarray:
    """Create a feathered [0, 1] alpha mask in analysis resolution."""
    fixed = cv2.dilate(mask, _ellipse(1))
    alpha = cv2.GaussianBlur(cv2.dilate(fixed, _ellipse(1)).astype(np.float32), (5, 5), 0.9)
    alpha[fixed != 0] = 1.0
    return np.clip(alpha, 0.0, 1.0)


def filter_overlapping_fills(
    alpha_mask: np.ndarray | None,
    vecfills: list[dict],
    tint_zones: list,
) -> tuple[list[dict], list]:
    """Drop vector fills and tint zones that overlap with the restored tone mask."""
    if alpha_mask is None or not np.any(alpha_mask > 0):
        return vecfills, tint_zones

    h, w = alpha_mask.shape[:2]
    active = alpha_mask > 0

    kept_fills: list[dict] = []
    for fill in vecfills:
        x0, y0, x1, y1 = fill["rect"]
        x0i, y0i = max(0, int(x0)), max(0, int(y0))
        x1i, y1i = min(w, int(x1)), min(h, int(y1))
        if x1i > x0i and y1i > y0i and np.any(active[y0i:y1i, x0i:x1i]):
            continue
        kept_fills.append(fill)

    kept_zones: list = []
    for zone in tint_zones:
        x0, y0, x1, y1 = zone.rect
        x0i, y0i = max(0, int(x0)), max(0, int(y0))
        x1i, y1i = min(w, int(x1)), min(h, int(y1))
        if x1i > x0i and y1i > y0i and np.any(active[y0i:y1i, x0i:x1i]):
            continue
        kept_zones.append(zone)

    return kept_fills, kept_zones


def detect_tone_restore(
    out_bgr: np.ndarray,
    current_bgr: np.ndarray,
    lines: list[dict],
    target_dpi: int,
    *,
    is_gray_or_color: bool = False,
    is_blank: bool = False,
    is_tinted: bool = False,
) -> ToneRestoreResult:
    """Detect diagram regions needing tone restoration and verify improvement."""
    if is_gray_or_color:
        return ToneRestoreResult(reason="manual_gray_or_color")
    if is_blank:
        return ToneRestoreResult(reason="blank_page")
    if is_tinted:
        return ToneRestoreResult(reason="tinted_page")

    h, w = out_bgr.shape[:2]
    if h < 64 or w < 64:
        return ToneRestoreResult(reason="too_small")

    scale = min(1.0, _ANALYSIS_DPI / float(max(1, target_dpi)))
    aw = max(1, int(round(w * scale)))
    ah = max(1, int(round(h * scale)))
    if aw < 150 or ah < 150:
        return ToneRestoreResult(reason="too_small")

    src_gray_full = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY) if out_bgr.ndim == 3 else out_bgr
    cur_gray_full = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY) if current_bgr.ndim == 3 else current_bgr

    if aw != w or ah != h:
        src_gray = cv2.resize(src_gray_full, (aw, ah), interpolation=cv2.INTER_AREA)
        cur_gray = cv2.resize(cur_gray_full, (aw, ah), interpolation=cv2.INTER_AREA)
    else:
        src_gray = src_gray_full
        cur_gray = cur_gray_full

    body = extract_body_lines(lines, aw / float(w), ah / float(h), aw, ah)
    mask, zones, loss = _find_tone_candidates(src_gray, cur_gray, body)
    if not zones:
        return ToneRestoreResult(loss_before=int(loss.sum()), loss_after=int(loss.sum()), reason="no_candidates")

    mask, expanded = _grow_diagrams(mask, zones, src_gray, body)
    alpha_small = _build_soft_alpha(mask)

    loss_before = int(loss.sum())
    loss_after = int((loss & (alpha_small < 1.0)).sum())
    if loss_after >= loss_before:
        return ToneRestoreResult(
            zones=[(z[0], z[1], z[2], z[3]) for z in zones],
            expanded_regions=expanded,
            loss_before=loss_before,
            loss_after=loss_after,
            adopted=False,
            reason="no_loss_reduction",
        )

    if aw != w or ah != h:
        alpha_full = cv2.resize(alpha_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        alpha_full = alpha_small
    alpha_full = np.clip(alpha_full, 0.0, 1.0).astype(np.float32)
    alpha_full[alpha_full < (1.0 / 255.0)] = 0.0

    if not np.any(alpha_full > 0):
        return ToneRestoreResult(reason="empty_alpha")

    # Machine verification at full resolution: outside the active mask, the
    # composited output must be pixel-identical to current_bgr.
    active = alpha_full > 0
    cand_bgr = out_bgr.copy()
    if cand_bgr.ndim == 2:
        cand_bgr = cv2.cvtColor(cand_bgr, cv2.COLOR_GRAY2BGR)
    # Grayscale restoration preserves neutral tone; color pages stay RGB if needed.
    from .mrc import _is_grayish

    if _is_grayish(cand_bgr):
        g = cv2.cvtColor(cand_bgr, cv2.COLOR_BGR2GRAY)
        cand_bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    a3 = alpha_full[..., None]
    verified_bgr = np.rint(
        current_bgr.astype(np.float32) * (1.0 - a3) + cand_bgr.astype(np.float32) * a3
    ).astype(np.uint8)
    outside_diff = int(np.any(verified_bgr != current_bgr, axis=-1)[~active].sum())
    if outside_diff != 0:
        return ToneRestoreResult(
            loss_before=loss_before,
            loss_after=loss_after,
            adopted=False,
            reason=f"outside_diff_{outside_diff}",
        )

    return ToneRestoreResult(
        alpha_mask=alpha_full,
        zones=[(z[0], z[1], z[2], z[3]) for z in zones],
        expanded_regions=expanded,
        loss_before=loss_before,
        loss_after=loss_after,
        adopted=True,
        reason="adopted",
    )
