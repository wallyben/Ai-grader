"""
Centering Analysis Engine
Detects inner border edges and computes left/right + top/bottom ratios
with PSA-standard scoring.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple


def _find_dominant_line(projection: np.ndarray, search_from: str = "start") -> Optional[int]:
    """
    Find the most prominent edge line in a 1D projection.

    Args:
        projection: Sum of edge magnitudes along one axis
        search_from: 'start' = find first strong line, 'end' = find last
    """
    if projection.max() == 0:
        return None

    # Smooth projection to reduce noise
    smooth = np.convolve(projection.astype(float),
                         np.ones(5) / 5, mode='same')
    threshold = smooth.max() * 0.25
    strong = np.where(smooth > threshold)[0]

    if len(strong) == 0:
        return None

    return int(strong[0]) if search_from == "start" else int(strong[-1])


def compute_centering(img: np.ndarray) -> Dict[str, Any]:
    """
    Detect inner printed border and compute centering ratios.

    Strategy:
    - Look in top/bottom 40% zones for horizontal border lines
    - Look in left/right 40% zones for vertical border lines
    - Compute margin ratios relative to card face

    Returns dict with:
        left_pct, right_pct, top_pct, bottom_pct: margin percentages
        lr_ratio, tb_ratio: human-readable strings (e.g. "57/43")
        score: 1.0–10.0
        borders: (top_b, bottom_b, left_b, right_b) in pixels
    """
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Multi-threshold edge map for robust border detection
    edges_low = cv2.Canny(blurred, 15, 60)
    edges_mid = cv2.Canny(blurred, 30, 100)
    edges = cv2.bitwise_or(edges_low, edges_mid)

    # Outer margin to skip card boundary noise (5%)
    outer = int(min(h, w) * 0.05)

    # ── TOP BORDER ──────────────────────────────────────────────────────────
    # Search in top 5%–40% of card
    top_start = outer
    top_end = int(h * 0.40)
    zone_top = edges[top_start:top_end, outer:w - outer]
    h_proj_top = np.sum(zone_top.astype(np.int32), axis=1)
    rel_top = _find_dominant_line(h_proj_top, "end")  # last strong line = inner border
    top_border = (rel_top + top_start) if rel_top is not None else int(h * 0.08)

    # ── BOTTOM BORDER ────────────────────────────────────────────────────────
    bot_start = int(h * 0.60)
    bot_end = h - outer
    zone_bot = edges[bot_start:bot_end, outer:w - outer]
    h_proj_bot = np.sum(zone_bot.astype(np.int32), axis=1)
    rel_bot = _find_dominant_line(h_proj_bot, "start")  # first strong line = inner border
    bottom_border = (rel_bot + bot_start) if rel_bot is not None else int(h * 0.92)

    # ── LEFT BORDER ──────────────────────────────────────────────────────────
    left_start = outer
    left_end = int(w * 0.40)
    zone_left = edges[outer:h - outer, left_start:left_end]
    v_proj_left = np.sum(zone_left.astype(np.int32), axis=0)
    rel_left = _find_dominant_line(v_proj_left, "end")
    left_border = (rel_left + left_start) if rel_left is not None else int(w * 0.08)

    # ── RIGHT BORDER ─────────────────────────────────────────────────────────
    right_start = int(w * 0.60)
    right_end = w - outer
    zone_right = edges[outer:h - outer, right_start:right_end]
    v_proj_right = np.sum(zone_right.astype(np.int32), axis=0)
    rel_right = _find_dominant_line(v_proj_right, "start")
    right_border = (rel_right + right_start) if rel_right is not None else int(w * 0.92)

    # ── SAFETY CLAMPS ────────────────────────────────────────────────────────
    top_border = max(outer, min(top_border, int(h * 0.45)))
    bottom_border = max(int(h * 0.55), min(bottom_border, h - outer))
    left_border = max(outer, min(left_border, int(w * 0.45)))
    right_border = max(int(w * 0.55), min(right_border, w - outer))

    # ── MARGIN CALCULATIONS ──────────────────────────────────────────────────
    left_margin = left_border
    right_margin = w - right_border
    top_margin = top_border
    bottom_margin = h - bottom_border

    total_h = left_margin + right_margin
    total_v = top_margin + bottom_margin

    if total_h < 4 or total_v < 4:
        # Degenerate case - assume perfect centering
        return _build_result(50.0, 50.0, 50.0, 50.0,
                              (top_border, bottom_border, left_border, right_border))

    left_pct = (left_margin / total_h) * 100.0
    right_pct = (right_margin / total_h) * 100.0
    top_pct = (top_margin / total_v) * 100.0
    bottom_pct = (bottom_margin / total_v) * 100.0

    return _build_result(left_pct, right_pct, top_pct, bottom_pct,
                          (top_border, bottom_border, left_border, right_border))


def _build_result(left_pct: float, right_pct: float,
                  top_pct: float, bottom_pct: float,
                  borders: Tuple) -> Dict[str, Any]:
    """Assemble centering result dict with score."""
    score = _score_centering(left_pct, right_pct, top_pct, bottom_pct)
    return {
        "left_pct": round(left_pct, 1),
        "right_pct": round(right_pct, 1),
        "top_pct": round(top_pct, 1),
        "bottom_pct": round(bottom_pct, 1),
        "lr_ratio": f"{round(left_pct)}/{round(right_pct)}",
        "tb_ratio": f"{round(top_pct)}/{round(bottom_pct)}",
        "score": score,
        "borders": borders,
    }


def _score_centering(left: float, right: float,
                     top: float, bottom: float) -> float:
    """
    Score centering using PSA-equivalent standards.

    PSA 10  = 55/45 or better on both axes
    PSA 9   = 60/40 or better
    PSA 8   = 65/35 or better
    PSA 7   = 70/30 or better
    PSA 6   = 75/25 or better
    PSA ≤5  = worse than 75/25
    """
    lr_dev = abs(left - right)    # 0 = perfect, 100 = extreme
    tb_dev = abs(top - bottom)
    max_dev = max(lr_dev, tb_dev)

    if max_dev <= 5.0:
        return 10.0
    elif max_dev <= 10.0:
        return 9.5
    elif max_dev <= 15.0:
        return 9.0
    elif max_dev <= 20.0:
        return 8.5
    elif max_dev <= 25.0:
        return 8.0
    elif max_dev <= 30.0:
        return 7.0
    elif max_dev <= 40.0:
        return 6.0
    elif max_dev <= 50.0:
        return 5.0
    elif max_dev <= 60.0:
        return 4.0
    elif max_dev <= 70.0:
        return 3.0
    else:
        return 2.0
