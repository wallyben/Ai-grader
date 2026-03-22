"""
Edge Detection & Analysis Engine
Scans all 4 card edges for whitening, chipping, and roughness.
Produces per-side defect classification and overall score.
"""

import cv2
import numpy as np
from typing import Dict, Any

EDGE_SCAN_WIDTH = 18   # px strip to analyze from each edge
EDGE_INNER_PAD = 5     # px skip at corners to avoid corner overlap


def _extract_strip(img: np.ndarray, side: str) -> np.ndarray:
    """Extract the edge strip for a given side."""
    h, w = img.shape[:2]
    p = EDGE_INNER_PAD
    s = EDGE_SCAN_WIDTH

    strips = {
        "top":    img[:s,    p:w - p],
        "bottom": img[h - s:, p:w - p],
        "left":   img[p:h - p, :s],
        "right":  img[p:h - p, w - s:],
    }
    return strips[side]


def _whitening_score(strip: np.ndarray) -> float:
    """
    Measure whitening intensity.
    Converts to LAB and measures proportion of pixels with high L* (brightness).
    Returns value in [0, 1].
    """
    lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0].astype(float)

    # L* in OpenCV is 0–255. >200 = very light/white
    white_ratio = float(np.sum(l_ch > 200)) / max(l_ch.size, 1)

    # Secondary: high saturation loss (desaturation = white/grey chipping)
    a_ch = lab[:, :, 1].astype(float) - 128.0
    b_ch = lab[:, :, 2].astype(float) - 128.0
    saturation = np.sqrt(a_ch ** 2 + b_ch ** 2)
    desaturated_ratio = float(np.sum(saturation < 15)) / max(saturation.size, 1)

    return min(1.0, (white_ratio * 0.6) + (desaturated_ratio * 0.15))


def _roughness_score(strip: np.ndarray, side: str) -> float:
    """
    Measure edge roughness via edge line irregularity.
    A clean cut edge has a smooth, consistent edge line.
    Chipping/roughness shows as variance in that line.
    Returns value in [0, 1].
    """
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)

    # Find edge pixels via Canny
    edges = cv2.Canny(gray, 20, 80)

    # Project to get the edge line profile
    if side in ("top", "bottom"):
        # Look at which rows have edge pixels along each column
        profile = []
        for col in range(edges.shape[1]):
            col_data = edges[:, col]
            nz = np.nonzero(col_data)[0]
            profile.append(float(nz[0]) if len(nz) > 0 else 0.0)
    else:
        # Look at which columns have edge pixels along each row
        profile = []
        for row in range(edges.shape[0]):
            row_data = edges[row, :]
            nz = np.nonzero(row_data)[0]
            profile.append(float(nz[0]) if len(nz) > 0 else 0.0)

    profile = np.array(profile, dtype=float)
    if len(profile) < 4 or np.std(profile) == 0:
        return 0.0

    # Roughness = coefficient of variation of edge positions
    roughness = np.std(profile) / (np.mean(np.abs(profile)) + 1.0)
    return float(min(1.0, roughness / 8.0))


def _chipping_score(strip: np.ndarray, side: str) -> float:
    """
    Detect chipping — sudden bright spots along the edge,
    indicating card stock exposed/missing at a specific point.
    """
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)

    if side in ("top", "bottom"):
        # Look at the very edge row
        edge_line = gray[0, :] if side == "top" else gray[-1, :]
    else:
        edge_line = gray[:, 0] if side == "left" else gray[:, -1]

    # Chip = very bright isolated pixels on the edge line
    bright = edge_line > 210
    chip_ratio = float(np.sum(bright)) / max(len(edge_line), 1)

    # Also look for isolated bright clusters (not uniform = chip not just color)
    if chip_ratio > 0:
        # Check if it's clustered (chips) vs uniform (white border on card design)
        kernel = np.ones(5)
        smoothed = np.convolve(bright.astype(float), kernel / 5.0, mode="same")
        local_max = np.max(smoothed)
        if local_max > 0.6:   # dense cluster = likely a chip
            return min(1.0, chip_ratio * 1.5)

    return min(1.0, chip_ratio * 0.5)


def _classify_defect(whitening: float, roughness: float, chipping: float) -> str:
    """Map defect metrics to categorical level."""
    combined = (whitening * 0.45) + (roughness * 0.30) + (chipping * 0.25)
    if combined > 0.18 or whitening > 0.25 or chipping > 0.30:
        return "severe"
    elif combined > 0.08 or whitening > 0.12 or chipping > 0.15:
        return "moderate"
    elif combined > 0.03 or whitening > 0.05 or roughness > 0.25:
        return "minor"
    return "clean"


def _score_side(whitening: float, roughness: float, chipping: float) -> float:
    """Score a single edge 1.0–10.0."""
    penalty = (whitening * 35.0) + (roughness * 20.0) + (chipping * 30.0)
    return float(max(1.0, min(10.0, round(10.0 - penalty, 1))))


def analyze_edges(img: np.ndarray) -> Dict[str, Any]:
    """
    Full edge analysis across all 4 sides.

    Returns:
        score: overall edge score 1–10
        sides: per-side breakdown
        avg_whitening, avg_roughness: aggregate metrics
        defect_locations: list of sides with defects
    """
    sides = ["top", "bottom", "left", "right"]
    results = {}
    totals = {"whitening": 0.0, "roughness": 0.0, "chipping": 0.0}
    defect_locations = []

    for side in sides:
        strip = _extract_strip(img, side)
        whitening = _whitening_score(strip)
        roughness = _roughness_score(strip, side)
        chipping = _chipping_score(strip, side)
        level = _classify_defect(whitening, roughness, chipping)
        side_score = _score_side(whitening, roughness, chipping)

        results[side] = {
            "whitening": round(whitening, 4),
            "roughness": round(roughness, 4),
            "chipping": round(chipping, 4),
            "defect_level": level,
            "score": side_score,
        }

        totals["whitening"] += whitening
        totals["roughness"] += roughness
        totals["chipping"] += chipping

        if level != "clean":
            defect_locations.append(side)

    n = len(sides)
    avg_whitening = totals["whitening"] / n
    avg_roughness = totals["roughness"] / n
    avg_chipping = totals["chipping"] / n

    # Overall score: weighted average + worst-side penalty
    avg_score = sum(r["score"] for r in results.values()) / n
    worst_score = min(r["score"] for r in results.values())
    overall_score = round((avg_score * 0.65) + (worst_score * 0.35), 1)

    return {
        "score": float(max(1.0, min(10.0, overall_score))),
        "sides": results,
        "avg_whitening": round(avg_whitening, 4),
        "avg_roughness": round(avg_roughness, 4),
        "avg_chipping": round(avg_chipping, 4),
        "defect_locations": defect_locations,
    }
