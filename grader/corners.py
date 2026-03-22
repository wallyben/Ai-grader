"""
Corner Analysis Engine
Extracts corner regions and measures:
  - Whitening / bleaching
  - Rounding / blunting
  - Sharpness via gradient magnitude
Uses PSA-equivalent corner grading logic.
"""

import cv2
import numpy as np
from typing import Dict, Any

CORNER_REGION_SIZE = 65   # px square extracted from each corner
CORNER_APEX_ZONE = 22     # px inner zone around the actual corner tip


def _extract_corner(img: np.ndarray, position: str) -> np.ndarray:
    """Extract a square region from the specified corner."""
    h, w = img.shape[:2]
    s = CORNER_REGION_SIZE
    regions = {
        "top_left":     img[:s, :s],
        "top_right":    img[:s, w - s:],
        "bottom_left":  img[h - s:, :s],
        "bottom_right": img[h - s:, w - s:],
    }
    return regions[position]


def _apex_zone(region: np.ndarray, position: str) -> np.ndarray:
    """Extract the small zone right at the corner tip."""
    h, w = region.shape[:2]
    z = CORNER_APEX_ZONE
    zones = {
        "top_left":     region[:z, :z],
        "top_right":    region[:z, w - z:],
        "bottom_left":  region[h - z:, :z],
        "bottom_right": region[h - z:, w - z:],
    }
    return zones[position]


def _measure_whitening(region: np.ndarray, position: str) -> float:
    """
    Detect whitening/bleaching at corner using LAB colorspace.
    Focus on apex zone for corner-specific analysis.
    """
    apex = _apex_zone(region, position)
    lab = cv2.cvtColor(apex, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0].astype(float)
    a_ch = lab[:, :, 1].astype(float) - 128.0
    b_ch = lab[:, :, 2].astype(float) - 128.0

    # High L* + low saturation = whitening
    bright_ratio = float(np.sum(l_ch > 195)) / max(l_ch.size, 1)
    saturation = np.sqrt(a_ch ** 2 + b_ch ** 2)
    desaturated = float(np.sum(saturation < 12)) / max(saturation.size, 1)

    whitening = (bright_ratio * 0.65) + (desaturated * 0.20)
    return float(min(1.0, whitening))


def _measure_rounding(region: np.ndarray, position: str) -> float:
    """
    Detect corner rounding by analyzing how much the actual corner
    deviates from a perfect right-angle.
    Uses the apex zone background ratio as a proxy.
    """
    apex = _apex_zone(region, position)
    gray = cv2.cvtColor(apex, cv2.COLOR_BGR2GRAY)

    # Binarize: background is usually bright (white/near-white card back)
    # or dark depending on card color, so use OTSU
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # For corner analysis we check if the extreme corner pixel cluster
    # is background (= worn away) or foreground (= intact card material)
    h, w = binary.shape
    z = max(h // 3, 4)

    corner_zones = {
        "top_left":     binary[:z, :z],
        "top_right":    binary[:z, w - z:],
        "bottom_left":  binary[h - z:, :z],
        "bottom_right": binary[h - z:, w - z:],
    }
    apex_tiny = corner_zones[position]
    total = apex_tiny.size
    if total == 0:
        return 0.0

    # Count pixels that look like background (extreme brightness or darkness)
    # A worn corner will have more background showing through
    mean_val = apex_tiny.mean()
    card_likely = 30 < mean_val < 225   # not pure black or pure white = card material
    if not card_likely:
        missing_ratio = float(np.sum(apex_tiny > 230) + np.sum(apex_tiny < 25)) / total
    else:
        missing_ratio = 0.0

    return float(min(1.0, missing_ratio * 1.2))


def _measure_sharpness(region: np.ndarray, position: str) -> float:
    """
    Measure corner sharpness via gradient magnitude in the apex zone.
    A sharp corner has high gradient at the corner point.
    A worn/rounded corner has diffuse, low gradients.
    Returns gradient magnitude (higher = sharper).
    """
    apex = _apex_zone(region, position)
    gray = cv2.cvtColor(apex, cv2.COLOR_BGR2GRAY).astype(float)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    return float(np.mean(magnitude))


def _classify_corner(whitening: float, rounding: float) -> str:
    combined = (whitening * 0.55) + (rounding * 0.45)
    if combined > 0.22 or whitening > 0.30:
        return "severe"
    elif combined > 0.11 or whitening > 0.15:
        return "moderate"
    elif combined > 0.04 or whitening > 0.06:
        return "minor"
    return "sharp"


def _score_corner(whitening: float, rounding: float, sharpness: float) -> float:
    """Score a single corner 1.0–10.0."""
    penalty = (whitening * 45.0) + (rounding * 30.0)
    # Slight sharpness bonus up to 1.0 point for very sharp corners
    sharpness_bonus = min(1.0, sharpness / 40.0)
    score = 10.0 - penalty + (sharpness_bonus * 0.3)
    return float(max(1.0, min(10.0, round(score, 1))))


def analyze_corners(img: np.ndarray) -> Dict[str, Any]:
    """
    Analyze all 4 corners for wear, whitening and rounding.

    Returns:
        score: overall corner score 1–10
        corners: per-corner breakdown
        defect_corners: list of corner names with defects
    """
    positions = ["top_left", "top_right", "bottom_left", "bottom_right"]
    results = {}
    defect_corners = []

    for pos in positions:
        region = _extract_corner(img, pos)
        whitening = _measure_whitening(region, pos)
        rounding = _measure_rounding(region, pos)
        sharpness = _measure_sharpness(region, pos)
        level = _classify_corner(whitening, rounding)
        corner_score = _score_corner(whitening, rounding, sharpness)

        results[pos] = {
            "whitening": round(whitening, 4),
            "rounding": round(rounding, 4),
            "sharpness": round(sharpness, 2),
            "defect_level": level,
            "score": corner_score,
        }

        if level != "sharp":
            defect_corners.append(pos)

    # Overall: average weighted with worst corner pull-down
    avg_score = sum(r["score"] for r in results.values()) / len(positions)
    worst_score = min(r["score"] for r in results.values())
    # Worst corner counts 40% — a single destroyed corner tanks the grade
    overall = round((avg_score * 0.60) + (worst_score * 0.40), 1)

    return {
        "score": float(max(1.0, min(10.0, overall))),
        "corners": results,
        "defect_corners": defect_corners,
    }
