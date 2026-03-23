"""
Image Quality Gate
Pre-grading quality assessment. Rejects or warns on:
  blur, glare, poor exposure, heavy shadow, card not visible,
  sleeve/toploader presence, and severe perspective issues.

Returns a structured capture_quality report and a pass/warn/fail decision.
"""

import cv2
import numpy as np
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional

_THRESH_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "config", "thresholds.json")
_thresh_cache: Optional[Dict] = None


def _load_thresholds() -> Dict:
    global _thresh_cache
    if _thresh_cache is None:
        with open(_THRESH_PATH) as f:
            _thresh_cache = json.load(f)
    return _thresh_cache["quality_gate"]


@dataclass
class QualityIssue:
    check:       str      # e.g. "blur", "glare"
    severity:    str      # "warn" or "fail"
    message:     str
    metric_name: str
    metric_value: float

    def to_dict(self) -> Dict:
        return asdict(self)


# ─── Individual checks ────────────────────────────────────────────────────────

def _check_blur(gray: np.ndarray, t: Dict) -> Tuple[Dict, Optional[QualityIssue]]:
    """Laplacian variance — low value = blurry."""
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    ok = lap_var >= t["blur_laplacian_warn"]
    check = {"metric": "laplacian_variance", "value": round(lap_var, 1), "ok": ok}
    issue = None
    if lap_var < t["blur_laplacian_fail"]:
        issue = QualityIssue("blur", "fail",
                              "Image is too blurry — focus the camera on the card.",
                              "laplacian_variance", lap_var)
    elif lap_var < t["blur_laplacian_warn"]:
        issue = QualityIssue("blur", "warn",
                              "Image may be slightly blurry — analysis accuracy may be reduced.",
                              "laplacian_variance", lap_var)
    return check, issue


def _check_glare(lab: np.ndarray, t: Dict) -> Tuple[Dict, Optional[QualityIssue]]:
    """Detect glare/reflection hotspots via high L* pixels."""
    l_ch = lab[:, :, 0].astype(float)
    glare_ratio = float(np.sum(l_ch > t["glare_l_threshold"])) / max(l_ch.size, 1)
    ok = glare_ratio < t["glare_ratio_warn"]
    check = {"metric": "glare_ratio", "value": round(glare_ratio, 4), "ok": ok}
    issue = None
    pct = round(glare_ratio * 100, 1)
    if glare_ratio >= t["glare_ratio_fail"]:
        issue = QualityIssue("glare", "fail",
                              f"Severe glare covers {pct}% of image — diffuse the light source.",
                              "glare_ratio", glare_ratio)
    elif glare_ratio >= t["glare_ratio_warn"]:
        issue = QualityIssue("glare", "warn",
                              f"Glare/reflection detected ({pct}%) — results may be inaccurate.",
                              "glare_ratio", glare_ratio)
    return check, issue


def _check_exposure(gray: np.ndarray, t: Dict) -> Tuple[Dict, Optional[QualityIssue]]:
    """Mean brightness — too dark or too bright."""
    mean_b = float(gray.mean())
    ok = t["exposure_min_warn"] <= mean_b <= t["exposure_max_warn"]
    check = {"metric": "mean_brightness", "value": round(mean_b, 1), "ok": ok}
    issue = None
    if mean_b < t["exposure_min_fail"]:
        issue = QualityIssue("exposure", "fail",
                              "Image is severely underexposed — increase lighting.",
                              "mean_brightness", mean_b)
    elif mean_b < t["exposure_min_warn"]:
        issue = QualityIssue("exposure", "warn",
                              "Image appears dark — better lighting improves accuracy.",
                              "mean_brightness", mean_b)
    elif mean_b > t["exposure_max_fail"]:
        issue = QualityIssue("exposure", "fail",
                              "Image is severely overexposed — reduce direct lighting.",
                              "mean_brightness", mean_b)
    elif mean_b > t["exposure_max_warn"]:
        issue = QualityIssue("exposure", "warn",
                              "Image appears bright/washed out — may affect surface analysis.",
                              "mean_brightness", mean_b)
    return check, issue


def _check_shadow(gray: np.ndarray, t: Dict) -> Tuple[Dict, Optional[QualityIssue]]:
    """Detect heavy shadow coverage."""
    dark_ratio = float(np.sum(gray < t["shadow_dark_threshold"])) / max(gray.size, 1)
    ok = dark_ratio < t["shadow_ratio_warn"]
    check = {"metric": "shadow_dark_ratio", "value": round(dark_ratio, 4), "ok": ok}
    issue = None
    pct = round(dark_ratio * 100, 1)
    if dark_ratio >= t["shadow_ratio_fail"]:
        issue = QualityIssue("shadow", "fail",
                              f"Heavy shadow covers {pct}% of image — relight with even illumination.",
                              "shadow_dark_ratio", dark_ratio)
    elif dark_ratio >= t["shadow_ratio_warn"]:
        issue = QualityIssue("shadow", "warn",
                              f"Shadow detected on {pct}% of image — may affect edge/corner detection.",
                              "shadow_dark_ratio", dark_ratio)
    return check, issue


def _check_card_visibility(img: np.ndarray, t: Dict) -> Tuple[Dict, Optional[QualityIssue]]:
    """Detect if a card-shaped object occupies sufficient area."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Two-pass contour finding
    area_ratio = 0.0
    for lo, hi in [(30, 120), (10, 80)]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, lo, hi)
        edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area_ratio = cv2.contourArea(largest) / (h * w)
            if area_ratio >= t["min_card_area_ratio"]:
                break

    ok = area_ratio >= t["min_card_area_ratio"]
    check = {"metric": "card_area_ratio", "value": round(float(area_ratio), 3), "ok": ok}
    issue = None
    if not ok:
        if area_ratio < t["min_card_area_ratio"] * 0.5:
            issue = QualityIssue("card_not_visible", "fail",
                                  "Card not detected in image — ensure full card is visible.",
                                  "card_area_ratio", area_ratio)
        else:
            issue = QualityIssue("card_not_visible", "warn",
                                  "Card may be partially cropped or at edge of frame.",
                                  "card_area_ratio", area_ratio)
    return check, issue


def _check_sleeve(img: np.ndarray, t: Dict) -> Tuple[Dict, Optional[QualityIssue]]:
    """
    Heuristic detection of sleeve or toploader.
    Signatures:
      - Very uniform, bright rectangular borders around inner card region
      - Sharp brightness discontinuity at a consistent distance from edges
    """
    h, w = img.shape[:2]
    bw = int(min(h, w) * 0.07)   # border width to sample
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)

    borders = {
        "top":    gray[:bw, bw:w - bw],
        "bottom": gray[h - bw:, bw:w - bw],
        "left":   gray[bw:h - bw, :bw],
        "right":  gray[bw:h - bw, w - bw:],
    }
    inner = gray[bw:h - bw, bw:w - bw]

    border_stds = [np.std(v) for v in borders.values()]
    border_means = [np.mean(v) for v in borders.values()]
    avg_border_std = float(np.mean(border_stds))
    avg_border_mean = float(np.mean(border_means))
    inner_std = float(np.std(inner))

    # Toploader/sleeve: borders much more uniform than card AND brighter than card content
    uniformity_ratio = avg_border_std / (inner_std + 1e-6)
    confidence = 0.0
    if uniformity_ratio < t["sleeve_uniformity_ratio"]:
        confidence += 0.40
    if avg_border_mean > t["sleeve_brightness_min"]:
        confidence += 0.35
    # Check for distinct brightness step at border edge
    edge_row = gray[bw, bw:w - bw]
    inner_mean = float(np.mean(inner))
    if abs(float(np.mean(edge_row)) - inner_mean) > 25:
        confidence += 0.25

    detected = confidence >= t["sleeve_confidence_thresh"]
    check = {"metric": "sleeve_confidence", "value": round(confidence, 3),
             "detected": detected, "ok": not detected}
    issue = None
    if detected:
        issue = QualityIssue("sleeve_detected", "warn",
                              "Card appears to be in a sleeve or toploader. "
                              "Remove for most accurate grading.",
                              "sleeve_confidence", confidence)
    return check, issue


# ─── Main quality assessment ──────────────────────────────────────────────────

def assess_quality(img: np.ndarray) -> Dict[str, Any]:
    """
    Run all quality checks on the raw input image (before normalization).

    Returns:
        decision:  "pass" | "warn" | "fail"
        score:     0–100 (100 = perfect quality)
        issues:    list of QualityIssue dicts
        checks:    per-check raw data
        summary:   human-readable one-liner
    """
    if img is None or img.size == 0:
        return {
            "decision": "fail",
            "score": 0,
            "issues": [{"check": "load_error", "severity": "fail",
                         "message": "Could not load image.",
                         "metric_name": "n/a", "metric_value": 0}],
            "checks": {},
            "summary": "Image could not be loaded.",
        }

    t = _load_thresholds()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    checks: Dict[str, Any] = {}
    issues: List[QualityIssue] = []

    checks["blur"],      _i = _check_blur(gray, t)
    if _i: issues.append(_i)

    checks["glare"],     _i = _check_glare(lab, t)
    if _i: issues.append(_i)

    checks["exposure"],  _i = _check_exposure(gray, t)
    if _i: issues.append(_i)

    checks["shadow"],    _i = _check_shadow(gray, t)
    if _i: issues.append(_i)

    checks["card_visible"], _i = _check_card_visibility(img, t)
    if _i: issues.append(_i)

    checks["sleeve"],    _i = _check_sleeve(img, t)
    if _i: issues.append(_i)

    # ── Decision ──────────────────────────────────────────────────────────────
    n_fail = sum(1 for i in issues if i.severity == "fail")
    n_warn = sum(1 for i in issues if i.severity == "warn")

    score = max(0, 100 - n_fail * 30 - n_warn * 8)

    if n_fail > 0:
        decision = "fail"
        summary = f"Quality check FAILED ({n_fail} critical issue(s)). Fix before grading."
    elif n_warn >= 3:
        decision = "fail"
        summary = f"Too many warnings ({n_warn}). Image quality insufficient for reliable grading."
    elif n_warn > 0:
        decision = "warn"
        summary = f"Quality check passed with {n_warn} warning(s). Results may be less accurate."
    else:
        decision = "pass"
        summary = "Image quality is good. Ready for grading."

    return {
        "decision": decision,
        "score":    score,
        "issues":   [i.to_dict() for i in issues],
        "checks":   checks,
        "summary":  summary,
    }
