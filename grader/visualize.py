"""
Visualization Engine
Generates annotated overlays and summary cards for the Streamlit UI.
All drawing is done in BGR (OpenCV native) — convert before display with bgr_to_rgb().
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple

# Colour palette (BGR)
C = {
    "green":    (50, 220, 80),
    "yellow":   (0, 220, 220),
    "orange":   (0, 145, 255),
    "red":      (0, 60, 220),
    "cyan":     (220, 220, 0),
    "white":    (255, 255, 255),
    "black":    (0, 0, 0),
    "grey":     (130, 130, 130),
    "blue":     (220, 80, 0),
    "dark_bg":  (30, 30, 30),
}

EDGE_SCAN_WIDTH = 18
CORNER_REGION_SIZE = 65

DEFECT_COLORS = {
    "clean":    C["green"],
    "sharp":    C["green"],
    "minor":    C["yellow"],
    "moderate": C["orange"],
    "severe":   C["red"],
}

DEFECT_ALPHA = {
    "clean":    0.10,
    "sharp":    0.10,
    "minor":    0.28,
    "moderate": 0.42,
    "severe":   0.58,
}


def _score_color(score: float) -> Tuple[int, int, int]:
    if score >= 9.0:
        return C["green"]
    elif score >= 7.0:
        return C["yellow"]
    elif score >= 5.0:
        return C["orange"]
    return C["red"]


def _blend_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                color: Tuple, alpha: float) -> np.ndarray:
    """Blend a semi-transparent rectangle onto the image."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def draw_centering_overlay(img: np.ndarray, centering: Dict) -> np.ndarray:
    """
    Overlay centering border lines + ratio annotations.
    Cyan = detected inner border, Green = perfect-center reference.
    """
    out = img.copy()
    h, w = out.shape[:2]
    top_b, bot_b, left_b, right_b = centering["borders"]

    # Detected border lines
    cv2.line(out, (0, top_b),  (w, top_b),  C["cyan"], 2)
    cv2.line(out, (0, bot_b),  (w, bot_b),  C["cyan"], 2)
    cv2.line(out, (left_b, 0), (left_b, h), C["cyan"], 2)
    cv2.line(out, (right_b, 0),(right_b, h),C["cyan"], 2)

    # Ideal centre lines
    cv2.line(out, (w // 2, 0), (w // 2, h), C["green"], 1)
    cv2.line(out, (0, h // 2), (w, h // 2), C["green"], 1)

    # Margin dimension arrows / labels
    score = centering["score"]
    col = _score_color(score)

    def _label(text: str, x: int, y: int) -> None:
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, C["black"], 3, cv2.LINE_AA)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, col, 1, cv2.LINE_AA)

    _label(f"LR: {centering['lr_ratio']}", 8, 22)
    _label(f"TB: {centering['tb_ratio']}", 8, 45)
    _label(f"CTR: {score}/10", 8, 68)

    return out


def draw_edge_overlay(img: np.ndarray, edges: Dict) -> np.ndarray:
    """
    Colour-coded edge strip overlays (green → red severity).
    """
    out = img.copy()
    h, w = out.shape[:2]
    s = EDGE_SCAN_WIDTH

    regions = {
        "top":    (0,     0,     w,  s),
        "bottom": (0,     h - s, w,  h),
        "left":   (0,     0,     s,  h),
        "right":  (w - s, 0,     w,  h),
    }

    for side, (x1, y1, x2, y2) in regions.items():
        level = edges["sides"][side]["defect_level"]
        color = DEFECT_COLORS[level]
        alpha = DEFECT_ALPHA[level]
        out = _blend_rect(out, x1, y1, x2, y2, color, alpha)
        # Outline
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), color, 1)

    score = edges["score"]
    col = _score_color(score)
    cv2.putText(out, f"Edges: {score}/10", (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C["black"], 3, cv2.LINE_AA)
    cv2.putText(out, f"Edges: {score}/10", (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    return out


def draw_corner_overlay(img: np.ndarray, corners: Dict) -> np.ndarray:
    """
    Colour-coded corner box overlays.
    """
    out = img.copy()
    h, w = out.shape[:2]
    s = CORNER_REGION_SIZE

    regions = {
        "top_left":     (0,     0,     s,  s),
        "top_right":    (w - s, 0,     w,  s),
        "bottom_left":  (0,     h - s, s,  h),
        "bottom_right": (w - s, h - s, w,  h),
    }

    for corner, (x1, y1, x2, y2) in regions.items():
        level = corners["corners"][corner]["defect_level"]
        color = DEFECT_COLORS[level]
        alpha = DEFECT_ALPHA[level]
        out = _blend_rect(out, x1, y1, x2, y2, color, alpha)
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), color, 2)

    score = corners["score"]
    col = _score_color(score)
    cv2.putText(out, f"Corners: {score}/10", (8, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C["black"], 3, cv2.LINE_AA)
    cv2.putText(out, f"Corners: {score}/10", (8, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    return out


def draw_surface_overlay(img: np.ndarray, surface: Dict) -> np.ndarray:
    """
    Heatmap overlay for scratches (red channel) and anomalies (orange).
    """
    out = img.copy()
    h, w = out.shape[:2]

    scratch_map = surface.get("scratch_map")
    anomaly_mask = surface.get("anomaly_mask")

    # Scratch layer — red tint
    if scratch_map is not None and scratch_map.max() > 0:
        red_layer = np.zeros_like(out)
        red_layer[:, :, 2] = scratch_map   # BGR red
        cv2.addWeighted(out, 0.72, red_layer, 0.28, 0, out)

    # Anomaly layer — orange tint
    if anomaly_mask is not None and anomaly_mask.max() > 0:
        orange_layer = np.zeros_like(out)
        orange_layer[:, :, 2] = (anomaly_mask.astype(float) * 0.9).astype(np.uint8)
        orange_layer[:, :, 1] = (anomaly_mask.astype(float) * 0.35).astype(np.uint8)
        cv2.addWeighted(out, 0.78, orange_layer, 0.22, 0, out)

    score = surface["score"]
    col = _score_color(score)
    cv2.putText(out, f"Surface: {score}/10", (8, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C["black"], 3, cv2.LINE_AA)
    cv2.putText(out, f"Surface: {score}/10", (8, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

    for i, flag in enumerate(surface.get("risk_flags", [])[:3]):
        y = h - 85 - i * 20
        cv2.putText(out, f"! {flag}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C["black"], 2, cv2.LINE_AA)
        cv2.putText(out, f"! {flag}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C["red"], 1, cv2.LINE_AA)
    return out


def create_full_defect_overlay(img: np.ndarray,
                                centering: Dict,
                                edges: Dict,
                                corners: Dict,
                                surface: Dict) -> np.ndarray:
    """Combined overlay: surface → corners → edges → centering (layered)."""
    out = draw_surface_overlay(img, surface)
    out = draw_corner_overlay(out, corners)
    out = draw_edge_overlay(out, edges)
    out = draw_centering_overlay(out, centering)
    return out


def create_grade_summary_image(grade_result: Dict,
                                img_width: int = 500) -> np.ndarray:
    """
    Dark-background grade summary card with score bars.
    """
    h = 380
    summary = np.full((h, img_width, 3), 28, dtype=np.uint8)

    grade = grade_result["grade"]
    col = _score_color(grade)

    # ── Header ──────────────────────────────────────────────────────────────
    cv2.putText(summary, "AI CARD GRADER", (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, C["grey"], 1, cv2.LINE_AA)

    # ── Grade number ─────────────────────────────────────────────────────────
    grade_str = str(grade) if grade != int(grade) else str(int(grade))
    cv2.putText(summary, grade_str, (18, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 3.8, col, 6, cv2.LINE_AA)
    cv2.putText(summary, "/ 10", (int(img_width * 0.52), 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, C["grey"], 1, cv2.LINE_AA)

    # Grade band
    cv2.putText(summary, grade_result["grade_band"], (18, 148),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (170, 170, 170), 1, cv2.LINE_AA)

    # Separator
    cv2.line(summary, (18, 162), (img_width - 18, 162), (70, 70, 70), 1)

    # ── Score bars ───────────────────────────────────────────────────────────
    labels = [
        ("Centering", grade_result["centering_score"]),
        ("Edges",     grade_result["edges_score"]),
        ("Corners",   grade_result["corners_score"]),
        ("Surface",   grade_result["surface_score"]),
    ]
    bar_x = 115
    bar_max = img_width - bar_x - 55
    y = 190

    for label, score in labels:
        bar_w = int((score / 10.0) * bar_max)
        bar_col = _score_color(score)
        cv2.putText(summary, f"{label}:", (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170, 170, 170), 1, cv2.LINE_AA)
        # Background bar
        cv2.rectangle(summary, (bar_x, y - 12), (bar_x + bar_max, y - 2),
                      (60, 60, 60), -1)
        # Score bar
        if bar_w > 0:
            cv2.rectangle(summary, (bar_x, y - 12), (bar_x + bar_w, y - 2),
                          bar_col, -1)
        cv2.putText(summary, f"{score}", (bar_x + bar_max + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, bar_col, 1, cv2.LINE_AA)
        y += 30

    # ── Confidence ───────────────────────────────────────────────────────────
    conf = grade_result["confidence"]
    conf_str = f"Confidence: {int(conf * 100)}%"
    cv2.putText(summary, conf_str, (18, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 140, 140), 1, cv2.LINE_AA)

    # ── Recommendation banner ────────────────────────────────────────────────
    rec = grade_result["recommendation"]
    rec_colors = {
        "GRADE":        C["green"],
        "CONDITIONAL":  C["yellow"],
        "DO NOT GRADE": C["red"],
    }
    rec_col = rec_colors.get(rec, C["grey"])

    cv2.rectangle(summary, (18, h - 55), (img_width - 18, h - 18),
                  (50, 50, 50), -1)
    cv2.putText(summary, f">> {rec}", (26, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, rec_col, 2, cv2.LINE_AA)

    return summary


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) to RGB (Streamlit / matplotlib)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
