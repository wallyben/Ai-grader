"""
Visualization Engine
Generates annotated overlays and summary cards for the Streamlit UI.
v2: added draw_defect_evidence(), create_grade_trace_image()
All drawing is done in BGR (OpenCV native) — convert before display with bgr_to_rgb().
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .defects import Defect

# Colour palette (BGR)
C = {
    "green":    (50,  220, 80),
    "yellow":   (0,   220, 220),
    "orange":   (0,   145, 255),
    "red":      (0,   60,  220),
    "cyan":     (220, 220, 0),
    "white":    (255, 255, 255),
    "black":    (0,   0,   0),
    "grey":     (130, 130, 130),
    "blue":     (220, 80,  0),
    "dark_bg":  (30,  30,  30),
    "mid_bg":   (50,  50,  50),
    "light_bg": (72,  72,  72),
}

EDGE_SCAN_WIDTH   = 18
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
SEVERITY_COLORS = {
    "minor":    C["yellow"],
    "moderate": C["orange"],
    "severe":   C["red"],
}


def _score_color(score: float) -> Tuple[int, int, int]:
    if score >= 9.0:  return C["green"]
    if score >= 7.0:  return C["yellow"]
    if score >= 5.0:  return C["orange"]
    return C["red"]


def _blend_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                color: Tuple, alpha: float) -> np.ndarray:
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def _put_text_outlined(img: np.ndarray, text: str, x: int, y: int,
                        scale: float, color: Tuple, thickness: int = 1) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, C["black"], thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


# ─── Existing overlays ────────────────────────────────────────────────────────

def draw_centering_overlay(img: np.ndarray, centering: Dict) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    top_b, bot_b, left_b, right_b = centering["borders"]

    cv2.line(out, (0, top_b),   (w, top_b),   C["cyan"], 2)
    cv2.line(out, (0, bot_b),   (w, bot_b),   C["cyan"], 2)
    cv2.line(out, (left_b, 0),  (left_b, h),  C["cyan"], 2)
    cv2.line(out, (right_b, 0), (right_b, h), C["cyan"], 2)
    cv2.line(out, (w // 2, 0),  (w // 2, h),  C["green"], 1)
    cv2.line(out, (0, h // 2),  (w, h // 2),  C["green"], 1)

    score = centering["score"]
    col   = _score_color(score)
    _put_text_outlined(out, f"LR: {centering['lr_ratio']}", 8, 22,  0.55, col, 1)
    _put_text_outlined(out, f"TB: {centering['tb_ratio']}", 8, 45,  0.55, col, 1)
    _put_text_outlined(out, f"CTR: {score}/10",             8, 68,  0.55, col, 1)
    return out


def draw_edge_overlay(img: np.ndarray, edges: Dict) -> np.ndarray:
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
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), color, 1)

    score = edges["score"]
    col   = _score_color(score)
    _put_text_outlined(out, f"Edges: {score}/10", 8, h - 10, 0.55, col, 1)
    return out


def draw_corner_overlay(img: np.ndarray, corners: Dict) -> np.ndarray:
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
    col   = _score_color(score)
    _put_text_outlined(out, f"Corners: {score}/10", 8, h - 35, 0.55, col, 1)
    return out


def draw_surface_overlay(img: np.ndarray, surface: Dict) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    scratch_map  = surface.get("scratch_map")
    anomaly_mask = surface.get("anomaly_mask")

    if scratch_map is not None and scratch_map.max() > 0:
        red_layer = np.zeros_like(out)
        red_layer[:, :, 2] = scratch_map
        cv2.addWeighted(out, 0.72, red_layer, 0.28, 0, out)

    if anomaly_mask is not None and anomaly_mask.max() > 0:
        orange_layer = np.zeros_like(out)
        orange_layer[:, :, 2] = (anomaly_mask.astype(float) * 0.9).astype(np.uint8)
        orange_layer[:, :, 1] = (anomaly_mask.astype(float) * 0.35).astype(np.uint8)
        cv2.addWeighted(out, 0.78, orange_layer, 0.22, 0, out)

    score = surface["score"]
    col   = _score_color(score)
    _put_text_outlined(out, f"Surface: {score}/10", 8, h - 60, 0.55, col, 1)
    for i, flag in enumerate(surface.get("risk_flags", [])[:3]):
        _put_text_outlined(out, f"! {flag}", 8, h - 85 - i * 20, 0.40, C["red"], 1)
    return out


def create_full_defect_overlay(img: np.ndarray, centering: Dict,
                                edges: Dict, corners: Dict,
                                surface: Dict) -> np.ndarray:
    out = draw_surface_overlay(img, surface)
    out = draw_corner_overlay(out, corners)
    out = draw_edge_overlay(out, edges)
    out = draw_centering_overlay(out, centering)
    return out


# ─── NEW: Defect evidence overlay ────────────────────────────────────────────

def draw_defect_evidence(img: np.ndarray, defects: List) -> np.ndarray:
    """
    Overlay labelled bounding boxes for every structured Defect object.
    Boxes are colour-coded by severity. Labels show type + severity.
    """
    out = img.copy()

    for defect in defects:
        # Support both Defect objects and dicts
        if hasattr(defect, "bbox"):
            bbox    = defect.bbox
            sev     = defect.severity
            d_type  = defect.defect_type
        else:
            bbox    = defect.get("bbox")
            sev     = defect.get("severity", "minor")
            d_type  = defect.get("defect_type", "unknown")

        if bbox is None:
            continue

        x, y, bw, bh = bbox
        color = SEVERITY_COLORS.get(sev, C["grey"])

        # Draw bounding box with dashed-like effect (solid 2px)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)

        # Label background + text
        label = f"{d_type.replace('_', ' ')} [{sev}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        lx, ly = x, max(0, y - 2)
        cv2.rectangle(out, (lx, ly - th - 2), (lx + tw + 4, ly + 2), color, -1)
        cv2.putText(out, label, (lx + 2, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["black"], 1, cv2.LINE_AA)

    return out


# ─── NEW: Grade trace image ───────────────────────────────────────────────────

def create_grade_trace_image(grade_result: Dict,
                               img_width: int = 600) -> np.ndarray:
    """
    Visual grade trace card:
      - Sub-scores with weight contribution
      - Weighted sum
      - Cap evaluation list (triggered / not triggered)
      - Effective cap
      - Final grade
    """
    trace = grade_result.get("grade_trace", {})
    if not trace:
        h = 80
        img = np.full((h, img_width, 3), 28, dtype=np.uint8)
        cv2.putText(img, "No grade trace available.", (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C["grey"], 1)
        return img

    sub  = trace.get("sub_scores", {})
    wts  = trace.get("weights", {})
    caps = trace.get("caps_evaluated", [])

    n_caps = len(caps)
    h = 320 + n_caps * 20
    img = np.full((h, img_width, 3), 28, dtype=np.uint8)

    # ── Header ────────────────────────────────────────────────────────────────
    cv2.putText(img, "GRADE TRACE", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, C["grey"], 1, cv2.LINE_AA)
    profile_name = trace.get("profile_used", "tcg_generic")
    cv2.putText(img, f"Profile: {profile_name}", (img_width - 200, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C["grey"], 1, cv2.LINE_AA)
    cv2.line(img, (12, 40), (img_width - 12, 40), C["light_bg"], 1)

    # ── Sub-score bars ────────────────────────────────────────────────────────
    y = 62
    bar_x = 145
    bar_max = img_width - bar_x - 80
    cv2.putText(img, "Sub-scores:", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (170, 170, 170), 1, cv2.LINE_AA)
    y += 22
    for cat in ["centering", "edges", "corners", "surface"]:
        score  = sub.get(cat, 0.0)
        weight = wts.get(cat, 0.0)
        contrib = score * weight
        col    = _score_color(score)
        bw     = int((score / 10.0) * bar_max)

        cv2.putText(img, f"{cat.title()}:", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.rectangle(img, (bar_x, y - 12), (bar_x + bar_max, y - 2), C["light_bg"], -1)
        if bw > 0:
            cv2.rectangle(img, (bar_x, y - 12), (bar_x + bw, y - 2), col, -1)
        cv2.putText(img, f"{score} × {int(weight*100)}% = {contrib:.2f}", (bar_x + bar_max + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
        y += 24

    # Weighted sum
    wsum = trace.get("weighted_sum", 0.0)
    cv2.line(img, (bar_x, y), (img_width - 12, y), C["light_bg"], 1)
    y += 18
    cv2.putText(img, f"Weighted sum (pre-cap):", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, f"{wsum:.3f}", (bar_x + bar_max + 5, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C["yellow"], 1, cv2.LINE_AA)
    y += 26

    # ── Cap evaluation ────────────────────────────────────────────────────────
    if caps:
        cv2.line(img, (12, y), (img_width - 12, y), C["light_bg"], 1)
        y += 16
        cv2.putText(img, "Cap rules evaluated:", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (170, 170, 170), 1, cv2.LINE_AA)
        y += 20
        for cap in caps:
            triggered = cap.get("triggered", False)
            icon_col  = C["red"] if triggered else C["green"]
            icon      = "X" if triggered else "v"
            desc      = cap.get("description", "")[:48]
            cap_val   = cap.get("cap", 0)

            cv2.putText(img, icon, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, icon_col, 1, cv2.LINE_AA)
            cv2.putText(img, f"Cap {cap_val}: {desc}", (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        C["red"] if triggered else (140, 140, 140), 1, cv2.LINE_AA)
            y += 19

    # ── Effective cap + final grade ───────────────────────────────────────────
    eff_cap = trace.get("effective_cap")
    post    = trace.get("post_cap_grade", wsum)
    final   = grade_result.get("grade", 0.0)

    y += 8
    cv2.line(img, (12, y), (img_width - 12, y), C["light_bg"], 1)
    y += 18
    if eff_cap is not None:
        cv2.putText(img, f"Effective cap applied:  {eff_cap}  →  {post:.3f}",
                    (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C["orange"], 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "No caps triggered.",
                    (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C["green"], 1, cv2.LINE_AA)
    y += 24
    final_col = _score_color(final)
    cv2.putText(img, f"Final rounded grade:  {final}",
                (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, final_col, 2, cv2.LINE_AA)

    return img


# ─── Grade summary ────────────────────────────────────────────────────────────

def create_grade_summary_image(grade_result: Dict, img_width: int = 500) -> np.ndarray:
    h = 380
    summary = np.full((h, img_width, 3), 28, dtype=np.uint8)

    grade = grade_result["grade"]
    col   = _score_color(grade)

    cv2.putText(summary, "AI CARD GRADER", (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, C["grey"], 1, cv2.LINE_AA)

    grade_str = str(grade) if grade != int(grade) else str(int(grade))
    cv2.putText(summary, grade_str, (18, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 3.8, col, 6, cv2.LINE_AA)
    cv2.putText(summary, "/ 10", (int(img_width * 0.52), 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, C["grey"], 1, cv2.LINE_AA)

    cv2.putText(summary, grade_result["grade_band"], (18, 148),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (170, 170, 170), 1, cv2.LINE_AA)

    # Profile name if present
    if "grade_trace" in grade_result:
        pname = grade_result["grade_trace"].get("profile_used", "")
        if pname:
            cv2.putText(summary, f"Profile: {pname}", (18, 164),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1, cv2.LINE_AA)

    cv2.line(summary, (18, 170), (img_width - 18, 170), (70, 70, 70), 1)

    labels = [
        ("Centering", grade_result["centering_score"]),
        ("Edges",     grade_result["edges_score"]),
        ("Corners",   grade_result["corners_score"]),
        ("Surface",   grade_result["surface_score"]),
    ]
    bar_x   = 115
    bar_max = img_width - bar_x - 55
    y = 198

    for label, score in labels:
        bw  = int((score / 10.0) * bar_max)
        bcol = _score_color(score)
        cv2.putText(summary, f"{label}:", (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.rectangle(summary, (bar_x, y - 12), (bar_x + bar_max, y - 2), (60, 60, 60), -1)
        if bw > 0:
            cv2.rectangle(summary, (bar_x, y - 12), (bar_x + bw, y - 2), bcol, -1)
        cv2.putText(summary, f"{score}", (bar_x + bar_max + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, bcol, 1, cv2.LINE_AA)
        y += 30

    conf = grade_result["confidence"]
    cv2.putText(summary, f"Confidence: {int(conf * 100)}%", (18, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 140, 140), 1, cv2.LINE_AA)

    rec     = grade_result["recommendation"]
    rec_col = {"GRADE": C["green"], "CONDITIONAL": C["yellow"],
               "DO NOT GRADE": C["red"]}.get(rec, C["grey"])
    cv2.rectangle(summary, (18, h - 55), (img_width - 18, h - 18), (50, 50, 50), -1)
    cv2.putText(summary, f">> {rec}", (26, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, rec_col, 2, cv2.LINE_AA)

    return summary


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) to RGB (Streamlit / matplotlib)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
