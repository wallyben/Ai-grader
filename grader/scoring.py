"""
Deterministic Grading Engine
Implements PSA/BGS-inspired rule-based grading with hard caps.
All outputs are reproducible given identical inputs — no randomness.
"""

from typing import Dict, Any, Tuple, List


# ─── WEIGHTS ──────────────────────────────────────────────────────────────────
# Based on real-world grading importance: corners/edges drive most grades
CATEGORY_WEIGHTS = {
    "centering": 0.20,
    "edges":     0.25,
    "corners":   0.30,
    "surface":   0.25,
}


# ─── HARD CAP RULES ───────────────────────────────────────────────────────────
# Each rule: (condition_fn, max_grade_allowed, description)
# Evaluated in order; the lowest cap from all triggered rules wins.

def _centering_caps(centering: Dict) -> List[Tuple[float, str]]:
    caps = []
    s = centering["score"]
    if s < 5.0:
        caps.append((4.0, "severe miscentering"))
    elif s <= 6.0:
        caps.append((6.0, "significant miscentering (>75/25)"))
    elif s <= 7.5:
        caps.append((8.0, "off-center (>65/35)"))
    elif s <= 9.0:
        caps.append((9.0, "slight centering issue (>60/40)"))
    return caps


def _edge_caps(edges: Dict) -> List[Tuple[float, str]]:
    caps = []
    for side, data in edges["sides"].items():
        level = data["defect_level"]
        if level == "severe":
            caps.append((6.0, f"severe edge damage — {side}"))
        elif level == "moderate":
            caps.append((8.0, f"edge whitening/chipping — {side}"))
        elif level == "minor":
            caps.append((9.0, f"minor edge wear — {side}"))
    return caps


def _corner_caps(corners: Dict) -> List[Tuple[float, str]]:
    caps = []
    for corner, data in corners["corners"].items():
        level = data["defect_level"]
        label = corner.replace("_", " ")
        if level == "severe":
            caps.append((5.0, f"severe corner damage — {label}"))
        elif level == "moderate":
            caps.append((7.0, f"corner wear — {label}"))
        elif level == "minor":
            caps.append((9.0, f"minor corner wear — {label}"))
    return caps


def _surface_caps(surface: Dict) -> List[Tuple[float, str]]:
    caps = []
    for flag in surface.get("risk_flags", []):
        if "heavy scratches" in flag:
            caps.append((6.0, flag))
        elif "scratch" in flag:
            caps.append((8.0, flag))
        elif "dent" in flag or "crease" in flag:
            caps.append((6.0, flag))
        elif "print line" in flag:
            caps.append((8.0, flag))
        elif "high surface noise" in flag:
            caps.append((7.0, flag))
    return caps


def apply_hard_caps(base_grade: float,
                     centering: Dict, edges: Dict,
                     corners: Dict, surface: Dict) -> Tuple[float, List[str]]:
    """
    Collect all applicable grade caps and apply the most restrictive one.
    Returns (capped_grade, list_of_triggered_cap_descriptions).
    """
    all_caps: List[Tuple[float, str]] = []
    all_caps.extend(_centering_caps(centering))
    all_caps.extend(_edge_caps(edges))
    all_caps.extend(_corner_caps(corners))
    all_caps.extend(_surface_caps(surface))

    if not all_caps:
        return base_grade, []

    # Find the strictest (lowest) cap
    min_cap = min(cap for cap, _ in all_caps)
    triggered = [desc for cap, desc in all_caps if cap <= base_grade or cap == min_cap]

    # Deduplicate while preserving order
    seen = set()
    unique_triggered = []
    for t in triggered:
        if t not in seen:
            seen.add(t)
            unique_triggered.append(t)

    final = min(base_grade, min_cap)
    return final, unique_triggered


def _compute_confidence(centering: Dict, edges: Dict,
                         corners: Dict, surface: Dict) -> float:
    """
    Confidence score reflects how certain we are about the grade.
    Reduced by:
    - Borderline scores in any category
    - Ambiguous surface readings
    - Multiple risk flags
    """
    conf = 0.94

    # Borderline centering is hard to distinguish
    c = centering["score"]
    if 7.5 <= c <= 9.0:
        conf -= 0.04

    # High surface noise makes analysis less reliable
    noise = surface.get("surface_noise", 0.0)
    if noise > 0.6:
        conf -= 0.10
    elif noise > 0.4:
        conf -= 0.05

    # Each risk flag adds uncertainty
    n_flags = len(surface.get("risk_flags", []))
    conf -= 0.04 * n_flags

    # Multiple defect categories
    defect_categories = sum([
        len(edges.get("defect_locations", [])) > 0,
        len(corners.get("defect_corners", [])) > 0,
        n_flags > 0,
    ])
    if defect_categories >= 2:
        conf -= 0.05

    return float(max(0.40, min(0.97, round(conf, 2))))


def _grade_band(grade: float) -> str:
    """PSA-style grade band label."""
    if grade >= 10.0:
        return "GEM MINT (PSA 10)"
    elif grade >= 9.5:
        return "GEM MINT+ (PSA 9.5)"
    elif grade >= 9.0:
        return "MINT (PSA 9)"
    elif grade >= 8.5:
        return "NM-MT+ (PSA 8.5)"
    elif grade >= 8.0:
        return "NM-MT (PSA 8)"
    elif grade >= 7.0:
        return "NEAR MINT (PSA 7)"
    elif grade >= 6.0:
        return "EX-MT (PSA 6)"
    elif grade >= 5.0:
        return "EXCELLENT (PSA 5)"
    elif grade >= 4.0:
        return "VG-EX (PSA 4)"
    elif grade >= 3.0:
        return "VERY GOOD (PSA 3)"
    elif grade >= 2.0:
        return "GOOD (PSA 2)"
    else:
        return "POOR (PSA 1)"


def _recommendation(grade: float, confidence: float,
                     caps: List[str]) -> Tuple[str, str]:
    """
    Should-I-Grade decision engine.

    GRADE        → high probability of positive ROI
    CONDITIONAL  → depends on card value/demand
    DO NOT GRADE → grading cost > likely slab premium
    """
    if grade >= 9.0:
        rec = "GRADE"
        reason = (f"Estimated grade {grade} qualifies as MINT or better. "
                  "Strong grading candidate with good ROI potential.")
    elif grade >= 8.0:
        rec = "CONDITIONAL"
        reason = (f"Estimated grade {grade}. "
                  "Consider grading only for high-demand or rare cards where "
                  "an 8 slab commands a meaningful premium.")
    elif grade >= 7.0:
        rec = "CONDITIONAL"
        reason = (f"Estimated grade {grade}. "
                  "Borderline — grading is only worthwhile for trophy/key cards. "
                  "Raw value likely similar to PSA 7 slab value for most sets.")
    else:
        rec = "DO NOT GRADE"
        reason = (f"Estimated grade {grade}. "
                  "Grading fees will likely exceed the slab premium at this grade. "
                  "Sell raw or pursue restoration options.")

    # Append cap notes
    if caps:
        cap_str = "; ".join(caps[:2])
        reason += f" | Caps triggered: {cap_str}"
        if len(caps) > 2:
            reason += f" (+{len(caps) - 2} more)"

    # Low confidence warning
    if confidence < 0.70:
        reason += " | ⚠ Low confidence — manual inspection strongly recommended."

    return rec, reason


def compute_grade(centering: Dict, edges: Dict,
                  corners: Dict, surface: Dict) -> Dict[str, Any]:
    """
    Compute the final deterministic grade from all analysis modules.

    Returns structured grade result including:
        grade, grade_band, per-category scores,
        caps_triggered, risk_flags, confidence, recommendation.
    """
    c_score = centering["score"]
    e_score = edges["score"]
    co_score = corners["score"]
    s_score = surface["score"]

    # Weighted base grade
    base_grade = (
        c_score  * CATEGORY_WEIGHTS["centering"] +
        e_score  * CATEGORY_WEIGHTS["edges"] +
        co_score * CATEGORY_WEIGHTS["corners"] +
        s_score  * CATEGORY_WEIGHTS["surface"]
    )

    # Apply hard caps
    final_grade_raw, caps_triggered = apply_hard_caps(
        base_grade, centering, edges, corners, surface
    )

    # Round to nearest 0.5 (PSA-style half-point grades)
    final_grade = round(final_grade_raw * 2) / 2
    final_grade = max(1.0, min(10.0, final_grade))

    confidence = _compute_confidence(centering, edges, corners, surface)
    band = _grade_band(final_grade)
    rec, rec_reason = _recommendation(final_grade, confidence, caps_triggered)

    # Collect all risk flags (from surface + caps)
    all_flags = list(surface.get("risk_flags", []))
    for cap in caps_triggered:
        if cap not in all_flags:
            all_flags.append(cap)

    return {
        "grade":            final_grade,
        "grade_band":       band,
        "centering_score":  c_score,
        "edges_score":      e_score,
        "corners_score":    co_score,
        "surface_score":    s_score,
        "centering_lr":     centering["lr_ratio"],
        "centering_tb":     centering["tb_ratio"],
        "caps_triggered":   caps_triggered,
        "risk_flags":       all_flags,
        "confidence":       confidence,
        "recommendation":   rec,
        "rec_reason":       rec_reason,
        "base_grade":       round(base_grade, 2),
    }
