"""
Deterministic Grading Engine
Implements PSA/BGS-inspired rule-based grading with hard caps.
Supports card profiles for type-specific thresholds.
All outputs are reproducible given identical inputs — no randomness.

New in v2: full grade_trace, profile-aware caps, structured cap evaluation log.
"""

from typing import Dict, Any, Tuple, List, Optional


# ─── Default weights (overridden by profile) ──────────────────────────────────
CATEGORY_WEIGHTS = {
    "centering": 0.20,
    "edges":     0.25,
    "corners":   0.30,
    "surface":   0.25,
}

DEFAULT_CAPS = {
    "severe_corner":          5.0,
    "moderate_corner":        7.0,
    "minor_corner":           9.0,
    "severe_edge":            6.0,
    "moderate_edge":          8.0,
    "minor_edge":             9.0,
    "centering_severe":       4.0,
    "centering_significant":  6.0,
    "centering_off":          8.0,
    "centering_slight":       9.0,
    "heavy_scratch":          6.0,
    "light_scratch":          8.0,
    "dent_crease":            6.0,
    "print_lines":            8.0,
    "high_surface_noise":     7.0,
}


def _get_caps(profile: Optional[Dict]) -> Dict:
    if profile and "caps" in profile:
        return profile["caps"]
    return DEFAULT_CAPS


def _get_weights(profile: Optional[Dict]) -> Dict:
    if profile and "scoring" in profile and "weights" in profile["scoring"]:
        return profile["scoring"]["weights"]
    return CATEGORY_WEIGHTS


# ─── Cap rule collectors ──────────────────────────────────────────────────────
# Each returns a list of (rule_key, cap_value, description) tuples.

def _centering_rules(centering: Dict, caps: Dict) -> List[Tuple[str, float, str]]:
    s = centering["score"]
    rules = []
    if s < 5.0:
        rules.append(("centering_severe",      caps["centering_severe"],
                       "severe miscentering (>75/25)"))
    elif s <= 6.0:
        rules.append(("centering_significant", caps["centering_significant"],
                       "significant miscentering (>70/30)"))
    elif s <= 7.5:
        rules.append(("centering_off",         caps["centering_off"],
                       "off-center (>65/35)"))
    elif s <= 9.0:
        rules.append(("centering_slight",      caps["centering_slight"],
                       "slight centering issue (>60/40)"))
    return rules


def _edge_rules(edges: Dict, caps: Dict) -> List[Tuple[str, float, str]]:
    rules = []
    for side, data in edges["sides"].items():
        level = data["defect_level"]
        if level == "severe":
            rules.append(("severe_edge", caps["severe_edge"],
                           f"severe edge damage — {side}"))
        elif level == "moderate":
            rules.append(("moderate_edge", caps["moderate_edge"],
                           f"edge whitening/chipping — {side}"))
        elif level == "minor":
            rules.append(("minor_edge", caps["minor_edge"],
                           f"minor edge wear — {side}"))
    return rules


def _corner_rules(corners: Dict, caps: Dict) -> List[Tuple[str, float, str]]:
    rules = []
    for pos, data in corners["corners"].items():
        label = pos.replace("_", " ")
        level = data["defect_level"]
        if level == "severe":
            rules.append(("severe_corner", caps["severe_corner"],
                           f"severe corner damage — {label}"))
        elif level == "moderate":
            rules.append(("moderate_corner", caps["moderate_corner"],
                           f"corner wear — {label}"))
        elif level == "minor":
            rules.append(("minor_corner", caps["minor_corner"],
                           f"minor corner wear — {label}"))
    return rules


def _surface_rules(surface: Dict, caps: Dict) -> List[Tuple[str, float, str]]:
    rules = []
    for flag in surface.get("risk_flags", []):
        if "heavy scratch" in flag:
            rules.append(("heavy_scratch", caps["heavy_scratch"], flag))
        elif "scratch" in flag:
            rules.append(("light_scratch", caps["light_scratch"], flag))
        elif "dent" in flag or "crease" in flag:
            rules.append(("dent_crease", caps["dent_crease"], flag))
        elif "print line" in flag:
            rules.append(("print_lines", caps["print_lines"], flag))
        elif "high surface noise" in flag:
            rules.append(("high_surface_noise", caps["high_surface_noise"], flag))
    return rules


# ─── Cap application with full trace ─────────────────────────────────────────

def apply_hard_caps(base_grade: float,
                     centering: Dict,
                     edges: Dict,
                     corners: Dict,
                     surface: Dict,
                     profile: Optional[Dict] = None) -> Tuple[float, List[str]]:
    """
    Backward-compatible wrapper — returns (final_grade, triggered_descriptions).
    """
    final, triggered, _, _ = _apply_caps_with_trace(
        base_grade, centering, edges, corners, surface, profile
    )
    return final, triggered


def _apply_caps_with_trace(
        base_grade: float,
        centering: Dict,
        edges: Dict,
        corners: Dict,
        surface: Dict,
        profile: Optional[Dict] = None
) -> Tuple[float, List[str], List[Dict], Optional[float]]:
    """
    Apply all caps and return:
        (final_grade, triggered_descs, full_cap_trace, effective_cap_value)

    full_cap_trace entries:
        { rule, cap, description, triggered, would_affect }
    """
    caps = _get_caps(profile)

    all_rules: List[Tuple[str, float, str]] = []
    all_rules.extend(_centering_rules(centering, caps))
    all_rules.extend(_edge_rules(edges, caps))
    all_rules.extend(_corner_rules(corners, caps))
    all_rules.extend(_surface_rules(surface, caps))

    cap_trace: List[Dict] = []
    triggered_rules: List[Tuple[float, str]] = []

    for rule_key, cap_val, desc in all_rules:
        would_lower = cap_val < base_grade
        cap_trace.append({
            "rule":        rule_key,
            "cap":         cap_val,
            "description": desc,
            "triggered":   would_lower,
        })
        if would_lower:
            triggered_rules.append((cap_val, desc))

    if triggered_rules:
        effective_cap = min(c for c, _ in triggered_rules)
        final_grade   = effective_cap
        triggered_descs = list(dict.fromkeys(d for _, d in triggered_rules))
    else:
        effective_cap = None
        final_grade   = base_grade
        triggered_descs = []

    return final_grade, triggered_descs, cap_trace, effective_cap


# ─── Confidence ───────────────────────────────────────────────────────────────

def _compute_confidence(centering: Dict, edges: Dict,
                         corners: Dict, surface: Dict) -> float:
    conf = 0.94
    if 7.5 <= centering["score"] <= 9.0:
        conf -= 0.04
    noise = surface.get("surface_noise", 0.0)
    if noise > 0.6:
        conf -= 0.10
    elif noise > 0.4:
        conf -= 0.05
    n_flags = len(surface.get("risk_flags", []))
    conf -= 0.04 * n_flags
    n_defect_cats = sum([
        len(edges.get("defect_locations", [])) > 0,
        len(corners.get("defect_corners", [])) > 0,
        n_flags > 0,
    ])
    if n_defect_cats >= 2:
        conf -= 0.05
    return float(max(0.40, min(0.97, round(conf, 2))))


# ─── Grade band ───────────────────────────────────────────────────────────────

def _grade_band(grade: float) -> str:
    if grade >= 10.0:  return "GEM MINT (PSA 10)"
    if grade >= 9.5:   return "GEM MINT+ (PSA 9.5)"
    if grade >= 9.0:   return "MINT (PSA 9)"
    if grade >= 8.5:   return "NM-MT+ (PSA 8.5)"
    if grade >= 8.0:   return "NM-MT (PSA 8)"
    if grade >= 7.0:   return "NEAR MINT (PSA 7)"
    if grade >= 6.0:   return "EX-MT (PSA 6)"
    if grade >= 5.0:   return "EXCELLENT (PSA 5)"
    if grade >= 4.0:   return "VG-EX (PSA 4)"
    if grade >= 3.0:   return "VERY GOOD (PSA 3)"
    if grade >= 2.0:   return "GOOD (PSA 2)"
    return "POOR (PSA 1)"


# ─── Recommendation ───────────────────────────────────────────────────────────

def _recommendation(grade: float, confidence: float, caps: List[str],
                     profile: Optional[Dict] = None) -> Tuple[str, str]:
    grade_thresh = 9.0
    cond_thresh  = 7.0
    if profile and "scoring" in profile:
        grade_thresh = profile["scoring"].get("grade_threshold", grade_thresh)
        cond_thresh  = profile["scoring"].get("conditional_threshold", cond_thresh)

    if grade >= grade_thresh:
        rec    = "GRADE"
        reason = (f"Estimated grade {grade} qualifies as MINT or better. "
                  "Strong grading candidate with good ROI potential.")
    elif grade >= cond_thresh:
        rec    = "CONDITIONAL"
        reason = (f"Estimated grade {grade}. Consider grading only for "
                  "high-demand or rare cards where the slab commands a premium.")
    else:
        rec    = "DO NOT GRADE"
        reason = (f"Estimated grade {grade}. Grading fees will likely exceed "
                  "the slab premium at this grade. Sell raw or hold.")

    if caps:
        reason += f" | Caps: {'; '.join(caps[:2])}"
        if len(caps) > 2:
            reason += f" (+{len(caps)-2} more)"
    if confidence < 0.70:
        reason += " | ⚠ Low confidence — manual inspection recommended."
    return rec, reason


# ─── Main entry point ─────────────────────────────────────────────────────────

def compute_grade(centering: Dict,
                   edges: Dict,
                   corners: Dict,
                   surface: Dict,
                   profile: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Compute the final deterministic grade from all analysis modules.

    New in v2:
      - profile-aware weights and caps
      - grade_trace field with full scoring audit trail
    """
    weights = _get_weights(profile)

    c_score  = centering["score"]
    e_score  = edges["score"]
    co_score = corners["score"]
    s_score  = surface["score"]

    base_grade = (
        c_score  * weights["centering"] +
        e_score  * weights["edges"] +
        co_score * weights["corners"] +
        s_score  * weights["surface"]
    )

    # Apply hard caps with full trace
    raw_capped, caps_triggered, cap_trace, effective_cap = _apply_caps_with_trace(
        base_grade, centering, edges, corners, surface, profile
    )

    # Round to nearest 0.5
    final_grade = round(round(raw_capped * 2) / 2, 1)
    final_grade = max(1.0, min(10.0, final_grade))

    confidence = _compute_confidence(centering, edges, corners, surface)
    band       = _grade_band(final_grade)
    rec, reason = _recommendation(final_grade, confidence, caps_triggered, profile)

    all_flags = list(surface.get("risk_flags", []))
    for cap in caps_triggered:
        if cap not in all_flags:
            all_flags.append(cap)

    # ── Full grade trace ──────────────────────────────────────────────────────
    grade_trace = {
        "sub_scores": {
            "centering": c_score,
            "edges":     e_score,
            "corners":   co_score,
            "surface":   s_score,
        },
        "weights": weights,
        "weighted_sum":     round(base_grade, 3),
        "caps_evaluated":   cap_trace,
        "effective_cap":    effective_cap,
        "post_cap_grade":   round(raw_capped, 3),
        "final_rounded_grade": final_grade,
        "profile_used":     profile.get("display_name", "tcg_generic") if profile else "tcg_generic",
    }

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
        "rec_reason":       reason,
        "base_grade":       round(base_grade, 2),
        "grade_trace":      grade_trace,
    }
