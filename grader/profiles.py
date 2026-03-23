"""
Card Profile System
Loads and applies card-type-specific grading profiles from config/profiles.json.
Profiles control: centering tolerances, whitening thresholds, corner severity,
surface sensitivity, scoring weights, cap thresholds, and recommendation bars.
"""

import json
import os
import copy
from typing import Dict, Any, List, Optional

_PROFILES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "config", "profiles.json")
_DEFAULT_PROFILE = "tcg_generic"

# ─── Cache ────────────────────────────────────────────────────────────────────
_profile_cache: Optional[Dict] = None


def _load_all() -> Dict:
    global _profile_cache
    if _profile_cache is None:
        with open(_PROFILES_PATH, "r") as f:
            _profile_cache = json.load(f)
    return _profile_cache


def list_profiles() -> List[str]:
    """Return sorted list of available profile keys."""
    return sorted(_load_all().keys())


def get_profile(name: str) -> Dict[str, Any]:
    """
    Return the full profile dict for the given key.
    Falls back to tcg_generic if name is not recognised.
    """
    all_profiles = _load_all()
    if name not in all_profiles:
        name = _DEFAULT_PROFILE
    return copy.deepcopy(all_profiles[name])


def get_default_profile() -> Dict[str, Any]:
    return get_profile(_DEFAULT_PROFILE)


def get_profile_display_names() -> Dict[str, str]:
    """Return {key: display_name} mapping for UI dropdowns."""
    return {k: v["display_name"] for k, v in _load_all().items()}


# ─── Centering rescoring ──────────────────────────────────────────────────────

def score_centering_with_profile(left_pct: float, right_pct: float,
                                  top_pct: float, bottom_pct: float,
                                  profile: Dict) -> float:
    """
    Re-score centering using profile-specific deviation thresholds.
    Replaces the default thresholds baked into centering.py.
    """
    t = profile["centering"]
    max_dev = max(abs(left_pct - right_pct), abs(top_pct - bottom_pct))

    if max_dev <= t["deviation_10"]:      return 10.0
    elif max_dev <= t["deviation_9_5"]:   return 9.5
    elif max_dev <= t["deviation_9"]:     return 9.0
    elif max_dev <= t["deviation_8_5"]:   return 8.5
    elif max_dev <= t["deviation_8"]:     return 8.0
    elif max_dev <= t["deviation_7"]:     return 7.0
    elif max_dev <= t["deviation_6"]:     return 6.0
    elif max_dev <= t["deviation_5"]:     return 5.0
    elif max_dev <= t["deviation_5"] * 1.25: return 4.0
    elif max_dev <= t["deviation_5"] * 1.5:  return 3.0
    else:                                  return 2.0


# ─── Edge reclassification ────────────────────────────────────────────────────

def classify_edge_with_profile(whitening: float, roughness: float,
                                 chipping: float, profile: Dict) -> str:
    """Return defect level for an edge side using profile thresholds."""
    t = profile["edges"]
    if (whitening >= t["whitening_severe"] or roughness >= t["roughness_severe"]
            or chipping >= t["chipping_severe"]):
        return "severe"
    if (whitening >= t["whitening_moderate"] or roughness >= t["roughness_moderate"]
            or chipping >= t["chipping_moderate"]):
        return "moderate"
    if (whitening >= t["whitening_minor"] or roughness >= t["roughness_minor"]
            or chipping >= t["chipping_minor"]):
        return "minor"
    return "clean"


def reclassify_edges(edge_data: Dict, profile: Dict) -> Dict:
    """
    Post-process edge analysis to reclassify defect levels using profile thresholds.
    Returns a new dict (original is not mutated).
    """
    result = copy.deepcopy(edge_data)
    for side, data in result["sides"].items():
        data["defect_level"] = classify_edge_with_profile(
            data["whitening"], data["roughness"], data["chipping"], profile
        )
    # Recompute defect_locations and aggregate score
    defect_locs = [s for s, d in result["sides"].items()
                   if d["defect_level"] != "clean"]
    result["defect_locations"] = defect_locs

    side_scores = [d["score"] for d in result["sides"].values()]
    avg = sum(side_scores) / len(side_scores)
    worst = min(side_scores)
    result["score"] = float(max(1.0, min(10.0, round(avg * 0.65 + worst * 0.35, 1))))
    return result


# ─── Corner reclassification ──────────────────────────────────────────────────

def classify_corner_with_profile(whitening: float, rounding: float,
                                   profile: Dict) -> str:
    """Return defect level for a corner using profile thresholds."""
    t = profile["corners"]
    if whitening >= t["whitening_severe"] or rounding >= t["rounding_severe"]:
        return "severe"
    if whitening >= t["whitening_moderate"] or rounding >= t["rounding_moderate"]:
        return "moderate"
    if whitening >= t["whitening_minor"] or rounding >= t["rounding_minor"]:
        return "minor"
    return "sharp"


def reclassify_corners(corner_data: Dict, profile: Dict) -> Dict:
    """
    Post-process corner analysis to reclassify defect levels using profile thresholds.
    """
    result = copy.deepcopy(corner_data)
    for pos, data in result["corners"].items():
        data["defect_level"] = classify_corner_with_profile(
            data["whitening"], data["rounding"], profile
        )
    defect_corners = [p for p, d in result["corners"].items()
                      if d["defect_level"] != "sharp"]
    result["defect_corners"] = defect_corners

    scores = [d["score"] for d in result["corners"].values()]
    avg = sum(scores) / len(scores)
    worst = min(scores)
    result["score"] = float(max(1.0, min(10.0, round(avg * 0.60 + worst * 0.40, 1))))
    return result


# ─── Surface risk flag rescoring ──────────────────────────────────────────────

def regenerate_surface_flags(surface_data: Dict, profile: Dict) -> Dict:
    """
    Re-evaluate surface risk flags and score using profile thresholds.
    """
    result = copy.deepcopy(surface_data)
    t = profile["surface"]
    sd = result["scratch_density"]
    pl = result["print_line_indicator"]
    ad = result["anomaly_density"]
    sn = result["surface_noise"]

    flags = []
    if sd > t["scratch_density_severe"]:
        flags.append("heavy scratches detected")
    elif sd > t["scratch_density_minor"]:
        flags.append("light scratches detected")
    if pl > t["print_line_warn"]:
        flags.append("print lines / manufacturing defect detected")
    if ad > t["anomaly_density_warn"]:
        flags.append("possible dent or crease detected")
    if sn > t["noise_warn"]:
        flags.append("high surface noise — possible damage")

    result["risk_flags"] = flags

    # Recompute score with profile penalties
    penalty = 0.0
    if sd > t["scratch_density_severe"]:
        penalty += (sd / t["scratch_density_severe"]) * 4.5
    elif sd > t["scratch_density_minor"]:
        penalty += (sd / t["scratch_density_moderate"]) * 2.0
    if pl > t["print_line_warn"]:
        penalty += (pl - t["print_line_warn"]) * 15.0
    if ad > t["anomaly_density_warn"]:
        penalty += (ad - t["anomaly_density_warn"]) * 80.0
    if sn > t["noise_warn"]:
        penalty += (sn - t["noise_warn"]) * 8.0

    result["score"] = float(max(1.0, min(10.0, round(10.0 - penalty, 1))))
    return result


# ─── Full analysis reclassification ──────────────────────────────────────────

def apply_profile_to_analysis(analysis: Dict, profile: Dict) -> Dict:
    """
    Apply profile-specific thresholds to reclassify all defect levels
    and re-score centering.

    This is a pure transformation — the original analysis dict is not mutated.
    """
    result = copy.deepcopy(analysis)

    # Centering: rescore with profile deviation thresholds
    c = result["centering"]
    result["centering"]["score"] = score_centering_with_profile(
        c["left_pct"], c["right_pct"], c["top_pct"], c["bottom_pct"], profile
    )

    # Edges: reclassify with profile whitening/roughness/chipping thresholds
    result["edges"] = reclassify_edges(result["edges"], profile)

    # Corners: reclassify with profile whitening/rounding thresholds
    result["corners"] = reclassify_corners(result["corners"], profile)

    # Surface: regenerate flags and score with profile sensitivity
    result["surface"] = regenerate_surface_flags(result["surface"], profile)

    return result
