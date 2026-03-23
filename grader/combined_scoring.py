"""
Front + Back Combined Scoring
Merges independent front and back card analyses into a single grading input.

Combination rules by category:
  Centering : worst of front/back (both sides matter)
  Edges     : worst of any of the 8 edge sides (front + back)
  Corners   : worst of any of the 8 corners (front + back)
  Surface   : front/back weighted by profile front_back_weight
"""

import copy
from typing import Dict, Any, Optional


# ─── Per-category combination helpers ────────────────────────────────────────

def _combine_centering(front: Dict, back: Dict,
                        fw: float, bw: float) -> Dict:
    """
    Centering: return the worse side (lower score) as the driving value.
    Both ratios are retained for display.
    """
    if front["score"] <= back["score"]:
        worse, better = front, back
    else:
        worse, better = back, front

    result = copy.deepcopy(worse)
    result["front_lr"] = front["lr_ratio"]
    result["front_tb"] = front["tb_ratio"]
    result["back_lr"]  = back["lr_ratio"]
    result["back_tb"]  = back["tb_ratio"]
    result["combination"] = "worse_side"
    return result


def _combine_edges(front: Dict, back: Dict) -> Dict:
    """
    Edges: for each of the 4 positions, take the worse side.
    The combined set of 4 'worst-of-8' sides then drives the overall score.
    """
    result = copy.deepcopy(front)
    for side in ["top", "bottom", "left", "right"]:
        fs = front["sides"][side]
        bs = back["sides"][side]
        # Lower score = worse
        if bs["score"] < fs["score"]:
            result["sides"][side] = copy.deepcopy(bs)
            result["sides"][side]["from"] = "back"
        else:
            result["sides"][side]["from"] = "front"

    # Recompute aggregate metrics
    all_whitening  = [s["whitening"]  for s in result["sides"].values()]
    all_roughness  = [s["roughness"]  for s in result["sides"].values()]
    all_chipping   = [s["chipping"]   for s in result["sides"].values()]
    all_scores     = [s["score"]      for s in result["sides"].values()]

    result["avg_whitening"]    = round(sum(all_whitening)  / 4, 4)
    result["avg_roughness"]    = round(sum(all_roughness)  / 4, 4)
    result["avg_chipping"]     = round(sum(all_chipping)   / 4, 4)
    result["defect_locations"] = [s for s, d in result["sides"].items()
                                   if d["defect_level"] != "clean"]
    avg   = sum(all_scores) / 4
    worst = min(all_scores)
    result["score"] = round(max(1.0, min(10.0, avg * 0.65 + worst * 0.35)), 1)
    result["combination"] = "worst_of_8_sides"
    return result


def _combine_corners(front: Dict, back: Dict) -> Dict:
    """
    Corners: for each of the 4 positions, take the worse side.
    """
    result = copy.deepcopy(front)
    for pos in ["top_left", "top_right", "bottom_left", "bottom_right"]:
        fc = front["corners"][pos]
        bc = back["corners"][pos]
        if bc["score"] < fc["score"]:
            result["corners"][pos] = copy.deepcopy(bc)
            result["corners"][pos]["from"] = "back"
        else:
            result["corners"][pos]["from"] = "front"

    all_scores = [c["score"] for c in result["corners"].values()]
    result["defect_corners"] = [p for p, d in result["corners"].items()
                                 if d["defect_level"] != "sharp"]
    avg   = sum(all_scores) / 4
    worst = min(all_scores)
    result["score"] = round(max(1.0, min(10.0, avg * 0.60 + worst * 0.40)), 1)
    result["combination"] = "worst_of_8_corners"
    return result


def _combine_surface(front: Dict, back: Dict,
                      fw: float, bw: float) -> Dict:
    """
    Surface: weighted average of front/back, but flags from either side apply.
    """
    import numpy as np

    def _weighted(f_val: float, b_val: float) -> float:
        return f_val * fw + b_val * bw

    result = copy.deepcopy(front)
    result["scratch_density"]       = round(_weighted(front["scratch_density"],
                                                       back["scratch_density"]), 5)
    result["print_line_indicator"]  = round(_weighted(front["print_line_indicator"],
                                                       back["print_line_indicator"]), 5)
    result["anomaly_density"]       = round(_weighted(front["anomaly_density"],
                                                       back["anomaly_density"]), 5)
    result["surface_noise"]         = round(_weighted(front["surface_noise"],
                                                       back["surface_noise"]), 5)

    # Merge flags from both sides (deduplicate)
    merged_flags = list(front.get("risk_flags", []))
    for flag in back.get("risk_flags", []):
        if flag not in merged_flags:
            merged_flags.append(flag)
    result["risk_flags"] = merged_flags

    # Combined score: weighted
    result["score"] = round(max(1.0, min(10.0,
        front["score"] * fw + back["score"] * bw)), 1)
    result["combination"] = f"weighted_front{int(fw*100)}_back{int(bw*100)}"
    return result


# ─── Public API ───────────────────────────────────────────────────────────────

def combine_front_back(front_analysis: Dict,
                        back_analysis: Optional[Dict],
                        profile: Dict) -> Dict[str, Any]:
    """
    Merge front and back analysis dicts into a unified scoring package.

    Args:
        front_analysis: output from independent front analysis (profile-applied)
        back_analysis:  output from independent back analysis, or None
        profile:        loaded profile dict (used for front/back weights)

    Returns dict with keys: centering, edges, corners, surface, metadata
    """
    if back_analysis is None:
        return {
            "centering": copy.deepcopy(front_analysis["centering"]),
            "edges":     copy.deepcopy(front_analysis["edges"]),
            "corners":   copy.deepcopy(front_analysis["corners"]),
            "surface":   copy.deepcopy(front_analysis["surface"]),
            "metadata": {
                "mode": "front_only",
                "front_weight": 1.0,
                "back_weight":  0.0,
            },
        }

    fw = profile["scoring"]["front_back_weight"]["front"]
    bw = profile["scoring"]["front_back_weight"]["back"]

    return {
        "centering": _combine_centering(front_analysis["centering"],
                                         back_analysis["centering"], fw, bw),
        "edges":     _combine_edges(front_analysis["edges"],
                                     back_analysis["edges"]),
        "corners":   _combine_corners(front_analysis["corners"],
                                       back_analysis["corners"]),
        "surface":   _combine_surface(front_analysis["surface"],
                                       back_analysis["surface"], fw, bw),
        "metadata": {
            "mode":         "front_back_combined",
            "front_weight": fw,
            "back_weight":  bw,
        },
    }
