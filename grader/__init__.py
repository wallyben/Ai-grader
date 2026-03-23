"""
AI Card Grader — Core Package  v3 (Phase 3)
Full production pipeline with quality gate, card profiles, defect evidence,
grade trace, front+back combined scoring, artifact export,
calibration engine, ROI analysis, PDF report generation,
and calibration/evaluation framework.
"""

import numpy as np
from typing import Dict, Any, Optional

from .preprocessing    import normalize_card, load_image, load_image_from_path
from .centering        import compute_centering
from .edges            import analyze_edges
from .corners          import analyze_corners
from .surface          import analyze_surface
from .scoring          import compute_grade
from .profiles         import (get_profile, get_default_profile, list_profiles,
                                get_profile_display_names, apply_profile_to_analysis)
from .quality_gate     import assess_quality
from .defects          import extract_defects, defects_to_dicts, summarise_defects
from .combined_scoring import combine_front_back
from .artifacts        import (save_grading_artifacts, build_download_zip,
                                get_csv_row, save_batch_csv)
from .visualize        import (
    draw_centering_overlay,
    draw_edge_overlay,
    draw_corner_overlay,
    draw_surface_overlay,
    create_full_defect_overlay,
    draw_defect_evidence,
    create_grade_trace_image,
    create_grade_summary_image,
    bgr_to_rgb,
)
# ── Phase 2 ───────────────────────────────────────────────────────────────────
from .calibration import calibrate_grade, compute_calibration_stats, CalibrationEngine
from .pricing     import build_pricing, validate_pricing, CardPricing, CardPricingBundle
from .roi         import compute_roi, rank_batch_by_roi, batch_roi_summary
from .report      import generate_report, generate_report_to_file
# ── Phase 3 ───────────────────────────────────────────────────────────────────
from .evaluation        import (EvaluationRecord, MetricsEngine,
                                 load_evaluation_dataset, save_evaluation_dataset)
from .decision_optimizer import DecisionOptimizer
from .eval_dashboard    import render_dashboard, render_evaluation_report


# ─── Full pipeline ────────────────────────────────────────────────────────────

def grade_card_full(
        front_img:         np.ndarray,
        back_img:          Optional[np.ndarray] = None,
        profile_name:      str = "tcg_generic",
        run_quality_gate:  bool = True,
        front_raw:         Optional[np.ndarray] = None,
        back_raw:          Optional[np.ndarray] = None,
        pricing:           Optional["CardPricingBundle"] = None,
        grading_cost:      float = 0.0,
        calibration_data:  Optional[list] = None,
) -> Dict[str, Any]:
    """
    Complete grading pipeline (Phase 2).

    Args:
        front_img:        Normalised front image (500×700 BGR)
        back_img:         Normalised back image or None
        profile_name:     Card type profile key (see list_profiles())
        run_quality_gate: If True, assess image quality before grading
        front_raw:        Pre-normalisation front image for quality gate
        back_raw:         Pre-normalisation back image for quality gate
        pricing:          Optional CardPricingBundle for ROI computation
        grading_cost:     Grading fee (used only if pricing is provided)
        calibration_data: Optional custom calibration dataset

    Returns comprehensive result dict:
        grade_result     – final grade, band, scores, caps, recommendation
        grade_trace      – full scoring audit trail (now includes calibrated_grade)
        analysis         – combined analysis (front + back)
        front_analysis   – raw front analysis
        back_analysis    – raw back analysis (or None)
        defects          – structured Defect objects (front side)
        quality_gate     – image quality report (or None)
        profile_used     – profile key
        visualizations   – all overlay images
        calibration      – calibration result dict (Phase 2)
        roi              – ROI result dict (Phase 2, only if pricing provided)
    """
    profile = get_profile(profile_name)

    # ── Quality gate (on raw pre-normalisation images when available) ─────────
    quality_report = None
    if run_quality_gate:
        gate_img = front_raw if front_raw is not None else front_img
        quality_report = assess_quality(gate_img)

    # ── Per-side analysis ─────────────────────────────────────────────────────
    front_analysis_raw = {
        "centering": compute_centering(front_img),
        "edges":     analyze_edges(front_img),
        "corners":   analyze_corners(front_img),
        "surface":   analyze_surface(front_img),
    }
    front_analysis = apply_profile_to_analysis(front_analysis_raw, profile)

    back_analysis = None
    if back_img is not None:
        back_analysis_raw = {
            "centering": compute_centering(back_img),
            "edges":     analyze_edges(back_img),
            "corners":   analyze_corners(back_img),
            "surface":   analyze_surface(back_img),
        }
        back_analysis = apply_profile_to_analysis(back_analysis_raw, profile)

    # ── Combined scoring ──────────────────────────────────────────────────────
    combined = combine_front_back(front_analysis, back_analysis, profile)

    # ── Grade computation ─────────────────────────────────────────────────────
    grade_result = compute_grade(
        combined["centering"],
        combined["edges"],
        combined["corners"],
        combined["surface"],
        profile=profile,
    )
    grade_trace = grade_result.pop("grade_trace", {})

    # ── Phase 2: Calibration ──────────────────────────────────────────────────
    raw_grade  = grade_result["grade"]
    raw_conf   = grade_result["confidence"]

    # Boost confidence using quality gate score when available
    quality_adjusted_conf = raw_conf
    if quality_report:
        q_score = quality_report.get("score", 100)
        if q_score < 60:
            quality_adjusted_conf = round(max(0.40, raw_conf - 0.08), 2)
        elif q_score < 80:
            quality_adjusted_conf = round(max(0.40, raw_conf - 0.04), 2)

    calibration_result = calibrate_grade(
        predicted_grade=raw_grade,
        confidence=quality_adjusted_conf,
        custom_data=calibration_data,
    )

    # Store calibrated grade and confidence back into grade_result
    grade_result["calibrated_grade"]      = calibration_result["calibrated_grade"]
    grade_result["calibrated_confidence"] = calibration_result["calibrated_confidence"]

    # Enrich grade_trace with calibration step
    grade_trace["calibrated_grade"]       = calibration_result["calibrated_grade"]
    grade_trace["calibration_adjustment"] = calibration_result["calibration_adjustment"]
    grade_trace["post_cap_grade"]         = grade_trace.get("post_cap_grade", raw_grade)

    # Use calibrated grade for recommendation if it differs
    calibrated = calibration_result["calibrated_grade"]
    calibrated_conf = calibration_result["calibrated_confidence"]
    if calibrated != grade_result["grade"]:
        # Re-run recommendation with calibrated grade + confidence
        from .scoring import _recommendation, _grade_band, _get_caps
        new_rec, new_reason = _recommendation(
            calibrated, calibrated_conf,
            grade_result.get("caps_triggered", []),
            profile,
        )
        grade_result["grade"]          = calibrated
        grade_result["grade_band"]     = _grade_band(calibrated)
        grade_result["recommendation"] = new_rec
        grade_result["rec_reason"]     = new_reason
        grade_result["confidence"]     = calibrated_conf

    # ── Phase 2: ROI ──────────────────────────────────────────────────────────
    roi_result = None
    if pricing is not None:
        p = pricing.pricing.fill_estimates()
        roi_result = compute_roi(
            estimated_grade=calibrated,
            confidence=calibrated_conf,
            raw_price=p.raw_price,
            grading_cost=pricing.grading_cost or grading_cost,
            psa_8_price=p.psa_8,
            psa_9_price=p.psa_9,
            psa_10_price=p.psa_10,
            card_name=p.card_name,
        )

    # ── Defect evidence ───────────────────────────────────────────────────────
    defects_front = extract_defects(front_analysis, side="front")
    defects_back  = extract_defects(back_analysis,  side="back") if back_analysis else []
    all_defects   = defects_front + defects_back

    # ── Visualizations (always on front image) ────────────────────────────────
    viz_centering = draw_centering_overlay(front_img, combined["centering"])
    viz_edges     = draw_edge_overlay(front_img, combined["edges"])
    viz_corners   = draw_corner_overlay(front_img, combined["corners"])
    viz_surface   = draw_surface_overlay(front_img, combined["surface"])
    viz_full      = create_full_defect_overlay(
        front_img, combined["centering"], combined["edges"],
        combined["corners"], combined["surface"]
    )
    viz_defects   = draw_defect_evidence(front_img, defects_front)
    viz_trace     = create_grade_trace_image({**grade_result, "grade_trace": grade_trace})
    viz_summary   = create_grade_summary_image({**grade_result, "grade_trace": grade_trace})

    return {
        "grade_result":    grade_result,
        "grade_trace":     grade_trace,
        "analysis":        combined,
        "front_analysis":  front_analysis,
        "back_analysis":   back_analysis,
        "defects":         all_defects,
        "defects_front":   defects_front,
        "defects_back":    defects_back,
        "quality_gate":    quality_report,
        "profile_used":    profile_name,
        "calibration":     calibration_result,
        "roi":             roi_result,
        "visualizations": {
            "centering":       viz_centering,
            "edges":           viz_edges,
            "corners":         viz_corners,
            "surface":         viz_surface,
            "full_overlay":    viz_full,
            "defect_evidence": viz_defects,
            "grade_trace":     viz_trace,
            "grade_summary":   viz_summary,
        },
    }


# ─── Backward-compatible wrapper ──────────────────────────────────────────────

def grade_card(front_img: np.ndarray,
               back_img: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Backward-compatible entry point (v1 API).
    Uses default profile, no quality gate.
    """
    result = grade_card_full(front_img, back_img,
                              profile_name="tcg_generic",
                              run_quality_gate=False)
    # Merge grade_trace back into grade_result for v1 consumers
    result["grade_result"]["grade_trace"] = result["grade_trace"]
    return result


def _pick_worse(a: Dict, b: Dict) -> Dict:
    """Return the dict with the lower 'score' (worse condition)."""
    return a if a["score"] <= b["score"] else b
