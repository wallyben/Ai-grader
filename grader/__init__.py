"""
AI Card Grader — Core Package
Exposes the top-level grade_card() pipeline and all sub-modules.
"""

from .preprocessing import normalize_card, load_image, load_image_from_path
from .centering import compute_centering
from .edges import analyze_edges
from .corners import analyze_corners
from .surface import analyze_surface
from .scoring import compute_grade
from .visualize import (
    draw_centering_overlay,
    draw_edge_overlay,
    draw_corner_overlay,
    draw_surface_overlay,
    create_full_defect_overlay,
    create_grade_summary_image,
    bgr_to_rgb,
)

import numpy as np
from typing import Dict, Any, Optional


def grade_card(front_img: np.ndarray,
               back_img: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Full grading pipeline.

    Args:
        front_img: Already-normalized front card image (BGR, 500×700).
        back_img:  Already-normalized back card image (optional).

    Returns:
        {
            "grade_result": { grade, grade_band, scores, caps, flags, … },
            "analysis":     { centering, edges, corners, surface },
            "visualizations": {
                centering, edges, corners, surface,
                full_overlay, grade_summary
            }
        }
    """
    # ── Front analysis ────────────────────────────────────────────────────────
    centering = compute_centering(front_img)
    edges     = analyze_edges(front_img)
    corners   = analyze_corners(front_img)
    surface   = analyze_surface(front_img)

    # ── Back analysis (merge worst scores) ───────────────────────────────────
    if back_img is not None:
        b_centering = compute_centering(back_img)
        b_edges     = analyze_edges(back_img)
        b_corners   = analyze_corners(back_img)
        b_surface   = analyze_surface(back_img)

        # Use whichever side has the worse (lower) score
        centering = _pick_worse(centering, b_centering)
        edges     = _pick_worse(edges, b_edges)
        corners   = _pick_worse(corners, b_corners)
        surface   = _pick_worse(surface, b_surface)

    # ── Grade computation ─────────────────────────────────────────────────────
    grade_result = compute_grade(centering, edges, corners, surface)

    # ── Visualizations (always on front image) ────────────────────────────────
    viz_centering    = draw_centering_overlay(front_img, centering)
    viz_edges        = draw_edge_overlay(front_img, edges)
    viz_corners      = draw_corner_overlay(front_img, corners)
    viz_surface      = draw_surface_overlay(front_img, surface)
    viz_full         = create_full_defect_overlay(front_img, centering, edges, corners, surface)
    viz_grade_summary = create_grade_summary_image(grade_result)

    return {
        "grade_result": grade_result,
        "analysis": {
            "centering": centering,
            "edges":     edges,
            "corners":   corners,
            "surface":   surface,
        },
        "visualizations": {
            "centering":     viz_centering,
            "edges":         viz_edges,
            "corners":       viz_corners,
            "surface":       viz_surface,
            "full_overlay":  viz_full,
            "grade_summary": viz_grade_summary,
        },
    }


def _pick_worse(a: Dict, b: Dict) -> Dict:
    """Return the dict with the lower 'score' (worse condition)."""
    return a if a["score"] <= b["score"] else b
