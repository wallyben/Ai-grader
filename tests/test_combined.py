"""Tests for combined front/back scoring and the full grade_card_full pipeline."""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grader.combined_scoring import combine_front_back
from grader.profiles  import apply_profile_to_analysis, get_profile
from grader.centering import compute_centering
from grader.edges     import analyze_edges
from grader.corners   import analyze_corners
from grader.surface   import analyze_surface
from grader           import grade_card_full
from tests.conftest   import make_clean_card, make_damaged_card


def _analysis(img, profile_name="tcg_generic"):
    p = get_profile(profile_name)
    raw = {
        "centering": compute_centering(img),
        "edges":     analyze_edges(img),
        "corners":   analyze_corners(img),
        "surface":   analyze_surface(img),
    }
    return apply_profile_to_analysis(raw, p)


class TestCombineFrontBackStructure:
    def test_front_only_returns_correct_keys(self):
        front = _analysis(make_clean_card())
        p     = get_profile("tcg_generic")
        r     = combine_front_back(front, None, p)
        for key in ["centering", "edges", "corners", "surface", "metadata"]:
            assert key in r

    def test_front_only_metadata_mode(self):
        front = _analysis(make_clean_card())
        p     = get_profile("tcg_generic")
        r     = combine_front_back(front, None, p)
        assert r["metadata"]["mode"] == "front_only"
        assert r["metadata"]["front_weight"] == 1.0
        assert r["metadata"]["back_weight"]  == 0.0

    def test_front_back_combined_metadata_mode(self):
        front = _analysis(make_clean_card())
        back  = _analysis(make_clean_card(seed=99))
        p     = get_profile("tcg_generic")
        r     = combine_front_back(front, back, p)
        assert r["metadata"]["mode"] == "front_back_combined"
        fw = p["scoring"]["front_back_weight"]["front"]
        bw = p["scoring"]["front_back_weight"]["back"]
        assert abs(r["metadata"]["front_weight"] - fw) < 1e-9
        assert abs(r["metadata"]["back_weight"]  - bw) < 1e-9

    def test_scores_in_valid_range(self):
        front = _analysis(make_clean_card())
        back  = _analysis(make_damaged_card())
        p     = get_profile("tcg_generic")
        r     = combine_front_back(front, back, p)
        for key in ["centering", "edges", "corners", "surface"]:
            assert 1.0 <= r[key]["score"] <= 10.0


class TestCombineEdgeLogic:
    def test_combined_edges_take_worst_side(self):
        """When back has worse edges, combined should reflect that."""
        front = _analysis(make_clean_card())
        back  = _analysis(make_damaged_card())   # has edge chipping
        p     = get_profile("tcg_generic")
        r     = combine_front_back(front, back, p)
        # Combined edge score should be <= clean front edge score
        assert r["edges"]["score"] <= front["edges"]["score"] + 0.5

    def test_all_four_sides_present_in_combined(self):
        front = _analysis(make_clean_card())
        back  = _analysis(make_damaged_card())
        p     = get_profile("tcg_generic")
        r     = combine_front_back(front, back, p)
        for side in ["top", "bottom", "left", "right"]:
            assert side in r["edges"]["sides"]

    def test_combined_sides_have_from_tag(self):
        front = _analysis(make_clean_card())
        back  = _analysis(make_damaged_card())
        p     = get_profile("tcg_generic")
        r     = combine_front_back(front, back, p)
        for side_data in r["edges"]["sides"].values():
            assert "from" in side_data
            assert side_data["from"] in {"front", "back"}


class TestCombineCornerLogic:
    def test_all_four_corners_present(self):
        front = _analysis(make_clean_card())
        back  = _analysis(make_damaged_card())
        p     = get_profile("tcg_generic")
        r     = combine_front_back(front, back, p)
        for pos in ["top_left", "top_right", "bottom_left", "bottom_right"]:
            assert pos in r["corners"]["corners"]

    def test_combined_corners_take_worst(self):
        front = _analysis(make_clean_card())
        back  = _analysis(make_damaged_card())   # has whitened corners
        p     = get_profile("tcg_generic")
        r     = combine_front_back(front, back, p)
        assert r["corners"]["score"] <= front["corners"]["score"] + 0.5


class TestCombineCenteringLogic:
    def test_combined_centering_uses_worse(self):
        front = _analysis(make_clean_card())
        back  = _analysis(make_damaged_card())
        p     = get_profile("tcg_generic")
        r     = combine_front_back(front, back, p)
        # Combined centering score = worse of front/back
        expected = min(front["centering"]["score"], back["centering"]["score"])
        assert abs(r["centering"]["score"] - expected) < 0.1

    def test_per_side_ratios_stored_when_back_present(self):
        front = _analysis(make_clean_card())
        back  = _analysis(make_clean_card(seed=55))
        p     = get_profile("tcg_generic")
        r     = combine_front_back(front, back, p)
        for key in ["front_lr", "front_tb", "back_lr", "back_tb"]:
            assert key in r["centering"]


class TestCombineSurfaceLogic:
    def test_combined_surface_score_is_weighted(self):
        front = _analysis(make_clean_card())
        back  = _analysis(make_damaged_card())
        p     = get_profile("tcg_generic")
        r     = combine_front_back(front, back, p)
        fw = p["scoring"]["front_back_weight"]["front"]
        bw = p["scoring"]["front_back_weight"]["back"]
        expected_approx = (front["surface"]["score"] * fw +
                            back["surface"]["score"] * bw)
        assert abs(r["surface"]["score"] - expected_approx) < 1.0

    def test_combined_flags_merge_both_sides(self):
        front = _analysis(make_clean_card())
        back  = _analysis(make_damaged_card())
        p     = get_profile("tcg_generic")
        r     = combine_front_back(front, back, p)
        # All back flags should appear in combined flags
        for flag in back["surface"]["risk_flags"]:
            assert flag in r["surface"]["risk_flags"]


class TestGradeCardFull:
    def test_returns_all_top_level_keys(self):
        img = make_clean_card()
        r   = grade_card_full(img)
        for key in ["grade_result", "grade_trace", "analysis", "front_analysis",
                    "back_analysis", "defects", "quality_gate", "profile_used",
                    "visualizations"]:
            assert key in r

    def test_grade_in_valid_range(self):
        img = make_clean_card()
        r   = grade_card_full(img)
        assert 1.0 <= r["grade_result"]["grade"] <= 10.0

    def test_grade_trace_present(self):
        img   = make_clean_card()
        r     = grade_card_full(img)
        trace = r["grade_trace"]
        for key in ["sub_scores", "weights", "weighted_sum",
                    "caps_evaluated", "final_rounded_grade"]:
            assert key in trace

    def test_caps_evaluated_is_list(self):
        img   = make_clean_card()
        r     = grade_card_full(img)
        caps  = r["grade_trace"]["caps_evaluated"]
        assert isinstance(caps, list)
        for cap in caps:
            assert "rule" in cap
            assert "triggered" in cap

    def test_defects_list_is_list(self):
        img = make_clean_card()
        r   = grade_card_full(img)
        assert isinstance(r["defects"], list)

    def test_quality_gate_runs_when_requested(self):
        img = make_clean_card()
        r   = grade_card_full(img, run_quality_gate=True)
        assert r["quality_gate"] is not None
        assert "decision" in r["quality_gate"]

    def test_quality_gate_skipped_when_not_requested(self):
        img = make_clean_card()
        r   = grade_card_full(img, run_quality_gate=False)
        assert r["quality_gate"] is None

    def test_profile_applied(self):
        img = make_clean_card()
        r_generic = grade_card_full(img, profile_name="tcg_generic",  run_quality_gate=False)
        r_chrome  = grade_card_full(img, profile_name="sports_chrome", run_quality_gate=False)
        assert r_generic["profile_used"] == "tcg_generic"
        assert r_chrome["profile_used"]  == "sports_chrome"

    def test_back_image_can_lower_grade(self):
        front = make_clean_card()
        back  = make_damaged_card()
        r_f   = grade_card_full(front, run_quality_gate=False)
        r_fb  = grade_card_full(front, back, run_quality_gate=False)
        assert r_fb["grade_result"]["grade"] <= r_f["grade_result"]["grade"] + 0.5

    def test_deterministic_with_profile(self):
        img  = make_clean_card()
        r1   = grade_card_full(img, profile_name="pokemon_modern", run_quality_gate=False)
        r2   = grade_card_full(img, profile_name="pokemon_modern", run_quality_gate=False)
        assert r1["grade_result"]["grade"] == r2["grade_result"]["grade"]

    def test_all_visualizations_present(self):
        img = make_clean_card()
        r   = grade_card_full(img, run_quality_gate=False)
        for key in ["centering", "edges", "corners", "surface",
                    "full_overlay", "defect_evidence", "grade_trace", "grade_summary"]:
            assert key in r["visualizations"]

    def test_all_visualizations_are_ndarrays(self):
        img = make_clean_card()
        r   = grade_card_full(img, run_quality_gate=False)
        for k, v in r["visualizations"].items():
            assert isinstance(v, np.ndarray), f"viz['{k}'] is not ndarray"
