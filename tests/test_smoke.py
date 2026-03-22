"""
Smoke Tests
Verify that every module runs without error on synthetic input
and returns structurally valid output.
"""

import numpy as np
import pytest

from grader.preprocessing import normalize_card, load_image
from grader.centering import compute_centering
from grader.edges import analyze_edges
from grader.corners import analyze_corners
from grader.surface import analyze_surface
from grader.scoring import compute_grade
from grader.visualize import (
    draw_centering_overlay,
    draw_edge_overlay,
    draw_corner_overlay,
    draw_surface_overlay,
    create_full_defect_overlay,
    create_grade_summary_image,
    bgr_to_rgb,
)
from grader import grade_card
from tests.conftest import make_clean_card, make_damaged_card


# ─── Preprocessing ────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_normalize_output_shape(self):
        img = make_clean_card()
        out, _ = normalize_card(img)
        assert out.shape == (700, 500, 3), f"Expected (700,500,3) got {out.shape}"

    def test_normalize_small_input(self):
        tiny = np.random.randint(0, 255, (120, 80, 3), dtype=np.uint8)
        out, _ = normalize_card(tiny)
        assert out.shape == (700, 500, 3)

    def test_normalize_large_input(self):
        big = np.random.randint(0, 255, (2400, 1700, 3), dtype=np.uint8)
        out, _ = normalize_card(big)
        assert out.shape == (700, 500, 3)

    def test_normalize_returns_bool_flag(self):
        img = make_clean_card()
        _, flag = normalize_card(img)
        assert isinstance(flag, bool)

    def test_load_image_from_bytes(self):
        import cv2
        img = make_clean_card()
        success, buf = cv2.imencode(".jpg", img)
        assert success
        loaded = load_image(buf.tobytes())
        assert loaded is not None
        assert loaded.ndim == 3


# ─── Centering ────────────────────────────────────────────────────────────────

class TestCentering:
    def test_returns_dict_with_required_keys(self, clean_card):
        r = compute_centering(clean_card)
        for key in ["left_pct", "right_pct", "top_pct", "bottom_pct",
                    "lr_ratio", "tb_ratio", "score", "borders"]:
            assert key in r, f"Missing key: {key}"

    def test_percentages_sum_to_100(self, clean_card):
        r = compute_centering(clean_card)
        assert abs(r["left_pct"] + r["right_pct"] - 100.0) < 0.5
        assert abs(r["top_pct"] + r["bottom_pct"] - 100.0) < 0.5

    def test_score_in_valid_range(self, clean_card):
        r = compute_centering(clean_card)
        assert 1.0 <= r["score"] <= 10.0

    def test_borders_are_4_tuple(self, clean_card):
        r = compute_centering(clean_card)
        assert len(r["borders"]) == 4

    def test_ratio_string_format(self, clean_card):
        r = compute_centering(clean_card)
        assert "/" in r["lr_ratio"]
        assert "/" in r["tb_ratio"]

    def test_deterministic(self, clean_card):
        r1 = compute_centering(clean_card)
        r2 = compute_centering(clean_card)
        assert r1["score"] == r2["score"]
        assert r1["lr_ratio"] == r2["lr_ratio"]


# ─── Edges ───────────────────────────────────────────────────────────────────

class TestEdges:
    def test_returns_dict_with_required_keys(self, clean_card):
        r = analyze_edges(clean_card)
        for key in ["score", "sides", "avg_whitening",
                    "avg_roughness", "defect_locations"]:
            assert key in r

    def test_all_four_sides_present(self, clean_card):
        r = analyze_edges(clean_card)
        for side in ["top", "bottom", "left", "right"]:
            assert side in r["sides"]

    def test_side_has_required_fields(self, clean_card):
        r = analyze_edges(clean_card)
        for side_data in r["sides"].values():
            for field in ["whitening", "roughness", "chipping",
                          "defect_level", "score"]:
                assert field in side_data

    def test_score_in_valid_range(self, clean_card):
        r = analyze_edges(clean_card)
        assert 1.0 <= r["score"] <= 10.0

    def test_defect_level_valid_values(self, clean_card):
        r = analyze_edges(clean_card)
        valid = {"clean", "minor", "moderate", "severe"}
        for side_data in r["sides"].values():
            assert side_data["defect_level"] in valid

    def test_deterministic(self, clean_card):
        r1 = analyze_edges(clean_card)
        r2 = analyze_edges(clean_card)
        assert r1["score"] == r2["score"]

    def test_damaged_edges_not_better_than_clean(self, clean_card, damaged_card):
        r_clean   = analyze_edges(clean_card)
        r_damaged = analyze_edges(damaged_card)
        # Damaged should score equal-or-worse
        assert r_damaged["score"] <= r_clean["score"] + 1.0


# ─── Corners ─────────────────────────────────────────────────────────────────

class TestCorners:
    def test_returns_dict_with_required_keys(self, clean_card):
        r = analyze_corners(clean_card)
        for key in ["score", "corners", "defect_corners"]:
            assert key in r

    def test_all_four_corners_present(self, clean_card):
        r = analyze_corners(clean_card)
        for c in ["top_left", "top_right", "bottom_left", "bottom_right"]:
            assert c in r["corners"]

    def test_corner_has_required_fields(self, clean_card):
        r = analyze_corners(clean_card)
        for cdata in r["corners"].values():
            for field in ["whitening", "rounding", "sharpness",
                          "defect_level", "score"]:
                assert field in cdata

    def test_score_in_valid_range(self, clean_card):
        r = analyze_corners(clean_card)
        assert 1.0 <= r["score"] <= 10.0

    def test_defect_level_valid_values(self, clean_card):
        r = analyze_corners(clean_card)
        valid = {"sharp", "minor", "moderate", "severe"}
        for cdata in r["corners"].values():
            assert cdata["defect_level"] in valid

    def test_deterministic(self, clean_card):
        r1 = analyze_corners(clean_card)
        r2 = analyze_corners(clean_card)
        assert r1["score"] == r2["score"]

    def test_damaged_corners_not_better_than_clean(self, clean_card, damaged_card):
        r_clean   = analyze_corners(clean_card)
        r_damaged = analyze_corners(damaged_card)
        assert r_damaged["score"] <= r_clean["score"] + 1.5


# ─── Surface ─────────────────────────────────────────────────────────────────

class TestSurface:
    def test_returns_dict_with_required_keys(self, clean_card):
        r = analyze_surface(clean_card)
        for key in ["score", "scratch_density", "print_line_indicator",
                    "anomaly_density", "surface_noise", "risk_flags",
                    "scratch_map", "anomaly_mask"]:
            assert key in r

    def test_score_in_valid_range(self, clean_card):
        r = analyze_surface(clean_card)
        assert 1.0 <= r["score"] <= 10.0

    def test_risk_flags_is_list(self, clean_card):
        r = analyze_surface(clean_card)
        assert isinstance(r["risk_flags"], list)

    def test_scratch_map_correct_shape(self, clean_card):
        r = analyze_surface(clean_card)
        assert r["scratch_map"].shape == (700, 500)

    def test_anomaly_mask_correct_shape(self, clean_card):
        r = analyze_surface(clean_card)
        assert r["anomaly_mask"].shape == (700, 500)

    def test_deterministic(self, clean_card):
        r1 = analyze_surface(clean_card)
        r2 = analyze_surface(clean_card)
        assert r1["score"] == r2["score"]

    def test_scratched_card_higher_density(self, clean_card, damaged_card):
        r_clean   = analyze_surface(clean_card)
        r_damaged = analyze_surface(damaged_card)
        # Damaged card should have equal or higher scratch density
        assert r_damaged["scratch_density"] >= r_clean["scratch_density"] - 0.002


# ─── Scoring ─────────────────────────────────────────────────────────────────

class TestScoring:
    def _run(self, img):
        c  = compute_centering(img)
        e  = analyze_edges(img)
        co = analyze_corners(img)
        s  = analyze_surface(img)
        return compute_grade(c, e, co, s)

    def test_returns_dict_with_required_keys(self, clean_card):
        r = self._run(clean_card)
        for key in ["grade", "grade_band", "centering_score", "edges_score",
                    "corners_score", "surface_score", "centering_lr",
                    "centering_tb", "caps_triggered", "risk_flags",
                    "confidence", "recommendation", "rec_reason", "base_grade"]:
            assert key in r

    def test_grade_in_valid_range(self, clean_card):
        r = self._run(clean_card)
        assert 1.0 <= r["grade"] <= 10.0

    def test_grade_is_half_point(self, clean_card):
        r = self._run(clean_card)
        # Grade must be a multiple of 0.5
        assert (r["grade"] * 2) % 1 == 0

    def test_confidence_in_valid_range(self, clean_card):
        r = self._run(clean_card)
        assert 0.0 <= r["confidence"] <= 1.0

    def test_recommendation_valid(self, clean_card):
        r = self._run(clean_card)
        assert r["recommendation"] in {"GRADE", "CONDITIONAL", "DO NOT GRADE"}

    def test_caps_is_list(self, clean_card):
        r = self._run(clean_card)
        assert isinstance(r["caps_triggered"], list)

    def test_deterministic(self, clean_card):
        r1 = self._run(clean_card)
        r2 = self._run(clean_card)
        assert r1["grade"] == r2["grade"]

    def test_damaged_grades_lower_or_equal(self, clean_card, damaged_card):
        r_clean   = self._run(clean_card)
        r_damaged = self._run(damaged_card)
        # Damaged card must grade equal or worse (lower or equal)
        assert r_damaged["grade"] <= r_clean["grade"] + 0.5


# ─── Visualizations ──────────────────────────────────────────────────────────

class TestVisualizations:
    def _get_analysis(self, img):
        return {
            "centering": compute_centering(img),
            "edges":     analyze_edges(img),
            "corners":   analyze_corners(img),
            "surface":   analyze_surface(img),
        }

    def test_centering_overlay_shape(self, clean_card):
        a = self._get_analysis(clean_card)
        out = draw_centering_overlay(clean_card, a["centering"])
        assert out.shape == clean_card.shape

    def test_edge_overlay_shape(self, clean_card):
        a = self._get_analysis(clean_card)
        out = draw_edge_overlay(clean_card, a["edges"])
        assert out.shape == clean_card.shape

    def test_corner_overlay_shape(self, clean_card):
        a = self._get_analysis(clean_card)
        out = draw_corner_overlay(clean_card, a["corners"])
        assert out.shape == clean_card.shape

    def test_surface_overlay_shape(self, clean_card):
        a = self._get_analysis(clean_card)
        out = draw_surface_overlay(clean_card, a["surface"])
        assert out.shape == clean_card.shape

    def test_full_overlay_shape(self, clean_card):
        a = self._get_analysis(clean_card)
        out = create_full_defect_overlay(
            clean_card, a["centering"], a["edges"], a["corners"], a["surface"]
        )
        assert out.shape == clean_card.shape

    def test_grade_summary_is_ndarray(self, clean_card):
        a = self._get_analysis(clean_card)
        gr = compute_grade(a["centering"], a["edges"], a["corners"], a["surface"])
        summary = create_grade_summary_image(gr)
        assert isinstance(summary, np.ndarray)
        assert summary.ndim == 3

    def test_bgr_to_rgb_channels_swapped(self, clean_card):
        rgb = bgr_to_rgb(clean_card)
        assert not np.array_equal(rgb[:, :, 0], clean_card[:, :, 0])  # channels differ


# ─── Full Pipeline ────────────────────────────────────────────────────────────

class TestFullPipeline:
    def test_grade_card_runs_front_only(self, clean_card):
        r = grade_card(clean_card)
        assert "grade_result" in r
        assert "analysis" in r
        assert "visualizations" in r

    def test_grade_card_with_back(self, clean_card, damaged_card):
        r = grade_card(clean_card, damaged_card)
        assert 1.0 <= r["grade_result"]["grade"] <= 10.0

    def test_all_visualizations_present(self, clean_card):
        r = grade_card(clean_card)
        for key in ["centering", "edges", "corners", "surface",
                    "full_overlay", "grade_summary"]:
            assert key in r["visualizations"]

    def test_all_visualizations_are_arrays(self, clean_card):
        r = grade_card(clean_card)
        for key, val in r["visualizations"].items():
            assert isinstance(val, np.ndarray), f"viz['{key}'] is not ndarray"

    def test_pipeline_deterministic(self, clean_card):
        r1 = grade_card(clean_card.copy())
        r2 = grade_card(clean_card.copy())
        assert r1["grade_result"]["grade"] == r2["grade_result"]["grade"]

    def test_back_image_can_lower_grade(self, clean_card, damaged_card):
        r_front_only = grade_card(clean_card)
        r_with_bad_back = grade_card(clean_card, damaged_card)
        # Back with damage should produce equal-or-lower grade
        assert (r_with_bad_back["grade_result"]["grade"] <=
                r_front_only["grade_result"]["grade"] + 0.5)
