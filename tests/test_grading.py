"""
Grading Logic Tests
Validates the deterministic rule engine:
  - Hard cap application
  - Grade band mapping
  - Recommendation engine
  - Score weighting
"""

import pytest
import numpy as np

from grader.scoring import (
    compute_grade,
    apply_hard_caps,
    _grade_band,
    _recommendation,
    CATEGORY_WEIGHTS,
)
from grader.centering import compute_centering, _score_centering
from tests.conftest import make_clean_card, make_damaged_card


# ─── Centering score unit tests ──────────────────────────────────────────────

class TestCenteringScoreLogic:
    def test_perfect_center_scores_10(self):
        assert _score_centering(50, 50, 50, 50) == 10.0

    def test_55_45_scores_9_or_better(self):
        assert _score_centering(55, 45, 50, 50) >= 9.0

    def test_60_40_scores_8_or_better(self):
        assert _score_centering(60, 40, 50, 50) >= 8.0

    def test_70_30_caps_at_7_or_less(self):
        assert _score_centering(70, 30, 50, 50) <= 7.0

    def test_80_20_scores_below_5(self):
        assert _score_centering(80, 20, 50, 50) <= 5.0

    def test_worst_axis_drives_score(self):
        # LR perfect but TB terrible
        s1 = _score_centering(50, 50, 75, 25)
        s2 = _score_centering(75, 25, 50, 50)
        assert s1 == s2  # symmetric


# ─── Hard cap logic ──────────────────────────────────────────────────────────

def _mock_centering(score: float) -> dict:
    return {
        "score": score, "left_pct": 50, "right_pct": 50,
        "top_pct": 50, "bottom_pct": 50,
        "lr_ratio": "50/50", "tb_ratio": "50/50",
        "borders": (56, 644, 40, 460),
    }

def _mock_edges(level: str = "clean") -> dict:
    side = {"whitening": 0, "roughness": 0, "chipping": 0,
            "defect_level": level, "score": 10.0}
    return {
        "score": 10.0,
        "sides": {s: side for s in ["top", "bottom", "left", "right"]},
        "avg_whitening": 0.0, "avg_roughness": 0.0, "avg_chipping": 0.0,
        "defect_locations": [] if level == "clean" else ["top"],
    }

def _mock_corners(level: str = "sharp") -> dict:
    corner = {"whitening": 0, "rounding": 0, "sharpness": 30,
              "defect_level": level, "score": 10.0}
    return {
        "score": 10.0,
        "corners": {c: corner for c in
                    ["top_left", "top_right", "bottom_left", "bottom_right"]},
        "defect_corners": [] if level == "sharp" else ["top_left"],
    }

def _mock_surface(flags: list = None) -> dict:
    return {
        "score": 10.0,
        "scratch_density": 0.0,
        "print_line_indicator": 0.0,
        "anomaly_density": 0.0,
        "surface_noise": 0.0,
        "scratch_map": np.zeros((700, 500), dtype=np.uint8),
        "anomaly_mask": np.zeros((700, 500), dtype=np.uint8),
        "risk_flags": flags or [],
    }


class TestHardCaps:
    def test_no_caps_on_perfect_card(self):
        c = _mock_centering(10.0)
        e = _mock_edges("clean")
        co = _mock_corners("sharp")
        s = _mock_surface([])
        final, caps = apply_hard_caps(10.0, c, e, co, s)
        assert final == 10.0
        assert caps == []

    def test_severe_corner_caps_at_5(self):
        c  = _mock_centering(10.0)
        e  = _mock_edges("clean")
        co = _mock_corners("severe")
        s  = _mock_surface([])
        final, caps = apply_hard_caps(10.0, c, e, co, s)
        assert final <= 5.0
        assert any("corner" in cap for cap in caps)

    def test_moderate_corner_caps_at_7(self):
        c  = _mock_centering(10.0)
        e  = _mock_edges("clean")
        co = _mock_corners("moderate")
        s  = _mock_surface([])
        final, caps = apply_hard_caps(10.0, c, e, co, s)
        assert final <= 7.0

    def test_severe_edge_caps_at_6(self):
        c  = _mock_centering(10.0)
        e  = _mock_edges("severe")
        co = _mock_corners("sharp")
        s  = _mock_surface([])
        final, caps = apply_hard_caps(10.0, c, e, co, s)
        assert final <= 6.0

    def test_moderate_edge_caps_at_8(self):
        c  = _mock_centering(10.0)
        e  = _mock_edges("moderate")
        co = _mock_corners("sharp")
        s  = _mock_surface([])
        final, caps = apply_hard_caps(10.0, c, e, co, s)
        assert final <= 8.0

    def test_scratch_flag_caps_at_8(self):
        c  = _mock_centering(10.0)
        e  = _mock_edges("clean")
        co = _mock_corners("sharp")
        s  = _mock_surface(["light scratches detected"])
        final, caps = apply_hard_caps(10.0, c, e, co, s)
        assert final <= 8.0

    def test_dent_flag_caps_at_6(self):
        c  = _mock_centering(10.0)
        e  = _mock_edges("clean")
        co = _mock_corners("sharp")
        s  = _mock_surface(["possible dent or crease detected"])
        final, caps = apply_hard_caps(10.0, c, e, co, s)
        assert final <= 6.0

    def test_multiple_caps_use_strictest(self):
        # Severe corner (cap 5) + moderate edge (cap 8) → should be 5
        c  = _mock_centering(10.0)
        e  = _mock_edges("moderate")
        co = _mock_corners("severe")
        s  = _mock_surface([])
        final, caps = apply_hard_caps(10.0, c, e, co, s)
        assert final <= 5.0

    def test_centering_off_center_caps_at_8(self):
        c  = _mock_centering(7.5)   # off-center triggers ≤8 cap
        e  = _mock_edges("clean")
        co = _mock_corners("sharp")
        s  = _mock_surface([])
        final, caps = apply_hard_caps(10.0, c, e, co, s)
        assert final <= 8.0

    def test_grade_never_exceeds_base(self):
        c  = _mock_centering(10.0)
        e  = _mock_edges("clean")
        co = _mock_corners("sharp")
        s  = _mock_surface([])
        for base in [5.0, 7.5, 8.5, 10.0]:
            final, _ = apply_hard_caps(base, c, e, co, s)
            assert final <= base


# ─── Grade band mapping ──────────────────────────────────────────────────────

class TestGradeBand:
    def test_grade_10_is_gem_mint(self):
        assert "GEM MINT" in _grade_band(10.0)

    def test_grade_9_is_mint(self):
        assert "MINT" in _grade_band(9.0)

    def test_grade_8_is_nm_mt(self):
        assert "NM" in _grade_band(8.0) or "MT" in _grade_band(8.0)

    def test_grade_7_is_near_mint(self):
        assert "NEAR MINT" in _grade_band(7.0)

    def test_grade_1_is_poor(self):
        assert "POOR" in _grade_band(1.0)

    def test_all_standard_grades_have_label(self):
        for g in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
            label = _grade_band(g)
            assert isinstance(label, str) and len(label) > 3


# ─── Recommendation engine ───────────────────────────────────────────────────

class TestRecommendation:
    def test_grade_9_or_above_recommends_grade(self):
        rec, _ = _recommendation(9.0, 0.9, [])
        assert rec == "GRADE"

    def test_grade_10_recommends_grade(self):
        rec, _ = _recommendation(10.0, 0.95, [])
        assert rec == "GRADE"

    def test_grade_8_is_conditional(self):
        rec, _ = _recommendation(8.0, 0.85, [])
        assert rec == "CONDITIONAL"

    def test_grade_7_is_conditional(self):
        rec, _ = _recommendation(7.0, 0.80, [])
        assert rec == "CONDITIONAL"

    def test_grade_6_do_not_grade(self):
        rec, _ = _recommendation(6.0, 0.80, [])
        assert rec == "DO NOT GRADE"

    def test_low_confidence_noted_in_reason(self):
        _, reason = _recommendation(9.0, 0.55, [])
        assert "confidence" in reason.lower() or "inspect" in reason.lower()

    def test_caps_appear_in_reason(self):
        _, reason = _recommendation(7.0, 0.80, ["corner wear — top left"])
        assert "corner" in reason.lower() or "cap" in reason.lower()


# ─── Full compute_grade integration ─────────────────────────────────────────

class TestComputeGrade:
    def test_perfect_mocks_produce_high_grade(self):
        c  = _mock_centering(10.0)
        e  = _mock_edges("clean")
        co = _mock_corners("sharp")
        s  = _mock_surface([])
        # Override individual scores to 10
        e["score"]  = 10.0
        co["score"] = 10.0
        s["score"]  = 10.0

        r = compute_grade(c, e, co, s)
        assert r["grade"] >= 9.5

    def test_category_weights_sum_to_1(self):
        total = sum(CATEGORY_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_grade_is_half_point_increment(self):
        img = make_clean_card()
        from grader import analyze_edges, analyze_corners
        from grader.centering import compute_centering
        from grader.surface import analyze_surface
        c  = compute_centering(img)
        e  = analyze_edges(img)
        co = analyze_corners(img)
        s  = analyze_surface(img)
        r  = compute_grade(c, e, co, s)
        assert (r["grade"] * 2) % 1 == 0

    def test_base_grade_field_present(self):
        c  = _mock_centering(10.0)
        e  = _mock_edges("clean")
        co = _mock_corners("sharp")
        s  = _mock_surface([])
        r = compute_grade(c, e, co, s)
        assert "base_grade" in r
        assert isinstance(r["base_grade"], float)

    def test_final_grade_le_base_grade(self):
        """After caps, final grade should never exceed base."""
        c  = _mock_centering(10.0)
        e  = _mock_edges("moderate")
        co = _mock_corners("moderate")
        s  = _mock_surface(["light scratches detected"])
        r = compute_grade(c, e, co, s)
        assert r["grade"] <= r["base_grade"] + 0.5  # 0.5 tolerance for rounding
