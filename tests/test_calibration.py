"""
Tests for calibration engine — Phase 2.
All tests are deterministic and require no external data.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grader.calibration import (
    CalibrationEngine,
    calibrate_grade,
    compute_calibration_stats,
    DEFAULT_CALIBRATION_DATA,
)


# ─── CalibrationEngine construction ──────────────────────────────────────────

class TestCalibrationEngine:
    def test_default_construction(self):
        engine = CalibrationEngine()
        assert engine.bias is not None
        assert engine.variance >= 0
        assert engine.std_dev >= 0

    def test_custom_data(self):
        data = [
            {"predicted_grade": 9.0, "actual_grade": 9},
            {"predicted_grade": 8.0, "actual_grade": 8},
            {"predicted_grade": 7.0, "actual_grade": 7},
        ]
        engine = CalibrationEngine(data)
        assert engine.bias == pytest.approx(0.0, abs=1e-6)
        assert engine.variance == pytest.approx(0.0, abs=1e-6)

    def test_biased_data(self):
        # System always over-grades by 0.5
        data = [
            {"predicted_grade": 9.5, "actual_grade": 9},
            {"predicted_grade": 8.5, "actual_grade": 8},
            {"predicted_grade": 7.5, "actual_grade": 7},
        ]
        engine = CalibrationEngine(data)
        assert engine.bias == pytest.approx(0.5, abs=1e-4)

    def test_summary_keys(self):
        engine = CalibrationEngine()
        s = engine.summary()
        for key in ("n_samples", "bias", "variance", "std_dev", "correction_points"):
            assert key in s

    def test_summary_n_samples(self):
        engine = CalibrationEngine()
        assert engine.summary()["n_samples"] == len(DEFAULT_CALIBRATION_DATA)


# ─── calibrate() output contract ─────────────────────────────────────────────

class TestCalibrateOutput:
    def setup_method(self):
        self.engine = CalibrationEngine()

    def test_output_keys(self):
        result = self.engine.calibrate(9.0, 0.90)
        required = {
            "calibrated_grade", "raw_calibrated", "calibration_adjustment",
            "calibrated_confidence", "bias", "variance", "std_dev",
        }
        assert required.issubset(result.keys())

    def test_calibrated_grade_range(self):
        for grade in [1.0, 4.0, 7.0, 8.5, 9.0, 10.0]:
            result = self.engine.calibrate(grade, 0.85)
            assert 1.0 <= result["calibrated_grade"] <= 10.0

    def test_calibrated_grade_half_steps(self):
        """Calibrated grade must be on 0.5 grid (PSA granularity)."""
        for grade in [5.0, 7.0, 8.5, 9.0]:
            result = self.engine.calibrate(grade, 0.88)
            cal = result["calibrated_grade"]
            assert (cal * 2) == round(cal * 2), f"{cal} is not a 0.5 step"

    def test_confidence_not_increased(self):
        """Calibration can only decrease or maintain confidence, never inflate."""
        result = self.engine.calibrate(9.0, 0.94)
        assert result["calibrated_confidence"] <= 0.94

    def test_confidence_lower_bound(self):
        result = self.engine.calibrate(5.0, 0.40)
        assert result["calibrated_confidence"] >= 0.40

    def test_adjustment_matches_delta(self):
        result = self.engine.calibrate(9.0, 0.90)
        expected_adj = result["calibrated_grade"] - 9.0
        assert result["calibration_adjustment"] == pytest.approx(expected_adj, abs=0.01)

    def test_deterministic(self):
        r1 = self.engine.calibrate(8.5, 0.88)
        r2 = self.engine.calibrate(8.5, 0.88)
        assert r1 == r2


# ─── Interpolation correctness ─────────────────────────────────────────────

class TestInterpolation:
    def test_exact_match(self):
        data = [
            {"predicted_grade": 9.0, "actual_grade": 9},
            {"predicted_grade": 8.0, "actual_grade": 7},
        ]
        engine = CalibrationEngine(data)
        result = engine.calibrate(9.0, 0.90)
        assert result["calibrated_grade"] == 9.0

    def test_clamp_below_range(self):
        data = [
            {"predicted_grade": 5.0, "actual_grade": 5},
            {"predicted_grade": 9.0, "actual_grade": 9},
        ]
        engine = CalibrationEngine(data)
        # Grade below min correction point → clamped to first entry
        result = engine.calibrate(1.0, 0.90)
        assert result["calibrated_grade"] >= 1.0

    def test_clamp_above_range(self):
        data = [
            {"predicted_grade": 5.0, "actual_grade": 5},
            {"predicted_grade": 9.0, "actual_grade": 9},
        ]
        engine = CalibrationEngine(data)
        result = engine.calibrate(10.0, 0.90)
        assert result["calibrated_grade"] <= 10.0


# ─── Module-level convenience functions ──────────────────────────────────────

class TestConvenienceFunctions:
    def test_calibrate_grade_default(self):
        result = calibrate_grade(8.5, 0.88)
        assert "calibrated_grade" in result
        assert 1.0 <= result["calibrated_grade"] <= 10.0

    def test_calibrate_grade_custom_data(self):
        data = [
            {"predicted_grade": 8.0, "actual_grade": 8},
            {"predicted_grade": 9.0, "actual_grade": 9},
        ]
        result = calibrate_grade(8.5, 0.90, custom_data=data)
        assert "calibrated_grade" in result

    def test_compute_calibration_stats(self):
        data = [
            {"predicted_grade": 9.0, "actual_grade": 9},
            {"predicted_grade": 8.0, "actual_grade": 8},
        ]
        stats = compute_calibration_stats(data)
        assert "bias" in stats
        assert "variance" in stats
        assert stats["n_samples"] == 2

    def test_zero_bias_perfect_calibration(self):
        data = [{"predicted_grade": float(g), "actual_grade": g} for g in range(1, 11)]
        stats = compute_calibration_stats(data)
        assert stats["bias"] == pytest.approx(0.0, abs=1e-6)
        assert stats["variance"] == pytest.approx(0.0, abs=1e-6)
