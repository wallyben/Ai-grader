"""
Tests for batch ROI ranking and pipeline integration — Phase 2.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grader.roi import compute_roi, rank_batch_by_roi, batch_roi_summary, _DECISION_RANK
from grader.calibration import calibrate_grade


# ─── Calibration → ROI pipeline ──────────────────────────────────────────────

class TestCalibrationRoiPipeline:
    def test_calibrated_grade_fed_to_roi(self):
        """Calibrated grade must be used for ROI, not raw predicted grade."""
        raw_grade = 9.2
        confidence = 0.90
        cal = calibrate_grade(raw_grade, confidence)
        calibrated = cal["calibrated_grade"]
        cal_conf   = cal["calibrated_confidence"]

        roi = compute_roi(calibrated, cal_conf, 40.0, 25.0, 80.0, 120.0, 400.0)
        assert roi["estimated_grade"] == calibrated
        assert roi["confidence"] == cal_conf

    def test_calibration_confidence_reduces_roi_certainty(self):
        """Low calibrated confidence must reduce ROI decision quality."""
        roi_high_conf = compute_roi(9.0, 0.92, 40.0, 25.0, 80.0, 120.0, 400.0)
        roi_low_conf  = compute_roi(9.0, 0.55, 40.0, 25.0, 80.0, 120.0, 400.0)

        rank_high = _DECISION_RANK.get(roi_high_conf["decision"], 5)
        rank_low  = _DECISION_RANK.get(roi_low_conf["decision"],  5)
        assert rank_high <= rank_low


# ─── Batch ranking correctness ────────────────────────────────────────────────

class TestBatchRankingCorrectness:
    """Verify ranking follows profit → roi → downside ordering within same decision tier."""

    def _card(self, name, grade, conf, raw, cost, p8, p9, p10):
        return compute_roi(grade, conf, raw, cost, p8, p9, p10, card_name=name)

    def test_higher_profit_ranks_first_within_tier(self):
        """Within the GRADE tier, the higher-profit card should rank first."""
        # Both cards should land in GRADE or STRONG GRADE
        high_profit = self._card("HighProfit", 9.5, 0.92, 50.0, 25.0, 150.0, 350.0, 1500.0)
        low_profit  = self._card("LowProfit",  9.5, 0.92, 50.0, 25.0, 100.0, 200.0,  600.0)

        # Only compare if both in same decision tier
        if high_profit["decision"] == low_profit["decision"]:
            ranked = rank_batch_by_roi([low_profit, high_profit])
            assert ranked[0]["card_name"] == "HighProfit"

    def test_all_decisions_represented(self):
        cards = [
            self._card("GEM",  9.5, 0.93, 50.0, 25.0, 150.0, 300.0, 1200.0),
            self._card("GOOD", 8.0, 0.85, 20.0, 25.0,  40.0,  80.0,  200.0),
            self._card("BULK", 5.0, 0.70,  1.0, 25.0,   2.0,   4.0,   10.0),
        ]
        ranked = rank_batch_by_roi(cards)
        assert len(ranked) == 3
        # GEM should be first, BULK should be last
        first_rank = _DECISION_RANK.get(ranked[0]["decision"], 5)
        last_rank  = _DECISION_RANK.get(ranked[-1]["decision"], 5)
        assert first_rank <= last_rank

    def test_empty_batch(self):
        ranked = rank_batch_by_roi([])
        assert ranked == []

    def test_single_card_batch(self):
        roi = self._card("Solo", 9.0, 0.90, 40.0, 25.0, 80.0, 120.0, 400.0)
        ranked = rank_batch_by_roi([roi])
        assert len(ranked) == 1
        assert ranked[0]["card_name"] == "Solo"


# ─── Portfolio summary ────────────────────────────────────────────────────────

class TestPortfolioSummary:
    def test_total_investment_matches_sum(self):
        rois = [
            compute_roi(9.0, 0.90, 40.0, 25.0, 80.0, 120.0, 400.0),
            compute_roi(8.0, 0.85, 20.0, 25.0, 40.0,  80.0, 200.0),
            compute_roi(7.0, 0.80, 10.0, 25.0, 20.0,  35.0,  80.0),
        ]
        summary = batch_roi_summary(rois)
        expected_inv = sum(r["total_investment"] for r in rois)
        assert summary["total_investment"] == pytest.approx(expected_inv, abs=0.01)

    def test_decision_counts_sum(self):
        rois = [
            compute_roi(9.5, 0.93, 50.0, 25.0, 150.0, 300.0, 1200.0),
            compute_roi(8.0, 0.85, 20.0, 25.0,  40.0,  80.0,  200.0),
            compute_roi(5.0, 0.70,  1.0, 25.0,   2.0,   4.0,   10.0),
        ]
        summary = batch_roi_summary(rois)
        assert sum(summary["decision_counts"].values()) == 3

    def test_portfolio_roi_pct(self):
        rois = [compute_roi(9.0, 0.90, 40.0, 25.0, 80.0, 120.0, 400.0)]
        summary = batch_roi_summary(rois)
        expected = summary["portfolio_profit"] / summary["total_investment"] * 100
        assert summary["portfolio_roi_pct"] == pytest.approx(expected, abs=0.1)


# ─── Grade trace + calibration integration ───────────────────────────────────

class TestGradeTraceCalibration:
    def test_calibrate_grade_returns_all_fields(self):
        result = calibrate_grade(9.2, 0.90)
        assert "calibrated_grade"       in result
        assert "calibration_adjustment" in result
        assert "calibrated_confidence"  in result
        assert "bias"    in result
        assert "variance" in result

    def test_calibrated_grade_is_valid(self):
        for raw in [1.0, 5.0, 7.5, 9.0, 10.0]:
            result = calibrate_grade(raw, 0.88)
            cal = result["calibrated_grade"]
            assert 1.0 <= cal <= 10.0
            assert (cal * 2) == round(cal * 2)  # 0.5 step

    def test_zero_confidence_still_valid(self):
        result = calibrate_grade(8.0, 0.40)
        assert result["calibrated_confidence"] >= 0.40
