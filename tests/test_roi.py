"""
Tests for ROI engine — Phase 2.
All tests are deterministic and require no external data.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grader.roi import (
    compute_roi,
    rank_batch_by_roi,
    batch_roi_summary,
    _grade_probabilities,
    _make_decision,
)


# ─── compute_roi output contract ─────────────────────────────────────────────

class TestComputeRoiContract:
    def _roi(self, **kwargs):
        defaults = dict(
            estimated_grade=9.0, confidence=0.90,
            raw_price=40.0, grading_cost=25.0,
            psa_8_price=80.0, psa_9_price=120.0, psa_10_price=400.0,
            card_name="Test Card",
        )
        defaults.update(kwargs)
        return compute_roi(**defaults)

    def test_required_keys(self):
        result = self._roi()
        required = {
            "card_name", "estimated_grade", "confidence",
            "raw_value", "psa_8_value", "psa_9_value", "psa_10_value",
            "grading_cost", "total_investment",
            "grade_probabilities", "expected_value", "profit",
            "roi_percent", "downside_loss", "downside_risk_pct",
            "decision", "confidence_adjusted", "best_case_profit",
            "worst_case_profit", "prices_estimated",
        }
        assert required.issubset(result.keys())

    def test_grade_probabilities_keys(self):
        result = self._roi()
        probs = result["grade_probabilities"]
        assert set(probs.keys()) == {"psa_10", "psa_9", "psa_8", "below_psa_8"}

    def test_probabilities_sum_to_one(self):
        for grade in [6.0, 8.0, 9.0, 9.5, 10.0]:
            result = self._roi(estimated_grade=grade)
            probs = result["grade_probabilities"]
            total = sum(probs.values())
            assert total == pytest.approx(1.0, abs=1e-6), \
                f"grade={grade}: probs sum={total}"

    def test_deterministic(self):
        r1 = self._roi()
        r2 = self._roi()
        assert r1["expected_value"] == r2["expected_value"]
        assert r1["roi_percent"]    == r2["roi_percent"]
        assert r1["decision"]       == r2["decision"]

    def test_total_investment(self):
        result = self._roi(raw_price=40.0, grading_cost=25.0)
        assert result["total_investment"] == pytest.approx(65.0)

    def test_profit_formula(self):
        result = self._roi()
        expected_profit = result["expected_value"] - result["total_investment"]
        assert result["profit"] == pytest.approx(expected_profit, abs=0.01)

    def test_roi_percent_formula(self):
        result = self._roi()
        expected_roi = result["profit"] / result["total_investment"] * 100
        assert result["roi_percent"] == pytest.approx(expected_roi, abs=0.1)

    def test_downside_loss_equals_grading_cost(self):
        """Downside loss = grading cost (worst case: sell raw, lose grading fee)."""
        result = self._roi(grading_cost=25.0)
        assert result["downside_loss"] == pytest.approx(25.0)

    def test_negative_raw_price_raises(self):
        with pytest.raises(ValueError):
            compute_roi(9.0, 0.90, -10.0, 25.0)

    def test_negative_grading_cost_raises(self):
        with pytest.raises(ValueError):
            compute_roi(9.0, 0.90, 40.0, -5.0)

    def test_card_name_stored(self):
        result = self._roi(card_name="Charizard Holo")
        assert result["card_name"] == "Charizard Holo"

    def test_unknown_card_name_default(self):
        result = compute_roi(9.0, 0.90, 40.0, 25.0)
        assert result["card_name"] == "Unknown"


# ─── Price estimation ─────────────────────────────────────────────────────────

class TestPriceEstimation:
    def test_prices_estimated_flag_true(self):
        result = compute_roi(9.0, 0.90, 40.0, 25.0)  # no psa prices supplied
        assert result["prices_estimated"] is True

    def test_prices_estimated_flag_false(self):
        result = compute_roi(9.0, 0.90, 40.0, 25.0,
                             psa_8_price=80.0, psa_9_price=120.0, psa_10_price=400.0)
        assert result["prices_estimated"] is False

    def test_estimated_prices_are_positive(self):
        result = compute_roi(9.0, 0.90, 40.0, 25.0)
        assert result["psa_8_value"]  > 0
        assert result["psa_9_value"]  > 0
        assert result["psa_10_value"] > 0


# ─── Grade probability model ──────────────────────────────────────────────────

class TestGradeProbabilities:
    def test_sum_to_one(self):
        for grade in [5.0, 7.0, 8.0, 9.0, 9.5, 10.0]:
            for conf in [0.65, 0.80, 0.90, 0.94]:
                probs = _grade_probabilities(grade, conf)
                total = sum(probs.values())
                assert total == pytest.approx(1.0, abs=1e-6)

    def test_all_non_negative(self):
        probs = _grade_probabilities(8.5, 0.88)
        for k, v in probs.items():
            assert v >= 0, f"{k} is negative"

    def test_gem_mint_dominates_at_high_grade(self):
        probs = _grade_probabilities(10.0, 0.95)
        assert probs["p_10"] + probs["p_9"] > 0.80

    def test_below_8_dominates_at_low_grade(self):
        probs = _grade_probabilities(4.0, 0.95)
        assert probs["p_below_8"] > 0.80

    def test_low_confidence_widens_distribution(self):
        # Use grade=8.5 which is close enough to the PSA 8 boundary for
        # low confidence to shift measurable mass into the below-8 tail.
        high_conf_probs = _grade_probabilities(8.5, 0.95)
        low_conf_probs  = _grade_probabilities(8.5, 0.65)
        # At low confidence, more mass in tails (higher p_below_8)
        assert low_conf_probs["p_below_8"] >= high_conf_probs["p_below_8"]

    def test_deterministic(self):
        p1 = _grade_probabilities(9.0, 0.90)
        p2 = _grade_probabilities(9.0, 0.90)
        assert p1 == p2


# ─── Decision engine ──────────────────────────────────────────────────────────

class TestDecisionEngine:
    def test_strong_grade(self):
        decision = _make_decision(
            roi_pct=200.0, profit=100.0, confidence=0.90,
            grade=9.0, p_below_8=0.05,
        )
        assert decision == "STRONG GRADE"

    def test_grade(self):
        decision = _make_decision(
            roi_pct=80.0, profit=40.0, confidence=0.85,
            grade=9.0, p_below_8=0.10,
        )
        assert decision == "GRADE"

    def test_conditional_low_roi(self):
        decision = _make_decision(
            roi_pct=15.0, profit=10.0, confidence=0.85,
            grade=8.0, p_below_8=0.20,
        )
        assert decision == "CONDITIONAL"

    def test_hold_raw_path(self):
        # Low confidence + slight positive profit + sub-40% ROI → HOLD RAW
        # (low confidence ceiling blocks GRADE, profit > 0 yields HOLD RAW)
        decision = _make_decision(
            roi_pct=1.2, profit=1.5, confidence=0.55,
            grade=8.5, p_below_8=0.10,
        )
        assert decision == "HOLD RAW"

    def test_do_not_grade_loss(self):
        decision = _make_decision(
            roi_pct=-30.0, profit=-15.0, confidence=0.85,
            grade=5.0, p_below_8=0.60,
        )
        assert decision == "DO NOT GRADE"

    def test_low_confidence_caps_at_conditional(self):
        # High ROI but low confidence → cannot be STRONG GRADE or GRADE
        decision = _make_decision(
            roi_pct=200.0, profit=100.0, confidence=0.55,
            grade=9.0, p_below_8=0.10,
        )
        assert decision in ("CONDITIONAL", "HOLD RAW", "DO NOT GRADE")

    def test_high_downside_risk_prevents_strong_grade(self):
        decision = _make_decision(
            roi_pct=200.0, profit=100.0, confidence=0.90,
            grade=9.0, p_below_8=0.50,  # > 0.40 threshold
        )
        assert decision != "STRONG GRADE"


# ─── Decision matches ROI compute_roi ─────────────────────────────────────────

class TestDecisionIntegration:
    def test_gem_mint_card_grades(self):
        """A high-grade, high-value card should get a positive decision."""
        result = compute_roi(
            estimated_grade=9.5, confidence=0.92,
            raw_price=50.0, grading_cost=25.0,
            psa_8_price=100.0, psa_9_price=200.0, psa_10_price=800.0,
        )
        assert result["decision"] in ("STRONG GRADE", "GRADE")

    def test_low_grade_bulk_card(self):
        """A low-grade cheap card should not be worth grading."""
        result = compute_roi(
            estimated_grade=5.0, confidence=0.75,
            raw_price=2.0, grading_cost=25.0,
            psa_8_price=5.0, psa_9_price=8.0, psa_10_price=20.0,
        )
        assert result["decision"] in ("HOLD RAW", "DO NOT GRADE")

    def test_high_confidence_better_than_low(self):
        """Same grade, high confidence should give better or equal decision tier."""
        from grader.roi import _DECISION_RANK
        high_conf = compute_roi(9.0, 0.92, 40.0, 25.0, 80.0, 150.0, 500.0)
        low_conf  = compute_roi(9.0, 0.60, 40.0, 25.0, 80.0, 150.0, 500.0)
        rank_high = _DECISION_RANK.get(high_conf["decision"], 5)
        rank_low  = _DECISION_RANK.get(low_conf["decision"],  5)
        assert rank_high <= rank_low


# ─── Batch ranking ────────────────────────────────────────────────────────────

class TestBatchRanking:
    def _make_roi(self, grade, confidence, raw, cost, p8, p9, p10, name="Card"):
        return compute_roi(grade, confidence, raw, cost, p8, p9, p10, card_name=name)

    def test_rank_returns_same_count(self):
        rois = [
            self._make_roi(9.0, 0.90, 40, 25, 80, 120, 400, "A"),
            self._make_roi(7.0, 0.80, 10, 25, 20, 35,  80,  "B"),
            self._make_roi(5.0, 0.70,  2, 25,  5,  8,  20,  "C"),
        ]
        ranked = rank_batch_by_roi(rois)
        assert len(ranked) == 3

    def test_best_decision_first(self):
        from grader.roi import _DECISION_RANK
        rois = [
            self._make_roi(5.0, 0.70,  2, 25,  5,  8,  20,  "Low"),
            self._make_roi(9.5, 0.93, 50, 25, 100, 200, 800, "High"),
            self._make_roi(7.5, 0.82, 20, 25,  40,  70, 200, "Mid"),
        ]
        ranked = rank_batch_by_roi(rois)
        ranks = [_DECISION_RANK.get(r["decision"], 5) for r in ranked]
        assert ranks == sorted(ranks), "Ranked list must be in ascending decision rank order"

    def test_input_not_modified(self):
        rois = [
            self._make_roi(9.0, 0.90, 40, 25, 80, 120, 400, "A"),
            self._make_roi(7.0, 0.80, 10, 25, 20, 35,  80,  "B"),
        ]
        original_order = [r["card_name"] for r in rois]
        rank_batch_by_roi(rois)
        assert [r["card_name"] for r in rois] == original_order

    def test_deterministic_ranking(self):
        rois = [
            self._make_roi(9.0, 0.90, 40, 25, 80, 120, 400, "A"),
            self._make_roi(7.0, 0.80, 10, 25, 20, 35,  80,  "B"),
            self._make_roi(5.0, 0.70,  2, 25,  5,  8,  20,  "C"),
        ]
        r1 = rank_batch_by_roi(rois)
        r2 = rank_batch_by_roi(rois)
        assert [r["card_name"] for r in r1] == [r["card_name"] for r in r2]


# ─── Batch portfolio summary ──────────────────────────────────────────────────

class TestBatchRoiSummary:
    def test_summary_keys(self):
        rois = [compute_roi(9.0, 0.90, 40.0, 25.0, 80.0, 120.0, 400.0)]
        summary = batch_roi_summary(rois)
        required = {
            "total_cards", "total_investment", "total_expected_value",
            "portfolio_profit", "portfolio_roi_pct",
            "submission_candidates", "decision_counts",
        }
        assert required.issubset(summary.keys())

    def test_empty_input(self):
        assert batch_roi_summary([]) == {}

    def test_portfolio_profit_formula(self):
        rois = [compute_roi(9.0, 0.90, 40.0, 25.0, 80.0, 120.0, 400.0)]
        summary = batch_roi_summary(rois)
        expected = round(summary["total_expected_value"] - summary["total_investment"], 2)
        assert summary["portfolio_profit"] == pytest.approx(expected, abs=0.01)

    def test_decision_counts_add_up(self):
        rois = [
            compute_roi(9.0, 0.90, 40.0, 25.0, 80.0, 120.0, 400.0),
            compute_roi(5.0, 0.70,  2.0, 25.0,  5.0,   8.0,  20.0),
        ]
        summary = batch_roi_summary(rois)
        assert sum(summary["decision_counts"].values()) == 2
