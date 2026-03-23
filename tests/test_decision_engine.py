"""
Tests for the enhanced decision engine and pricing module — Phase 2.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grader.pricing import (
    CardPricing,
    CardPricingBundle,
    build_pricing,
    validate_pricing,
)
from grader.roi import compute_roi, rank_batch_by_roi, batch_roi_summary


# ─── Pricing module ───────────────────────────────────────────────────────────

class TestCardPricing:
    def test_build_pricing_complete(self):
        bundle = build_pricing("Charizard", 40.0, psa_8=80.0, psa_9=120.0, psa_10=400.0)
        assert isinstance(bundle, CardPricingBundle)
        assert bundle.pricing.raw_price == 40.0
        assert bundle.pricing.psa_8 == 80.0

    def test_build_pricing_partial(self):
        bundle = build_pricing("Bulbasaur", 10.0)
        p = bundle.pricing
        assert p.psa_8 is None
        assert p.psa_9 is None

    def test_fill_estimates_complete(self):
        pricing = CardPricing("Pikachu", 20.0)
        filled = pricing.fill_estimates()
        assert filled.psa_8  > 0
        assert filled.psa_9  > 0
        assert filled.psa_10 > 0
        assert filled.source == "estimated"

    def test_fill_estimates_preserves_existing(self):
        pricing = CardPricing("Pikachu", 20.0, psa_9=100.0)
        filled = pricing.fill_estimates()
        assert filled.psa_9 == 100.0  # kept as-is

    def test_is_complete_false(self):
        pricing = CardPricing("X", 10.0, psa_8=20.0)
        assert not pricing.is_complete()

    def test_is_complete_true(self):
        pricing = CardPricing("X", 10.0, psa_8=20.0, psa_9=30.0, psa_10=60.0)
        assert pricing.is_complete()

    def test_to_dict_keys(self):
        pricing = CardPricing("X", 10.0)
        d = pricing.to_dict()
        assert "card_name" in d
        assert "raw_price" in d
        assert "source"    in d

    def test_grading_cost_in_bundle(self):
        bundle = build_pricing("X", 10.0, grading_cost=25.0)
        assert bundle.grading_cost == 25.0


class TestValidatePricing:
    def test_valid_pricing(self):
        p = CardPricing("X", 40.0, psa_8=80.0, psa_9=120.0, psa_10=400.0)
        result = validate_pricing(p)
        assert result["valid"] is True
        assert result["issues"] == []

    def test_zero_raw_price_invalid(self):
        p = CardPricing("X", 0.0)
        result = validate_pricing(p)
        assert result["valid"] is False

    def test_negative_raw_price_invalid(self):
        p = CardPricing("X", -5.0)
        result = validate_pricing(p)
        assert result["valid"] is False

    def test_psa9_less_than_psa8_invalid(self):
        p = CardPricing("X", 40.0, psa_8=100.0, psa_9=80.0, psa_10=200.0)
        result = validate_pricing(p)
        assert result["valid"] is False
        assert any("PSA 9" in issue for issue in result["issues"])

    def test_psa10_less_than_psa9_invalid(self):
        p = CardPricing("X", 40.0, psa_8=80.0, psa_9=120.0, psa_10=100.0)
        result = validate_pricing(p)
        assert result["valid"] is False

    def test_psa8_below_raw_warns(self):
        p = CardPricing("X", 40.0, psa_8=30.0, psa_9=60.0, psa_10=120.0)
        result = validate_pricing(p)
        # Should warn but not fail (unusual but can happen for damaged/low-demand cards)
        assert result["valid"] is True
        assert len(result["warnings"]) > 0


# ─── Decision tier integration ────────────────────────────────────────────────

class TestDecisionTiers:
    """End-to-end tests covering all five decision tiers."""

    def _roi(self, grade, conf, raw, cost, p8, p9, p10):
        return compute_roi(grade, conf, raw, cost, p8, p9, p10)

    def test_strong_grade_scenario(self):
        """PSA 10 pop report gem: high premium, great confidence."""
        r = self._roi(9.5, 0.93, 50.0, 25.0, 150.0, 300.0, 1200.0)
        assert r["decision"] == "STRONG GRADE"

    def test_grade_scenario(self):
        """Solid mid-tier card with positive ROI."""
        r = self._roi(9.0, 0.88, 30.0, 25.0, 60.0, 120.0, 400.0)
        assert r["decision"] in ("GRADE", "STRONG GRADE")

    def test_conditional_scenario(self):
        """Solid mid-grade card where grading makes marginal sense."""
        r = self._roi(8.0, 0.85, 20.0, 25.0, 60.0, 100.0, 250.0)
        assert r["decision"] in ("CONDITIONAL", "GRADE")

    def test_hold_raw_scenario(self):
        """Low-confidence card with tiny slab premium — hold raw."""
        r = self._roi(8.5, 0.55, 100.0, 25.0, 120.0, 145.0, 200.0)
        assert r["decision"] in ("HOLD RAW", "CONDITIONAL")

    def test_do_not_grade_scenario(self):
        """Bulk common card — grading clearly unprofitable."""
        r = self._roi(5.0, 0.72, 1.0, 25.0, 2.0, 4.0, 10.0)
        assert r["decision"] == "DO NOT GRADE"


# ─── Batch ranking integration ────────────────────────────────────────────────

class TestBatchRankingIntegration:
    def test_ranked_batch_ordered_by_decision(self):
        from grader.roi import _DECISION_RANK
        rois = [
            compute_roi(5.0, 0.70, 1.0, 25.0, 2.0, 4.0, 10.0, card_name="Bulk"),
            compute_roi(9.5, 0.93, 50.0, 25.0, 150.0, 300.0, 1200.0, card_name="Gem"),
            compute_roi(7.5, 0.80, 15.0, 25.0, 25.0, 45.0, 120.0, card_name="Mid"),
        ]
        ranked = rank_batch_by_roi(rois)
        assert ranked[0]["card_name"] == "Gem"
        assert ranked[-1]["card_name"] == "Bulk"

    def test_portfolio_summary_is_correct(self):
        rois = [
            compute_roi(9.0, 0.90, 40.0, 25.0, 80.0, 150.0, 500.0),
            compute_roi(8.0, 0.85, 20.0, 25.0, 40.0,  80.0, 200.0),
        ]
        summary = batch_roi_summary(rois)
        assert summary["total_cards"] == 2
        assert summary["total_investment"] == pytest.approx(
            rois[0]["total_investment"] + rois[1]["total_investment"], abs=0.01
        )

    def test_submission_candidates_count(self):
        rois = [
            compute_roi(9.5, 0.93, 50.0, 25.0, 150.0, 300.0, 1200.0),  # STRONG/GRADE
            compute_roi(5.0, 0.70,  1.0, 25.0,   2.0,   4.0,   10.0),  # DO NOT GRADE
        ]
        summary = batch_roi_summary(rois)
        assert summary["submission_candidates"] >= 1
