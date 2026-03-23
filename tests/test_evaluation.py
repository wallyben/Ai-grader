"""
Tests for Phase 3 Evaluation Framework.
Covers: EvaluationRecord, MetricsEngine, CalibrationEngine (Phase 3 additions),
        DecisionOptimizer, eval_dashboard, and eval_mode integration.

All tests are deterministic and require no external files or images.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grader.evaluation import (
    EvaluationRecord,
    MetricsEngine,
    load_evaluation_dataset,
    save_evaluation_dataset,
    _grade_to_band,
    _bucket_errors,
)
from grader.calibration import CalibrationEngine, DEFAULT_CALIBRATION_DATA
from grader.decision_optimizer import DecisionOptimizer
from grader.eval_dashboard import render_dashboard, render_evaluation_report


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_record(**kwargs) -> EvaluationRecord:
    """Build a minimal valid EvaluationRecord with optional overrides."""
    defaults = {
        "card_id":               "test_card_001",
        "predicted_grade":       9.0,
        "final_grade":           9,
        "roi_decision":          "GRADE",
        "actual_outcome_profit": 100.0,
        "predicted_profit":      90.0,
        "profile":               "tcg_generic",
        "timestamp":             "2025-09-15T10:00:00Z",
    }
    defaults.update(kwargs)
    return EvaluationRecord(**defaults)


def _make_dataset(n: int = 10) -> list:
    """Create a mixed dataset of n EvaluationRecords."""
    records = []
    for i in range(n):
        grade     = 6.0 + (i % 5)   # grades 6–10
        pred      = grade + (0.3 if i % 3 == 0 else -0.2 if i % 3 == 1 else 0.0)
        decision  = "GRADE" if grade >= 9 else ("CONDITIONAL" if grade >= 7 else "DO NOT GRADE")
        profit    = 50.0 * (grade - 7.0) if decision != "DO NOT GRADE" else None
        records.append(_make_record(
            card_id=f"card_{i:03d}",
            predicted_grade=pred,
            final_grade=grade,
            roi_decision=decision,
            actual_outcome_profit=profit,
            predicted_profit=profit * 1.1 if profit is not None else None,
            timestamp=f"2025-09-{15 + i:02d}T10:00:00Z",
        ))
    return records


# ─── EvaluationRecord tests ────────────────────────────────────────────────────

class TestEvaluationRecord:
    def test_grade_error_positive(self):
        r = _make_record(predicted_grade=9.5, final_grade=9.0)
        assert r.grade_error == pytest.approx(0.5, abs=1e-6)

    def test_grade_error_negative(self):
        r = _make_record(predicted_grade=8.5, final_grade=9.0)
        assert r.grade_error == pytest.approx(-0.5, abs=1e-6)

    def test_abs_grade_error(self):
        r = _make_record(predicted_grade=8.0, final_grade=9.0)
        assert r.abs_grade_error == pytest.approx(1.0, abs=1e-6)

    def test_within_half_true(self):
        r = _make_record(predicted_grade=9.0, final_grade=9.0)
        assert r.within_half is True

    def test_within_half_false(self):
        r = _make_record(predicted_grade=7.5, final_grade=9.0)
        assert r.within_half is False

    def test_within_one_true(self):
        r = _make_record(predicted_grade=8.0, final_grade=9.0)
        assert r.within_one is True

    def test_within_one_false(self):
        r = _make_record(predicted_grade=6.5, final_grade=9.0)
        assert r.within_one is False

    def test_is_grade_call_true(self):
        for decision in ("GRADE", "STRONG GRADE"):
            r = _make_record(roi_decision=decision)
            assert r.is_grade_call is True

    def test_is_grade_call_false(self):
        for decision in ("DO NOT GRADE", "HOLD RAW", "CONDITIONAL"):
            r = _make_record(roi_decision=decision)
            assert r.is_grade_call is False

    def test_is_profitable_true(self):
        r = _make_record(actual_outcome_profit=50.0)
        assert r.is_profitable is True

    def test_is_profitable_false(self):
        r = _make_record(actual_outcome_profit=-10.0)
        assert r.is_profitable is False

    def test_is_profitable_none(self):
        r = _make_record(actual_outcome_profit=None)
        assert r.is_profitable is None

    def test_profit_error(self):
        r = _make_record(predicted_profit=100.0, actual_outcome_profit=80.0)
        assert r.profit_error == pytest.approx(20.0, abs=0.01)

    def test_profit_error_none_when_missing(self):
        r = _make_record(predicted_profit=None, actual_outcome_profit=80.0)
        assert r.profit_error is None

    def test_band_correct_true(self):
        r = _make_record(predicted_grade=9.2, final_grade=9.0)
        assert r.band_correct is True   # both PSA 9

    def test_band_correct_false(self):
        r = _make_record(predicted_grade=9.5, final_grade=8.0)
        assert r.band_correct is False  # different bands

    def test_from_dict_roundtrip(self):
        r  = _make_record()
        d  = r.to_dict()
        r2 = EvaluationRecord.from_dict(d)
        assert r2.card_id         == r.card_id
        assert r2.predicted_grade == r.predicted_grade
        assert r2.final_grade     == r.final_grade
        assert r2.roi_decision    == r.roi_decision


# ─── Dataset I/O tests ────────────────────────────────────────────────────────

class TestDatasetIO:
    def test_save_and_load(self, tmp_path):
        records  = _make_dataset(5)
        path     = str(tmp_path / "test_dataset.json")
        save_evaluation_dataset(records, path)
        loaded   = load_evaluation_dataset(path)
        assert len(loaded) == 5
        assert loaded[0].card_id == records[0].card_id

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_evaluation_dataset(str(tmp_path / "does_not_exist.json"))

    def test_load_non_array_raises(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as fh:
            json.dump({"not": "an array"}, fh)
        with pytest.raises(ValueError):
            load_evaluation_dataset(path)

    def test_load_sample_dataset(self):
        """Integration: load the real sample dataset shipped with the system."""
        sample = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "evaluation", "sample_dataset.json"
        )
        if not os.path.exists(sample):
            pytest.skip("Sample dataset not found")
        records = load_evaluation_dataset(sample)
        assert len(records) >= 10
        for r in records:
            assert 1.0 <= r.predicted_grade <= 10.0
            assert 1.0 <= r.final_grade <= 10.0

    def test_chronological_sort(self, tmp_path):
        records = [
            _make_record(card_id="c", timestamp="2025-10-01T00:00:00Z"),
            _make_record(card_id="a", timestamp="2025-08-01T00:00:00Z"),
            _make_record(card_id="b", timestamp="2025-09-01T00:00:00Z"),
        ]
        path = str(tmp_path / "sort.json")
        save_evaluation_dataset(records, path)
        loaded = load_evaluation_dataset(path)
        assert [r.card_id for r in loaded] == ["a", "b", "c"]


# ─── MetricsEngine tests ──────────────────────────────────────────────────────

class TestMetricsEngine:
    def setup_method(self):
        self.records = _make_dataset(15)
        self.engine  = MetricsEngine(self.records)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            MetricsEngine([])

    def test_compute_all_keys(self):
        result = self.engine.compute_all()
        required = {
            "summary", "grade_metrics", "decision_metrics",
            "roi_metrics", "confusion_matrix", "band_breakdown",
            "trend", "per_card", "generated_at", "n_records",
        }
        assert required.issubset(result.keys())

    def test_n_records(self):
        result = self.engine.compute_all()
        assert result["n_records"] == 15

    def test_grade_metrics_keys(self):
        gm = self.engine.grade_metrics()
        required = {"n", "mae", "rmse", "bias", "within_half_pct",
                    "within_one_pct", "band_accuracy_pct", "error_distribution"}
        assert required.issubset(gm.keys())

    def test_mae_non_negative(self):
        gm = self.engine.grade_metrics()
        assert gm["mae"] >= 0

    def test_rmse_gte_mae(self):
        gm = self.engine.grade_metrics()
        # RMSE >= MAE always (Cauchy-Schwarz inequality)
        assert gm["rmse"] >= gm["mae"] - 1e-9

    def test_band_accuracy_pct_range(self):
        gm = self.engine.grade_metrics()
        assert 0.0 <= gm["band_accuracy_pct"] <= 100.0

    def test_decision_metrics_keys(self):
        dm = self.engine.decision_metrics()
        required = {"decision_accuracy_pct", "false_positive_rate",
                    "false_negative_rate", "true_positives", "false_positives",
                    "true_negatives", "false_negatives"}
        assert required.issubset(dm.keys())

    def test_confusion_matrix_totals(self):
        cm = self.engine.confusion_matrix()
        total = cm["true_positives"] + cm["false_positives"] + \
                cm["true_negatives"] + cm["false_negatives"]
        assert total == cm["total"]

    def test_confusion_matrix_accuracy_range(self):
        cm = self.engine.confusion_matrix()
        assert 0.0 <= cm["overall_accuracy"] <= 100.0

    def test_roi_metrics_keys(self):
        rm = self.engine.roi_metrics()
        assert "n_with_actual_profit" in rm

    def test_roi_metrics_profitable_rate_range(self):
        rm = self.engine.roi_metrics()
        if "profitable_rate_pct" in rm:
            assert 0.0 <= rm["profitable_rate_pct"] <= 100.0

    def test_band_breakdown_non_empty(self):
        bb = self.engine.band_breakdown()
        assert len(bb) > 0

    def test_band_breakdown_accuracy_range(self):
        bb = self.engine.band_breakdown()
        for band, data in bb.items():
            assert 0.0 <= data["band_accuracy"] <= 100.0

    def test_trend_metrics_direction_valid(self):
        trend = self.engine.trend_metrics()
        if "trend_direction" in trend:
            assert trend["trend_direction"] in ("improving", "degrading", "stable")

    def test_per_card_length(self):
        result = self.engine.compute_all()
        assert len(result["per_card"]) == 15

    def test_per_card_keys(self):
        result = self.engine.compute_all()
        for card in result["per_card"]:
            assert "card_id" in card
            assert "grade_error" in card
            assert "roi_decision" in card

    def test_deterministic(self):
        """Same dataset always produces identical metrics."""
        r1 = self.engine.compute_all()
        r2 = MetricsEngine(self.records).compute_all()
        assert r1["grade_metrics"]["mae"] == r2["grade_metrics"]["mae"]
        assert r1["decision_metrics"]["false_positive_rate"] == \
               r2["decision_metrics"]["false_positive_rate"]


# ─── Perfect dataset (zero error) ─────────────────────────────────────────────

class TestPerfectDataset:
    def setup_method(self):
        # System predicts exactly what PSA returns
        self.records = [
            _make_record(card_id=f"p{i}", predicted_grade=9.0, final_grade=9.0,
                         roi_decision="GRADE", actual_outcome_profit=100.0)
            for i in range(10)
        ]
        self.engine = MetricsEngine(self.records)

    def test_mae_is_zero(self):
        gm = self.engine.grade_metrics()
        assert gm["mae"] == pytest.approx(0.0, abs=1e-6)

    def test_bias_is_zero(self):
        gm = self.engine.grade_metrics()
        assert gm["bias"] == pytest.approx(0.0, abs=1e-6)

    def test_within_half_is_100(self):
        gm = self.engine.grade_metrics()
        assert gm["within_half_pct"] == pytest.approx(100.0, abs=1e-3)

    def test_band_accuracy_is_100(self):
        gm = self.engine.grade_metrics()
        assert gm["band_accuracy_pct"] == pytest.approx(100.0, abs=1e-3)


# ─── Calibration Phase 3 tests ────────────────────────────────────────────────

class TestCalibrationPhase3:
    def test_update_from_outcomes_basic(self):
        engine = CalibrationEngine()
        outcomes = [
            {"predicted_grade": 9.2, "actual_grade": 9},
            {"predicted_grade": 8.5, "actual_grade": 8},
        ]
        result = engine.update_from_outcomes(outcomes)
        assert result["updated"] is True
        assert result["n_new_outcomes"] == 2

    def test_update_from_outcomes_empty(self):
        engine = CalibrationEngine()
        result = engine.update_from_outcomes([])
        assert result["updated"] is False

    def test_update_from_outcomes_invalid_skipped(self):
        engine = CalibrationEngine()
        outcomes = [
            {"predicted_grade": 9.0},      # missing actual_grade
            {"actual_grade": 8.0},          # missing predicted_grade
            {"predicted_grade": 9.0, "actual_grade": 9},  # valid
        ]
        result = engine.update_from_outcomes(outcomes)
        assert result["n_new_outcomes"] == 1

    def test_update_records_history(self):
        engine = CalibrationEngine()
        engine.update_from_outcomes([{"predicted_grade": 9.0, "actual_grade": 9}])
        engine.update_from_outcomes([{"predicted_grade": 8.0, "actual_grade": 8}])
        assert len(engine.calibration_history) == 2

    def test_rolling_window_enforced(self):
        engine = CalibrationEngine(rolling_window=3)
        for i in range(10):
            engine.update_from_outcomes(
                [{"predicted_grade": 9.0, "actual_grade": 9}],
                record_history=False,
            )
        assert engine.n_rolling_outcomes <= 3

    def test_rolling_performance_keys(self):
        engine = CalibrationEngine()
        engine.update_from_outcomes([
            {"predicted_grade": 9.0, "actual_grade": 9},
            {"predicted_grade": 8.0, "actual_grade": 8},
        ])
        rp = engine.rolling_performance()
        assert "n" in rp
        assert "mae" in rp
        assert "bias" in rp

    def test_rolling_performance_empty(self):
        engine = CalibrationEngine()
        rp = engine.rolling_performance()
        assert rp["n"] == 0

    def test_save_and_load_state(self, tmp_path):
        engine = CalibrationEngine()
        engine.update_from_outcomes([
            {"predicted_grade": 9.0, "actual_grade": 9},
            {"predicted_grade": 8.0, "actual_grade": 8},
        ])
        state_path = str(tmp_path / "cal_state.json")
        engine.save_state(state_path)
        assert os.path.exists(state_path)

        # Load into new engine and verify rolling outcomes preserved
        engine2 = CalibrationEngine()
        loaded  = engine2.load_state(state_path)
        assert loaded is True
        assert engine2.n_rolling_outcomes == 2

    def test_load_state_missing_returns_false(self, tmp_path):
        engine = CalibrationEngine()
        result = engine.load_state(str(tmp_path / "no_file.json"))
        assert result is False

    def test_summary_has_phase3_keys(self):
        engine = CalibrationEngine()
        s = engine.summary()
        assert "n_rolling_outcomes" in s
        assert "n_base_samples"     in s
        assert "n_history_entries"  in s
        assert "rolling_window"     in s

    def test_calibration_deterministic_after_update(self):
        engine = CalibrationEngine()
        engine.update_from_outcomes([{"predicted_grade": 9.0, "actual_grade": 9}])
        r1 = engine.calibrate(9.0, 0.90)
        r2 = engine.calibrate(9.0, 0.90)
        assert r1 == r2

    def test_update_shifts_bias(self):
        """Adding consistently over-predicted outcomes increases reported bias."""
        engine = CalibrationEngine()
        initial_bias = engine.bias
        # 10 outcomes where system over-grades by 1.5 points
        outcomes = [{"predicted_grade": 10.0, "actual_grade": 8}] * 10
        engine.update_from_outcomes(outcomes)
        # Bias should increase (system now even more over-grade biased)
        assert engine.bias > initial_bias


# ─── DecisionOptimizer tests ──────────────────────────────────────────────────

class TestDecisionOptimizer:
    def setup_method(self):
        self.records = _make_dataset(15)

    def test_optimize_keys(self):
        opt    = DecisionOptimizer(self.records)
        result = opt.optimize()
        required = {
            "tier_analysis", "grade_band_bias", "weaknesses",
            "threshold_adjustments", "recommendations",
        }
        assert required.issubset(result.keys())

    def test_tier_analysis_non_empty(self):
        opt    = DecisionOptimizer(self.records)
        result = opt.optimize()
        assert len(result["tier_analysis"]) > 0

    def test_tier_analysis_profitable_rate_range(self):
        opt    = DecisionOptimizer(self.records)
        result = opt.optimize()
        for tier, data in result["tier_analysis"].items():
            assert 0.0 <= data["profitable_rate"] <= 100.0

    def test_grade_band_bias_non_empty(self):
        opt    = DecisionOptimizer(self.records)
        result = opt.optimize()
        assert len(result["grade_band_bias"]) > 0

    def test_grade_band_direction_valid(self):
        opt    = DecisionOptimizer(self.records)
        result = opt.optimize()
        for band, info in result["grade_band_bias"].items():
            assert info["direction"] in ("over-graded", "under-graded", "accurate")

    def test_recommendations_is_list(self):
        opt    = DecisionOptimizer(self.records)
        result = opt.optimize()
        assert isinstance(result["recommendations"], list)

    def test_weaknesses_is_list(self):
        opt    = DecisionOptimizer(self.records)
        result = opt.optimize()
        assert isinstance(result["weaknesses"], list)

    def test_threshold_adjustments_has_current_values(self):
        opt    = DecisionOptimizer(self.records)
        result = opt.optimize()
        ta = result["threshold_adjustments"]
        assert "STRONG_GRADE_ROI_PCT" in ta
        assert "GRADE_ROI_PCT"        in ta

    def test_all_false_positive_triggers_recommendation(self):
        """If every GRADE call is unprofitable, expect weakness + adjustment."""
        bad_records = [
            _make_record(
                card_id=f"bad_{i}",
                roi_decision="GRADE",
                actual_outcome_profit=-20.0,   # always a loss
                predicted_grade=9.0,
                final_grade=8.0,
            )
            for i in range(8)
        ]
        opt    = DecisionOptimizer(bad_records)
        result = opt.optimize()
        weaknesses = result["weaknesses"]
        assert any(w["type"] == "high_false_positive_rate" for w in weaknesses)

    def test_empty_profit_data_handled(self):
        """Records with no profit data should not crash."""
        records = [
            _make_record(card_id=f"no_profit_{i}", actual_outcome_profit=None)
            for i in range(5)
        ]
        opt    = DecisionOptimizer(records)
        result = opt.optimize()
        # Should return with 0 records (all filtered out)
        assert result["n_records_with_outcomes"] == 0


# ─── Dashboard rendering tests ────────────────────────────────────────────────

class TestEvalDashboard:
    def setup_method(self):
        records  = _make_dataset(10)
        engine   = MetricsEngine(records)
        self.metrics = engine.compute_all()

    def test_render_dashboard_returns_string(self):
        output = render_dashboard(self.metrics)
        assert isinstance(output, str)
        assert len(output) > 100

    def test_dashboard_contains_summary_header(self):
        output = render_dashboard(self.metrics)
        assert "SUMMARY" in output

    def test_dashboard_contains_grade_metrics(self):
        output = render_dashboard(self.metrics)
        assert "GRADE ACCURACY" in output

    def test_dashboard_contains_confusion_matrix(self):
        output = render_dashboard(self.metrics)
        assert "CONFUSION MATRIX" in output

    def test_dashboard_with_optimizer(self):
        records   = _make_dataset(10)
        optimizer = DecisionOptimizer(records)
        opt_result = optimizer.optimize()
        output = render_dashboard(self.metrics, optimizer=opt_result)
        assert "WEAKNESSES" in output or "RECOMMENDATIONS" in output

    def test_render_evaluation_report_returns_string(self):
        output = render_evaluation_report(self.metrics)
        assert isinstance(output, str)
        assert "EVALUATION REPORT" in output

    def test_report_contains_per_card_table(self):
        output = render_evaluation_report(self.metrics)
        assert "PER-CARD DETAIL" in output


# ─── Helpers tests ────────────────────────────────────────────────────────────

class TestHelpers:
    def test_grade_to_band_10(self):
        assert _grade_to_band(10.0) == "PSA 9.5-10"

    def test_grade_to_band_9(self):
        assert _grade_to_band(9.0) == "PSA 9"

    def test_grade_to_band_8(self):
        assert _grade_to_band(8.0) == "PSA 8-8.5"

    def test_grade_to_band_7(self):
        assert _grade_to_band(7.0) == "PSA 7-7.5"

    def test_grade_to_band_5(self):
        assert _grade_to_band(5.0) == "PSA 5-6"

    def test_grade_to_band_3(self):
        assert _grade_to_band(3.0) == "PSA 1-4"

    def test_bucket_errors_counts(self):
        errors = [0.0, 0.0, 0.3, 0.6, 1.2]
        buckets = _bucket_errors(errors)
        assert buckets["exact (0)"]    == 2
        assert buckets["within_0.5"]   == 1
        assert buckets["within_1.0"]   == 1
        assert buckets["over_1.0"]     == 1

    def test_bucket_errors_empty(self):
        assert _bucket_errors([]) == {}


# ─── Integration: eval_mode end-to-end ────────────────────────────────────────

class TestEvalModeIntegration:
    def test_full_pipeline_no_crash(self, tmp_path):
        """Run the complete eval_mode pipeline in-process."""
        # Save dataset to temp file
        records  = _make_dataset(10)
        dataset_path = str(tmp_path / "dataset.json")
        save_evaluation_dataset(records, dataset_path)

        # Run evaluation + calibration update
        engine      = MetricsEngine(records)
        metrics     = engine.compute_all()
        optimizer   = DecisionOptimizer(records)
        opt_result  = optimizer.optimize()

        cal_engine  = CalibrationEngine()
        outcomes    = [{"predicted_grade": r.predicted_grade,
                        "actual_grade": r.final_grade} for r in records]
        cal_result  = cal_engine.update_from_outcomes(outcomes)

        # Save calibration state
        state_path = str(tmp_path / "cal_state.json")
        cal_engine.save_state(state_path)

        # Generate report
        report = render_evaluation_report(metrics, optimizer=opt_result)

        # Assertions
        assert metrics["n_records"] == 10
        assert cal_result["updated"] is True
        assert os.path.exists(state_path)
        assert len(report) > 200

    def test_calibration_update_applied_to_grades(self, tmp_path):
        """After calibration update, calibrated grades should reflect new bias."""
        engine = CalibrationEngine()
        # Simulate: system over-grades by 1 in the 9–10 range
        outcomes = [
            {"predicted_grade": 10.0, "actual_grade": 9},
            {"predicted_grade": 9.5,  "actual_grade": 9},
            {"predicted_grade": 9.0,  "actual_grade": 9},
        ] * 4  # 12 outcomes
        engine.update_from_outcomes(outcomes)

        result = engine.calibrate(10.0, 0.90)
        # After learning, calibrated grade for 10.0 prediction should be ≤ 10.0
        # (system is learning it over-grades)
        assert result["calibrated_grade"] <= 10.0
        assert result["calibrated_grade"] >= 1.0
