"""
Evaluation Engine — Phase 3
Measures real-world accuracy of the grading system from submitted card outcomes.

Computes:
- Grade error metrics (MAE, RMSE, bias, band accuracy)
- Decision accuracy (correct GRADE vs DO NOT GRADE calls)
- ROI accuracy (predicted vs actual profit)
- False positive / false negative rates
- Confusion matrix
- Rolling trend analysis over time

All computations are deterministic — no randomness, no ML.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ─── Decision groupings ────────────────────────────────────────────────────────
# "Grade" calls: system recommended submitting for grading
GRADE_DECISIONS = {"STRONG GRADE", "GRADE"}
# "Conditional" and "No-grade" calls
NO_GRADE_DECISIONS = {"DO NOT GRADE", "HOLD RAW", "CONDITIONAL"}


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class EvaluationRecord:
    """
    One real-world grading outcome record.

    Minimum required: card_id, predicted_grade, final_grade, roi_decision.
    Optional profit fields enable ROI accuracy metrics.
    """
    card_id:               str
    predicted_grade:       float
    final_grade:           float
    roi_decision:          str          # STRONG GRADE | GRADE | CONDITIONAL | HOLD RAW | DO NOT GRADE
    actual_outcome_profit: Optional[float] = None   # Actual P&L after sale
    predicted_profit:      Optional[float] = None   # AI system profit prediction
    card_name:             Optional[str] = None
    profile:               Optional[str] = None
    timestamp:             Optional[str] = None
    notes:                 Optional[str] = None

    # ── Derived fields (computed on demand) ───────────────────────────────────

    @property
    def grade_error(self) -> float:
        """Signed prediction error: positive = system over-graded."""
        return round(self.predicted_grade - self.final_grade, 3)

    @property
    def abs_grade_error(self) -> float:
        """Absolute grade prediction error."""
        return round(abs(self.grade_error), 3)

    @property
    def grade_band(self) -> str:
        """Bucketed grade band for the FINAL (actual) grade."""
        return _grade_to_band(self.final_grade)

    @property
    def predicted_band(self) -> str:
        """Bucketed grade band for the predicted grade."""
        return _grade_to_band(self.predicted_grade)

    @property
    def band_correct(self) -> bool:
        """True if predicted and actual grades are in the same PSA band."""
        return self.grade_band == self.predicted_band

    @property
    def within_half(self) -> bool:
        """True if prediction is within ±0.5 of final grade."""
        return self.abs_grade_error <= 0.5

    @property
    def within_one(self) -> bool:
        """True if prediction is within ±1.0 of final grade."""
        return self.abs_grade_error <= 1.0

    @property
    def is_grade_call(self) -> bool:
        """True if system made a GRADE or STRONG GRADE recommendation."""
        return self.roi_decision in GRADE_DECISIONS

    @property
    def is_profitable(self) -> Optional[bool]:
        """True if actual profit > 0. None when profit data is unavailable."""
        if self.actual_outcome_profit is None:
            return None
        return self.actual_outcome_profit > 0

    @property
    def profit_error(self) -> Optional[float]:
        """Signed profit prediction error. None if data unavailable."""
        if self.predicted_profit is None or self.actual_outcome_profit is None:
            return None
        return round(self.predicted_profit - self.actual_outcome_profit, 2)

    @property
    def abs_profit_error(self) -> Optional[float]:
        """Absolute profit prediction error."""
        if self.profit_error is None:
            return None
        return round(abs(self.profit_error), 2)

    # ── Serialisation ─────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: Dict) -> "EvaluationRecord":
        return cls(
            card_id               = str(d["card_id"]),
            predicted_grade       = float(d["predicted_grade"]),
            final_grade           = float(d["final_grade"]),
            roi_decision          = str(d["roi_decision"]),
            actual_outcome_profit = _optional_float(d.get("actual_outcome_profit")),
            predicted_profit      = _optional_float(d.get("predicted_profit")),
            card_name             = d.get("card_name"),
            profile               = d.get("profile"),
            timestamp             = d.get("timestamp"),
            notes                 = d.get("notes"),
        )

    def to_dict(self) -> Dict:
        return {
            "card_id":               self.card_id,
            "card_name":             self.card_name,
            "predicted_grade":       self.predicted_grade,
            "final_grade":           self.final_grade,
            "roi_decision":          self.roi_decision,
            "actual_outcome_profit": self.actual_outcome_profit,
            "predicted_profit":      self.predicted_profit,
            "profile":               self.profile,
            "timestamp":             self.timestamp,
            "notes":                 self.notes,
        }


# ─── Dataset loader ───────────────────────────────────────────────────────────

def load_evaluation_dataset(path: str) -> List[EvaluationRecord]:
    """
    Load evaluation records from a JSON file.

    The file must be a JSON array of objects matching the evaluation schema.
    Returns a list of EvaluationRecord objects sorted by timestamp (oldest first).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError("Evaluation dataset must be a JSON array of records.")

    records = [EvaluationRecord.from_dict(item) for item in raw]

    # Sort chronologically so trend analysis is meaningful
    records.sort(key=lambda r: r.timestamp or "")
    return records


def save_evaluation_dataset(records: List[EvaluationRecord], path: str) -> None:
    """Persist evaluation records to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([r.to_dict() for r in records], fh, indent=2)


# ─── Core Metrics Engine ──────────────────────────────────────────────────────

class MetricsEngine:
    """
    Computes all evaluation metrics from a list of EvaluationRecord objects.

    Usage:
        records = load_evaluation_dataset("data/evaluation/sample_dataset.json")
        engine  = MetricsEngine(records)
        metrics = engine.compute_all()
    """

    def __init__(self, records: List[EvaluationRecord]) -> None:
        if not records:
            raise ValueError("Cannot compute metrics on empty record set.")
        self.records = records

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute_all(self) -> Dict[str, Any]:
        """Compute and return the complete metrics report as a dict."""
        return {
            "summary":           self.summary(),
            "grade_metrics":     self.grade_metrics(),
            "decision_metrics":  self.decision_metrics(),
            "roi_metrics":       self.roi_metrics(),
            "confusion_matrix":  self.confusion_matrix(),
            "band_breakdown":    self.band_breakdown(),
            "trend":             self.trend_metrics(),
            "per_card":          self._per_card_detail(),
            "generated_at":      datetime.utcnow().isoformat() + "Z",
            "n_records":         len(self.records),
        }

    # ── Summary ────────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """High-level KPI summary."""
        gm  = self.grade_metrics()
        dm  = self.decision_metrics()
        rm  = self.roi_metrics()

        return {
            "total_cards":          len(self.records),
            "mae":                  gm["mae"],
            "grade_within_half":    gm["within_half_pct"],
            "grade_within_one":     gm["within_one_pct"],
            "band_accuracy_pct":    gm["band_accuracy_pct"],
            "decision_accuracy_pct": dm["decision_accuracy_pct"],
            "false_positive_rate":  dm["false_positive_rate"],
            "false_negative_rate":  dm["false_negative_rate"],
            "roi_mae":              rm.get("profit_mae"),
            "total_actual_profit":  rm.get("total_actual_profit"),
            "total_predicted_profit": rm.get("total_predicted_profit"),
        }

    # ── Grade accuracy ─────────────────────────────────────────────────────────

    def grade_metrics(self) -> Dict[str, Any]:
        """Grade prediction accuracy metrics."""
        errors     = [r.grade_error     for r in self.records]
        abs_errors = [r.abs_grade_error for r in self.records]
        n          = len(self.records)

        mae  = round(sum(abs_errors) / n, 4)
        bias = round(sum(errors) / n, 4)          # + = system over-grades
        rmse = round(math.sqrt(sum(e ** 2 for e in errors) / n), 4)

        within_half  = sum(1 for r in self.records if r.within_half)
        within_one   = sum(1 for r in self.records if r.within_one)
        band_correct = sum(1 for r in self.records if r.band_correct)

        # Over/under grading breakdown
        over_graded  = sum(1 for e in errors if e > 0.25)
        under_graded = sum(1 for e in errors if e < -0.25)
        exact        = n - over_graded - under_graded

        # Grade-level error distribution
        error_dist = _bucket_errors(abs_errors)

        return {
            "n":                  n,
            "mae":                mae,
            "rmse":               rmse,
            "bias":               bias,
            "within_half_pct":    round(within_half  / n * 100, 1),
            "within_one_pct":     round(within_one   / n * 100, 1),
            "band_accuracy_pct":  round(band_correct / n * 100, 1),
            "exact_count":        exact,
            "over_graded_count":  over_graded,
            "under_graded_count": under_graded,
            "error_distribution": error_dist,
            "max_error":          round(max(abs_errors), 2),
            "min_error":          round(min(abs_errors), 2),
        }

    # ── Decision accuracy ──────────────────────────────────────────────────────

    def decision_metrics(self) -> Dict[str, Any]:
        """
        Decision accuracy from the ROI recommendation engine.

        Definitions:
        - True Positive  (TP): GRADE call + actual profit > 0
        - False Positive (FP): GRADE call + actual profit <= 0  ← BAD call
        - True Negative  (TN): NO-GRADE call + actual profit <= 0
        - False Negative (FN): NO-GRADE call + actual profit > 0  ← missed card
        Records with no profit data are excluded from FP/FN analysis.
        """
        records_with_profit = [r for r in self.records
                                if r.actual_outcome_profit is not None]
        n_all = len(self.records)
        n_profit = len(records_with_profit)

        # Total decision accuracy (grade within ±1 AND decision directionally correct)
        correct_decisions = sum(
            1 for r in self.records if _decision_correct(r)
        )
        decision_accuracy = round(correct_decisions / n_all * 100, 1)

        # Confusion matrix counts (only where profit known)
        tp = sum(1 for r in records_with_profit
                 if r.is_grade_call and r.is_profitable)
        fp = sum(1 for r in records_with_profit
                 if r.is_grade_call and not r.is_profitable)
        tn = sum(1 for r in records_with_profit
                 if not r.is_grade_call and not r.is_profitable)
        fn = sum(1 for r in records_with_profit
                 if not r.is_grade_call and r.is_profitable)

        # Rates (guard against division by zero)
        fpr = round(fp / (fp + tn) * 100, 1) if (fp + tn) > 0 else 0.0
        fnr = round(fn / (fn + tp) * 100, 1) if (fn + tp) > 0 else 0.0
        precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else 0.0
        recall    = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0.0

        # Per-decision-tier breakdown
        tier_counts: Dict[str, int] = {}
        for r in self.records:
            tier_counts[r.roi_decision] = tier_counts.get(r.roi_decision, 0) + 1

        return {
            "n_total":               n_all,
            "n_with_profit_data":    n_profit,
            "decision_accuracy_pct": decision_accuracy,
            "correct_decisions":     correct_decisions,
            "true_positives":        tp,
            "false_positives":       fp,
            "true_negatives":        tn,
            "false_negatives":       fn,
            "false_positive_rate":   fpr,
            "false_negative_rate":   fnr,
            "precision_pct":         precision,
            "recall_pct":            recall,
            "tier_counts":           tier_counts,
        }

    # ── ROI accuracy ───────────────────────────────────────────────────────────

    def roi_metrics(self) -> Dict[str, Any]:
        """Predicted vs actual profit accuracy."""
        records_with_both = [
            r for r in self.records
            if r.actual_outcome_profit is not None and r.predicted_profit is not None
        ]
        records_with_actual = [
            r for r in self.records
            if r.actual_outcome_profit is not None
        ]

        if not records_with_actual:
            return {"n": 0, "note": "No profit data available."}

        total_actual    = round(sum(r.actual_outcome_profit for r in records_with_actual), 2)
        total_predicted = None
        profit_mae      = None
        profit_bias     = None

        if records_with_both:
            profit_errors = [r.profit_error for r in records_with_both]
            abs_errors    = [r.abs_profit_error for r in records_with_both]
            profit_mae    = round(sum(abs_errors) / len(abs_errors), 2)
            profit_bias   = round(sum(profit_errors) / len(profit_errors), 2)
            total_predicted = round(
                sum(r.predicted_profit for r in records_with_both), 2
            )

        # Profitable / unprofitable breakdown
        profitable   = [r for r in records_with_actual if r.is_profitable]
        unprofitable = [r for r in records_with_actual if not r.is_profitable]

        # Tier-level profit summary
        tier_profit: Dict[str, Dict] = {}
        for r in records_with_actual:
            t = r.roi_decision
            if t not in tier_profit:
                tier_profit[t] = {"count": 0, "total_profit": 0.0, "profits": []}
            tier_profit[t]["count"] += 1
            tier_profit[t]["total_profit"] += r.actual_outcome_profit
            tier_profit[t]["profits"].append(r.actual_outcome_profit)

        for t, data in tier_profit.items():
            profits = data.pop("profits")
            data["total_profit"] = round(data["total_profit"], 2)
            data["avg_profit"]   = round(data["total_profit"] / data["count"], 2)
            data["profitable_pct"] = round(
                sum(1 for p in profits if p > 0) / len(profits) * 100, 1
            )

        return {
            "n_with_actual_profit":    len(records_with_actual),
            "n_with_both":             len(records_with_both),
            "total_actual_profit":     total_actual,
            "total_predicted_profit":  total_predicted,
            "profit_mae":              profit_mae,
            "profit_bias":             profit_bias,    # + = system over-estimates profit
            "profitable_cards":        len(profitable),
            "unprofitable_cards":      len(unprofitable),
            "profitable_rate_pct":     round(len(profitable) / len(records_with_actual) * 100, 1),
            "tier_profit_breakdown":   tier_profit,
        }

    # ── Confusion matrix ───────────────────────────────────────────────────────

    def confusion_matrix(self) -> Dict[str, Any]:
        """
        Binary confusion matrix: GRADE call vs actual profitability.

        Rows = predicted (GRADE / NO-GRADE)
        Cols = actual (Profitable / Not Profitable)
        Only includes records with known actual profit.
        """
        dm = self.decision_metrics()
        tp = dm["true_positives"]
        fp = dm["false_positives"]
        tn = dm["true_negatives"]
        fn = dm["false_negatives"]

        total = tp + fp + tn + fn
        overall_acc = round((tp + tn) / total * 100, 1) if total > 0 else 0.0

        return {
            "true_positives":   tp,
            "false_positives":  fp,
            "true_negatives":   tn,
            "false_negatives":  fn,
            "total":            total,
            "overall_accuracy": overall_acc,
            "table": {
                "predicted_grade_actual_profit":       tp,
                "predicted_grade_actual_loss":         fp,
                "predicted_no_grade_actual_loss":      tn,
                "predicted_no_grade_actual_profit":    fn,
            }
        }

    # ── Grade band breakdown ───────────────────────────────────────────────────

    def band_breakdown(self) -> Dict[str, Any]:
        """
        Per-grade-band accuracy analysis.

        Shows which grade ranges the system is most/least accurate at.
        Useful for detecting systematic bias at specific grade levels.
        """
        bands: Dict[str, Dict] = {}

        for r in self.records:
            band = r.grade_band
            if band not in bands:
                bands[band] = {
                    "count":   0,
                    "errors":  [],
                    "correct": 0,
                }
            bands[band]["count"] += 1
            bands[band]["errors"].append(r.abs_grade_error)
            if r.band_correct:
                bands[band]["correct"] += 1

        result = {}
        for band, data in sorted(bands.items(), key=lambda x: -_band_order(x[0])):
            errors = data["errors"]
            n      = data["count"]
            result[band] = {
                "count":           n,
                "mae":             round(sum(errors) / n, 3),
                "band_accuracy":   round(data["correct"] / n * 100, 1),
                "within_half_pct": round(
                    sum(1 for e in errors if e <= 0.5) / n * 100, 1
                ),
            }

        return result

    # ── Rolling trend ──────────────────────────────────────────────────────────

    def trend_metrics(self, window: int = 5) -> Dict[str, Any]:
        """
        Rolling performance metrics over time (sorted by timestamp).

        Returns rolling MAE and bias computed over a sliding window.
        Useful for detecting whether system accuracy is improving or degrading.
        """
        if len(self.records) < 2:
            return {"note": "Insufficient records for trend analysis."}

        # Build time series
        series = []
        for i, r in enumerate(self.records):
            start = max(0, i - window + 1)
            window_records = self.records[start: i + 1]
            w_errors = [rec.grade_error for rec in window_records]
            w_abs    = [rec.abs_grade_error for rec in window_records]
            w_n      = len(window_records)
            series.append({
                "index":       i,
                "card_id":     r.card_id,
                "timestamp":   r.timestamp,
                "rolling_mae": round(sum(w_abs) / w_n, 3),
                "rolling_bias": round(sum(w_errors) / w_n, 3),
                "grade_error": r.grade_error,
            })

        # Overall trend direction: compare first-half vs second-half MAE
        mid   = len(series) // 2
        first_half_mae  = sum(s["rolling_mae"] for s in series[:mid])  / mid if mid else 0
        second_half_mae = sum(s["rolling_mae"] for s in series[mid:]) / (len(series) - mid) if len(series) > mid else 0

        trend_direction = "improving" if second_half_mae < first_half_mae else (
            "degrading" if second_half_mae > first_half_mae else "stable"
        )

        return {
            "window_size":     window,
            "series":          series,
            "trend_direction": trend_direction,
            "first_half_mae":  round(first_half_mae, 3),
            "second_half_mae": round(second_half_mae, 3),
        }

    # ── Per-card detail ────────────────────────────────────────────────────────

    def _per_card_detail(self) -> List[Dict]:
        """Full per-record breakdown for reporting."""
        rows = []
        for r in self.records:
            row = {
                "card_id":         r.card_id,
                "card_name":       r.card_name or r.card_id,
                "predicted_grade": r.predicted_grade,
                "final_grade":     r.final_grade,
                "grade_error":     r.grade_error,
                "band_correct":    r.band_correct,
                "within_half":     r.within_half,
                "roi_decision":    r.roi_decision,
                "is_grade_call":   r.is_grade_call,
                "actual_profit":   r.actual_outcome_profit,
                "predicted_profit": r.predicted_profit,
                "profit_error":    r.profit_error,
                "is_profitable":   r.is_profitable,
                "timestamp":       r.timestamp,
            }
            rows.append(row)
        return rows


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _optional_float(v: Any) -> Optional[float]:
    """Convert to float or None."""
    return float(v) if v is not None else None


def _grade_to_band(grade: float) -> str:
    """Map a numeric grade to a PSA-style band bucket."""
    if grade >= 9.5:  return "PSA 9.5-10"
    if grade >= 9.0:  return "PSA 9"
    if grade >= 8.0:  return "PSA 8-8.5"
    if grade >= 7.0:  return "PSA 7-7.5"
    if grade >= 5.0:  return "PSA 5-6"
    return "PSA 1-4"


def _band_order(band: str) -> int:
    """Sort key for grade bands (higher = better)."""
    order = {
        "PSA 9.5-10": 6, "PSA 9": 5, "PSA 8-8.5": 4,
        "PSA 7-7.5": 3,  "PSA 5-6": 2, "PSA 1-4": 1,
    }
    return order.get(band, 0)


def _bucket_errors(abs_errors: List[float]) -> Dict[str, int]:
    """Count absolute errors by bucket for distribution analysis."""
    n = len(abs_errors)
    if n == 0:
        return {}
    return {
        "exact (0)":          sum(1 for e in abs_errors if e == 0.0),
        "within_0.5":         sum(1 for e in abs_errors if 0.0 < e <= 0.5),
        "within_1.0":         sum(1 for e in abs_errors if 0.5 < e <= 1.0),
        "over_1.0":           sum(1 for e in abs_errors if e > 1.0),
    }


def _decision_correct(r: EvaluationRecord) -> bool:
    """
    A decision is 'correct' if:
    - System predicted GRADE/STRONG GRADE AND grade was within ±1 of final
    - System predicted no-grade AND grade was outside top tier (final < 9)
    - Or grade error was within ±0.5 regardless of decision
    """
    if r.within_half:
        return True
    if r.is_grade_call and r.within_one:
        return True
    if not r.is_grade_call and r.final_grade < 9.0:
        return True
    return False
