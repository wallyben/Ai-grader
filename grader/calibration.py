"""
Calibration Engine — Phase 2 + Phase 3
Aligns system grades to real-world PSA outcomes using bias/variance correction
and piecewise linear interpolation.

Phase 3 additions:
- update_from_outcomes(): ingest real evaluation records to re-fit calibration
- Rolling performance tracking (n most-recent outcomes)
- Calibration history log (every time the model is updated)
- save_state() / load_state() for persistence between sessions

Apply AFTER scoring but BEFORE final output to correct systematic over/under-grading.
All outputs are fully deterministic — no randomness.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ─── Default calibration dataset ──────────────────────────────────────────────
# Derived from observed PSA alignment: system tends to be slightly optimistic
# at high grades (9+) and accurate at mid-range (5–8).
DEFAULT_CALIBRATION_DATA: List[Dict] = [
    {"predicted_grade": 10.0, "actual_grade": 10},
    {"predicted_grade": 9.8,  "actual_grade": 10},
    {"predicted_grade": 9.5,  "actual_grade": 9},
    {"predicted_grade": 9.2,  "actual_grade": 9},
    {"predicted_grade": 9.0,  "actual_grade": 9},
    {"predicted_grade": 8.7,  "actual_grade": 8},
    {"predicted_grade": 8.5,  "actual_grade": 8},
    {"predicted_grade": 8.0,  "actual_grade": 8},
    {"predicted_grade": 7.5,  "actual_grade": 7},
    {"predicted_grade": 7.0,  "actual_grade": 7},
    {"predicted_grade": 6.5,  "actual_grade": 6},
    {"predicted_grade": 6.0,  "actual_grade": 6},
    {"predicted_grade": 5.0,  "actual_grade": 5},
    {"predicted_grade": 4.0,  "actual_grade": 4},
    {"predicted_grade": 3.0,  "actual_grade": 3},
]

# Maximum number of recent outcomes to retain in the rolling window
DEFAULT_ROLLING_WINDOW = 50


class CalibrationEngine:
    """
    Aligns predicted grades to real-world PSA outcomes.

    Phase 2 behaviour (original):
    1. Fit: compute bias, variance from calibration dataset.
    2. Build piecewise linear correction table.
    3. calibrate(predicted, confidence) → calibrated grade + confidence adjustment.

    Phase 3 additions:
    4. update_from_outcomes(): absorb new real outcomes, re-fit automatically.
    5. Rolling history: keep the N most recent outcomes for rolling stats.
    6. Calibration history log: every re-fit is recorded with timestamp + stats.
    7. save_state() / load_state(): persist and restore calibration state to JSON.
    """

    def __init__(
        self,
        calibration_data:   Optional[List[Dict]] = None,
        rolling_window:     int = DEFAULT_ROLLING_WINDOW,
    ) -> None:
        # Core data: base calibration points (from config or defaults)
        self._base_data: List[Dict] = list(calibration_data or DEFAULT_CALIBRATION_DATA)

        # Rolling outcomes: real-world records added via update_from_outcomes()
        # Format: [{"predicted_grade": float, "actual_grade": float}, ...]
        self._rolling_outcomes: List[Dict] = []
        self._rolling_window: int = rolling_window

        # Calibration history: log of every re-fit event
        self._calibration_history: List[Dict] = []

        # Internal state
        self._bias: float = 0.0
        self._variance: float = 0.0
        self._correction_table: List[Tuple[float, float]] = []

        self._fit()

    # ── Fitting ────────────────────────────────────────────────────────────────

    def _fit(self) -> None:
        """
        Re-fit bias, variance and correction table from merged data.

        Merges base calibration data with recent rolling outcomes.
        Rolling outcomes take precedence (they are more recent and real).
        """
        merged = list(self._base_data) + list(self._rolling_outcomes)
        if not merged:
            return

        errors = [d["predicted_grade"] - d["actual_grade"] for d in merged]
        n = len(errors)
        self._bias     = sum(errors) / n
        self._variance = sum((e - self._bias) ** 2 for e in errors) / n

        # Piecewise correction table: sorted by predicted_grade
        # De-duplicate by averaging actual grades for identical predicted values
        _table_map: Dict[float, List[float]] = {}
        for d in merged:
            pg = d["predicted_grade"]
            _table_map.setdefault(pg, []).append(float(d["actual_grade"]))

        self._correction_table = sorted(
            [(pg, sum(actuals) / len(actuals)) for pg, actuals in _table_map.items()],
            key=lambda x: x[0],
        )

    # ── Interpolation ──────────────────────────────────────────────────────────

    def _interpolate(self, predicted: float) -> float:
        """Piecewise linear interpolation from correction table."""
        if not self._correction_table:
            return predicted

        # Clamp to range
        if predicted <= self._correction_table[0][0]:
            return self._correction_table[0][1]
        if predicted >= self._correction_table[-1][0]:
            return self._correction_table[-1][1]

        # Find bounding segment
        for i in range(len(self._correction_table) - 1):
            x0, y0 = self._correction_table[i]
            x1, y1 = self._correction_table[i + 1]
            if x0 <= predicted <= x1:
                if x1 == x0:
                    return y0
                t = (predicted - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)

        return predicted

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def bias(self) -> float:
        """Mean prediction error (positive = system over-grades)."""
        return round(self._bias, 4)

    @property
    def variance(self) -> float:
        """Variance of prediction errors."""
        return round(self._variance, 4)

    @property
    def std_dev(self) -> float:
        """Standard deviation of prediction errors."""
        return round(math.sqrt(self._variance), 4)

    @property
    def rolling_window(self) -> int:
        return self._rolling_window

    @property
    def n_rolling_outcomes(self) -> int:
        return len(self._rolling_outcomes)

    @property
    def calibration_history(self) -> List[Dict]:
        """Read-only copy of the calibration update log."""
        return list(self._calibration_history)

    # ── Public API ─────────────────────────────────────────────────────────────

    def calibrate(self, predicted_grade: float, confidence: float) -> Dict[str, Any]:
        """
        Apply calibration to a predicted grade.

        Returns:
            calibrated_grade        – PSA-aligned grade (rounded to 0.5)
            raw_calibrated          – Continuous calibrated value before rounding
            calibration_adjustment  – delta vs predicted_grade
            calibrated_confidence   – confidence penalised by calibration variance
            bias                    – fitted model bias
            variance                – fitted model variance
            std_dev                 – standard deviation
        """
        raw = self._interpolate(predicted_grade)

        # Round to nearest 0.5 to match PSA granularity
        calibrated = round(round(raw * 2) / 2, 1)
        calibrated = max(1.0, min(10.0, calibrated))

        adjustment = round(calibrated - predicted_grade, 2)

        # Confidence adjustment: penalise by calibration variance
        # High variance = less reliable calibration = lower confidence
        variance_penalty = min(0.15, self._variance * 0.08)
        calibrated_conf  = round(max(0.40, confidence - variance_penalty), 2)

        return {
            "calibrated_grade":       calibrated,
            "raw_calibrated":         round(raw, 3),
            "calibration_adjustment": adjustment,
            "calibrated_confidence":  calibrated_conf,
            "bias":                   self.bias,
            "variance":               self.variance,
            "std_dev":                self.std_dev,
        }

    # ── Phase 3: Feedback loop ─────────────────────────────────────────────────

    def update_from_outcomes(
        self,
        outcomes: List[Dict],
        record_history: bool = True,
    ) -> Dict[str, Any]:
        """
        Absorb real-world grading outcomes and re-fit the calibration model.

        Each outcome must have:
            predicted_grade: float  — what the AI predicted
            actual_grade:    float  — what PSA returned

        Optional:
            card_id, timestamp, notes

        Rolling window behaviour:
            - New outcomes are appended to the rolling buffer.
            - If the buffer exceeds rolling_window, oldest outcomes are dropped.
            - Base calibration data is always retained as the prior.

        Returns:
            A summary of the re-fit: old bias, new bias, delta, n_outcomes.
        """
        if not outcomes:
            return {"updated": False, "reason": "No outcomes provided."}

        # Validate and normalise
        new_points: List[Dict] = []
        for o in outcomes:
            if "predicted_grade" not in o or "actual_grade" not in o:
                continue
            new_points.append({
                "predicted_grade": float(o["predicted_grade"]),
                "actual_grade":    float(o["actual_grade"]),
            })

        if not new_points:
            return {"updated": False, "reason": "No valid outcome records."}

        old_bias     = self.bias
        old_variance = self.variance
        old_n        = self.n_rolling_outcomes

        # Append to rolling buffer
        self._rolling_outcomes.extend(new_points)

        # Trim to rolling window (keep most recent)
        if len(self._rolling_outcomes) > self._rolling_window:
            self._rolling_outcomes = self._rolling_outcomes[-self._rolling_window:]

        # Re-fit
        self._fit()

        new_bias     = self.bias
        new_variance = self.variance
        bias_delta   = round(new_bias - old_bias, 4)

        result = {
            "updated":        True,
            "n_new_outcomes": len(new_points),
            "n_rolling_total": self.n_rolling_outcomes,
            "old_bias":       old_bias,
            "new_bias":       new_bias,
            "bias_delta":     bias_delta,
            "old_variance":   old_variance,
            "new_variance":   new_variance,
            "timestamp":      datetime.utcnow().isoformat() + "Z",
        }

        if record_history:
            self._calibration_history.append(result)

        return result

    def rolling_performance(self) -> Dict[str, Any]:
        """
        Compute rolling accuracy stats from the most-recent outcomes buffer.

        Returns per-outcome error stats for the rolling window only,
        useful for detecting recent drift vs baseline calibration.
        """
        if not self._rolling_outcomes:
            return {"n": 0, "note": "No rolling outcomes recorded yet."}

        errors     = [d["predicted_grade"] - d["actual_grade"] for d in self._rolling_outcomes]
        abs_errors = [abs(e) for e in errors]
        n          = len(errors)

        return {
            "n":          n,
            "bias":       round(sum(errors) / n, 4),
            "mae":        round(sum(abs_errors) / n, 4),
            "variance":   round(sum((e - sum(errors) / n) ** 2 for e in errors) / n, 4),
            "max_error":  round(max(abs_errors), 2),
            "min_error":  round(min(abs_errors), 2),
            "within_0.5_pct": round(sum(1 for e in abs_errors if e <= 0.5) / n * 100, 1),
        }

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_state(self, path: str) -> None:
        """
        Persist calibration engine state to a JSON file.

        Saves: rolling_outcomes, calibration_history, rolling_window.
        Base calibration data is NOT saved (it is loaded from config at startup).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = {
            "rolling_window":       self._rolling_window,
            "rolling_outcomes":     self._rolling_outcomes,
            "calibration_history":  self._calibration_history,
            "saved_at":             datetime.utcnow().isoformat() + "Z",
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)

    def load_state(self, path: str) -> bool:
        """
        Load calibration state from a previously saved JSON file.

        Returns True on success, False if file not found.
        Re-fits the model after loading.
        """
        if not os.path.exists(path):
            return False

        with open(path, "r", encoding="utf-8") as fh:
            state = json.load(fh)

        self._rolling_window        = state.get("rolling_window", self._rolling_window)
        self._rolling_outcomes      = state.get("rolling_outcomes", [])
        self._calibration_history   = state.get("calibration_history", [])

        self._fit()
        return True

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return calibration model diagnostics."""
        return {
            "n_samples":          len(self._base_data) + len(self._rolling_outcomes),
            "n_base_samples":     len(self._base_data),
            "n_rolling_outcomes": self.n_rolling_outcomes,
            "rolling_window":     self._rolling_window,
            "bias":               self.bias,
            "variance":           self.variance,
            "std_dev":            self.std_dev,
            "correction_points":  len(self._correction_table),
            "n_history_entries":  len(self._calibration_history),
            "grade_range": (
                self._correction_table[0][0]  if self._correction_table else None,
                self._correction_table[-1][0] if self._correction_table else None,
            ),
        }


# ─── Module-level default engine ──────────────────────────────────────────────

_default_engine = CalibrationEngine()


def calibrate_grade(
    predicted_grade: float,
    confidence:      float,
    custom_data:     Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Convenience function.
    Uses default calibration data unless custom_data is provided.
    """
    engine = CalibrationEngine(custom_data) if custom_data else _default_engine
    return engine.calibrate(predicted_grade, confidence)


def compute_calibration_stats(calibration_data: List[Dict]) -> Dict[str, Any]:
    """Fit a calibration engine on given data and return diagnostics."""
    engine = CalibrationEngine(calibration_data)
    return engine.summary()
