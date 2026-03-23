"""
Calibration Engine — Phase 2
Aligns system grades to real-world PSA outcomes using bias/variance correction
and piecewise linear interpolation.

Apply AFTER scoring but BEFORE final output to correct systematic over/under-grading.
All outputs are fully deterministic — no randomness.
"""

import math
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


class CalibrationEngine:
    """
    Aligns predicted grades to real-world PSA outcomes.

    Steps:
    1. Fit: compute bias, variance from calibration dataset.
    2. Build piecewise linear correction table.
    3. calibrate(predicted, confidence) → calibrated grade + confidence adjustment.
    """

    def __init__(self, calibration_data: Optional[List[Dict]] = None) -> None:
        self._data: List[Dict] = calibration_data or DEFAULT_CALIBRATION_DATA
        self._bias: float = 0.0
        self._variance: float = 0.0
        self._correction_table: List[Tuple[float, float]] = []
        self._fit()

    # ── Fitting ────────────────────────────────────────────────────────────────

    def _fit(self) -> None:
        """Compute bias, variance and build correction table."""
        if not self._data:
            return

        errors = [d["predicted_grade"] - d["actual_grade"] for d in self._data]
        n = len(errors)
        self._bias = sum(errors) / n
        self._variance = sum((e - self._bias) ** 2 for e in errors) / n

        # Piecewise correction table: sorted by predicted_grade
        points = sorted(self._data, key=lambda d: d["predicted_grade"])
        self._correction_table = [
            (d["predicted_grade"], float(d["actual_grade"])) for d in points
        ]

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
        calibrated_conf = round(max(0.40, confidence - variance_penalty), 2)

        return {
            "calibrated_grade":       calibrated,
            "raw_calibrated":         round(raw, 3),
            "calibration_adjustment": adjustment,
            "calibrated_confidence":  calibrated_conf,
            "bias":                   self.bias,
            "variance":               self.variance,
            "std_dev":                self.std_dev,
        }

    def summary(self) -> Dict[str, Any]:
        """Return calibration model diagnostics."""
        return {
            "n_samples":         len(self._data),
            "bias":              self.bias,
            "variance":          self.variance,
            "std_dev":           self.std_dev,
            "correction_points": len(self._correction_table),
            "grade_range":       (
                self._correction_table[0][0] if self._correction_table else None,
                self._correction_table[-1][0] if self._correction_table else None,
            ),
        }


# ─── Module-level default engine ──────────────────────────────────────────────

_default_engine = CalibrationEngine()


def calibrate_grade(
    predicted_grade: float,
    confidence: float,
    custom_data: Optional[List[Dict]] = None,
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
