"""
Decision Optimizer — Phase 3
Analyses real-world ROI outcomes to detect systematic decision errors
and recommend threshold adjustments for the decision engine.

Operates entirely on evaluation data — no ML, no randomness.
Produces actionable threshold recommendations based on observed profitability
patterns per decision tier and grade band.

Does NOT mutate roi.py directly. Returns adjustment recommendations that can
be applied by the caller (or committed to roi.py after review).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .evaluation import EvaluationRecord, GRADE_DECISIONS, NO_GRADE_DECISIONS


# ─── Score weight constants ────────────────────────────────────────────────────
# Used when accumulating a per-tier "trust score" from outcomes
CORRECT_GRADE_REWARD:    float =  1.0   # Correct GRADE call → profitable
CORRECT_NO_GRADE_REWARD: float =  0.5   # Correct NO-GRADE call
FALSE_POSITIVE_PENALTY:  float = -2.0   # GRADE call → loss
FALSE_NEGATIVE_PENALTY:  float = -1.0   # NO-GRADE call → missed profit

# Minimum records required before recommending threshold changes
MIN_RECORDS_FOR_ADJUSTMENT: int = 5

# Threshold change step sizes
ROI_THRESHOLD_STEP:     float = 5.0    # Adjust ROI% thresholds by this increment
PROFIT_THRESHOLD_STEP:  float = 10.0   # Adjust minimum profit thresholds by this


class DecisionOptimizer:
    """
    Analyses evaluation outcomes to score decision accuracy per tier
    and recommend threshold adjustments.

    Usage:
        optimizer = DecisionOptimizer(records)
        report    = optimizer.optimize()
        print(report["recommendations"])
    """

    def __init__(self, records: List[EvaluationRecord]) -> None:
        self.records = [r for r in records if r.actual_outcome_profit is not None]

    # ── Public API ─────────────────────────────────────────────────────────────

    def optimize(self) -> Dict[str, Any]:
        """
        Run full optimization analysis.

        Returns:
            tier_scores          — trust score per decision tier
            tier_analysis        — detailed accuracy breakdown per tier
            recommendations      — list of threshold adjustment recommendations
            grade_band_bias      — where the system over/under-grades
            weaknesses           — identified systematic weaknesses
            threshold_adjustments — concrete parameter changes suggested
        """
        tier_analysis   = self._analyse_tiers()
        grade_band_bias = self._grade_band_bias()
        weaknesses      = self._identify_weaknesses(tier_analysis, grade_band_bias)
        adjustments     = self._recommend_adjustments(tier_analysis, grade_band_bias)

        return {
            "n_records_with_outcomes": len(self.records),
            "tier_analysis":           tier_analysis,
            "grade_band_bias":         grade_band_bias,
            "weaknesses":              weaknesses,
            "threshold_adjustments":   adjustments,
            "recommendations":         self._narrative_recommendations(
                                           weaknesses, adjustments
                                       ),
        }

    # ── Tier analysis ──────────────────────────────────────────────────────────

    def _analyse_tiers(self) -> Dict[str, Dict]:
        """Score and analyse each decision tier independently."""
        tiers: Dict[str, Dict] = {}

        for r in self.records:
            t = r.roi_decision
            if t not in tiers:
                tiers[t] = {
                    "count":            0,
                    "profitable":       0,
                    "loss":             0,
                    "total_profit":     0.0,
                    "score":            0.0,
                    "profits":          [],
                    "grade_errors":     [],
                }
            data = tiers[t]
            data["count"]        += 1
            data["total_profit"] += r.actual_outcome_profit
            data["profits"].append(r.actual_outcome_profit)
            data["grade_errors"].append(r.grade_error)

            if r.is_profitable:
                data["profitable"] += 1
                # Reward for correct grade calls, partial reward for no-grade calls
                if r.is_grade_call:
                    data["score"] += CORRECT_GRADE_REWARD
                else:
                    data["score"] += FALSE_NEGATIVE_PENALTY  # missed profitable card
            else:
                data["loss"] += 1
                if r.is_grade_call:
                    data["score"] += FALSE_POSITIVE_PENALTY  # wrong GRADE call
                else:
                    data["score"] += CORRECT_NO_GRADE_REWARD

        # Compute summary stats per tier
        result = {}
        for t, data in tiers.items():
            n       = data["count"]
            profits = data.pop("profits")
            errors  = data.pop("grade_errors")

            data["total_profit"]     = round(data["total_profit"], 2)
            data["avg_profit"]       = round(data["total_profit"] / n, 2)
            data["profitable_rate"]  = round(data["profitable"] / n * 100, 1)
            data["normalised_score"] = round(data["score"] / n, 3)  # per-card score
            data["avg_grade_error"]  = round(sum(errors) / n, 3)    # +ve = over-graded
            data["mae"]              = round(sum(abs(e) for e in errors) / n, 3)

            # Quartile profit analysis
            sorted_profits = sorted(profits)
            mid_idx = n // 2
            data["median_profit"] = round(sorted_profits[mid_idx], 2)
            data["min_profit"]    = round(sorted_profits[0], 2)
            data["max_profit"]    = round(sorted_profits[-1], 2)

            result[t] = data

        return result

    # ── Grade band bias detection ──────────────────────────────────────────────

    def _grade_band_bias(self) -> Dict[str, Dict]:
        """
        Detect systematic over/under-grading by final grade band.

        Returns per-band bias (mean signed error) and direction.
        """
        bands: Dict[str, List[float]] = {}
        for r in self.records:
            band = _grade_to_band(r.final_grade)
            bands.setdefault(band, []).append(r.grade_error)

        result = {}
        for band, errors in bands.items():
            n    = len(errors)
            bias = sum(errors) / n
            result[band] = {
                "count":     n,
                "bias":      round(bias, 3),    # + = system over-grades in this band
                "mae":       round(sum(abs(e) for e in errors) / n, 3),
                "direction": "over-graded" if bias > 0.15 else (
                             "under-graded" if bias < -0.15 else "accurate"),
            }

        return result

    # ── Weakness identification ────────────────────────────────────────────────

    def _identify_weaknesses(
        self,
        tier_analysis:   Dict[str, Dict],
        grade_band_bias: Dict[str, Dict],
    ) -> List[Dict]:
        """
        Identify systematic weaknesses from tier and band analysis.

        Returns a list of weakness dicts, each with:
            type, description, severity, affected_tier/band, evidence
        """
        weaknesses = []

        # Weakness 1: High false-positive rate in GRADE/STRONG GRADE tiers
        for tier in ("GRADE", "STRONG GRADE"):
            if tier not in tier_analysis:
                continue
            data = tier_analysis[tier]
            if data["count"] < MIN_RECORDS_FOR_ADJUSTMENT:
                continue
            fp_rate = 100 - data["profitable_rate"]
            if fp_rate >= 30:
                weaknesses.append({
                    "type":         "high_false_positive_rate",
                    "tier":         tier,
                    "severity":     "high" if fp_rate >= 50 else "medium",
                    "description":  (
                        f"{tier} tier has {fp_rate:.0f}% unprofitable outcomes. "
                        "System is recommending too many cards for grading."
                    ),
                    "evidence":     f"{data['loss']}/{data['count']} bad calls",
                })

        # Weakness 2: Missed profitable cards in CONDITIONAL/HOLD RAW
        for tier in ("CONDITIONAL", "HOLD RAW"):
            if tier not in tier_analysis:
                continue
            data = tier_analysis[tier]
            if data["count"] < MIN_RECORDS_FOR_ADJUSTMENT:
                continue
            fn_rate = data["profitable_rate"]  # profitable but not recommended to grade
            if fn_rate >= 50:
                weaknesses.append({
                    "type":        "high_false_negative_rate",
                    "tier":        tier,
                    "severity":    "medium",
                    "description": (
                        f"{tier} tier has {fn_rate:.0f}% profitable outcomes. "
                        "System is under-recommending grading for viable cards."
                    ),
                    "evidence":    f"{data['profitable']}/{data['count']} profitable",
                })

        # Weakness 3: Systematic grade bias by band
        for band, info in grade_band_bias.items():
            if info["count"] < 3:
                continue
            if abs(info["bias"]) >= 0.4:
                weaknesses.append({
                    "type":        "grade_band_bias",
                    "band":        band,
                    "severity":    "high" if abs(info["bias"]) >= 0.7 else "medium",
                    "description": (
                        f"System consistently {info['direction']} cards in {band} "
                        f"by {abs(info['bias']):.2f} grade points on average."
                    ),
                    "evidence":    f"bias={info['bias']:+.3f} over {info['count']} cards",
                })

        # Weakness 4: Overall low score for a tier
        for tier, data in tier_analysis.items():
            if data["count"] < MIN_RECORDS_FOR_ADJUSTMENT:
                continue
            if data["normalised_score"] < -0.5:
                weaknesses.append({
                    "type":        "poor_tier_performance",
                    "tier":        tier,
                    "severity":    "high",
                    "description": (
                        f"{tier} decisions are net-negative "
                        f"(score per card: {data['normalised_score']:.2f}). "
                        "Threshold review strongly recommended."
                    ),
                    "evidence":    f"avg_profit={data['avg_profit']:.0f}",
                })

        return weaknesses

    # ── Threshold recommendations ──────────────────────────────────────────────

    def _recommend_adjustments(
        self,
        tier_analysis:   Dict[str, Dict],
        grade_band_bias: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """
        Recommend concrete threshold adjustments for the decision engine.

        Returns parameter names and suggested new values.
        These map directly to constants in roi.py.
        """
        from .roi import (
            STRONG_GRADE_ROI_PCT,
            GRADE_ROI_PCT,
            CONDITIONAL_ROI_PCT,
            STRONG_GRADE_MIN_PROFIT,
        )

        adjustments: Dict[str, Any] = {
            "STRONG_GRADE_ROI_PCT":    STRONG_GRADE_ROI_PCT,
            "GRADE_ROI_PCT":           GRADE_ROI_PCT,
            "CONDITIONAL_ROI_PCT":     CONDITIONAL_ROI_PCT,
            "STRONG_GRADE_MIN_PROFIT": STRONG_GRADE_MIN_PROFIT,
            "changes": [],
        }

        # If STRONG GRADE has many bad calls → raise ROI threshold
        if "STRONG GRADE" in tier_analysis:
            sg = tier_analysis["STRONG GRADE"]
            if sg["count"] >= MIN_RECORDS_FOR_ADJUSTMENT:
                fp_rate = 100 - sg["profitable_rate"]
                if fp_rate >= 30:
                    new_val = round(STRONG_GRADE_ROI_PCT + ROI_THRESHOLD_STEP, 1)
                    adjustments["STRONG_GRADE_ROI_PCT"] = new_val
                    adjustments["changes"].append({
                        "param":  "STRONG_GRADE_ROI_PCT",
                        "from":   STRONG_GRADE_ROI_PCT,
                        "to":     new_val,
                        "reason": f"STRONG GRADE FP rate {fp_rate:.0f}% — raise ROI threshold",
                    })
                    # Also raise minimum profit floor
                    new_profit = round(STRONG_GRADE_MIN_PROFIT + PROFIT_THRESHOLD_STEP, 1)
                    adjustments["STRONG_GRADE_MIN_PROFIT"] = new_profit
                    adjustments["changes"].append({
                        "param":  "STRONG_GRADE_MIN_PROFIT",
                        "from":   STRONG_GRADE_MIN_PROFIT,
                        "to":     new_profit,
                        "reason": "Raising profit floor alongside ROI threshold",
                    })

        # If GRADE has many bad calls → raise GRADE threshold
        if "GRADE" in tier_analysis:
            g = tier_analysis["GRADE"]
            if g["count"] >= MIN_RECORDS_FOR_ADJUSTMENT:
                fp_rate = 100 - g["profitable_rate"]
                if fp_rate >= 30:
                    new_val = round(GRADE_ROI_PCT + ROI_THRESHOLD_STEP, 1)
                    adjustments["GRADE_ROI_PCT"] = new_val
                    adjustments["changes"].append({
                        "param":  "GRADE_ROI_PCT",
                        "from":   GRADE_ROI_PCT,
                        "to":     new_val,
                        "reason": f"GRADE FP rate {fp_rate:.0f}% — raise ROI threshold",
                    })

        # If CONDITIONAL has high profitable rate → lower GRADE threshold
        # (we're leaving money on the table by classifying them as CONDITIONAL)
        if "CONDITIONAL" in tier_analysis:
            c = tier_analysis["CONDITIONAL"]
            if c["count"] >= MIN_RECORDS_FOR_ADJUSTMENT and c["profitable_rate"] >= 70:
                new_val = round(max(20.0, GRADE_ROI_PCT - ROI_THRESHOLD_STEP), 1)
                adjustments["GRADE_ROI_PCT"] = new_val
                adjustments["changes"].append({
                    "param":  "GRADE_ROI_PCT",
                    "from":   GRADE_ROI_PCT,
                    "to":     new_val,
                    "reason": (
                        f"CONDITIONAL profitable rate {c['profitable_rate']:.0f}% — "
                        "lower GRADE threshold to capture more value"
                    ),
                })

        return adjustments

    # ── Narrative ─────────────────────────────────────────────────────────────

    def _narrative_recommendations(
        self,
        weaknesses:  List[Dict],
        adjustments: Dict[str, Any],
    ) -> List[str]:
        """Convert weaknesses + adjustments into human-readable recommendations."""
        recs = []

        if not weaknesses and not adjustments["changes"]:
            recs.append(
                "System performance is within acceptable bounds. "
                "No threshold adjustments recommended at this time."
            )
            return recs

        for w in weaknesses:
            recs.append(f"[{w['severity'].upper()}] {w['description']}")

        for change in adjustments["changes"]:
            recs.append(
                f"ADJUST {change['param']}: {change['from']} → {change['to']} "
                f"({change['reason']})"
            )

        return recs


# ─── Helper ───────────────────────────────────────────────────────────────────

def _grade_to_band(grade: float) -> str:
    if grade >= 9.5:  return "PSA 9.5-10"
    if grade >= 9.0:  return "PSA 9"
    if grade >= 8.0:  return "PSA 8-8.5"
    if grade >= 7.0:  return "PSA 7-7.5"
    if grade >= 5.0:  return "PSA 5-6"
    return "PSA 1-4"
