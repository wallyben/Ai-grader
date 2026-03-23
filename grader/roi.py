"""
ROI Engine — Phase 2
Computes expected value, profit, ROI%, and grading decision from pricing + grade data.

Decision ladder:
  STRONG GRADE  — high ROI + high profit + high confidence
  GRADE         — positive ROI with reasonable confidence
  CONDITIONAL   — marginal ROI or borderline confidence
  HOLD RAW      — grading cost exceeds slab premium, but card has raw value
  DO NOT GRADE  — clearly unprofitable to grade

All calculations are deterministic and fully auditable.
No randomness. No black-box ML.
"""

from typing import Any, Dict, List, Optional

# ─── Decision thresholds ──────────────────────────────────────────────────────
STRONG_GRADE_ROI_PCT   = 150.0   # ROI% for STRONG GRADE tier
GRADE_ROI_PCT          = 40.0    # ROI% for GRADE tier
CONDITIONAL_ROI_PCT    = 5.0     # ROI% for CONDITIONAL tier
STRONG_GRADE_MIN_PROFIT = 60.0   # Minimum absolute profit for STRONG GRADE
MIN_CONFIDENCE_FULL     = 0.65   # Below this: cap at CONDITIONAL
HIGH_RISK_THRESHOLD     = 0.40   # p(below PSA 8) above this = "high risk"


# ─── Core ROI computation ─────────────────────────────────────────────────────

def compute_roi(
    estimated_grade:  float,
    confidence:       float,
    raw_price:        float,
    grading_cost:     float,
    psa_8_price:      Optional[float] = None,
    psa_9_price:      Optional[float] = None,
    psa_10_price:     Optional[float] = None,
    card_name:        Optional[str]   = None,
) -> Dict[str, Any]:
    """
    Compute ROI for grading a card.

    Args:
        estimated_grade:  Calibrated grade (1–10)
        confidence:       Calibrated confidence (0–1)
        raw_price:        Current raw card value
        grading_cost:     Fee to have the card graded (e.g. PSA service level)
        psa_8_price:      Market value of a PSA 8 slab (optional)
        psa_9_price:      Market value of a PSA 9 slab (optional)
        psa_10_price:     Market value of a PSA 10 slab (optional)
        card_name:        Optional card name for display

    Returns:
        Comprehensive ROI dict including expected_value, profit, roi_percent,
        grade_probabilities, decision, downside metrics.
    """
    if raw_price < 0:
        raise ValueError("raw_price must be non-negative")
    if grading_cost < 0:
        raise ValueError("grading_cost must be non-negative")

    total_investment = raw_price + grading_cost

    # ── Fill missing price tiers with multiplier estimates ────────────────────
    p8  = psa_8_price  if psa_8_price  is not None else round(raw_price * 2.0, 2)
    p9  = psa_9_price  if psa_9_price  is not None else round(raw_price * 3.5, 2)
    p10 = psa_10_price if psa_10_price is not None else round(raw_price * 8.0, 2)
    prices_estimated = (psa_8_price is None or psa_9_price is None or psa_10_price is None)

    # ── Grade probability distribution ────────────────────────────────────────
    probs = _grade_probabilities(estimated_grade, confidence)

    # ── Expected value ─────────────────────────────────────────────────────────
    # Weighted sum over outcomes.
    # Downside scenario: grade < PSA 8 → assume card sells at raw_price
    ev = (
        probs["p_10"]      * p10 +
        probs["p_9"]       * p9  +
        probs["p_8"]       * p8  +
        probs["p_below_8"] * raw_price
    )
    ev = round(ev, 2)

    profit      = round(ev - total_investment, 2)
    roi_percent = round((profit / total_investment * 100) if total_investment > 0 else 0.0, 1)

    # ── Downside risk ──────────────────────────────────────────────────────────
    # Maximum loss = grading cost (if grade < 8, sell raw, recoup raw_price)
    downside_loss     = round(grading_cost, 2)
    downside_risk_pct = round(probs["p_below_8"] * 100, 1)

    # ── Best-case / worst-case ─────────────────────────────────────────────────
    best_case_profit  = round(p10 - total_investment, 2)
    worst_case_profit = round(raw_price - total_investment, 2)

    # ── Decision ──────────────────────────────────────────────────────────────
    decision = _make_decision(
        roi_pct=roi_percent,
        profit=profit,
        confidence=confidence,
        grade=estimated_grade,
        p_below_8=probs["p_below_8"],
    )

    return {
        "card_name":          card_name or "Unknown",
        "estimated_grade":    estimated_grade,
        "confidence":         confidence,
        # --- Prices ---
        "raw_value":          round(raw_price, 2),
        "psa_8_value":        round(p8,  2),
        "psa_9_value":        round(p9,  2),
        "psa_10_value":       round(p10, 2),
        "grading_cost":       round(grading_cost, 2),
        "total_investment":   round(total_investment, 2),
        "prices_estimated":   prices_estimated,
        # --- Grade probabilities ---
        "grade_probabilities": {
            "psa_10":      round(probs["p_10"],      3),
            "psa_9":       round(probs["p_9"],       3),
            "psa_8":       round(probs["p_8"],       3),
            "below_psa_8": round(probs["p_below_8"], 3),
        },
        # --- Financial outputs ---
        "expected_value":     ev,
        "profit":             profit,
        "roi_percent":        roi_percent,
        "best_case_profit":   best_case_profit,
        "worst_case_profit":  worst_case_profit,
        "downside_loss":      downside_loss,
        "downside_risk_pct":  downside_risk_pct,
        # --- Decision ---
        "decision":           decision,
        "confidence_adjusted": confidence < 0.90,
    }


# ─── Grade probability model ──────────────────────────────────────────────────

def _grade_probabilities(grade: float, confidence: float) -> Dict[str, float]:
    """
    Deterministic triangular probability model.

    Maps (estimated_grade, confidence) → P(PSA 10 | 9 | 8 | <8).

    The grade estimate has uncertainty proportional to (1 - confidence).
    A triangular distribution centered on `grade` models the spread.
    Higher confidence = tighter distribution = less probability mass in tails.
    """
    # Spread: at 100% confidence spread=0, at 65% spread≈0.875
    spread = (1.0 - confidence) * 2.5

    def _tri_cdf(x: float, center: float, half_width: float) -> float:
        """Triangular CDF: P(grade_outcome <= x)."""
        if half_width <= 0.0:
            return 1.0 if x >= center else 0.0
        lo = center - half_width
        hi = center + half_width
        if x <= lo:
            return 0.0
        if x >= hi:
            return 1.0
        if x <= center:
            return ((x - lo) ** 2) / ((hi - lo) * (center - lo))
        return 1.0 - ((hi - x) ** 2) / ((hi - lo) * (hi - center))

    # PSA grade boundaries: <8, [8–8.5), [9–9.5), 10
    p_below_8 = _tri_cdf(7.75,  grade, spread)
    p_8       = _tri_cdf(8.75,  grade, spread) - _tri_cdf(7.75, grade, spread)
    p_9       = _tri_cdf(9.75,  grade, spread) - _tri_cdf(8.75, grade, spread)
    p_10      = 1.0 - _tri_cdf(9.75,  grade, spread)

    # Clamp negatives (numerical edge cases at boundary)
    p_below_8 = max(0.0, p_below_8)
    p_8       = max(0.0, p_8)
    p_9       = max(0.0, p_9)
    p_10      = max(0.0, p_10)

    # Normalise to ensure probabilities sum to 1
    total = p_below_8 + p_8 + p_9 + p_10
    if total > 0:
        p_below_8 /= total
        p_8       /= total
        p_9       /= total
        p_10      /= total

    return {
        "p_below_8": p_below_8,
        "p_8":       p_8,
        "p_9":       p_9,
        "p_10":      p_10,
    }


# ─── Decision engine ──────────────────────────────────────────────────────────

def _make_decision(
    roi_pct:   float,
    profit:    float,
    confidence: float,
    grade:     float,
    p_below_8: float,
) -> str:
    """
    Five-tier grading decision based on financial metrics and risk.

    Decision factors:
    - ROI%: primary signal for profitability
    - Profit: absolute floor (high ROI on tiny value cards still matters)
    - Confidence: uncertainty penalty — low confidence caps upside
    - Grade: minimum grade threshold for grading viability
    - p_below_8: downside risk guard
    """
    high_risk = p_below_8 > HIGH_RISK_THRESHOLD

    # Low confidence ceiling: don't recommend GRADE or STRONG GRADE if uncertain
    if confidence < MIN_CONFIDENCE_FULL:
        if roi_pct >= GRADE_ROI_PCT and not high_risk:
            return "CONDITIONAL"
        if profit > 0:
            return "HOLD RAW"
        return "DO NOT GRADE"

    # Main decision ladder (descending threshold)
    if (not high_risk
            and roi_pct >= STRONG_GRADE_ROI_PCT
            and profit >= STRONG_GRADE_MIN_PROFIT):
        return "STRONG GRADE"

    if roi_pct >= GRADE_ROI_PCT and not high_risk:
        return "GRADE"

    if roi_pct >= CONDITIONAL_ROI_PCT or (profit > 0 and grade >= 7.0):
        return "CONDITIONAL"

    if profit > 0:
        return "HOLD RAW"

    return "DO NOT GRADE"


# ─── Batch ranking ─────────────────────────────────────────────────────────────

# Decision priority order (lower = submit first)
_DECISION_RANK: Dict[str, int] = {
    "STRONG GRADE": 0,
    "GRADE":        1,
    "CONDITIONAL":  2,
    "HOLD RAW":     3,
    "DO NOT GRADE": 4,
}


def rank_batch_by_roi(roi_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rank a list of ROI result dicts for optimal batch submission ordering.

    Sort order (primary → tertiary):
    1. Decision tier (STRONG GRADE first, DO NOT GRADE last)
    2. Highest absolute profit
    3. Highest ROI%
    4. Lowest downside risk %

    Returns new sorted list — does NOT modify input.
    """
    def _key(r: Dict[str, Any]) -> tuple:
        return (
            _DECISION_RANK.get(r.get("decision", "DO NOT GRADE"), 5),
            -r.get("profit", 0.0),
            -r.get("roi_percent", 0.0),
             r.get("downside_risk_pct", 100.0),
        )

    return sorted(roi_results, key=_key)


def batch_roi_summary(roi_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute portfolio-level summary statistics for a ranked batch.

    Returns:
        total_investment, total_expected_value, portfolio_profit,
        portfolio_roi_pct, submission_candidates (GRADE + STRONG GRADE),
        decision_counts
    """
    valid = [r for r in roi_results if r.get("expected_value") is not None]
    if not valid:
        return {}

    total_investment = sum(r["total_investment"] for r in valid)
    total_ev         = sum(r["expected_value"]   for r in valid)
    total_profit     = round(total_ev - total_investment, 2)
    portfolio_roi    = round(
        (total_profit / total_investment * 100) if total_investment > 0 else 0.0, 1
    )

    decision_counts: Dict[str, int] = {}
    for r in valid:
        d = r.get("decision", "Unknown")
        decision_counts[d] = decision_counts.get(d, 0) + 1

    candidates = [
        r for r in valid
        if r.get("decision") in ("STRONG GRADE", "GRADE")
    ]

    return {
        "total_cards":        len(valid),
        "total_investment":   round(total_investment, 2),
        "total_expected_value": round(total_ev, 2),
        "portfolio_profit":   total_profit,
        "portfolio_roi_pct":  portfolio_roi,
        "submission_candidates": len(candidates),
        "decision_counts":    decision_counts,
    }
