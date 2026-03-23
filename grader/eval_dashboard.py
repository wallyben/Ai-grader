"""
Evaluation Dashboard — Phase 3
Formats evaluation metrics into a structured text dashboard for terminal output.

Produces:
- Accuracy summary header
- Grade accuracy table
- Decision accuracy + confusion matrix
- ROI performance table
- Grade band breakdown
- Trend analysis
- Weaknesses + recommendations
- Full evaluation report (for file output)

No external dependencies beyond the standard library.
All output is plain text / ASCII — works in any terminal or log file.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_DIVIDER   = "─" * 72
_SEPARATOR = "═" * 72
_TICK      = "✓"
_CROSS     = "✗"
_WARN      = "⚠"


# ─── Public API ───────────────────────────────────────────────────────────────

def render_dashboard(
    metrics:    Dict[str, Any],
    optimizer:  Optional[Dict[str, Any]] = None,
    title:      str = "AI CARD GRADER — EVALUATION DASHBOARD",
) -> str:
    """
    Render a complete evaluation dashboard as a multi-line string.

    Args:
        metrics:   Output from MetricsEngine.compute_all()
        optimizer: Output from DecisionOptimizer.optimize() (optional)
        title:     Dashboard title string

    Returns:
        Complete formatted dashboard string (print or write to file).
    """
    lines: List[str] = []

    lines.append(_SEPARATOR)
    lines.append(_centre(title))
    lines.append(_SEPARATOR)

    lines.extend(_render_summary(metrics))
    lines.extend(_render_grade_metrics(metrics.get("grade_metrics", {})))
    lines.extend(_render_decision_metrics(metrics.get("decision_metrics", {})))
    lines.extend(_render_confusion_matrix(metrics.get("confusion_matrix", {})))
    lines.extend(_render_roi_metrics(metrics.get("roi_metrics", {})))
    lines.extend(_render_band_breakdown(metrics.get("band_breakdown", {})))
    lines.extend(_render_trend(metrics.get("trend", {})))

    if optimizer:
        lines.extend(_render_optimizer(optimizer))

    lines.append(_SEPARATOR)
    lines.append(f"  Generated: {metrics.get('generated_at', 'N/A')}")
    lines.append(f"  Records:   {metrics.get('n_records', 0)}")
    lines.append(_SEPARATOR)

    return "\n".join(lines)


def render_evaluation_report(
    metrics:   Dict[str, Any],
    optimizer: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a plain-text evaluation report suitable for saving to file.

    Includes all sections plus per-card detail table.
    """
    lines: List[str] = []

    lines.append("=" * 72)
    lines.append(_centre("AI CARD GRADER — EVALUATION REPORT"))
    lines.append("=" * 72)
    lines.append("")

    # Header stats
    s = metrics.get("summary", {})
    lines.append(f"  Report Date:    {metrics.get('generated_at', 'N/A')}")
    lines.append(f"  Total Records:  {metrics.get('n_records', 0)}")
    lines.append(f"  Grade MAE:      {s.get('mae', 'N/A')}")
    lines.append(f"  Decision Acc:   {s.get('decision_accuracy_pct', 'N/A')}%")
    lines.append("")

    # Core sections
    lines.extend(_render_summary(metrics))
    lines.extend(_render_grade_metrics(metrics.get("grade_metrics", {})))
    lines.extend(_render_decision_metrics(metrics.get("decision_metrics", {})))
    lines.extend(_render_confusion_matrix(metrics.get("confusion_matrix", {})))
    lines.extend(_render_roi_metrics(metrics.get("roi_metrics", {})))
    lines.extend(_render_band_breakdown(metrics.get("band_breakdown", {})))
    lines.extend(_render_trend(metrics.get("trend", {})))

    if optimizer:
        lines.extend(_render_optimizer(optimizer))

    # Per-card table
    lines.extend(_render_per_card_table(metrics.get("per_card", [])))

    lines.append("=" * 72)
    lines.append("")
    return "\n".join(lines)


# ─── Section renderers ─────────────────────────────────────────────────────────

def _render_summary(metrics: Dict) -> List[str]:
    s = metrics.get("summary", {})
    if not s:
        return []

    lines = [
        "",
        _DIVIDER,
        "  SUMMARY",
        _DIVIDER,
    ]

    mae   = s.get("mae", "-")
    w05   = s.get("grade_within_half", "-")
    w1    = s.get("grade_within_one",  "-")
    band  = s.get("band_accuracy_pct", "-")
    dec   = s.get("decision_accuracy_pct", "-")
    fpr   = s.get("false_positive_rate", "-")
    fnr   = s.get("false_negative_rate", "-")
    roi_m = s.get("roi_mae", "-")
    total_p = s.get("total_actual_profit", "-")

    lines += [
        f"  Grade MAE:              {mae:>8}   (avg absolute grade error)",
        f"  Grade Within ±0.5:      {w05:>7}%  (% cards within half a grade)",
        f"  Grade Within ±1.0:      {w1:>7}%  (% cards within one grade)",
        f"  Band Accuracy:          {band:>7}%  (% cards in correct PSA band)",
        f"  Decision Accuracy:      {dec:>7}%  (grade + direction correct)",
        f"  False Positive Rate:    {fpr:>7}%  (bad GRADE calls)",
        f"  False Negative Rate:    {fnr:>7}%  (missed profitable cards)",
        f"  Profit MAE:             {_fmt_currency(roi_m):>8}   (avg profit prediction error)",
        f"  Total Actual Profit:    {_fmt_currency(total_p):>8}",
    ]
    return lines


def _render_grade_metrics(gm: Dict) -> List[str]:
    if not gm:
        return []

    lines = [
        "",
        _DIVIDER,
        "  GRADE ACCURACY METRICS",
        _DIVIDER,
        f"  N Cards:            {gm.get('n', '-'):>6}",
        f"  MAE:                {gm.get('mae', '-'):>6}   (mean absolute error)",
        f"  RMSE:               {gm.get('rmse', '-'):>6}   (root mean squared error)",
        f"  Bias:              {gm.get('bias', '-'):>+7.3f}  (+ = system over-grades)",
        f"  Within ±0.5:        {gm.get('within_half_pct', '-'):>5}%",
        f"  Within ±1.0:        {gm.get('within_one_pct', '-'):>5}%",
        f"  Band Accuracy:      {gm.get('band_accuracy_pct', '-'):>5}%",
        f"  Max Error:          {gm.get('max_error', '-'):>6}",
        "",
        "  Error Distribution:",
    ]

    dist = gm.get("error_distribution", {})
    for label, count in dist.items():
        lines.append(f"    {label:<18} {count:>4}")

    over  = gm.get("over_graded_count", 0)
    under = gm.get("under_graded_count", 0)
    exact = gm.get("exact_count", 0)
    total = gm.get("n", 1)

    lines += [
        "",
        "  Grading Direction:",
        f"    Over-graded:  {over:>4}  ({over/total*100:.1f}%)",
        f"    Under-graded: {under:>4}  ({under/total*100:.1f}%)",
        f"    Accurate:     {exact:>4}  ({exact/total*100:.1f}%)",
    ]
    return lines


def _render_decision_metrics(dm: Dict) -> List[str]:
    if not dm:
        return []

    lines = [
        "",
        _DIVIDER,
        "  DECISION ACCURACY",
        _DIVIDER,
        f"  Total Cards:        {dm.get('n_total', '-'):>6}",
        f"  With Profit Data:   {dm.get('n_with_profit_data', '-'):>6}",
        f"  Decision Accuracy:  {dm.get('decision_accuracy_pct', '-'):>5}%",
        f"  False Positive Rate:{dm.get('false_positive_rate', '-'):>5}%  (GRADE call → loss)",
        f"  False Negative Rate:{dm.get('false_negative_rate', '-'):>5}%  (NO-GRADE → profit)",
        f"  Precision:          {dm.get('precision_pct', '-'):>5}%",
        f"  Recall:             {dm.get('recall_pct', '-'):>5}%",
        "",
        "  Decision Tier Counts:",
    ]

    for tier, count in sorted(dm.get("tier_counts", {}).items(),
                               key=lambda x: _tier_sort(x[0])):
        lines.append(f"    {tier:<20} {count:>4}")

    return lines


def _render_confusion_matrix(cm: Dict) -> List[str]:
    if not cm or not cm.get("total"):
        return []

    tp = cm.get("true_positives",  0)
    fp = cm.get("false_positives", 0)
    tn = cm.get("true_negatives",  0)
    fn = cm.get("false_negatives", 0)
    acc = cm.get("overall_accuracy", 0)

    lines = [
        "",
        _DIVIDER,
        "  CONFUSION MATRIX  (GRADE call vs actual profitability)",
        _DIVIDER,
        "",
        "                    ACTUAL PROFIT   ACTUAL LOSS",
        f"  Predicted GRADE       {tp:>6} TP      {fp:>6} FP",
        f"  Predicted NO-GRADE    {fn:>6} FN      {tn:>6} TN",
        "",
        f"  Overall Accuracy:  {acc}%",
        f"  Total (with data): {cm.get('total', 0)}",
    ]
    return lines


def _render_roi_metrics(rm: Dict) -> List[str]:
    if not rm or rm.get("n", rm.get("n_with_actual_profit", 0)) == 0:
        return []

    lines = [
        "",
        _DIVIDER,
        "  ROI PERFORMANCE",
        _DIVIDER,
        f"  Cards with Profit Data:  {rm.get('n_with_actual_profit', '-'):>6}",
        f"  Total Actual Profit:     {_fmt_currency(rm.get('total_actual_profit')):>8}",
        f"  Total Predicted Profit:  {_fmt_currency(rm.get('total_predicted_profit')):>8}",
        f"  Profit MAE:              {_fmt_currency(rm.get('profit_mae')):>8}",
        f"  Profit Bias:             {_fmt_currency(rm.get('profit_bias')):>8}  (+ = overestimates)",
        f"  Profitable Cards:        {rm.get('profitable_cards', '-'):>6}",
        f"  Unprofitable Cards:      {rm.get('unprofitable_cards', '-'):>6}",
        f"  Profitable Rate:         {rm.get('profitable_rate_pct', '-'):>5}%",
        "",
        "  Profit by Decision Tier:",
        f"  {'Tier':<22} {'Count':>5}  {'Total':>9}  {'Avg':>7}  {'Win%':>6}",
        "  " + "-" * 56,
    ]

    for tier, data in sorted(
        rm.get("tier_profit_breakdown", {}).items(),
        key=lambda x: _tier_sort(x[0])
    ):
        lines.append(
            f"  {tier:<22} {data['count']:>5}  "
            f"{_fmt_currency(data['total_profit']):>9}  "
            f"{_fmt_currency(data['avg_profit']):>7}  "
            f"{data['profitable_pct']:>5}%"
        )

    return lines


def _render_band_breakdown(bb: Dict) -> List[str]:
    if not bb:
        return []

    lines = [
        "",
        _DIVIDER,
        "  GRADE BAND ACCURACY BREAKDOWN",
        _DIVIDER,
        f"  {'Band':<16} {'Count':>5}  {'MAE':>6}  {'BandAcc':>8}  {'W±0.5':>6}",
        "  " + "-" * 50,
    ]

    for band, data in bb.items():
        lines.append(
            f"  {band:<16} {data['count']:>5}  "
            f"{data['mae']:>6.3f}  "
            f"{data['band_accuracy']:>7}%  "
            f"{data['within_half_pct']:>5}%"
        )

    return lines


def _render_trend(trend: Dict) -> List[str]:
    if not trend or "note" in trend:
        return []

    direction  = trend.get("trend_direction", "stable")
    first_mae  = trend.get("first_half_mae", 0)
    second_mae = trend.get("second_half_mae", 0)
    window     = trend.get("window_size", 5)

    direction_symbol = {"improving": _TICK, "degrading": _CROSS, "stable": "~"}.get(
        direction, "~"
    )

    lines = [
        "",
        _DIVIDER,
        f"  TREND ANALYSIS  (rolling window: {window} cards)",
        _DIVIDER,
        f"  Trend Direction:  {direction_symbol} {direction.upper()}",
        f"  First-half MAE:   {first_mae:.3f}",
        f"  Second-half MAE:  {second_mae:.3f}",
        f"  MAE Delta:        {second_mae - first_mae:+.3f}",
        "",
        "  Rolling Performance (last 10 records):",
        f"  {'#':<4} {'Card':<30} {'Error':>7}  {'Rolling MAE':>11}  {'Bias':>7}",
        "  " + "-" * 65,
    ]

    series = trend.get("series", [])[-10:]
    for s in series:
        lines.append(
            f"  {s['index']+1:<4} {(s['card_id'] or '')[:29]:<30} "
            f"{s['grade_error']:>+7.2f}  "
            f"{s['rolling_mae']:>11.3f}  "
            f"{s['rolling_bias']:>+7.3f}"
        )

    return lines


def _render_optimizer(opt: Dict) -> List[str]:
    if not opt:
        return []

    lines = [
        "",
        _DIVIDER,
        "  WEAKNESSES & RECOMMENDATIONS",
        _DIVIDER,
    ]

    weaknesses = opt.get("weaknesses", [])
    if weaknesses:
        lines.append("  Identified Weaknesses:")
        for w in weaknesses:
            sev    = w.get("severity", "").upper()
            desc   = w.get("description", "")
            evid   = w.get("evidence", "")
            lines.append(f"    [{sev}] {desc}")
            if evid:
                lines.append(f"           Evidence: {evid}")
    else:
        lines.append("  No significant weaknesses detected.")

    adjustments = opt.get("threshold_adjustments", {})
    changes = adjustments.get("changes", [])
    if changes:
        lines.append("")
        lines.append("  Recommended Threshold Adjustments:")
        for c in changes:
            lines.append(
                f"    {c['param']:<30} {c['from']:>7} → {c['to']:<7}   {c['reason']}"
            )

    recs = opt.get("recommendations", [])
    if recs:
        lines.append("")
        lines.append("  Action Items:")
        for i, rec in enumerate(recs, 1):
            lines.append(f"    {i}. {rec}")

    # Per-tier performance scores
    tier_analysis = opt.get("tier_analysis", {})
    if tier_analysis:
        lines += [
            "",
            "  Decision Tier Performance Scores:",
            f"  {'Tier':<22} {'Count':>5}  {'Score/card':>10}  {'Profitable':>10}  {'Avg Profit':>10}",
            "  " + "-" * 65,
        ]
        for tier, data in sorted(tier_analysis.items(), key=lambda x: _tier_sort(x[0])):
            score = data.get("normalised_score", 0)
            score_str = f"{score:>+.2f}"
            lines.append(
                f"  {tier:<22} {data['count']:>5}  "
                f"{score_str:>10}  "
                f"{data['profitable_rate']:>9}%  "
                f"{_fmt_currency(data['avg_profit']):>10}"
            )

    # Grade band bias
    band_bias = opt.get("grade_band_bias", {})
    if band_bias:
        lines += [
            "",
            "  Grade Band Bias Detection:",
            f"  {'Band':<16} {'Count':>5}  {'Bias':>7}  {'MAE':>6}  {'Direction':>14}",
            "  " + "-" * 55,
        ]
        for band, info in band_bias.items():
            lines.append(
                f"  {band:<16} {info['count']:>5}  "
                f"{info['bias']:>+7.3f}  "
                f"{info['mae']:>6.3f}  "
                f"{info['direction']:>14}"
            )

    return lines


def _render_per_card_table(per_card: List[Dict]) -> List[str]:
    if not per_card:
        return []

    lines = [
        "",
        _DIVIDER,
        "  PER-CARD DETAIL",
        _DIVIDER,
        f"  {'Card':<34} {'Pred':>5} {'Actual':>6} {'Err':>5} {'Decision':<22} {'Profit':>8}",
        "  " + "-" * 90,
    ]

    for r in per_card:
        name    = (r.get("card_name") or r.get("card_id", ""))[:33]
        pred    = r.get("predicted_grade", 0)
        actual  = r.get("final_grade", 0)
        err     = r.get("grade_error", 0)
        dec     = (r.get("roi_decision") or "")[:21]
        profit  = r.get("actual_profit")

        tick = _TICK if r.get("within_half") else ("~" if r.get("within_one") else _CROSS)

        lines.append(
            f"  {name:<34} {pred:>5.1f} {actual:>6.1f} {err:>+5.1f} "
            f"{tick} {dec:<21} {_fmt_currency(profit):>8}"
        )

    return lines


# ─── Formatting helpers ───────────────────────────────────────────────────────

def _fmt_currency(val: Any) -> str:
    """Format a value as currency string, or '-' if None."""
    if val is None:
        return "-"
    try:
        v = float(val)
        return f"${v:+.0f}" if v != 0 else "$0"
    except (TypeError, ValueError):
        return str(val)


def _centre(text: str, width: int = 72) -> str:
    return text.center(width)


def _tier_sort(tier: str) -> int:
    order = {
        "STRONG GRADE": 0, "GRADE": 1, "CONDITIONAL": 2,
        "HOLD RAW": 3, "DO NOT GRADE": 4,
    }
    return order.get(tier, 5)
