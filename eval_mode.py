#!/usr/bin/env python3
"""
Evaluation Mode — Phase 3 CLI
Runs the AI Card Grader evaluation framework against a real-world outcome dataset.

Usage:
    python eval_mode.py                                      # default sample dataset
    python eval_mode.py --dataset data/evaluation/my_data.json
    python eval_mode.py --dataset data/evaluation/sample_dataset.json --report
    python eval_mode.py --dataset data/evaluation/sample_dataset.json --update-calibration
    python eval_mode.py --dataset data/evaluation/sample_dataset.json --report --output report.txt

Options:
    --dataset PATH           Path to evaluation JSON dataset (default: sample_dataset.json)
    --report                 Save full evaluation report to a text file
    --output PATH            Output path for report (default: evaluation_report.txt)
    --update-calibration     Re-fit calibration engine from dataset outcomes
    --calibration-state PATH Path to load/save calibration state (default: data/calibration_state.json)
    --no-optimizer           Skip decision optimizer analysis
    --quiet                  Suppress dashboard output (only show summary)
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Make grader importable when run from project root ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grader.evaluation import (
    EvaluationRecord,
    MetricsEngine,
    load_evaluation_dataset,
)
from grader.calibration import CalibrationEngine
from grader.decision_optimizer import DecisionOptimizer
from grader.eval_dashboard import render_dashboard, render_evaluation_report


# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_DATASET   = os.path.join("data", "evaluation", "sample_dataset.json")
DEFAULT_CAL_STATE = os.path.join("data", "calibration_state.json")
DEFAULT_REPORT    = "evaluation_report.txt"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset_path = args.dataset or DEFAULT_DATASET
    print(f"\n[eval_mode] Loading dataset: {dataset_path}")

    try:
        records = load_evaluation_dataset(dataset_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return 1

    if not records:
        print("[ERROR] Dataset is empty.")
        return 1

    print(f"[eval_mode] Loaded {len(records)} evaluation records.")

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("[eval_mode] Computing metrics...")
    engine  = MetricsEngine(records)
    metrics = engine.compute_all()

    # ── Decision optimizer ────────────────────────────────────────────────────
    optimizer_result = None
    if not args.no_optimizer:
        print("[eval_mode] Running decision optimizer...")
        optimizer = DecisionOptimizer(records)
        optimizer_result = optimizer.optimize()

    # ── Calibration update ────────────────────────────────────────────────────
    calibration_update = None
    if args.update_calibration:
        print("[eval_mode] Updating calibration from outcomes...")
        cal_engine = CalibrationEngine()

        # Load existing state if present
        cal_state_path = args.calibration_state or DEFAULT_CAL_STATE
        if os.path.exists(cal_state_path):
            cal_engine.load_state(cal_state_path)
            print(f"  Loaded calibration state: {cal_state_path}")

        # Build outcome dicts from evaluation records
        outcomes = [
            {"predicted_grade": r.predicted_grade, "actual_grade": r.final_grade}
            for r in records
        ]
        calibration_update = cal_engine.update_from_outcomes(outcomes)

        # Save updated state
        os.makedirs(os.path.dirname(os.path.abspath(cal_state_path)), exist_ok=True)
        cal_engine.save_state(cal_state_path)
        print(f"  Calibration updated: bias {calibration_update['old_bias']:+.4f} → "
              f"{calibration_update['new_bias']:+.4f} "
              f"(delta: {calibration_update['bias_delta']:+.4f})")
        print(f"  Calibration state saved: {cal_state_path}")

        # Show rolling performance
        rolling = cal_engine.rolling_performance()
        print(f"  Rolling window: {rolling.get('n', 0)} outcomes | "
              f"MAE: {rolling.get('mae', '-')} | "
              f"Within ±0.5: {rolling.get('within_0.5_pct', '-')}%")

    # ── Dashboard output ──────────────────────────────────────────────────────
    if not args.quiet:
        dashboard = render_dashboard(metrics, optimizer=optimizer_result)
        print()
        print(dashboard)

    # ── Report output ─────────────────────────────────────────────────────────
    if args.report:
        report_path = args.output or DEFAULT_REPORT
        print(f"\n[eval_mode] Generating report: {report_path}")
        report = render_evaluation_report(metrics, optimizer=optimizer_result)
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(report)
            if calibration_update:
                fh.write("\n")
                fh.write("═" * 72 + "\n")
                fh.write("  CALIBRATION UPDATE\n")
                fh.write("─" * 72 + "\n")
                fh.write(f"  Old Bias:    {calibration_update['old_bias']:+.4f}\n")
                fh.write(f"  New Bias:    {calibration_update['new_bias']:+.4f}\n")
                fh.write(f"  Bias Delta:  {calibration_update['bias_delta']:+.4f}\n")
                fh.write(f"  N Outcomes:  {calibration_update['n_new_outcomes']}\n")
                fh.write(f"  Timestamp:   {calibration_update['timestamp']}\n")
        print(f"[eval_mode] Report saved: {report_path}")

    # ── Exit summary ──────────────────────────────────────────────────────────
    _print_exit_summary(metrics, optimizer_result)

    return 0


def _print_exit_summary(metrics: dict, optimizer_result) -> None:
    s   = metrics.get("summary", {})
    mae = s.get("mae", "-")
    dec = s.get("decision_accuracy_pct", "-")
    fpr = s.get("false_positive_rate", "-")
    pft = s.get("total_actual_profit", "-")

    print()
    print("─" * 50)
    print("  EVALUATION COMPLETE")
    print("─" * 50)
    print(f"  Grade MAE:          {mae}")
    print(f"  Decision Accuracy:  {dec}%")
    print(f"  False Positive Rate:{fpr}%")
    if pft is not None and pft != "-":
        print(f"  Total Profit:       ${pft:+.0f}")

    if optimizer_result:
        weaknesses = optimizer_result.get("weaknesses", [])
        changes    = optimizer_result.get("threshold_adjustments", {}).get("changes", [])
        if weaknesses:
            print(f"  Weaknesses Found:   {len(weaknesses)}")
        if changes:
            print(f"  Threshold Changes:  {len(changes)} recommended")
    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Card Grader — Evaluation Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", "-d",
        metavar="PATH",
        help=f"Path to evaluation JSON file (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--report", "-r",
        action="store_true",
        help="Save full evaluation report to a text file",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        help=f"Output path for report (default: {DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--update-calibration",
        action="store_true",
        help="Re-fit calibration engine using dataset outcomes",
    )
    parser.add_argument(
        "--calibration-state",
        metavar="PATH",
        help=f"Path to load/save calibration state (default: {DEFAULT_CAL_STATE})",
    )
    parser.add_argument(
        "--no-optimizer",
        action="store_true",
        help="Skip decision optimizer analysis",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress full dashboard output",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
