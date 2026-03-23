#!/usr/bin/env python3
"""
Batch Card Grading CLI
Processes a folder of card images, grades each one, saves artifacts,
and exports a ranked CSV.

Usage:
    # Single images (one per card):
    python batch_grade.py --input cards/ --output results/ --profile pokemon_modern

    # Front/back pairs (filename must contain _front or _back):
    python batch_grade.py --input cards/ --output results/ --pairs

    # Dry run (no files saved):
    python batch_grade.py --input cards/ --dry-run
"""

import argparse
import os
import sys
import glob
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure grader package is importable when run from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from grader import (
    grade_card_full, list_profiles,
    load_image_from_path, normalize_card,
    save_grading_artifacts, save_batch_csv, get_csv_row,
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _find_images(directory: str) -> List[str]:
    """Return sorted list of image paths in directory."""
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))
    return sorted(set(paths))


def _find_front_back_pairs(images: List[str]) -> Dict[str, Tuple[str, Optional[str]]]:
    """
    Detect front/back pairs by filename convention.
    Accepted patterns: *_front.*, *_back.*, *_f.*, *_b.*
    Unmatched images treated as front-only.
    """
    front_map: Dict[str, str] = {}
    back_map:  Dict[str, str] = {}
    singles:   List[str]      = []

    FRONT_SUFFIXES = ("_front", "_f", "-front", "-f")
    BACK_SUFFIXES  = ("_back",  "_b", "-back",  "-b")

    for path in images:
        stem = Path(path).stem.lower()
        matched = False
        for suf in FRONT_SUFFIXES:
            if stem.endswith(suf):
                base = stem[: -len(suf)]
                front_map[base] = path
                matched = True
                break
        if not matched:
            for suf in BACK_SUFFIXES:
                if stem.endswith(suf):
                    base = stem[: -len(suf)]
                    back_map[base] = path
                    matched = True
                    break
        if not matched:
            singles.append(path)

    pairs: Dict[str, Tuple[str, Optional[str]]] = {}
    all_bases = set(front_map) | set(back_map)
    for base in sorted(all_bases):
        front = front_map.get(base)
        back  = back_map.get(base)
        if front:
            pairs[base] = (front, back)
        else:
            # Back without a front — treat back as front
            pairs[base] = (back, None)

    for path in singles:
        card_id = Path(path).stem
        pairs[card_id] = (path, None)

    return pairs


def _grade_one(card_id: str, front_path: str,
               back_path: Optional[str],
               profile_name: str,
               output_dir: str,
               dry_run: bool,
               verbose: bool) -> Optional[Dict]:
    """Grade a single card and optionally save artifacts."""
    # Load + normalize front
    front_raw = load_image_from_path(front_path)
    if front_raw is None:
        print(f"  [SKIP] Cannot load: {front_path}")
        return None
    front_norm, _ = normalize_card(front_raw)

    # Load + normalize back
    back_norm = None
    if back_path and os.path.isfile(back_path):
        back_raw = load_image_from_path(back_path)
        if back_raw is not None:
            back_norm, _ = normalize_card(back_raw)

    # Grade
    results = grade_card_full(
        front_norm, back_norm,
        profile_name=profile_name,
        run_quality_gate=True,
        front_raw=front_raw,
        back_raw=back_raw if back_path else None,
    )
    results["profile_used"] = profile_name

    gr = results["grade_result"]
    qg = results.get("quality_gate", {}) or {}

    if verbose:
        caps_str = ", ".join(gr.get("caps_triggered", [])) or "none"
        qd       = qg.get("decision", "n/a")
        print(f"  Grade: {gr['grade']:4}  {gr['grade_band']:<20}  "
              f"Rec: {gr['recommendation']:<14}  "
              f"Conf: {int(gr['confidence']*100):3}%  "
              f"QGate: {qd}  "
              f"Caps: [{caps_str}]")

    # Save artifacts
    if not dry_run:
        save_grading_artifacts(card_id, front_norm, back_norm, results, output_dir)

    csv_row = get_csv_row(card_id, results)
    return csv_row


# ─── Comparison / ranking ────────────────────────────────────────────────────

def _print_ranking(rows: List[Dict]) -> None:
    valid  = [r for r in rows if r.get("grade") is not None]
    errors = [r for r in rows if r.get("grade") is None]
    valid.sort(key=lambda r: -float(r["grade"]))

    print("\n" + "═" * 72)
    print(f"{'Rank':<5} {'Card ID':<28} {'Grade':<7} {'Band':<22} {'Rec'}")
    print("─" * 72)
    for i, row in enumerate(valid, 1):
        rec_short = {"GRADE": "GRADE ✓",
                     "CONDITIONAL": "COND ~",
                     "DO NOT GRADE": "SKIP ✗"}.get(row["recommendation"], row["recommendation"])
        print(f"{i:<5} {row['card_id']:<28} {row['grade']:<7} "
              f"{str(row.get('grade_band','')):<22} {rec_short}")

    if errors:
        print(f"\n  {len(errors)} card(s) failed to process.")
    print("═" * 72)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch AI card grader — grade a folder of card images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",   "-i", required=True,
                        help="Input directory containing card images.")
    parser.add_argument("--output",  "-o", default="results",
                        help="Output directory for artifacts and CSV. Default: results/")
    parser.add_argument("--profile", "-p", default="tcg_generic",
                        choices=list_profiles(),
                        help="Card grading profile. Default: tcg_generic")
    parser.add_argument("--pairs",   action="store_true",
                        help="Detect front/back pairs by _front/_back filename suffix.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Grade cards but do not write output files.")
    parser.add_argument("--quiet",   "-q", action="store_true",
                        help="Suppress per-card detail; show only summary.")
    args = parser.parse_args()

    # ── Discover images ───────────────────────────────────────────────────────
    if not os.path.isdir(args.input):
        print(f"Error: input directory '{args.input}' not found.")
        sys.exit(1)

    images = _find_images(args.input)
    if not images:
        print(f"No images found in '{args.input}'. "
              f"Supported: {', '.join(IMAGE_EXTS)}")
        sys.exit(1)

    if args.pairs:
        pairs = _find_front_back_pairs(images)
        print(f"Found {len(pairs)} card(s) ({len(images)} image files, pair mode).")
    else:
        pairs = {Path(p).stem: (p, None) for p in images}
        print(f"Found {len(pairs)} card image(s) (single-image mode).")

    print(f"Profile: {args.profile}")
    if args.dry_run:
        print("DRY RUN — no files will be written.")
    else:
        os.makedirs(args.output, exist_ok=True)
        print(f"Output:  {os.path.abspath(args.output)}")
    print()

    # ── Process each card ─────────────────────────────────────────────────────
    csv_rows: List[Dict] = []
    ok = err = 0

    for card_id, (front_path, back_path) in pairs.items():
        print(f"[{ok + err + 1}/{len(pairs)}] {card_id}", end="")
        if not args.quiet:
            print()  # newline before verbose line
        else:
            print(" ... ", end="", flush=True)

        try:
            row = _grade_one(card_id, front_path, back_path,
                              args.profile, args.output,
                              args.dry_run, not args.quiet)
            if row:
                csv_rows.append(row)
                ok += 1
                if args.quiet:
                    g = row.get("grade", "?")
                    r = row.get("recommendation", "?")
                    print(f"Grade {g} | {r}")
        except Exception as exc:
            err += 1
            print(f"  ERROR: {exc}")
            if not args.quiet:
                traceback.print_exc()
            csv_rows.append({"card_id": card_id, "grade": None,
                              "error": str(exc)})

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if not args.dry_run and csv_rows:
        csv_path = os.path.join(args.output, "grading_summary.csv")
        save_batch_csv(csv_rows, csv_path)
        print(f"\nSummary CSV → {csv_path}")

    # ── Ranked summary ────────────────────────────────────────────────────────
    _print_ranking(csv_rows)
    print(f"\nDone. {ok} graded, {err} error(s).")
    if not args.dry_run:
        print(f"Artifacts → {os.path.abspath(args.output)}/")


if __name__ == "__main__":
    main()
