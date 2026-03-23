"""
Artifact Export System
Saves all grading outputs to disk: normalized images, overlays, debug masks,
grading JSON, and per-card CSV rows. Also provides in-memory ZIP for UI download.
"""

import cv2
import json
import os
import io
import csv
import zipfile
import datetime
from typing import Dict, Any, Optional, List
import numpy as np


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy types and arrays to JSON-safe Python types."""
    if isinstance(obj, np.ndarray):
        return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _save_img(path: str, img: np.ndarray, quality: int = 92) -> None:
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])


# ─── Single-card artifact save ────────────────────────────────────────────────

def save_grading_artifacts(card_id: str,
                            front_norm: np.ndarray,
                            back_norm: Optional[np.ndarray],
                            results: Dict[str, Any],
                            output_dir: str,
                            quality: int = 92) -> str:
    """
    Save all grading artifacts for a single card to output_dir/card_id/.

    Saved files:
      front_normalized.jpg
      back_normalized.jpg          (if back provided)
      overlay_centering.jpg
      overlay_edges.jpg
      overlay_corners.jpg
      overlay_surface.jpg
      overlay_full.jpg
      grade_summary.jpg
      mask_scratches.jpg
      mask_anomalies.jpg
      grade_result.json

    Returns the path to the card directory.
    """
    card_dir = os.path.join(output_dir, card_id)
    os.makedirs(card_dir, exist_ok=True)

    # ── Normalized images ─────────────────────────────────────────────────────
    _save_img(os.path.join(card_dir, "front_normalized.jpg"), front_norm, quality)
    if back_norm is not None:
        _save_img(os.path.join(card_dir, "back_normalized.jpg"), back_norm, quality)

    # ── Overlay images ────────────────────────────────────────────────────────
    viz_names = {
        "centering":     "overlay_centering.jpg",
        "edges":         "overlay_edges.jpg",
        "corners":       "overlay_corners.jpg",
        "surface":       "overlay_surface.jpg",
        "full_overlay":  "overlay_full.jpg",
        "grade_summary": "grade_summary.jpg",
    }
    for key, fname in viz_names.items():
        img = results.get("visualizations", {}).get(key)
        if isinstance(img, np.ndarray):
            _save_img(os.path.join(card_dir, fname), img, quality)

    # ── Debug masks ───────────────────────────────────────────────────────────
    surface = results.get("analysis", {}).get("surface", {})
    for mask_key, fname in [("scratch_map", "mask_scratches.jpg"),
                              ("anomaly_mask", "mask_anomalies.jpg")]:
        mask = surface.get(mask_key)
        if isinstance(mask, np.ndarray):
            cv2.imwrite(os.path.join(card_dir, fname), mask)

    # ── Grade JSON ────────────────────────────────────────────────────────────
    grade_data = _json_safe({
        "card_id":   card_id,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "grade_result": results.get("grade_result", {}),
        "grade_trace":  results.get("grade_trace", {}),
        "analysis": {
            k: _json_safe(v)
            for k, v in results.get("analysis", {}).items()
        },
        "quality_gate": results.get("quality_gate"),
        "defects": [d if isinstance(d, dict) else d.to_dict()
                    for d in results.get("defects", [])],
        "profile_used": results.get("profile_used", "tcg_generic"),
    })
    with open(os.path.join(card_dir, "grade_result.json"), "w") as f:
        json.dump(grade_data, f, indent=2)

    return card_dir


# ─── CSV row ──────────────────────────────────────────────────────────────────

def get_csv_row(card_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
    """Build a flat dict suitable for CSV export."""
    gr = results.get("grade_result", {})
    qg = results.get("quality_gate", {})
    defects = results.get("defects", [])
    def _sev(d): return d.severity if not isinstance(d, dict) else d.get("severity")
    n_severe   = sum(1 for d in defects if _sev(d) == "severe")
    n_moderate = sum(1 for d in defects if _sev(d) == "moderate")
    n_minor    = sum(1 for d in defects if _sev(d) == "minor")

    return {
        "card_id":            card_id,
        "grade":              gr.get("grade"),
        "grade_band":         gr.get("grade_band"),
        "recommendation":     gr.get("recommendation"),
        "confidence":         gr.get("confidence"),
        "centering_score":    gr.get("centering_score"),
        "edges_score":        gr.get("edges_score"),
        "corners_score":      gr.get("corners_score"),
        "surface_score":      gr.get("surface_score"),
        "centering_lr":       gr.get("centering_lr"),
        "centering_tb":       gr.get("centering_tb"),
        "caps_triggered":     "|".join(gr.get("caps_triggered", [])),
        "risk_flags":         "|".join(gr.get("risk_flags", [])),
        "defects_severe":     n_severe,
        "defects_moderate":   n_moderate,
        "defects_minor":      n_minor,
        "quality_decision":   qg.get("decision") if qg else "not_run",
        "quality_score":      qg.get("score") if qg else None,
        "profile_used":       results.get("profile_used", "tcg_generic"),
        "base_grade":         gr.get("base_grade"),
    }


# ─── Batch CSV ────────────────────────────────────────────────────────────────

CSV_FIELDNAMES = [
    "card_id", "grade", "grade_band", "recommendation", "confidence",
    "centering_score", "edges_score", "corners_score", "surface_score",
    "centering_lr", "centering_tb", "caps_triggered", "risk_flags",
    "defects_severe", "defects_moderate", "defects_minor",
    "quality_decision", "quality_score", "profile_used", "base_grade",
]


def save_batch_csv(rows: List[Dict], output_path: str) -> None:
    """Write a list of CSV row dicts to a CSV file, sorted best grade first."""
    def _sort_key(r):
        g = r.get("grade")
        return -float(g) if g is not None else 999

    rows_sorted = sorted(rows, key=_sort_key)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_sorted)


# ─── In-memory ZIP for UI download ───────────────────────────────────────────

def build_download_zip(card_id: str,
                        front_norm: np.ndarray,
                        back_norm: Optional[np.ndarray],
                        results: Dict[str, Any]) -> bytes:
    """
    Build an in-memory ZIP of all grading artifacts for UI download.
    Returns raw bytes suitable for st.download_button.
    """
    buf = io.BytesIO()

    def _encode_jpg(img: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return encoded.tobytes() if ok else b""

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Normalized images
        zf.writestr(f"{card_id}/front_normalized.jpg", _encode_jpg(front_norm))
        if back_norm is not None:
            zf.writestr(f"{card_id}/back_normalized.jpg", _encode_jpg(back_norm))

        # Overlays
        for key, fname in [
            ("centering",    "overlay_centering.jpg"),
            ("edges",        "overlay_edges.jpg"),
            ("corners",      "overlay_corners.jpg"),
            ("surface",      "overlay_surface.jpg"),
            ("full_overlay", "overlay_full.jpg"),
            ("grade_summary","grade_summary.jpg"),
        ]:
            img = results.get("visualizations", {}).get(key)
            if isinstance(img, np.ndarray):
                zf.writestr(f"{card_id}/{fname}", _encode_jpg(img))

        # Debug masks
        surface = results.get("analysis", {}).get("surface", {})
        for mkey, fname in [("scratch_map", "mask_scratches.jpg"),
                              ("anomaly_mask", "mask_anomalies.jpg")]:
            mask = surface.get(mkey)
            if isinstance(mask, np.ndarray):
                ok, enc = cv2.imencode(".jpg", mask)
                if ok:
                    zf.writestr(f"{card_id}/{fname}", enc.tobytes())

        # JSON
        grade_data = _json_safe({
            "card_id":   card_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "grade_result": results.get("grade_result", {}),
            "grade_trace":  results.get("grade_trace", {}),
            "defects":      [d if isinstance(d, dict) else d.to_dict()
                              for d in results.get("defects", [])],
            "profile_used": results.get("profile_used", "tcg_generic"),
        })
        zf.writestr(f"{card_id}/grade_result.json",
                    json.dumps(grade_data, indent=2))

    buf.seek(0)
    return buf.read()
