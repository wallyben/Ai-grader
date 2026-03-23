"""
Defect Evidence Objects
Converts raw analysis results into structured, auditable Defect records.
Every detected issue becomes a first-class object with type, location,
severity, bounding box, and human-readable explanation.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple
import copy

# Card dimensions (must match preprocessing constants)
CARD_W = 500
CARD_H = 700
CORNER_SIZE  = 65    # must match corners.py
EDGE_WIDTH   = 18    # must match edges.py


@dataclass
class Defect:
    """Single grading defect with full evidence record."""
    defect_type:  str                          # e.g. "corner_whitening"
    category:     str                          # "centering"|"edge"|"corner"|"surface"
    side:         str                          # "front"|"back"
    location:     str                          # human-readable location
    severity:     str                          # "minor"|"moderate"|"severe"
    bbox:         Optional[Tuple[int,int,int,int]]  # (x, y, w, h) in 500×700 space
    explanation:  str                          # plain-English description
    metric:       str                          # e.g. "whitening"
    metric_value: float                        # raw metric
    cap_impact:   Optional[float] = None       # grade cap triggered by this defect, if any

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox) if self.bbox else None
        return d


# ─── Bounding box helpers ─────────────────────────────────────────────────────

def _corner_bbox(position: str) -> Tuple[int, int, int, int]:
    s = CORNER_SIZE
    bboxes = {
        "top_left":     (0,          0,          s, s),
        "top_right":    (CARD_W - s, 0,          s, s),
        "bottom_left":  (0,          CARD_H - s, s, s),
        "bottom_right": (CARD_W - s, CARD_H - s, s, s),
    }
    return bboxes[position]


def _edge_bbox(side: str) -> Tuple[int, int, int, int]:
    s = EDGE_WIDTH
    bboxes = {
        "top":    (0,          0,          CARD_W, s),
        "bottom": (0,          CARD_H - s, CARD_W, s),
        "left":   (0,          0,          s,      CARD_H),
        "right":  (CARD_W - s, 0,          s,      CARD_H),
    }
    return bboxes[side]


def _severity_from_level(level: str) -> str:
    """Map defect_level string to normalised severity."""
    mapping = {"severe": "severe", "moderate": "moderate",
                "minor": "minor", "clean": None, "sharp": None}
    return mapping.get(level, None)


# ─── Extractor functions ──────────────────────────────────────────────────────

def _extract_centering_defects(centering: Dict, side: str) -> List[Defect]:
    defects = []
    score = centering["score"]
    lr = centering["lr_ratio"]
    tb = centering["tb_ratio"]
    lr_dev = abs(centering["left_pct"] - centering["right_pct"])
    tb_dev = abs(centering["top_pct"] - centering["bottom_pct"])
    max_dev = max(lr_dev, tb_dev)

    if score < 5.0:
        sev = "severe"
    elif score < 7.5:
        sev = "moderate"
    elif score < 9.5:
        sev = "minor"
    else:
        return defects  # perfect centering

    axis = "L/R" if lr_dev >= tb_dev else "T/B"
    defects.append(Defect(
        defect_type="centering",
        category="centering",
        side=side,
        location="card border",
        severity=sev,
        bbox=None,
        explanation=(f"{axis} centering is {sev} ({lr} L/R, {tb} T/B). "
                     f"Maximum deviation {max_dev:.1f}pp from 50/50."),
        metric="max_deviation_pp",
        metric_value=round(max_dev, 2),
    ))
    return defects


def _extract_edge_defects(edges: Dict, side: str) -> List[Defect]:
    defects = []
    for edge_side, data in edges["sides"].items():
        sev = _severity_from_level(data["defect_level"])
        if sev is None:
            continue
        bbox = _edge_bbox(edge_side)
        dom_metric = "whitening"
        dom_val = data["whitening"]
        if data["chipping"] > data["whitening"] and data["chipping"] > data["roughness"]:
            dom_metric, dom_val = "chipping", data["chipping"]
        elif data["roughness"] > data["whitening"]:
            dom_metric, dom_val = "roughness", data["roughness"]

        defects.append(Defect(
            defect_type=f"edge_{dom_metric}",
            category="edge",
            side=side,
            location=f"{edge_side} edge",
            severity=sev,
            bbox=bbox,
            explanation=(f"{sev.title()} {dom_metric} on {edge_side} edge "
                         f"(w={data['whitening']:.3f} r={data['roughness']:.3f} "
                         f"c={data['chipping']:.3f})."),
            metric=dom_metric,
            metric_value=round(dom_val, 4),
        ))
    return defects


def _extract_corner_defects(corners: Dict, side: str) -> List[Defect]:
    defects = []
    for pos, data in corners["corners"].items():
        sev = _severity_from_level(data["defect_level"])
        if sev is None:
            continue
        bbox = _corner_bbox(pos)
        dom_metric = "whitening" if data["whitening"] >= data["rounding"] else "rounding"
        dom_val = data[dom_metric]
        label = pos.replace("_", " ")
        defects.append(Defect(
            defect_type=f"corner_{dom_metric}",
            category="corner",
            side=side,
            location=f"{label} corner",
            severity=sev,
            bbox=bbox,
            explanation=(f"{sev.title()} corner wear at {label} "
                         f"(whitening={data['whitening']:.3f} "
                         f"rounding={data['rounding']:.3f})."),
            metric=dom_metric,
            metric_value=round(dom_val, 4),
        ))
    return defects


def _extract_surface_defects(surface: Dict, side: str) -> List[Defect]:
    defects = []
    sd = surface["scratch_density"]
    pl = surface["print_line_indicator"]
    ad = surface["anomaly_density"]

    if sd > 0.0:
        sev = "severe" if sd > 0.04 else ("moderate" if sd > 0.015 else "minor")
        if sev != "clean":
            defects.append(Defect(
                defect_type="surface_scratch",
                category="surface",
                side=side,
                location="card face",
                severity=sev,
                bbox=(18, 18, CARD_W - 36, CARD_H - 36),
                explanation=f"Scratches detected on card face (density={sd:.5f}).",
                metric="scratch_density",
                metric_value=round(sd, 5),
            ))

    if pl > 0.40:
        sev = "moderate" if pl > 0.65 else "minor"
        defects.append(Defect(
            defect_type="print_lines",
            category="surface",
            side=side,
            location="card face",
            severity=sev,
            bbox=(18, 18, CARD_W - 36, CARD_H - 36),
            explanation=f"Print lines or manufacturing banding detected (score={pl:.3f}).",
            metric="print_line_indicator",
            metric_value=round(pl, 4),
        ))

    if ad > 0.04:
        sev = "moderate" if ad > 0.08 else "minor"
        defects.append(Defect(
            defect_type="surface_anomaly",
            category="surface",
            side=side,
            location="card face",
            severity=sev,
            bbox=(18, 18, CARD_W - 36, CARD_H - 36),
            explanation=f"Surface anomalies detected — possible dent or crease (density={ad:.5f}).",
            metric="anomaly_density",
            metric_value=round(ad, 5),
        ))

    return defects


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_defects(analysis: Dict, side: str = "front") -> List[Defect]:
    """
    Extract structured Defect objects from a complete analysis result dict.

    Args:
        analysis: dict with keys centering, edges, corners, surface
        side: "front" or "back"

    Returns:
        List[Defect] sorted by severity (severe first)
    """
    defects: List[Defect] = []
    defects.extend(_extract_centering_defects(analysis["centering"], side))
    defects.extend(_extract_edge_defects(analysis["edges"], side))
    defects.extend(_extract_corner_defects(analysis["corners"], side))
    defects.extend(_extract_surface_defects(analysis["surface"], side))

    sev_order = {"severe": 0, "moderate": 1, "minor": 2}
    defects.sort(key=lambda d: sev_order.get(d.severity, 9))
    return defects


def defects_to_dicts(defects: List[Defect]) -> List[Dict]:
    """Serialise defect list to JSON-safe dicts."""
    return [d.to_dict() for d in defects]


def summarise_defects(defects: List[Defect]) -> Dict[str, int]:
    """Return count per severity level."""
    counts = {"severe": 0, "moderate": 0, "minor": 0}
    for d in defects:
        if d.severity in counts:
            counts[d.severity] += 1
    return counts
