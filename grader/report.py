"""
PDF Report Generator — Phase 2
Generates professional per-card grading reports using ReportLab.

Sections:
  1. Header — card name, date, profile
  2. Card images — normalized front/back
  3. Grade block — final grade, band, confidence, recommendation
  4. Score breakdown — centering, edges, corners, surface
  5. Defect overlay images
  6. Grade trace — audit trail (sub-scores → caps → final)
  7. ROI Analysis — expected value, profit, decision (if pricing provided)
  8. Defect list — structured evidence

Requires: reportlab (pip install reportlab)
"""

import io
from datetime import date
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# ── ReportLab imports ─────────────────────────────────────────────────────────
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        Image as RLImage,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    _REPORTLAB_AVAILABLE = True
except ImportError:
    _REPORTLAB_AVAILABLE = False


# ─── Colour palette ───────────────────────────────────────────────────────────
_C_DARK   = colors.HexColor("#0d1117")
_C_PANEL  = colors.HexColor("#161b22")
_C_GREEN  = colors.HexColor("#22c55e")
_C_BLUE   = colors.HexColor("#60a5fa")
_C_AMBER  = colors.HexColor("#f59e0b")
_C_RED    = colors.HexColor("#ef4444")
_C_GREY   = colors.HexColor("#6b7280")
_C_WHITE  = colors.HexColor("#f9fafb")
_C_BORDER = colors.HexColor("#30363d")


def _grade_color(grade: float):
    if grade >= 9:   return _C_GREEN
    if grade >= 7:   return _C_BLUE
    if grade >= 5:   return _C_AMBER
    return _C_RED


def _decision_color(decision: str):
    return {
        "STRONG GRADE": _C_GREEN,
        "GRADE":        _C_GREEN,
        "CONDITIONAL":  _C_AMBER,
        "HOLD RAW":     _C_AMBER,
        "DO NOT GRADE": _C_RED,
    }.get(decision, _C_GREY)


def _score_color(score: float):
    if score >= 9:  return _C_GREEN
    if score >= 7:  return _C_BLUE
    if score >= 5:  return _C_AMBER
    return _C_RED


# ─── Image helpers ────────────────────────────────────────────────────────────

def _cv2_to_rl_image(
    bgr: np.ndarray,
    width_mm: float = 70.0,
    max_height_mm: float = 100.0,
) -> Optional["RLImage"]:
    """Convert a BGR numpy array to a ReportLab Image flowable."""
    if bgr is None or not _REPORTLAB_AVAILABLE:
        return None
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image as PILImage
        pil = PILImage.fromarray(rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=88)
        buf.seek(0)

        h_px, w_px = bgr.shape[:2]
        aspect = h_px / w_px if w_px > 0 else 1.4
        w_pt = width_mm * mm
        h_pt = w_pt * aspect
        if h_pt > max_height_mm * mm:
            h_pt = max_height_mm * mm
            w_pt = h_pt / aspect

        return RLImage(buf, width=w_pt, height=h_pt)
    except Exception:
        return None


# ─── Style registry ───────────────────────────────────────────────────────────

def _make_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"],
            fontSize=20, textColor=_C_WHITE, spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"],
            fontSize=10, textColor=_C_GREY, spaceAfter=8,
        ),
        "section": ParagraphStyle(
            "section", parent=base["Heading2"],
            fontSize=12, textColor=_C_BLUE, spaceAfter=4, spaceBefore=10,
            borderPad=2,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontSize=9, textColor=_C_WHITE, spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"],
            fontSize=8, textColor=_C_GREY, spaceAfter=2,
        ),
        "grade_big": ParagraphStyle(
            "grade_big", parent=base["Normal"],
            fontSize=36, textColor=_C_GREEN, alignment=TA_CENTER, fontName="Helvetica-Bold",
        ),
        "center": ParagraphStyle(
            "center", parent=base["Normal"],
            fontSize=9, textColor=_C_WHITE, alignment=TA_CENTER,
        ),
        "label": ParagraphStyle(
            "label", parent=base["Normal"],
            fontSize=8, textColor=_C_GREY, alignment=TA_CENTER,
        ),
    }


# ─── Section builders ─────────────────────────────────────────────────────────

def _build_header(styles, card_name: str, profile: str) -> List:
    today = date.today().strftime("%d %b %Y")
    return [
        Paragraph(f"AI Card Grader — Grading Report", styles["title"]),
        Paragraph(
            f"Card: <b>{card_name or 'Unknown'}</b> &nbsp;|&nbsp; "
            f"Profile: {profile} &nbsp;|&nbsp; Date: {today}",
            styles["subtitle"],
        ),
        HRFlowable(width="100%", thickness=1, color=_C_BORDER, spaceAfter=8),
    ]


def _build_grade_block(styles, gr: Dict, calibration: Optional[Dict]) -> List:
    grade      = gr.get("grade", 0)
    band       = gr.get("grade_band", "")
    confidence = gr.get("confidence", 0)
    rec        = gr.get("recommendation", "")

    g_color = _grade_color(grade)
    r_color = _decision_color(rec)

    grade_str = str(int(grade)) if grade == int(grade) else str(grade)

    cal_line = ""
    if calibration:
        adj = calibration.get("calibration_adjustment", 0)
        sign = "+" if adj > 0 else ""
        cal_line = f"Calibration adjustment: {sign}{adj} | Bias: {calibration.get('bias',0)}"

    data = [
        [
            Paragraph(f"<font color='#{g_color.hexval()[2:]}' size='28'><b>{grade_str}</b></font><br/>"
                      f"<font size='10' color='#9ca3af'>/10</font>", styles["center"]),
            Paragraph(
                f"<b><font color='#{g_color.hexval()[2:]}'>{band}</font></b><br/>"
                f"<font size='8' color='#9ca3af'>Confidence: {int(confidence * 100)}%</font><br/>"
                f"<br/>"
                f"<font color='#{r_color.hexval()[2:]}' size='10'><b>▶ {rec}</b></font><br/>"
                f"<font size='7' color='#9ca3af'>{gr.get('rec_reason', '')[:120]}</font>",
                styles["body"],
            ),
        ],
    ]
    t = Table(data, colWidths=[45 * mm, 130 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), _C_PANEL),
        ("GRID",         (0, 0), (-1, -1), 0.5, _C_BORDER),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", (0, 0), (-1, -1), 4),
    ]))
    elements = [
        Paragraph("Grade Result", styles["section"]),
        t,
    ]
    if cal_line:
        elements.append(Paragraph(cal_line, styles["small"]))
    return elements


def _build_score_breakdown(styles, gr: Dict) -> List:
    rows = [
        ["Category", "Score /10", "Details", "Status"],
        [
            "Centering",
            f"{gr.get('centering_score', 0)}/10",
            f"LR {gr.get('centering_lr', '?')} · TB {gr.get('centering_tb', '?')}",
            _score_badge(gr.get("centering_score", 0)),
        ],
        [
            "Edges",
            f"{gr.get('edges_score', 0)}/10",
            "See defect list",
            _score_badge(gr.get("edges_score", 0)),
        ],
        [
            "Corners",
            f"{gr.get('corners_score', 0)}/10",
            "See defect list",
            _score_badge(gr.get("corners_score", 0)),
        ],
        [
            "Surface",
            f"{gr.get('surface_score', 0)}/10",
            "See defect list",
            _score_badge(gr.get("surface_score", 0)),
        ],
    ]

    t = Table(rows, colWidths=[40 * mm, 25 * mm, 80 * mm, 30 * mm])
    style = [
        ("BACKGROUND",   (0, 0), (-1, 0),  _C_PANEL),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  _C_BLUE),
        ("BACKGROUND",   (0, 1), (-1, -1), colors.HexColor("#0d1117")),
        ("TEXTCOLOR",    (0, 1), (-1, -1), _C_WHITE),
        ("GRID",         (0, 0), (-1, -1), 0.4, _C_BORDER),
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]
    t.setStyle(TableStyle(style))

    elements = [Paragraph("Score Breakdown", styles["section"]), t]

    if gr.get("caps_triggered"):
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(
            "⚠ Grade Caps Applied: " + " | ".join(gr["caps_triggered"]),
            styles["small"],
        ))

    return elements


def _score_badge(score: float) -> str:
    if score >= 9:  return "✅ Excellent"
    if score >= 7:  return "⚠ Good"
    if score >= 5:  return "△ Fair"
    return "✗ Poor"


def _build_grade_trace(styles, grade_trace: Dict) -> List:
    if not grade_trace:
        return []

    sub = grade_trace.get("sub_scores", {})
    rows = [
        ["Step", "Value"],
        ["Centering score",   f"{sub.get('centering', 0):.2f}"],
        ["Edges score",       f"{sub.get('edges', 0):.2f}"],
        ["Corners score",     f"{sub.get('corners', 0):.2f}"],
        ["Surface score",     f"{sub.get('surface', 0):.2f}"],
        ["Weighted sum",      f"{grade_trace.get('weighted_sum', 0):.3f}"],
        ["Effective cap",     str(grade_trace.get('effective_cap', 'None'))],
        ["Post-cap grade",    f"{grade_trace.get('post_cap_grade', 0):.3f}"],
        ["Calibrated grade",  str(grade_trace.get('calibrated_grade', 'N/A'))],
        ["Final grade",       str(grade_trace.get('final_rounded_grade', 0))],
    ]

    t = Table(rows, colWidths=[90 * mm, 85 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  _C_PANEL),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  _C_BLUE),
        ("BACKGROUND",   (0, 1), (-1, -1), colors.HexColor("#0d1117")),
        ("TEXTCOLOR",    (0, 1), (-1, -1), _C_WHITE),
        ("GRID",         (0, 0), (-1, -1), 0.4, _C_BORDER),
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    return [Paragraph("Grade Trace (Audit Trail)", styles["section"]), t]


def _build_roi_section(styles, roi: Dict) -> List:
    if not roi:
        return []

    d_color = _decision_color(roi.get("decision", ""))

    probs = roi.get("grade_probabilities", {})
    prob_text = (
        f"PSA 10: {probs.get('psa_10', 0)*100:.1f}% | "
        f"PSA 9: {probs.get('psa_9', 0)*100:.1f}% | "
        f"PSA 8: {probs.get('psa_8', 0)*100:.1f}% | "
        f"Below 8: {probs.get('below_psa_8', 0)*100:.1f}%"
    )

    rows = [
        ["Metric", "Value"],
        ["Raw Value",          f"€{roi.get('raw_value', 0):.2f}"],
        ["PSA 8 Market Value", f"€{roi.get('psa_8_value', 0):.2f}"],
        ["PSA 9 Market Value", f"€{roi.get('psa_9_value', 0):.2f}"],
        ["PSA 10 Market Value",f"€{roi.get('psa_10_value', 0):.2f}"],
        ["Grading Cost",       f"€{roi.get('grading_cost', 0):.2f}"],
        ["Total Investment",   f"€{roi.get('total_investment', 0):.2f}"],
        ["Expected Value",     f"€{roi.get('expected_value', 0):.2f}"],
        ["Profit",             f"€{roi.get('profit', 0):.2f}"],
        ["ROI %",              f"{roi.get('roi_percent', 0):.1f}%"],
        ["Downside Risk",      f"{roi.get('downside_risk_pct', 0):.1f}% chance below PSA 8"],
        ["Decision",           roi.get("decision", "—")],
    ]

    t = Table(rows, colWidths=[80 * mm, 95 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  _C_PANEL),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  _C_BLUE),
        ("BACKGROUND",   (0, 1), (-1, -2), colors.HexColor("#0d1117")),
        ("BACKGROUND",   (0, -1),(-1, -1), _C_PANEL),
        ("TEXTCOLOR",    (0, 1), (-1, -1), _C_WHITE),
        ("GRID",         (0, 0), (-1, -1), 0.4, _C_BORDER),
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTNAME",     (1, -1),(-1, -1), "Helvetica-Bold"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))

    note = ""
    if roi.get("prices_estimated"):
        note = "⚠ Some PSA prices were estimated using default multipliers. Enter actual market prices for accurate ROI."

    elements = [
        Paragraph("ROI Analysis", styles["section"]),
        Paragraph(f"Grade probability distribution: {prob_text}", styles["small"]),
        Spacer(1, 4),
        t,
    ]
    if note:
        elements.append(Paragraph(note, styles["small"]))
    return elements


def _build_defect_list(styles, defects: List[Dict]) -> List:
    if not defects:
        return [
            Paragraph("Defect Evidence", styles["section"]),
            Paragraph("No defects recorded.", styles["body"]),
        ]

    severity_order = {"severe": 0, "moderate": 1, "minor": 2}
    sorted_defects = sorted(defects, key=lambda d: severity_order.get(d.get("severity", "minor"), 3))

    rows = [["Type", "Location", "Side", "Severity", "Metric Value"]]
    for d in sorted_defects:
        rows.append([
            d.get("defect_type", "").replace("_", " ").title(),
            d.get("location", ""),
            d.get("side", ""),
            d.get("severity", "").upper(),
            f"{d.get('metric_value', 0):.4f}",
        ])

    col_w = [55 * mm, 35 * mm, 20 * mm, 22 * mm, 25 * mm]
    t = Table(rows, colWidths=col_w)
    style_cmds = [
        ("BACKGROUND",   (0, 0), (-1, 0),  _C_PANEL),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  _C_BLUE),
        ("BACKGROUND",   (0, 1), (-1, -1), colors.HexColor("#0d1117")),
        ("TEXTCOLOR",    (0, 1), (-1, -1), _C_WHITE),
        ("GRID",         (0, 0), (-1, -1), 0.4, _C_BORDER),
        ("FONTSIZE",     (0, 0), (-1, -1), 7),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]
    # Colour-code severity column
    for i, d in enumerate(sorted_defects, 1):
        sev = d.get("severity", "")
        color = {"severe": _C_RED, "moderate": _C_AMBER, "minor": _C_BLUE}.get(sev, _C_WHITE)
        style_cmds.append(("TEXTCOLOR", (3, i), (3, i), color))
        style_cmds.append(("FONTNAME",  (3, i), (3, i), "Helvetica-Bold"))

    t.setStyle(TableStyle(style_cmds))
    return [Paragraph("Defect Evidence", styles["section"]), t]


def _build_image_row(styles, images: List[tuple]) -> List:
    """
    images: list of (label, bgr_array) tuples.
    Renders up to 3 images side-by-side.
    """
    filtered = [(label, img) for label, img in images if img is not None]
    if not filtered:
        return []

    max_w = 55.0
    cells = []
    for label, bgr in filtered[:3]:
        rl_img = _cv2_to_rl_image(bgr, width_mm=max_w, max_height_mm=80.0)
        if rl_img:
            cells.append([rl_img, Paragraph(label, styles["label"])])
        else:
            cells.append([Paragraph(f"[{label}]", styles["small"]), Paragraph("", styles["label"])])

    # Pad to 3 columns
    while len(cells) < 3:
        cells.append(["", ""])

    # Build 2-row table (image row + label row)
    img_row   = [c[0] for c in cells]
    label_row = [c[1] for c in cells]
    col_w = [max_w * mm] * 3

    t = Table([img_row, label_row], colWidths=col_w)
    t.setStyle(TableStyle([
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, 0),  "BOTTOM"),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
    ]))
    return [t]


# ─── Main report builder ──────────────────────────────────────────────────────

def generate_report(
    card_name:   str,
    results:     Dict[str, Any],
    roi:         Optional[Dict[str, Any]] = None,
    calibration: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Generate a PDF grading report and return it as bytes.

    Args:
        card_name:   Display name for the card.
        results:     Full results dict from grade_card_full().
        roi:         Optional ROI dict from compute_roi().
        calibration: Optional calibration dict from calibrate_grade().

    Returns:
        PDF bytes ready for writing to disk or serving as download.

    Raises:
        ImportError if reportlab is not installed.
    """
    if not _REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab is required for PDF report generation. "
            "Install with: pip install reportlab"
        )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title=f"AI Card Grader — {card_name}",
        author="AI Card Grader v2",
    )

    styles  = _make_styles()
    gr      = results.get("grade_result", {})
    trace   = results.get("grade_trace",  {})
    viz     = results.get("visualizations", {})
    defects = results.get("defects", [])
    profile = results.get("profile_used", "tcg_generic")

    front_norm = results.get("_front_norm")
    back_norm  = results.get("_back_norm")

    # Include calibrated_grade in trace display
    if calibration:
        trace = dict(trace)
        trace["calibrated_grade"] = calibration.get("calibrated_grade", "N/A")

    story: List = []

    # ── 1. Header ──────────────────────────────────────────────────────────────
    story.extend(_build_header(styles, card_name, profile))
    story.append(Spacer(1, 6))

    # ── 2. Card images ─────────────────────────────────────────────────────────
    story.append(Paragraph("Card Images", styles["section"]))
    image_pairs: List[tuple] = []
    if front_norm is not None:
        image_pairs.append(("Front (Normalized)", front_norm))
    if back_norm is not None:
        image_pairs.append(("Back (Normalized)", back_norm))
    if viz.get("full_overlay") is not None:
        image_pairs.append(("Full Defect Overlay", viz["full_overlay"]))
    story.extend(_build_image_row(styles, image_pairs))
    story.append(Spacer(1, 6))

    # ── 3. Grade block ─────────────────────────────────────────────────────────
    story.extend(_build_grade_block(styles, gr, calibration))
    story.append(Spacer(1, 6))

    # ── 4. Score breakdown ─────────────────────────────────────────────────────
    story.extend(_build_score_breakdown(styles, gr))
    story.append(Spacer(1, 6))

    # ── 5. Analysis overlays ───────────────────────────────────────────────────
    story.append(Paragraph("Defect Overlays", styles["section"]))
    overlay_pairs: List[tuple] = []
    if viz.get("centering") is not None:
        overlay_pairs.append(("Centering", viz["centering"]))
    if viz.get("corners") is not None:
        overlay_pairs.append(("Corners",   viz["corners"]))
    if viz.get("surface") is not None:
        overlay_pairs.append(("Surface",   viz["surface"]))
    story.extend(_build_image_row(styles, overlay_pairs))
    story.append(Spacer(1, 6))

    # ── 6. Grade trace ─────────────────────────────────────────────────────────
    story.extend(_build_grade_trace(styles, trace))
    story.append(Spacer(1, 6))

    # ── 7. ROI section (optional) ──────────────────────────────────────────────
    if roi:
        story.extend(_build_roi_section(styles, roi))
        story.append(Spacer(1, 6))

    # ── 8. Defect list ─────────────────────────────────────────────────────────
    defect_dicts = []
    for d in defects:
        if isinstance(d, dict):
            defect_dicts.append(d)
        elif hasattr(d, "to_dict"):
            defect_dicts.append(d.to_dict())
        elif hasattr(d, "__dict__"):
            defect_dicts.append(d.__dict__)

    story.extend(_build_defect_list(styles, defect_dicts))

    # ── Footer ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_C_BORDER))
    story.append(Paragraph(
        "Generated by AI Card Grader v2 · PSA-style automated analysis · "
        "For informational purposes only. Not a substitute for professional grading.",
        styles["small"],
    ))

    doc.build(story)
    return buf.getvalue()


def generate_report_to_file(
    card_name: str,
    results:   Dict[str, Any],
    output_path: str,
    roi:         Optional[Dict[str, Any]] = None,
    calibration: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate PDF and write to output_path. Returns output_path."""
    pdf_bytes = generate_report(card_name, results, roi=roi, calibration=calibration)
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
    return output_path
