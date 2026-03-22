"""
AI Card Grader — Streamlit UI
Production-grade interface for trading card grading analysis.
Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import traceback

from grader import grade_card, load_image, normalize_card, bgr_to_rgb

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Card Grader",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .stMetric { background: #1a1a2e; border-radius: 8px; padding: 8px; }
    .grade-banner { border-radius: 10px; padding: 18px 22px; margin-bottom: 8px; }
    div[data-testid="stImage"] img { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar() -> bool:
    with st.sidebar:
        st.markdown("## 🃏 AI Card Grader")
        st.markdown("*PSA-style AI analysis*")
        st.divider()

        show_debug = st.checkbox("Show raw data / debug", value=False)

        st.divider()
        st.markdown("### 📷 Image Tips")
        st.markdown("""
- Lay card flat on neutral background
- Even, diffuse lighting (no glare)
- Full card visible, no cropping
- Minimum 800 × 600 px recommended
        """)

        st.divider()
        st.markdown("### 📊 Grade Scale")
        grade_table = [
            ("10",   "💎 Gem Mint"),
            ("9–9.5","✨ Mint"),
            ("8–8.5","⭐ NM-MT"),
            ("7",    "🔵 Near Mint"),
            ("6",    "🟡 EX-MT"),
            ("5",    "🟠 Excellent"),
            ("≤4",   "🔴 VG or below"),
        ]
        for g, lbl in grade_table:
            st.markdown(f"`{g}` {lbl}")

        st.divider()
        st.markdown("### 🔒 Cap Rules")
        st.markdown("""
| Defect | Max Grade |
|---|---|
| Severe corner | 5 |
| Off-center >65/35 | 8 |
| Edge whitening | 8 |
| Heavy scratch | 6 |
| Surface dent | 6 |
        """)

    return show_debug


# ─── Grade Banner ─────────────────────────────────────────────────────────────
def render_grade_banner(gr: dict) -> None:
    grade = gr["grade"]
    band  = gr["grade_band"]
    rec   = gr["recommendation"]
    conf  = gr["confidence"]

    palette = {
        "high":   ("#0d2b1a", "#22c55e"),
        "mid":    ("#0d1f3c", "#60a5fa"),
        "low":    ("#2b1a00", "#f59e0b"),
        "fail":   ("#2b0a0a", "#ef4444"),
    }
    tier = ("high" if grade >= 9 else "mid" if grade >= 7
            else "low" if grade >= 5 else "fail")
    bg, fg = palette[tier]

    rec_fg = {"GRADE": "#22c55e", "CONDITIONAL": "#f59e0b",
              "DO NOT GRADE": "#ef4444"}.get(rec, "#ffffff")

    grade_disp = str(grade) if grade != int(grade) else str(int(grade))

    st.markdown(f"""
<div class="grade-banner" style="background:{bg}; border:1px solid {fg}33;">
  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
    <div>
      <span style="font-size:3.8em; font-weight:900; color:{fg}; letter-spacing:-2px;">{grade_disp}</span>
      <span style="font-size:1.2em; color:#888; margin-left:6px;">/10</span>
    </div>
    <div style="text-align:right;">
      <div style="font-size:1.35em; font-weight:700; color:{fg};">{band}</div>
      <div style="font-size:0.85em; color:#999; margin-top:2px;">
        Confidence: {int(conf * 100)}%
      </div>
    </div>
  </div>
  <div style="margin-top:12px; background:rgba(0,0,0,0.35); border-radius:6px;
              padding:9px 14px; border-left:3px solid {rec_fg};">
    <span style="color:{rec_fg}; font-size:1.05em; font-weight:700;">▶ {rec}</span>
    <span style="color:#aaa; font-size:0.82em; margin-left:10px;">{gr['rec_reason']}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Score Metrics Row ────────────────────────────────────────────────────────
def render_score_metrics(gr: dict, analysis: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    items = [
        (c1, "🎯 Centering", gr["centering_score"],
         f"LR {gr['centering_lr']} · TB {gr['centering_tb']}"),
        (c2, "🔲 Edges",     gr["edges_score"],
         f"Whitening {analysis['edges']['avg_whitening']:.3f}"),
        (c3, "📐 Corners",   gr["corners_score"],
         f"Defects: {len(analysis['corners']['defect_corners'])} corner(s)"),
        (c4, "✨ Surface",   gr["surface_score"],
         f"Scratches {analysis['surface']['scratch_density']:.4f}"),
    ]
    for col, label, score, detail in items:
        with col:
            st.metric(label, f"{score}/10", delta=detail)

    # Flags / caps
    if gr["caps_triggered"]:
        caps_md = " &nbsp;|&nbsp; ".join(
            f"🔒 {c}" for c in gr["caps_triggered"]
        )
        st.error(f"**Grade Caps Applied:** {caps_md}")

    if gr["risk_flags"]:
        flags_md = " &nbsp;|&nbsp; ".join(
            f"⚠️ {f.title()}" for f in gr["risk_flags"]
        )
        st.warning(f"**Risk Flags:** {flags_md}")


# ─── Tab: Overview ────────────────────────────────────────────────────────────
def tab_overview(front_norm: np.ndarray, viz: dict,
                 gr: dict, analysis: dict, detected: bool) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Normalized Card")
        if not detected:
            st.caption("⚠️ Auto-boundary detection fell back — "
                       "ensure full card is visible in image.")
        st.image(bgr_to_rgb(front_norm), use_container_width=True)

    with c2:
        st.subheader("Grade Summary")
        st.image(bgr_to_rgb(viz["grade_summary"]), use_container_width=True)

    st.divider()
    st.subheader("📋 Full Breakdown")

    rows = [
        ("Centering", gr["centering_score"],
         f"LR {gr['centering_lr']} · TB {gr['centering_tb']}",
         analysis["centering"]["score"]),
        ("Edges", gr["edges_score"],
         f"Avg whitening {analysis['edges']['avg_whitening']:.4f} · "
         f"Defects: {', '.join(analysis['edges']['defect_locations']) or 'none'}",
         analysis["edges"]["score"]),
        ("Corners", gr["corners_score"],
         f"Defect corners: {', '.join(analysis['corners']['defect_corners']) or 'none'}",
         analysis["corners"]["score"]),
        ("Surface", gr["surface_score"],
         f"Scratches {analysis['surface']['scratch_density']:.5f} · "
         f"Anomalies {analysis['surface']['anomaly_density']:.5f}",
         analysis["surface"]["score"]),
    ]

    header_cols = st.columns([2, 1.5, 5, 2])
    for col, h in zip(header_cols, ["Category", "Score", "Details", "Status"]):
        col.markdown(f"**{h}**")
    st.markdown("---")

    for cat, score, detail, _ in rows:
        cc = st.columns([2, 1.5, 5, 2])
        cc[0].markdown(f"**{cat}**")
        cc[1].markdown(f"`{score}/10`")
        cc[2].markdown(detail)
        cc[3].markdown(_score_badge(score))


# ─── Tab: Centering ──────────────────────────────────────────────────────────
def tab_centering(front_norm: np.ndarray, viz: dict, centering: dict) -> None:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Centering Overlay")
        st.image(bgr_to_rgb(viz["centering"]), use_container_width=True,
                 caption="Cyan = detected border  |  Green = ideal centre")
    with c2:
        st.subheader("Metrics")
        st.metric("Centering Score", f"{centering['score']}/10")
        st.markdown("**Left / Right:**")
        st.progress(centering["left_pct"] / 100)
        st.code(f"{centering['lr_ratio']}  (L/R%)", language="")

        st.markdown("**Top / Bottom:**")
        st.progress(centering["top_pct"] / 100)
        st.code(f"{centering['tb_ratio']}  (T/B%)", language="")

        st.divider()
        st.markdown("**PSA Centering Thresholds:**")
        standards = {
            "PSA 10": "55/45 or better",
            "PSA 9":  "60/40 or better",
            "PSA 8":  "65/35 or better",
            "PSA 7":  "70/30 or better",
        }
        for g, s in standards.items():
            st.markdown(f"- `{g}` → {s}")


# ─── Tab: Edges & Corners ─────────────────────────────────────────────────────
def tab_edges_corners(front_norm: np.ndarray, viz: dict,
                      edges: dict, corners: dict) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Edge Analysis")
        st.image(bgr_to_rgb(viz["edges"]), use_container_width=True,
                 caption="🟢 Clean  🟡 Minor  🟠 Moderate  🔴 Severe")
        st.markdown(f"**Overall Edge Score: `{edges['score']}/10`**")
        st.divider()
        for side, data in edges["sides"].items():
            lvl = data["defect_level"]
            icon = {"clean": "✅", "minor": "⚠️",
                    "moderate": "🟠", "severe": "🔴"}.get(lvl, "❓")
            st.markdown(
                f"{icon} **{side.title()}** — {lvl.upper()} &nbsp;"
                f"*(whitening={data['whitening']:.4f} · "
                f"roughness={data['roughness']:.4f} · "
                f"chipping={data['chipping']:.4f})*"
            )

    with c2:
        st.subheader("Corner Analysis")
        st.image(bgr_to_rgb(viz["corners"]), use_container_width=True,
                 caption="Corner wear visualization")
        st.markdown(f"**Overall Corner Score: `{corners['score']}/10`**")
        st.divider()
        for corner, data in corners["corners"].items():
            lvl = data["defect_level"]
            icon = {"sharp": "✅", "minor": "⚠️",
                    "moderate": "🟠", "severe": "🔴"}.get(lvl, "❓")
            label = corner.replace("_", " ").title()
            st.markdown(
                f"{icon} **{label}** — {lvl.upper()} &nbsp;"
                f"*(whitening={data['whitening']:.4f} · "
                f"rounding={data['rounding']:.4f})*"
            )


# ─── Tab: Surface ────────────────────────────────────────────────────────────
def tab_surface(front_norm: np.ndarray, viz: dict, surface: dict) -> None:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Surface Defect Map")
        st.image(bgr_to_rgb(viz["surface"]), use_container_width=True,
                 caption="Red = scratches  |  Orange = anomalies / dents")

    with c2:
        st.subheader("Surface Metrics")
        st.metric("Surface Score", f"{surface['score']}/10")
        st.divider()

        metrics = [
            ("Scratch Density",       surface["scratch_density"],   0.015, 0.04),
            ("Print Line Indicator",  surface["print_line_indicator"], 0.3,  0.6),
            ("Anomaly Density",       surface["anomaly_density"],   0.04, 0.08),
            ("Surface Noise",         surface["surface_noise"],     0.35, 0.65),
        ]
        for name, val, warn, bad in metrics:
            icon = "🟢" if val < warn else ("🟡" if val < bad else "🔴")
            st.markdown(f"{icon} **{name}:** `{val:.5f}`")

        st.divider()
        if surface["risk_flags"]:
            for flag in surface["risk_flags"]:
                st.error(f"⚠️ {flag.title()}")
        else:
            st.success("✅ No significant surface defects detected")


# ─── Tab: Full Overlay ───────────────────────────────────────────────────────
def tab_full_overlay(front_norm: np.ndarray, viz: dict) -> None:
    st.subheader("Combined Defect Overlay")
    st.markdown("All detected defects layered on the original image.")
    c1, c2 = st.columns(2)
    with c1:
        st.image(bgr_to_rgb(front_norm), caption="Original",
                 use_container_width=True)
    with c2:
        st.image(bgr_to_rgb(viz["full_overlay"]), caption="All Defects",
                 use_container_width=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _score_badge(score: float) -> str:
    if score >= 9:
        return "✅ Excellent"
    elif score >= 7:
        return "⚠️ Good"
    elif score >= 5:
        return "🟠 Fair"
    return "🔴 Poor"


def _show_landing() -> None:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 📷 For Best Results")
        st.markdown("""
- Flat, steady surface
- No direct flash or strong glare
- Full card visible, slight padding OK
- 1+ MP resolution preferred
        """)
    with c2:
        st.markdown("### 🔬 What We Detect")
        st.markdown("""
- Border centering (L/R, T/B ratios)
- Edge whitening, chipping, roughness
- Corner wear, rounding, whitening
- Surface scratches, print lines, dents
        """)
    with c3:
        st.markdown("### 📈 Grade → Value")
        st.markdown("""
- **10** → Premium slab — grade it
- **9** → High demand → grade for value
- **8** → Card-dependent → conditional
- **7** → Low ROI most sets
- **≤6** → Sell raw or hold
        """)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    show_debug = render_sidebar()

    st.title("🃏 AI Card Grader")
    st.markdown("*Automated PSA-style grading analysis — fully local, no paid APIs*")

    col_front, col_back = st.columns(2)
    with col_front:
        st.subheader("Front of Card")
        front_file = st.file_uploader(
            "Upload front image", type=["jpg", "jpeg", "png", "webp"], key="front"
        )
    with col_back:
        st.subheader("Back of Card *(optional)*")
        back_file = st.file_uploader(
            "Upload back image", type=["jpg", "jpeg", "png", "webp"], key="back"
        )

    if front_file is None:
        _show_landing()
        return

    # ── Load & normalize ────────────────────────────────────────────────────
    with st.spinner("🔬 Running grading analysis…"):
        try:
            front_raw = load_image(front_file.read())
            if front_raw is None:
                st.error("Could not decode front image. Please try a different file.")
                return

            front_norm, detected = normalize_card(front_raw)

            back_norm = None
            if back_file is not None:
                back_raw = load_image(back_file.read())
                if back_raw is not None:
                    back_norm, _ = normalize_card(back_raw)

            results = grade_card(front_norm, back_norm)

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            if show_debug:
                st.code(traceback.format_exc())
            return

    gr      = results["grade_result"]
    analysis = results["analysis"]
    viz     = results["visualizations"]

    st.markdown("---")

    # ── Grade banner ─────────────────────────────────────────────────────────
    render_grade_banner(gr)
    render_score_metrics(gr, analysis)

    st.markdown("---")

    # ── Analysis tabs ────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "📊 Overview",
        "🎯 Centering",
        "🔲 Edges & Corners",
        "✨ Surface",
        "🗺️ Full Overlay",
    ])

    with t1:
        tab_overview(front_norm, viz, gr, analysis, detected)
    with t2:
        tab_centering(front_norm, viz, analysis["centering"])
    with t3:
        tab_edges_corners(front_norm, viz, analysis["edges"], analysis["corners"])
    with t4:
        tab_surface(front_norm, viz, analysis["surface"])
    with t5:
        tab_full_overlay(front_norm, viz)

    # ── Debug ────────────────────────────────────────────────────────────────
    if show_debug:
        with st.expander("🔧 Raw Analysis Data"):
            def _safe(d):
                return {k: v for k, v in d.items()
                        if not isinstance(v, np.ndarray)}

            st.json({
                "grade_result": _safe(gr),
                "centering":    _safe(analysis["centering"]),
                "edges":        {
                    **_safe(analysis["edges"]),
                    "sides": analysis["edges"]["sides"],
                },
                "corners":      analysis["corners"],
                "surface":      _safe(analysis["surface"]),
            })


if __name__ == "__main__":
    main()
