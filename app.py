"""
AI Card Grader v2 — Streamlit UI
Production-grade interface with quality gate, card profiles, defect evidence,
grade trace, front/back comparison, card comparison mode, and artifact download.
Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import io
import traceback
from typing import Optional, Dict, Any

from grader import (
    grade_card_full, load_image, normalize_card,
    list_profiles, get_profile_display_names,
    build_download_zip, defects_to_dicts,
    bgr_to_rgb,
    build_pricing, CardPricingBundle,
    generate_report,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Card Grader",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1rem; }
.stMetric { background: #1a1a2e; border-radius: 8px; padding: 8px; }
div[data-testid="stImage"] img { border-radius: 6px; }
.defect-row { padding: 4px 8px; border-radius: 4px; margin: 2px 0; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
if "comparison_cards" not in st.session_state:
    st.session_state.comparison_cards = []


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar() -> tuple:
    with st.sidebar:
        st.markdown("## 🃏 AI Card Grader v2")
        st.markdown("*PSA-style AI analysis · Zero cloud dependency*")
        st.divider()

        # Profile selector
        st.markdown("### 🎴 Card Profile")
        disp_names = get_profile_display_names()
        profile_keys = list(disp_names.keys())
        profile_labels = [disp_names[k] for k in profile_keys]
        sel_idx = st.selectbox(
            "Select card type", range(len(profile_keys)),
            format_func=lambda i: profile_labels[i],
            index=profile_keys.index("tcg_generic"),
        )
        profile_name = profile_keys[sel_idx]

        st.divider()
        show_debug = st.checkbox("Show raw data / debug", False)
        run_qgate  = st.checkbox("Run quality gate", True)

        # ── ROI / Pricing ─────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 💰 ROI Analysis *(optional)*")
        st.caption("Enter pricing to get a grading ROI decision.")

        enable_roi = st.checkbox("Enable ROI analysis", False)
        pricing_bundle: Optional[CardPricingBundle] = None

        if enable_roi:
            card_name_roi = st.text_input("Card name", placeholder="e.g. Charizard Holo #4",
                                           key="roi_card_name")
            raw_price = st.number_input("Raw (ungraded) value ($)", min_value=0.0,
                                         value=0.0, step=1.0, format="%.2f")
            grading_cost = st.number_input("Grading fee ($)", min_value=0.0,
                                            value=25.0, step=1.0, format="%.2f")

            st.caption("PSA slab prices *(leave 0 to auto-estimate)*")
            psa8  = st.number_input("PSA 8 ($)", min_value=0.0, value=0.0,
                                     step=1.0, format="%.2f", key="psa8")
            psa9  = st.number_input("PSA 9 ($)", min_value=0.0, value=0.0,
                                     step=1.0, format="%.2f", key="psa9")
            psa10 = st.number_input("PSA 10 ($)", min_value=0.0, value=0.0,
                                     step=1.0, format="%.2f", key="psa10")

            if raw_price > 0:
                pricing_bundle = build_pricing(
                    card_name=card_name_roi or "Unknown",
                    raw_price=raw_price,
                    grading_cost=grading_cost,
                    psa_8=psa8  if psa8  > 0 else None,
                    psa_9=psa9  if psa9  > 0 else None,
                    psa_10=psa10 if psa10 > 0 else None,
                )
            else:
                st.warning("Enter a raw value > 0 to compute ROI.")

        st.divider()
        st.markdown("### 📷 Image Tips")
        st.markdown("""
- Flat on neutral background
- Diffuse, even lighting (no glare)
- Full card in frame, no cropping
- ≥ 800 × 600 px recommended
- Remove from sleeve/toploader
        """)
        st.divider()
        st.markdown("### 📊 Grade Scale")
        for g, lbl in [("10","💎 Gem Mint"),("9–9.5","✨ Mint"),("8–8.5","⭐ NM-MT"),
                        ("7","🔵 Near Mint"),("6","🟡 EX-MT"),("5","🟠 Excellent"),("≤4","🔴 VG or below")]:
            st.markdown(f"`{g}` {lbl}")

    return profile_name, show_debug, run_qgate, pricing_bundle


# ─── Quality Gate Widget ──────────────────────────────────────────────────────
def render_quality_gate(qg: Dict) -> None:
    decision = qg.get("decision", "pass")
    score    = qg.get("score", 100)
    summary  = qg.get("summary", "")
    issues   = qg.get("issues", [])

    if decision == "pass":
        st.success(f"✅ **Quality Gate: PASS** (score {score}/100) — {summary}")
    elif decision == "warn":
        st.warning(f"⚠️ **Quality Gate: WARN** (score {score}/100) — {summary}")
    else:
        st.error(f"❌ **Quality Gate: FAIL** (score {score}/100) — {summary}")

    if issues:
        with st.expander("Quality issues detail", expanded=(decision == "fail")):
            for issue in issues:
                sev = issue.get("severity", "warn")
                icon = "❌" if sev == "fail" else "⚠️"
                st.markdown(
                    f"{icon} **{issue['check'].replace('_',' ').title()}** — "
                    f"{issue['message']} "
                    f"*(metric: {issue['metric_name']} = {issue['metric_value']:.3f})*"
                )


# ─── Grade Banner ─────────────────────────────────────────────────────────────
def render_grade_banner(gr: Dict) -> None:
    grade = gr["grade"]
    band  = gr["grade_band"]
    rec   = gr["recommendation"]
    conf  = gr["confidence"]

    tier_map = {
        "high": (grade >= 9,    "#0d2b1a", "#22c55e"),
        "mid":  (grade >= 7,    "#0d1f3c", "#60a5fa"),
        "low":  (grade >= 5,    "#2b1a00", "#f59e0b"),
        "fail": (True,          "#2b0a0a", "#ef4444"),
    }
    bg, fg = next((b, f) for t, (cond, b, f) in tier_map.items() if cond)
    rec_fg = {"GRADE": "#22c55e", "CONDITIONAL": "#f59e0b",
              "DO NOT GRADE": "#ef4444"}.get(rec, "#ffffff")
    grade_disp = str(grade) if grade != int(grade) else str(int(grade))

    st.markdown(f"""
<div style="background:{bg};border:1px solid {fg}33;border-radius:10px;
            padding:18px 22px;margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
    <div>
      <span style="font-size:3.8em;font-weight:900;color:{fg};letter-spacing:-2px;">{grade_disp}</span>
      <span style="font-size:1.2em;color:#888;margin-left:6px;">/10</span>
    </div>
    <div style="text-align:right;">
      <div style="font-size:1.35em;font-weight:700;color:{fg};">{band}</div>
      <div style="font-size:0.85em;color:#999;margin-top:2px;">Confidence: {int(conf*100)}%</div>
    </div>
  </div>
  <div style="margin-top:12px;background:rgba(0,0,0,0.35);border-radius:6px;
              padding:9px 14px;border-left:3px solid {rec_fg};">
    <span style="color:{rec_fg};font-size:1.05em;font-weight:700;">▶ {rec}</span>
    <span style="color:#aaa;font-size:0.82em;margin-left:10px;">{gr['rec_reason']}</span>
  </div>
</div>
""", unsafe_allow_html=True)


def render_score_metrics(gr: Dict, analysis: Dict) -> None:
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

    if gr["caps_triggered"]:
        st.error("**🔒 Grade Caps Applied:** " +
                 " | ".join(f"🔒 {c}" for c in gr["caps_triggered"]))
    if gr["risk_flags"]:
        st.warning("**⚠️ Risk Flags:** " +
                   " | ".join(f"⚠️ {f.title()}" for f in gr["risk_flags"]))


# ─── Analysis Tabs ────────────────────────────────────────────────────────────

def tab_overview(front_norm, viz, gr, analysis, detected, profile_name) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Normalised Card")
        if not detected:
            st.caption("⚠️ Auto-boundary detection fell back — ensure full card is visible.")
        st.image(bgr_to_rgb(front_norm), use_container_width=True)
        st.caption(f"Profile: **{profile_name}**")
    with c2:
        st.subheader("Grade Summary")
        st.image(bgr_to_rgb(viz["grade_summary"]), use_container_width=True)

    st.divider()
    st.subheader("📋 Full Breakdown")
    header = st.columns([2, 1.5, 5, 2])
    for col, h in zip(header, ["Category", "Score", "Details", "Status"]):
        col.markdown(f"**{h}**")
    st.markdown("---")
    rows = [
        ("Centering", gr["centering_score"],
         f"LR {gr['centering_lr']} · TB {gr['centering_tb']}"),
        ("Edges",     gr["edges_score"],
         f"Avg whitening {analysis['edges']['avg_whitening']:.4f} · "
         f"Defects: {', '.join(analysis['edges']['defect_locations']) or 'none'}"),
        ("Corners",   gr["corners_score"],
         f"Defect corners: {', '.join(analysis['corners']['defect_corners']) or 'none'}"),
        ("Surface",   gr["surface_score"],
         f"Scratches {analysis['surface']['scratch_density']:.5f} · "
         f"Anomalies {analysis['surface']['anomaly_density']:.5f}"),
    ]
    for cat, score, detail in rows:
        cc = st.columns([2, 1.5, 5, 2])
        cc[0].markdown(f"**{cat}**")
        cc[1].markdown(f"`{score}/10`")
        cc[2].markdown(detail)
        cc[3].markdown(_score_badge(score))


def tab_front_back(front_norm, back_norm, front_analysis, back_analysis, viz) -> None:
    """Side-by-side front/back comparison."""
    col_f, col_b = st.columns(2)
    with col_f:
        st.subheader("Front")
        st.image(bgr_to_rgb(front_norm), use_container_width=True)
        if front_analysis:
            _render_side_scores("Front", front_analysis)
    with col_b:
        st.subheader("Back")
        if back_norm is not None:
            st.image(bgr_to_rgb(back_norm), use_container_width=True)
            if back_analysis:
                _render_side_scores("Back", back_analysis)
        else:
            st.info("No back image uploaded.")
            st.markdown("Upload a back image for combined front+back grading.")


def _render_side_scores(label: str, analysis: Dict) -> None:
    st.markdown(f"**{label} sub-scores:**")
    items = [
        ("Centering", analysis["centering"]["score"]),
        ("Edges",     analysis["edges"]["score"]),
        ("Corners",   analysis["corners"]["score"]),
        ("Surface",   analysis["surface"]["score"]),
    ]
    cols = st.columns(4)
    for col, (name, score) in zip(cols, items):
        col.metric(name, f"{score}/10")


def tab_centering(front_norm, viz, centering) -> None:
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
        if "front_lr" in centering:
            st.divider()
            st.markdown("**Per-side centering:**")
            st.markdown(f"- Front: LR `{centering['front_lr']}` · TB `{centering['front_tb']}`")
            st.markdown(f"- Back:  LR `{centering['back_lr']}` · TB `{centering['back_tb']}`")
        st.divider()
        st.markdown("**PSA Thresholds:**")
        for g, s in [("PSA 10","55/45 or better"),("PSA 9","60/40"),
                     ("PSA 8","65/35"),("PSA 7","70/30")]:
            st.markdown(f"- `{g}` → {s}")


def tab_edges_corners(front_norm, viz, edges, corners) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Edge Analysis")
        st.image(bgr_to_rgb(viz["edges"]), use_container_width=True,
                 caption="🟢 Clean  🟡 Minor  🟠 Moderate  🔴 Severe")
        st.markdown(f"**Overall Edge Score: `{edges['score']}/10`**")
        st.divider()
        for side, data in edges["sides"].items():
            lvl  = data["defect_level"]
            icon = {"clean":"✅","minor":"⚠️","moderate":"🟠","severe":"🔴"}.get(lvl,"❓")
            from_tag = f" *(from {data.get('from','front')})*" if "from" in data else ""
            st.markdown(
                f"{icon} **{side.title()}**{from_tag} — {lvl.upper()} "
                f"*(w={data['whitening']:.4f} r={data['roughness']:.4f} "
                f"c={data['chipping']:.4f})*"
            )
    with c2:
        st.subheader("Corner Analysis")
        st.image(bgr_to_rgb(viz["corners"]), use_container_width=True,
                 caption="Corner wear visualisation")
        st.markdown(f"**Overall Corner Score: `{corners['score']}/10`**")
        st.divider()
        for corner, data in corners["corners"].items():
            lvl  = data["defect_level"]
            icon = {"sharp":"✅","minor":"⚠️","moderate":"🟠","severe":"🔴"}.get(lvl,"❓")
            from_tag = f" *(from {data.get('from','front')})*" if "from" in data else ""
            st.markdown(
                f"{icon} **{corner.replace('_',' ').title()}**{from_tag} — {lvl.upper()} "
                f"*(w={data['whitening']:.4f} r={data['rounding']:.4f})*"
            )


def tab_surface(front_norm, viz, surface) -> None:
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
            ("Scratch Density",      surface["scratch_density"],  0.015, 0.04),
            ("Print Line Indicator", surface["print_line_indicator"], 0.3, 0.6),
            ("Anomaly Density",      surface["anomaly_density"],  0.04, 0.08),
            ("Surface Noise",        surface["surface_noise"],    0.35, 0.65),
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


def tab_full_overlay(front_norm, viz) -> None:
    st.subheader("Combined Defect Overlay")
    st.markdown("*All detected defects layered on the original image.*")
    c1, c2 = st.columns(2)
    with c1:
        st.image(bgr_to_rgb(front_norm), caption="Original", use_container_width=True)
    with c2:
        st.image(bgr_to_rgb(viz["full_overlay"]), caption="All Defects",
                 use_container_width=True)


def tab_defect_evidence(defects: list, viz) -> None:
    """Structured defect evidence list with bounding-box overlay."""
    st.subheader("Defect Evidence")
    st.markdown("*Every detected issue as a structured record with location and severity.*")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.image(bgr_to_rgb(viz["defect_evidence"]), use_container_width=True,
                 caption="Colour-coded bounding boxes per defect")
    with c2:
        if not defects:
            st.success("✅ No defects recorded.")
            return

        dicts = [d if isinstance(d, dict) else d.to_dict() for d in defects]
        severity_map = {"severe": ("🔴","#4a0a0a"), "moderate": ("🟠","#3a2a00"),
                        "minor": ("⚠️","#2a2a00")}

        counts = {"severe": 0, "moderate": 0, "minor": 0}
        for d in dicts:
            if d["severity"] in counts:
                counts[d["severity"]] += 1
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("🔴 Severe",   counts["severe"])
        mc2.metric("🟠 Moderate", counts["moderate"])
        mc3.metric("⚠️ Minor",    counts["minor"])

        st.divider()
        for d in dicts:
            sev = d.get("severity", "minor")
            icon, bg = severity_map.get(sev, ("❓", "#1a1a1a"))
            st.markdown(
                f'<div style="background:{bg};padding:6px 10px;border-radius:5px;margin:3px 0;">'
                f'{icon} <b>{d["defect_type"].replace("_"," ").title()}</b> '
                f'— {d["location"]} ({d["side"]}) <br/>'
                f'<span style="color:#aaa;font-size:0.82em;">{d["explanation"]}'
                f' | {d["metric"]}={d["metric_value"]:.4f}</span></div>',
                unsafe_allow_html=True
            )


def tab_grade_trace(gr: Dict, grade_trace: Dict, viz,
                    calibration: Optional[Dict] = None) -> None:
    """Full grade trace audit view."""
    st.subheader("Grade Trace")
    st.markdown("*Complete scoring audit trail — from sub-scores through cap evaluation to final grade.*")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.image(bgr_to_rgb(viz["grade_trace"]), use_container_width=True)
    with c2:
        st.markdown("**Scoring Summary:**")
        st.markdown(f"- Weighted sum (pre-cap): `{grade_trace.get('weighted_sum', 0):.3f}`")
        eff_cap = grade_trace.get("effective_cap")
        if eff_cap is not None:
            st.markdown(f"- Effective cap: `{eff_cap}`")
            st.markdown(f"- Post-cap: `{grade_trace.get('post_cap_grade', 0):.3f}`")
        else:
            st.markdown("- No caps applied")
        st.markdown(f"- **Final rounded grade: `{grade_trace.get('final_rounded_grade', 0)}`**")
        st.divider()
        caps = grade_trace.get("caps_evaluated", [])
        if caps:
            triggered_caps = [c for c in caps if c.get("triggered")]
            safe_caps      = [c for c in caps if not c.get("triggered")]
            if triggered_caps:
                st.markdown("**🔒 Triggered caps:**")
                for c in triggered_caps:
                    st.error(f"Cap {c['cap']}: {c['description']}")
            if safe_caps:
                with st.expander(f"Non-triggered cap rules ({len(safe_caps)})"):
                    for c in safe_caps:
                        st.markdown(f"✅ Cap {c['cap']}: {c['description']}")

    # ── Calibration block ─────────────────────────────────────────────────────
    if calibration:
        st.divider()
        st.markdown("**🎯 Calibration (PSA Alignment)**")
        cal_g  = calibration.get("calibrated_grade", "—")
        adj    = calibration.get("calibration_adjustment", 0.0)
        bias   = calibration.get("bias", 0.0)
        cal_c  = calibration.get("calibrated_confidence", "—")
        # Derive raw grade from calibrated grade and adjustment
        raw_g  = round(cal_g - adj, 1) if isinstance(cal_g, float) and isinstance(adj, float) else "—"

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Raw Grade",        raw_g)
        cc2.metric("Calibrated Grade", cal_g,
                   delta=f"{adj:+.1f}" if isinstance(adj, float) and adj != 0 else "no change")
        cc3.metric("Calibrated Confidence",
                   f"{int(cal_c*100)}%" if isinstance(cal_c, float) else cal_c)
        st.caption(
            f"Dataset bias: `{bias:+.3f}` · "
            f"Adjustment applied: `{adj:+.2f}` · "
            f"Std dev: `{calibration.get('std_dev', 0):.3f}`"
        )


def tab_roi(roi: Dict) -> None:
    """ROI analysis display."""
    decision = roi["decision"]
    decision_colors = {
        "STRONG GRADE": ("#0d2b1a", "#22c55e"),
        "GRADE":        ("#0d2b1a", "#22c55e"),
        "CONDITIONAL":  ("#2b1f00", "#f59e0b"),
        "HOLD RAW":     ("#2b1f00", "#f59e0b"),
        "DO NOT GRADE": ("#2b0a0a", "#ef4444"),
    }
    bg, fg = decision_colors.get(decision, ("#1a1a2e", "#ffffff"))

    st.markdown(f"""
<div style="background:{bg};border:1px solid {fg}33;border-radius:10px;
            padding:16px 20px;margin-bottom:12px;">
  <div style="font-size:1.6em;font-weight:900;color:{fg};">{decision}</div>
  <div style="color:#aaa;font-size:0.9em;margin-top:4px;">
    Card: <b>{roi['card_name']}</b> &nbsp;·&nbsp;
    Grade: <b>{roi['estimated_grade']}</b> &nbsp;·&nbsp;
    Confidence: <b>{int(roi['confidence']*100)}%</b>
    {"&nbsp;·&nbsp;<i>Prices partially estimated</i>" if roi.get('prices_estimated') else ""}
  </div>
</div>
""", unsafe_allow_html=True)

    # Financial metrics
    c1, c2, c3, c4 = st.columns(4)
    profit_delta = f"+${roi['profit']:.2f}" if roi["profit"] >= 0 else f"-${abs(roi['profit']):.2f}"
    c1.metric("Expected Value",  f"${roi['expected_value']:.2f}")
    c2.metric("Est. Profit",     profit_delta)
    c3.metric("ROI",             f"{roi['roi_percent']:.1f}%")
    c4.metric("Downside Risk",   f"{roi['downside_risk_pct']:.1f}%")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Price Inputs**")
        price_rows = [
            ("Raw value",     roi["raw_value"]),
            ("Grading fee",   roi["grading_cost"]),
            ("Total invested",roi["total_investment"]),
            ("PSA 8 value",   roi["psa_8_value"]),
            ("PSA 9 value",   roi["psa_9_value"]),
            ("PSA 10 value",  roi["psa_10_value"]),
        ]
        for label, val in price_rows:
            st.markdown(f"- **{label}:** `${val:.2f}`")
        if roi.get("prices_estimated"):
            st.caption("⚠️ One or more PSA prices were estimated using default multipliers.")

    with col_r:
        st.markdown("**Grade Probability Distribution**")
        probs = roi["grade_probabilities"]
        for tier, key in [("PSA 10", "psa_10"), ("PSA 9", "psa_9"),
                           ("PSA 8", "psa_8"), ("Below PSA 8", "below_psa_8")]:
            pct = probs[key] * 100
            bar_color = "#22c55e" if key == "psa_10" else "#60a5fa" if key == "psa_9" else \
                        "#f59e0b" if key == "psa_8" else "#ef4444"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
                f'<span style="width:90px;font-size:0.85em;">{tier}</span>'
                f'<div style="flex:1;background:#1a1a2e;border-radius:4px;height:14px;">'
                f'<div style="width:{pct:.1f}%;background:{bar_color};height:100%;border-radius:4px;"></div>'
                f'</div>'
                f'<span style="width:42px;text-align:right;font-size:0.85em;">{pct:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("**Scenario Analysis**")
    sc1, sc2 = st.columns(2)
    sc1.metric("Best case (PSA 10)",  f"${roi['best_case_profit']:+.2f}")
    sc2.metric("Worst case (<PSA 8)", f"${roi['worst_case_profit']:+.2f}")
    if roi.get("confidence_adjusted"):
        st.info("ℹ️ Confidence below 90% — decision capped conservatively.")


# ─── Comparison Mode ──────────────────────────────────────────────────────────

def render_comparison_section() -> None:
    """Card comparison — multiple cards ranked best to worst."""
    st.divider()
    st.subheader("🆚 Card Comparison")

    if not st.session_state.comparison_cards:
        st.info("Grade a card above, then click **Add to Comparison** to start comparing multiple cards.")
        return

    cards = st.session_state.comparison_cards
    st.markdown(f"**{len(cards)} card(s) in comparison.** Sorted best → worst.")

    if st.button("🗑 Clear Comparison"):
        st.session_state.comparison_cards = []
        st.rerun()

    # Sort by grade desc
    sorted_cards = sorted(cards, key=lambda c: -c["grade"])

    # Header row
    cols = st.columns([1, 2, 1.5, 1.5, 1.5, 1.5, 2])
    for col, h in zip(cols, ["Rank","Name","Grade","Band","Centering","Corners","Rec"]):
        col.markdown(f"**{h}**")
    st.markdown("---")

    for i, card in enumerate(sorted_cards, 1):
        cc = st.columns([1, 2, 1.5, 1.5, 1.5, 1.5, 2])
        grade = card["grade"]
        grade_col = "#22c55e" if grade >= 9 else "#60a5fa" if grade >= 7 else "#ef4444"
        cc[0].markdown(f"**{i}**")
        cc[1].markdown(card["name"])
        cc[2].markdown(f"<span style='color:{grade_col};font-weight:700;font-size:1.1em;'>"
                        f"{grade}</span>", unsafe_allow_html=True)
        cc[3].markdown(card.get("band", "—"))
        cc[4].markdown(f"`{card.get('centering_lr','?')}`")
        cc[5].markdown(f"`{card.get('corners_score','?')}/10`")
        rec_icons = {"GRADE":"✅","CONDITIONAL":"⚠️","DO NOT GRADE":"❌"}
        cc[6].markdown(f"{rec_icons.get(card.get('rec',''),'?')} {card.get('rec','')}")


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _score_badge(score: float) -> str:
    if score >= 9:  return "✅ Excellent"
    if score >= 7:  return "⚠️ Good"
    if score >= 5:  return "🟠 Fair"
    return "🔴 Poor"


def _show_landing() -> None:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 📷 Best Results")
        st.markdown("- Flat, neutral background\n- Diffuse lighting, no glare\n"
                    "- Full card in frame\n- Remove from sleeve/toploader\n- ≥ 1 MP")
    with c2:
        st.markdown("### 🔬 What We Detect")
        st.markdown("- Border centering (L/R, T/B)\n- Edge whitening/chipping\n"
                    "- Corner wear/rounding\n- Surface scratches/dents/print lines")
    with c3:
        st.markdown("### 🎴 Card Profiles")
        st.markdown("- **Pokémon Modern** — strict centering\n"
                    "- **Pokémon Vintage** — lenient center, critical surface\n"
                    "- **Sports Chrome** — high surface sensitivity\n"
                    "- **Sports Paper** — standard tolerances\n"
                    "- **TCG Generic** — balanced defaults")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    profile_name, show_debug, run_qgate, pricing_bundle = render_sidebar()

    st.title("🃏 AI Card Grader v2")
    st.markdown("*Automated PSA-style grading · Card profiles · Defect evidence · Grade trace*")

    # ── Upload section ────────────────────────────────────────────────────────
    col_front, col_back = st.columns(2)
    with col_front:
        st.subheader("Front of Card")
        front_file = st.file_uploader("Upload front image",
                                       type=["jpg","jpeg","png","webp"], key="front")
    with col_back:
        st.subheader("Back of Card *(optional)*")
        back_file = st.file_uploader("Upload back image",
                                      type=["jpg","jpeg","png","webp"], key="back")

    if front_file is None:
        _show_landing()
        render_comparison_section()
        return

    # ── Load & normalize ──────────────────────────────────────────────────────
    with st.spinner("🔬 Running grading analysis…"):
        try:
            front_raw = load_image(front_file.read())
            if front_raw is None:
                st.error("Could not decode front image. Please try a different file.")
                return
            front_norm, detected = normalize_card(front_raw)

            back_raw  = None
            back_norm = None
            if back_file is not None:
                back_raw = load_image(back_file.read())
                if back_raw is not None:
                    back_norm, _ = normalize_card(back_raw)

            results = grade_card_full(
                front_norm, back_norm,
                profile_name=profile_name,
                run_quality_gate=run_qgate,
                front_raw=front_raw,
                back_raw=back_raw,
                pricing=pricing_bundle,
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            if show_debug:
                st.code(traceback.format_exc())
            return

    gr           = results["grade_result"]
    grade_trace  = results["grade_trace"]
    analysis     = results["analysis"]
    front_an     = results["front_analysis"]
    back_an      = results["back_analysis"]
    viz          = results["visualizations"]
    defects      = results["defects"]
    qg           = results.get("quality_gate")
    roi_result   = results.get("roi")
    cal_result   = results.get("calibration")

    st.markdown("---")

    # ── Quality gate ──────────────────────────────────────────────────────────
    if qg and run_qgate:
        render_quality_gate(qg)
        if qg.get("decision") == "fail":
            st.error("⛔ Image quality insufficient for reliable grading. "
                     "Please recapture the card image.")
            if not st.checkbox("Grade anyway (not recommended)"):
                render_comparison_section()
                return

    # ── Grade banner ──────────────────────────────────────────────────────────
    render_grade_banner(gr)
    render_score_metrics(gr, analysis)
    st.markdown("---")

    # ── Analysis tabs ─────────────────────────────────────────────────────────
    tab_labels = [
        "📊 Overview",
        "📸 Front vs Back",
        "🎯 Centering",
        "🔲 Edges & Corners",
        "✨ Surface",
        "🗺️ Full Overlay",
        "🧬 Defect Evidence",
        "📈 Grade Trace",
    ]
    if roi_result is not None:
        tab_labels.append("💰 ROI Analysis")
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        tab_overview(front_norm, viz, gr, analysis, detected, profile_name)
    with tabs[1]:
        tab_front_back(front_norm, back_norm, front_an, back_an, viz)
    with tabs[2]:
        tab_centering(front_norm, viz, analysis["centering"])
    with tabs[3]:
        tab_edges_corners(front_norm, viz, analysis["edges"], analysis["corners"])
    with tabs[4]:
        tab_surface(front_norm, viz, analysis["surface"])
    with tabs[5]:
        tab_full_overlay(front_norm, viz)
    with tabs[6]:
        tab_defect_evidence(defects, viz)
    with tabs[7]:
        tab_grade_trace(gr, grade_trace, viz, cal_result)
    if roi_result is not None:
        with tabs[8]:
            tab_roi(roi_result)

    # ── Add to comparison ─────────────────────────────────────────────────────
    st.markdown("---")
    ccol1, ccol2, _ = st.columns([2, 2, 4])
    with ccol1:
        card_label = st.text_input("Card label for comparison",
                                    placeholder="e.g. Charizard Holo #4",
                                    label_visibility="collapsed")
    with ccol2:
        if st.button("➕ Add to Comparison"):
            label = card_label.strip() or f"Card {len(st.session_state.comparison_cards)+1}"
            st.session_state.comparison_cards.append({
                "name":          label,
                "grade":         gr["grade"],
                "band":          gr["grade_band"],
                "rec":           gr["recommendation"],
                "centering_lr":  gr["centering_lr"],
                "corners_score": gr["corners_score"],
                "profile":       profile_name,
            })
            st.success(f"Added **{label}** to comparison.")

    # ── Download artifacts ────────────────────────────────────────────────────
    card_id = (card_label.strip().replace(" ", "_") or "card") if card_label else "card"
    dl1, dl2 = st.columns(2)

    zip_bytes = build_download_zip(card_id, front_norm, back_norm, results)
    with dl1:
        st.download_button(
            label="⬇️ Download Grading Artifacts (ZIP)",
            data=zip_bytes,
            file_name=f"{card_id}_grade_artifacts.zip",
            mime="application/zip",
            use_container_width=True,
        )

    with dl2:
        try:
            display_name = card_label.strip() if card_label and card_label.strip() else card_id
            pdf_bytes = generate_report(
                card_name=display_name,
                results=results,
                roi=roi_result,
                calibration=cal_result,
            )
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"{card_id}_grade_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as pdf_err:
            st.warning(f"PDF generation unavailable: {pdf_err}")

    # ── Comparison section ────────────────────────────────────────────────────
    render_comparison_section()

    # ── Debug ─────────────────────────────────────────────────────────────────
    if show_debug:
        with st.expander("🔧 Raw Analysis Data"):
            def _safe(d):
                return {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}
            st.json({
                "grade_result":  _safe(gr),
                "grade_trace":   grade_trace,
                "centering":     _safe(analysis["centering"]),
                "edges":         {**_safe(analysis["edges"]), "sides": analysis["edges"]["sides"]},
                "corners":       analysis["corners"],
                "surface":       _safe(analysis["surface"]),
                "defects":       defects_to_dicts(defects),
                "quality_gate":  qg,
                "profile_used":  results.get("profile_used"),
            })


if __name__ == "__main__":
    main()
