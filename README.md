# 🃏 AI Card Grader

> Production-ready AI-assisted trading card grading system.
> PSA-style analysis. Fully local. Zero paid APIs.

---

## Features

| Feature | Description |
|---|---|
| **Card Normalization** | Auto-detects card boundaries, perspective-corrects, resizes to 500×700 |
| **Centering Engine** | Measures inner border ratios (L/R, T/B) to 0.1% precision |
| **Edge Analysis** | Detects whitening, chipping, roughness on all 4 sides |
| **Corner Analysis** | Measures rounding, whitening, and sharpness at all 4 corners |
| **Surface Analysis** | FFT print-line detection, morphological scratch detection, anomaly heatmaps |
| **Deterministic Grading** | Rule-based hard caps + weighted scoring (no randomness) |
| **Visual Overlays** | Colour-coded defect maps on every analysis layer |
| **Card Profiles** | Pokémon Modern/Vintage, Sports Chrome/Paper, TCG Generic — per-type thresholds |
| **Front + Back Grading** | Upload both sides for combined worst-of scoring |
| **Defect Evidence** | Structured defect list with bounding-box overlays and severity |
| **Grade Trace** | Full audit trail from sub-scores through cap evaluation to final grade |
| **Quality Gate** | Image quality check before grading (blur, brightness, contrast, size) |
| **PSA Calibration** | Piecewise linear bias correction aligning system grades to real PSA outcomes |
| **ROI Analysis** | Expected value, profit, ROI%, and 5-tier grading decision (STRONG GRADE → DO NOT GRADE) |
| **PDF Report** | Professional per-card grading report with images, scores, ROI, and defect list |
| **Batch CLI** | `python batch_grade.py cards/` — grade a folder of cards to CSV |
| **Streamlit UI** | Full interactive web interface with comparison mode and artifact download |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the UI

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

### 3. Run tests

```bash
pytest tests/ -v
```

---

## Grading Logic

### Score Weights

| Category | Weight |
|---|---|
| Corners | 30% |
| Edges | 25% |
| Surface | 25% |
| Centering | 20% |

### Hard Caps (PSA-inspired)

| Defect | Grade Cap |
|---|---|
| Severe corner damage | ≤ 5 |
| Moderate corner wear | ≤ 7 |
| Severe edge damage | ≤ 6 |
| Moderate edge whitening | ≤ 8 |
| Centering > 65/35 | ≤ 8 |
| Centering > 75/25 | ≤ 6 |
| Heavy scratches | ≤ 6 |
| Light scratches | ≤ 8 |
| Surface dent/crease | ≤ 6 |

### Centering Standards

| PSA Grade | Centering Threshold |
|---|---|
| PSA 10 | 55/45 or better |
| PSA 9  | 60/40 or better |
| PSA 8  | 65/35 or better |
| PSA 7  | 70/30 or better |

### Recommendation Engine

Basic recommendation (no pricing):

| Grade | Decision |
|---|---|
| ≥ 9.0 | **GRADE** — strong ROI potential |
| 8.0–8.5 | **CONDITIONAL** — grade if high-demand card |
| 7.0–7.5 | **CONDITIONAL** — borderline ROI |
| ≤ 6.5 | **DO NOT GRADE** — grading cost > slab premium |

ROI-based decision (with pricing, Phase 2):

| Decision | Criteria |
|---|---|
| **STRONG GRADE** | ROI > 150% and profit > $60 and low downside risk |
| **GRADE** | ROI > 40% and low downside risk |
| **CONDITIONAL** | ROI > 5% or positive profit on grade ≥ 7 |
| **HOLD RAW** | Positive profit but doesn't meet grading thresholds |
| **DO NOT GRADE** | Negative expected profit |

---

## Project Structure

```
/
├── app.py                    # Streamlit UI entry point
├── batch_grade.py            # CLI batch grading tool
├── requirements.txt
├── README.md
├── grader/
│   ├── __init__.py           # grade_card_full() pipeline + public API
│   ├── preprocessing.py      # Boundary detection + perspective warp
│   ├── centering.py          # Inner border detection + ratio scoring
│   ├── edges.py              # Edge whitening / chipping / roughness
│   ├── corners.py            # Corner wear / rounding / whitening
│   ├── surface.py            # Scratches / print lines / anomalies
│   ├── scoring.py            # Deterministic grading engine + caps
│   ├── profiles.py           # Card type profiles (Pokémon, Sports, etc.)
│   ├── combined_scoring.py   # Front + back worst-of merge
│   ├── quality_gate.py       # Image quality assessment
│   ├── defects.py            # Structured defect extraction
│   ├── artifacts.py          # ZIP export + CSV batch output
│   ├── visualize.py          # Overlay drawing + summary card
│   ├── calibration.py        # PSA alignment engine (Phase 2)
│   ├── pricing.py            # CardPricing / CardPricingBundle (Phase 2)
│   ├── roi.py                # ROI engine + 5-tier decision (Phase 2)
│   └── report.py             # PDF report generator (Phase 2)
└── tests/                    # 314 tests, 0 failures
    ├── conftest.py
    ├── test_smoke.py
    ├── test_grading.py
    ├── test_combined.py
    ├── test_defects.py
    ├── test_profiles.py
    ├── test_quality_gate.py
    ├── test_batch.py
    ├── test_calibration.py   # Phase 2
    ├── test_roi.py           # Phase 2
    ├── test_decision_engine.py # Phase 2
    └── test_batch_roi.py     # Phase 2
```

---

## Example Output

```json
{
  "grade": 8.5,
  "grade_band": "NM-MT (PSA 8)",
  "centering_score": 9.0,
  "edges_score": 8.5,
  "corners_score": 7.5,
  "surface_score": 9.0,
  "centering_lr": "53/47",
  "centering_tb": "51/49",
  "caps_triggered": ["minor corner wear — top left"],
  "risk_flags": ["minor corner wear — top left"],
  "confidence": 0.87,
  "recommendation": "CONDITIONAL",
  "rec_reason": "Estimated grade 8.5. Consider grading only for high-demand cards..."
}
```

---

## How It Works

### Image Normalization
1. Bilateral filter + Canny edge detection to find card boundary
2. Adaptive threshold fallback for tricky backgrounds
3. 4-point perspective transform to remove skew
4. Resize to standard 500×700 px

### Centering
- Scans top/bottom 40% zones for horizontal border lines
- Scans left/right 40% zones for vertical border lines
- Multi-threshold Canny edge map for robustness
- Margin ratios → deviation from 50/50 → PSA-standard scoring

### Edges
- Extracts 18px strip from each side
- LAB colorspace whitening detection (L\* > 200)
- Edge line profile variance → roughness metric
- Cluster-aware chip detection on extreme edge pixels

### Corners
- Extracts 65×65 px from each corner
- 22px apex zone for tip-focused analysis
- OTSU binarization → missing-corner rounding detection
- Sobel gradient magnitude → sharpness score

### Surface
- Morphological BLACKHAT (H+V) scratch detection
- FFT frequency band analysis for print-line detection
- Laplacian + deviation-from-smooth → anomaly/dent heatmap
- Worst-4 scoring with heavy scratch penalty

---

## Upgrade Roadmap

✅ = completed

1. ✅ **Batch Mode** — `python batch_grade.py cards/` to process folders
2. ✅ **PDF Report Export** — professional per-card grading reports via ReportLab
3. ✅ **PSA Calibration** — piecewise linear bias correction using real-world alignment data
4. ✅ **ROI Analysis** — expected value, profit, and 5-tier grading decision engine
5. **ML Corner Classifier** — train a lightweight CNN on labelled corner crops (sharp / minor / moderate / severe)
6. **Gloss/Holo Detection** — separate analysis path for foil/holo cards with different surface reflection
7. **Card Set Recognition** — OCR + template matching to identify card set and pull live market prices
8. **Multi-card Detection** — detect and grade multiple cards in a single photo

---

## Notes

- All analysis is purely local — no network calls, no cloud APIs
- Results are 100% deterministic for identical input images
- The grading engine is rule-based and fully auditable
- Designed for real-world use, not just demos
