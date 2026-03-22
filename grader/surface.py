"""
Surface Analysis Engine
Detects scratches, print lines, dents, and surface anomalies using:
  - Morphological BLACKHAT operations (scratch detection)
  - FFT frequency analysis (print lines)
  - Laplacian variance (surface noise / dent approximation)
  - Local deviation mapping (reflection inconsistencies)
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, List


def _get_inner_region(img: np.ndarray, margin_pct: float = 0.08) -> Tuple[np.ndarray, int, int]:
    """
    Strip the outer margin to focus analysis on the card face (artwork),
    avoiding edge/corner overlap with other engines.
    Returns (inner_img, y_offset, x_offset).
    """
    h, w = img.shape[:2]
    margin_y = int(h * margin_pct)
    margin_x = int(w * margin_pct)
    inner = img[margin_y:h - margin_y, margin_x:w - margin_x]
    return inner, margin_y, margin_x


def detect_scratches(inner: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Detect linear scratches using morphological BLACKHAT operations.
    BLACKHAT reveals dark structures finer than the structuring element
    on a bright background (and inverse via complement for light scratches).

    Returns (density [0,1], binary scratch map).
    """
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)

    # Directional kernels for H/V scratches
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    # Diagonal approximation with elongated cross kernel
    k_d1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (21, 21))

    blackhat_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_h)
    blackhat_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_v)

    # Also catch light scratches on dark backgrounds
    tophat_h = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_h)
    tophat_v = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_v)

    # Combine all orientations
    combined = cv2.add(blackhat_h, blackhat_v)
    combined = cv2.add(combined, tophat_h)
    combined = cv2.add(combined, tophat_v)

    # Threshold to binary scratch map
    _, scratch_binary = cv2.threshold(combined, 12, 255, cv2.THRESH_BINARY)

    # Remove isolated noise pixels
    kernel_clean = np.ones((2, 2), np.uint8)
    scratch_binary = cv2.morphologyEx(scratch_binary, cv2.MORPH_OPEN, kernel_clean)

    density = float(np.sum(scratch_binary > 0)) / max(scratch_binary.size, 1)
    return density, scratch_binary


def detect_print_lines(inner: np.ndarray) -> float:
    """
    Detect print defects (lines, banding) via FFT frequency analysis.
    Print lines manifest as strong periodic energy along horizontal/vertical
    frequency bands in the Fourier spectrum.
    Returns indicator in [0, 1].
    """
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Zero-mean the image to reduce DC component dominance
    gray -= gray.mean()

    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    rows, cols = magnitude.shape
    cy, cx = rows // 2, cols // 2

    # Mask DC (center region)
    dc_radius = max(8, int(min(rows, cols) * 0.04))
    y_idx, x_idx = np.ogrid[:rows, :cols]
    dc_mask = ((x_idx - cx) ** 2 + (y_idx - cy) ** 2) > dc_radius ** 2

    # Measure energy in strict horizontal/vertical bands (print line signature)
    band_hw = 4
    h_band = magnitude[cy - band_hw:cy + band_hw, :] * dc_mask[cy - band_hw:cy + band_hw, :]
    v_band = magnitude[:, cx - band_hw:cx + band_hw] * dc_mask[:, cx - band_hw:cx + band_hw]

    h_energy = float(np.mean(h_band[h_band > 0])) if np.any(h_band > 0) else 0.0
    v_energy = float(np.mean(v_band[v_band > 0])) if np.any(v_band > 0) else 0.0
    background_energy = float(np.mean(magnitude[dc_mask])) if np.any(dc_mask) else 1.0

    if background_energy == 0:
        return 0.0

    ratio = max(h_energy, v_energy) / (background_energy + 1e-6)
    # Normalize: typical range 1–8 for print lines
    return float(min(1.0, max(0.0, (ratio - 1.5) / 7.0)))


def detect_surface_anomalies(inner: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Detect dents, creases, and surface irregularities via:
    1. Local Laplacian spike detection
    2. Local mean deviation from smoothed surface

    Returns (anomaly density [0,1], anomaly mask).
    """
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Method 1: Laplacian spikes
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)

    # Method 2: Deviation from a heavily smoothed (ideal) surface
    smooth = cv2.GaussianBlur(gray, (31, 31), 0)
    deviation = np.abs(gray - smooth)

    # Combine
    combined_score = (lap_abs / (lap_abs.max() + 1e-6) * 0.5 +
                      deviation / (deviation.max() + 1e-6) * 0.5)

    # Use 93rd percentile threshold to isolate real anomalies
    threshold = np.percentile(combined_score, 93)
    anomaly_mask = (combined_score > threshold).astype(np.uint8) * 255

    # Clean noise
    kernel = np.ones((4, 4), np.uint8)
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)

    density = float(np.sum(anomaly_mask > 0)) / max(anomaly_mask.size, 1)
    return density, anomaly_mask


def compute_surface_noise(inner: np.ndarray) -> float:
    """
    Overall surface quality via Laplacian variance.
    High variance can indicate rough/damaged surface.
    Returns normalized value [0, 1].
    """
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Scale: <500 = smooth, 500–3000 = moderate, >3000 = noisy/damaged
    return float(min(1.0, lap_var / 4000.0))


def _generate_risk_flags(scratch_density: float,
                          print_line_indicator: float,
                          anomaly_density: float,
                          surface_noise: float) -> List[str]:
    flags = []
    if scratch_density > 0.015:
        severity = "heavy" if scratch_density > 0.04 else "light"
        flags.append(f"{severity} scratches detected")
    if print_line_indicator > 0.45:
        flags.append("print lines / manufacturing defect detected")
    if anomaly_density > 0.06:
        flags.append("possible dent or crease detected")
    if surface_noise > 0.65:
        flags.append("high surface noise — possible damage")
    return flags


def _score_surface(scratch_density: float,
                   print_line_indicator: float,
                   anomaly_density: float,
                   surface_noise: float) -> float:
    """Deterministic surface score 1–10."""
    penalty = 0.0
    penalty += scratch_density * 220.0       # Scratches: heavy penalty
    penalty += print_line_indicator * 12.0   # Print lines: moderate
    penalty += anomaly_density * 90.0        # Anomalies: moderate-heavy
    penalty += max(0.0, surface_noise - 0.5) * 8.0  # Noise only above baseline

    score = 10.0 - penalty
    return float(max(1.0, min(10.0, round(score, 1))))


def analyze_surface(img: np.ndarray) -> Dict[str, Any]:
    """
    Full surface analysis pipeline.

    Returns:
        score: surface quality score 1–10
        scratch_density, print_line_indicator, anomaly_density: raw metrics
        surface_noise: overall noise level
        scratch_map: binary map (full image size, uint8)
        anomaly_mask: binary map (full image size, uint8)
        risk_flags: list of human-readable warnings
    """
    h, w = img.shape[:2]
    inner, y_off, x_off = _get_inner_region(img, margin_pct=0.08)

    scratch_density, scratch_map_inner = detect_scratches(inner)
    print_line_indicator = detect_print_lines(inner)
    anomaly_density, anomaly_mask_inner = detect_surface_anomalies(inner)
    surface_noise = compute_surface_noise(inner)

    # Embed inner maps back into full-size canvases for visualization
    scratch_map_full = np.zeros((h, w), dtype=np.uint8)
    scratch_map_full[y_off:h - y_off, x_off:w - x_off] = scratch_map_inner

    anomaly_mask_full = np.zeros((h, w), dtype=np.uint8)
    anomaly_mask_full[y_off:h - y_off, x_off:w - x_off] = anomaly_mask_inner

    score = _score_surface(scratch_density, print_line_indicator,
                            anomaly_density, surface_noise)
    risk_flags = _generate_risk_flags(scratch_density, print_line_indicator,
                                       anomaly_density, surface_noise)

    return {
        "score": score,
        "scratch_density": round(float(scratch_density), 5),
        "print_line_indicator": round(float(print_line_indicator), 5),
        "anomaly_density": round(float(anomaly_density), 5),
        "surface_noise": round(float(surface_noise), 5),
        "scratch_map": scratch_map_full,
        "anomaly_mask": anomaly_mask_full,
        "risk_flags": risk_flags,
    }
