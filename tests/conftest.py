"""
Shared test fixtures and synthetic card image generators.
All images are generated deterministically using a fixed seed.
"""

import numpy as np
import cv2
import pytest
import sys
import os

# Ensure the repo root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CARD_W = 500
CARD_H = 700


def make_clean_card(seed: int = 42) -> np.ndarray:
    """
    Synthetic clean card: light grey face, black inner border, no defects.
    Deterministic given the same seed.
    """
    rng = np.random.default_rng(seed)
    img = np.full((CARD_H, CARD_W, 3), 195, dtype=np.uint8)

    # Slight colour texture to simulate print
    noise = rng.integers(0, 12, (CARD_H, CARD_W, 3), dtype=np.uint8)
    img = cv2.add(img, noise)

    # Inner border (centred, simulating card border)
    margin = 38
    cv2.rectangle(img,
                  (margin, margin),
                  (CARD_W - margin, CARD_H - margin),
                  (20, 20, 20), 3)

    return img


def make_damaged_card(seed: int = 42) -> np.ndarray:
    """
    Synthetic card with explicit defects:
    - Corner whitening (top-left, bottom-right)
    - Edge chipping (top edge)
    - Scratch line across face
    - Off-centre border
    """
    img = make_clean_card(seed)

    # Corner whitening — bleach top-left corner
    img[:35, :35] = 245

    # Corner whitening — bleach bottom-right
    img[CARD_H - 35:, CARD_W - 35:] = 242

    # Edge chipping — bright spots along top edge
    img[:12, 80:110] = 250
    img[:12, 300:330] = 248

    # Scratch — thin bright line
    cv2.line(img, (80, 200), (380, 420), (230, 230, 230), 2)

    # Additional scratch
    cv2.line(img, (150, 150), (200, 550), (225, 225, 225), 1)

    return img


@pytest.fixture
def clean_card():
    return make_clean_card()


@pytest.fixture
def damaged_card():
    return make_damaged_card()
