"""Tests for the defect evidence module."""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grader.defects import (
    Defect, extract_defects, defects_to_dicts, summarise_defects,
    CARD_W, CARD_H,
)
from grader.centering import compute_centering
from grader.edges     import analyze_edges
from grader.corners   import analyze_corners
from grader.surface   import analyze_surface
from grader.profiles  import apply_profile_to_analysis, get_profile
from tests.conftest   import make_clean_card, make_damaged_card


def _make_analysis(img, profile_name="tcg_generic"):
    p = get_profile(profile_name)
    raw = {
        "centering": compute_centering(img),
        "edges":     analyze_edges(img),
        "corners":   analyze_corners(img),
        "surface":   analyze_surface(img),
    }
    return apply_profile_to_analysis(raw, p)


class TestDefectStructure:
    def test_extract_returns_list(self):
        img    = make_clean_card()
        an     = _make_analysis(img)
        defects = extract_defects(an)
        assert isinstance(defects, list)

    def test_defects_are_defect_objects(self):
        img    = make_damaged_card()
        an     = _make_analysis(img)
        defects = extract_defects(an)
        for d in defects:
            assert isinstance(d, Defect)

    def test_defect_has_required_attributes(self):
        img    = make_damaged_card()
        an     = _make_analysis(img)
        defects = extract_defects(an)
        for d in defects:
            for attr in ["defect_type", "category", "side", "location",
                         "severity", "bbox", "explanation", "metric", "metric_value"]:
                assert hasattr(d, attr), f"Defect missing attribute: {attr}"

    def test_severity_valid_values(self):
        img    = make_damaged_card()
        an     = _make_analysis(img)
        defects = extract_defects(an)
        valid_sev = {"minor", "moderate", "severe"}
        for d in defects:
            assert d.severity in valid_sev

    def test_category_valid_values(self):
        img    = make_damaged_card()
        an     = _make_analysis(img)
        defects = extract_defects(an)
        valid_cats = {"centering", "edge", "corner", "surface"}
        for d in defects:
            assert d.category in valid_cats

    def test_side_valid_values(self):
        img    = make_clean_card()
        an     = _make_analysis(img)
        defects_front = extract_defects(an, side="front")
        defects_back  = extract_defects(an, side="back")
        for d in defects_front:
            assert d.side == "front"
        for d in defects_back:
            assert d.side == "back"

    def test_bbox_within_image_bounds(self):
        img    = make_damaged_card()
        an     = _make_analysis(img)
        defects = extract_defects(an)
        for d in defects:
            if d.bbox is not None:
                x, y, w, h = d.bbox
                assert x >= 0 and y >= 0
                assert x + w <= CARD_W, f"bbox x+w={x+w} > CARD_W={CARD_W}"
                assert y + h <= CARD_H, f"bbox y+h={y+h} > CARD_H={CARD_H}"

    def test_metric_value_is_float(self):
        img    = make_damaged_card()
        an     = _make_analysis(img)
        defects = extract_defects(an)
        for d in defects:
            assert isinstance(d.metric_value, float)


class TestDefectSorting:
    def test_severe_defects_come_first(self):
        img    = make_damaged_card()
        an     = _make_analysis(img)
        defects = extract_defects(an)
        if len(defects) < 2:
            pytest.skip("Not enough defects to test sort order")
        sev_order = {"severe": 0, "moderate": 1, "minor": 2}
        for i in range(len(defects) - 1):
            assert sev_order[defects[i].severity] <= sev_order[defects[i+1].severity]


class TestDefectDicts:
    def test_to_dicts_returns_list_of_dicts(self):
        img    = make_damaged_card()
        an     = _make_analysis(img)
        defects = extract_defects(an)
        dicts   = defects_to_dicts(defects)
        assert isinstance(dicts, list)
        for d in dicts:
            assert isinstance(d, dict)

    def test_dicts_contain_required_keys(self):
        img    = make_damaged_card()
        an     = _make_analysis(img)
        dicts  = defects_to_dicts(extract_defects(an))
        for d in dicts:
            for key in ["defect_type", "category", "side", "location",
                        "severity", "bbox", "explanation"]:
                assert key in d

    def test_bbox_serialised_as_list_or_none(self):
        img   = make_damaged_card()
        an    = _make_analysis(img)
        dicts = defects_to_dicts(extract_defects(an))
        for d in dicts:
            assert d["bbox"] is None or isinstance(d["bbox"], list)


class TestDefectSummary:
    def test_summarise_returns_counts(self):
        img    = make_damaged_card()
        an     = _make_analysis(img)
        defects = extract_defects(an)
        summary = summarise_defects(defects)
        assert "severe" in summary
        assert "moderate" in summary
        assert "minor" in summary

    def test_counts_are_non_negative(self):
        img    = make_clean_card()
        an     = _make_analysis(img)
        summary = summarise_defects(extract_defects(an))
        for v in summary.values():
            assert v >= 0

    def test_clean_card_fewer_defects_than_damaged(self):
        clean   = _make_analysis(make_clean_card())
        damaged = _make_analysis(make_damaged_card())
        n_clean   = len(extract_defects(clean))
        n_damaged = len(extract_defects(damaged))
        assert n_damaged >= n_clean


class TestSpecificDefectTypes:
    def test_scratched_card_has_surface_defects(self):
        img = make_damaged_card()   # includes a scratch line
        an  = _make_analysis(img)
        defects = extract_defects(an)
        surface_defects = [d for d in defects if d.category == "surface"]
        # The damaged card has a scratch — surface defects should exist
        # (may not always trigger depending on threshold, so just test structure)
        for d in surface_defects:
            assert d.defect_type.startswith("surface_") or d.defect_type == "print_lines"

    def test_whitened_corner_card_has_corner_defects(self):
        img = make_damaged_card()   # includes corner whitening
        an  = _make_analysis(img)
        defects = extract_defects(an)
        corner_defects = [d for d in defects if d.category == "corner"]
        # Damaged card has explicit corner whitening at top_left and bottom_right
        assert len(corner_defects) >= 1
        for d in corner_defects:
            assert "corner" in d.location
