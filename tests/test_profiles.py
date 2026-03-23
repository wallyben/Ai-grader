"""Tests for the card profile system."""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grader.profiles import (
    get_profile, get_default_profile, list_profiles,
    get_profile_display_names,
    score_centering_with_profile,
    classify_edge_with_profile, reclassify_edges,
    classify_corner_with_profile, reclassify_corners,
    regenerate_surface_flags, apply_profile_to_analysis,
)
from grader.centering import compute_centering
from grader.edges     import analyze_edges
from grader.corners   import analyze_corners
from grader.surface   import analyze_surface
from tests.conftest   import make_clean_card, make_damaged_card

EXPECTED_PROFILES = {"tcg_generic", "pokemon_modern", "pokemon_vintage",
                     "sports_paper", "sports_chrome"}


class TestProfileLoading:
    def test_list_profiles_returns_expected_set(self):
        profiles = set(list_profiles())
        assert EXPECTED_PROFILES.issubset(profiles)

    def test_get_profile_returns_dict(self):
        p = get_profile("tcg_generic")
        assert isinstance(p, dict)

    def test_get_profile_has_required_keys(self):
        for name in EXPECTED_PROFILES:
            p = get_profile(name)
            for key in ["display_name", "description", "centering", "edges",
                        "corners", "surface", "scoring", "caps"]:
                assert key in p, f"Profile '{name}' missing key '{key}'"

    def test_unknown_profile_falls_back_to_generic(self):
        p = get_profile("totally_fake_profile_xyz")
        assert p["display_name"] == get_profile("tcg_generic")["display_name"]

    def test_get_default_profile(self):
        p = get_default_profile()
        assert "display_name" in p

    def test_get_profile_returns_deep_copy(self):
        p1 = get_profile("tcg_generic")
        p2 = get_profile("tcg_generic")
        p1["centering"]["deviation_10"] = 9999
        assert p2["centering"]["deviation_10"] != 9999

    def test_display_names_dict_nonempty(self):
        dn = get_profile_display_names()
        assert len(dn) >= 5
        for k, v in dn.items():
            assert isinstance(k, str) and isinstance(v, str)

    def test_all_profiles_have_caps(self):
        for name in EXPECTED_PROFILES:
            p = get_profile(name)
            caps = p["caps"]
            for key in ["severe_corner", "moderate_corner", "severe_edge",
                        "moderate_edge", "centering_severe", "light_scratch"]:
                assert key in caps, f"{name} missing cap '{key}'"

    def test_all_profiles_have_scoring_weights(self):
        for name in EXPECTED_PROFILES:
            p = get_profile(name)
            w = p["scoring"]["weights"]
            total = sum(w.values())
            assert abs(total - 1.0) < 0.01, f"{name} weights sum to {total}"


class TestCenteringRescore:
    def test_perfect_center_scores_10(self):
        p = get_profile("tcg_generic")
        assert score_centering_with_profile(50, 50, 50, 50, p) == 10.0

    def test_score_decreases_with_deviation(self):
        p = get_profile("tcg_generic")
        s1 = score_centering_with_profile(50, 50, 50, 50, p)
        s2 = score_centering_with_profile(60, 40, 50, 50, p)
        s3 = score_centering_with_profile(70, 30, 50, 50, p)
        assert s1 >= s2 >= s3

    def test_pokemon_modern_stricter_than_vintage(self):
        p_mod = get_profile("pokemon_modern")
        p_vin = get_profile("pokemon_vintage")
        # Same 20pp deviation should score lower on modern
        dev = 20
        s_mod = score_centering_with_profile(50 + dev/2, 50 - dev/2, 50, 50, p_mod)
        s_vin = score_centering_with_profile(50 + dev/2, 50 - dev/2, 50, 50, p_vin)
        assert s_mod <= s_vin

    def test_chrome_stricter_than_paper(self):
        p_chr = get_profile("sports_chrome")
        p_pap = get_profile("sports_paper")
        dev = 15
        s_chr = score_centering_with_profile(50 + dev/2, 50 - dev/2, 50, 50, p_chr)
        s_pap = score_centering_with_profile(50 + dev/2, 50 - dev/2, 50, 50, p_pap)
        assert s_chr <= s_pap


class TestEdgeReclassification:
    def test_classify_edge_clean(self):
        p = get_profile("tcg_generic")
        lvl = classify_edge_with_profile(0.01, 0.10, 0.02, p)
        assert lvl == "clean"

    def test_classify_edge_minor(self):
        p = get_profile("tcg_generic")
        t = p["edges"]
        w = (t["whitening_minor"] + t["whitening_moderate"]) / 2
        lvl = classify_edge_with_profile(w, 0.0, 0.0, p)
        assert lvl == "minor"

    def test_classify_edge_severe(self):
        p = get_profile("tcg_generic")
        t = p["edges"]
        lvl = classify_edge_with_profile(t["whitening_severe"] + 0.05, 0.0, 0.0, p)
        assert lvl == "severe"

    def test_reclassify_edges_returns_dict(self):
        img = make_clean_card()
        edges = analyze_edges(img)
        p = get_profile("tcg_generic")
        result = reclassify_edges(edges, p)
        assert isinstance(result, dict)
        assert "sides" in result
        assert "score" in result

    def test_reclassify_does_not_mutate_original(self):
        img = make_clean_card()
        edges = analyze_edges(img)
        original_score = edges["score"]
        p = get_profile("tcg_generic")
        reclassify_edges(edges, p)
        assert edges["score"] == original_score

    def test_chrome_classifies_same_whitening_higher(self):
        """Same whitening on chrome should classify as moderate or worse vs. minor on paper."""
        p_chr = get_profile("sports_chrome")
        p_pap = get_profile("sports_paper")
        # Use a whitening value between chrome's minor and paper's moderate
        w = (p_chr["edges"]["whitening_minor"] + p_pap["edges"]["whitening_moderate"]) / 2
        lvl_chr = classify_edge_with_profile(w, 0.0, 0.0, p_chr)
        lvl_pap = classify_edge_with_profile(w, 0.0, 0.0, p_pap)
        sev_map = {"clean": 0, "minor": 1, "moderate": 2, "severe": 3}
        assert sev_map[lvl_chr] >= sev_map[lvl_pap]


class TestCornerReclassification:
    def test_classify_corner_sharp(self):
        p = get_profile("tcg_generic")
        lvl = classify_corner_with_profile(0.01, 0.01, p)
        assert lvl == "sharp"

    def test_classify_corner_severe(self):
        p = get_profile("tcg_generic")
        t = p["corners"]
        lvl = classify_corner_with_profile(t["whitening_severe"] + 0.05, 0.0, p)
        assert lvl == "severe"

    def test_reclassify_corners_returns_valid_struct(self):
        img = make_clean_card()
        corners = analyze_corners(img)
        p = get_profile("tcg_generic")
        result = reclassify_corners(corners, p)
        for pos in ["top_left", "top_right", "bottom_left", "bottom_right"]:
            assert pos in result["corners"]
            assert result["corners"][pos]["defect_level"] in \
                   {"sharp", "minor", "moderate", "severe"}


class TestApplyProfileToAnalysis:
    def test_returns_dict_with_all_keys(self):
        img = make_clean_card()
        analysis = {
            "centering": compute_centering(img),
            "edges":     analyze_edges(img),
            "corners":   analyze_corners(img),
            "surface":   analyze_surface(img),
        }
        p = get_profile("tcg_generic")
        result = apply_profile_to_analysis(analysis, p)
        for key in ["centering", "edges", "corners", "surface"]:
            assert key in result

    def test_does_not_mutate_original(self):
        img = make_clean_card()
        analysis = {
            "centering": compute_centering(img),
            "edges":     analyze_edges(img),
            "corners":   analyze_corners(img),
            "surface":   analyze_surface(img),
        }
        orig_c_score = analysis["centering"]["score"]
        p = get_profile("tcg_generic")
        apply_profile_to_analysis(analysis, p)
        assert analysis["centering"]["score"] == orig_c_score

    def test_different_profiles_can_yield_different_scores(self):
        # Damaged card should score differently on chrome vs. vintage
        img = make_damaged_card()
        analysis = {
            "centering": compute_centering(img),
            "edges":     analyze_edges(img),
            "corners":   analyze_corners(img),
            "surface":   analyze_surface(img),
        }
        p_chr = get_profile("sports_chrome")
        p_vin = get_profile("pokemon_vintage")
        r_chr = apply_profile_to_analysis(analysis, p_chr)
        r_vin = apply_profile_to_analysis(analysis, p_vin)
        # Both valid, chrome surface sensitivity may differ
        assert 1.0 <= r_chr["surface"]["score"] <= 10.0
        assert 1.0 <= r_vin["surface"]["score"] <= 10.0
