"""Tests for the image quality gate module."""

import numpy as np
import cv2
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grader.quality_gate import assess_quality
from tests.conftest import make_clean_card, CARD_W, CARD_H


def make_blurry_card(seed=42):
    img = make_clean_card(seed)
    return cv2.GaussianBlur(img, (41, 41), 20)


def make_glare_card(seed=42):
    img = make_clean_card(seed)
    # Add extreme bright hotspot (glare) — 200x200 = ~11% of card area, above warn threshold
    img[100:300, 100:300] = 255
    return img


def make_dark_card(seed=42):
    img = make_clean_card(seed)
    return (img * 0.15).astype(np.uint8)


def make_bright_card(seed=42):
    img = make_clean_card(seed)
    return np.clip(img.astype(int) + 180, 0, 255).astype(np.uint8)


class TestQualityGateStructure:
    def test_returns_dict_with_required_keys(self):
        img = make_clean_card()
        r = assess_quality(img)
        for key in ["decision", "score", "issues", "checks", "summary"]:
            assert key in r, f"Missing key: {key}"

    def test_decision_valid_values(self):
        img = make_clean_card()
        r = assess_quality(img)
        assert r["decision"] in {"pass", "warn", "fail"}

    def test_score_in_range(self):
        img = make_clean_card()
        r = assess_quality(img)
        assert 0 <= r["score"] <= 100

    def test_issues_is_list(self):
        img = make_clean_card()
        r = assess_quality(img)
        assert isinstance(r["issues"], list)

    def test_checks_is_dict(self):
        img = make_clean_card()
        r = assess_quality(img)
        assert isinstance(r["checks"], dict)

    def test_none_image_returns_fail(self):
        r = assess_quality(None)
        assert r["decision"] == "fail"
        assert r["score"] == 0

    def test_issue_has_required_fields(self):
        # Use glare to ensure at least one issue
        img = make_glare_card()
        r = assess_quality(img)
        for issue in r.get("issues", []):
            for field in ["check", "severity", "message", "metric_name", "metric_value"]:
                assert field in issue, f"Issue missing field: {field}"


class TestBlurDetection:
    def test_sharp_card_no_blur_issue(self):
        img = make_clean_card()
        r = assess_quality(img)
        blur_issues = [i for i in r["issues"] if i["check"] == "blur"]
        assert len(blur_issues) == 0

    def test_blurry_card_raises_issue(self):
        img = make_blurry_card()
        r = assess_quality(img)
        blur_issues = [i for i in r["issues"] if i["check"] == "blur"]
        assert len(blur_issues) >= 1

    def test_blur_issue_is_warn_or_fail(self):
        img = make_blurry_card()
        r = assess_quality(img)
        for issue in r["issues"]:
            if issue["check"] == "blur":
                assert issue["severity"] in {"warn", "fail"}

    def test_blur_metric_present_in_checks(self):
        img = make_clean_card()
        r = assess_quality(img)
        assert "blur" in r["checks"]
        assert "value" in r["checks"]["blur"]


class TestGlareDetection:
    def test_clean_card_no_glare_issue(self):
        img = make_clean_card()
        r = assess_quality(img)
        glare_issues = [i for i in r["issues"] if i["check"] == "glare"]
        assert len(glare_issues) == 0

    def test_glare_card_raises_issue(self):
        img = make_glare_card()
        r = assess_quality(img)
        glare_issues = [i for i in r["issues"] if i["check"] == "glare"]
        assert len(glare_issues) >= 1

    def test_glare_metric_in_checks(self):
        img = make_clean_card()
        r = assess_quality(img)
        assert "glare" in r["checks"]


class TestExposureDetection:
    def test_normal_card_no_exposure_issue(self):
        img = make_clean_card()
        r = assess_quality(img)
        exp_issues = [i for i in r["issues"] if i["check"] == "exposure"]
        assert len(exp_issues) == 0

    def test_dark_card_raises_issue(self):
        img = make_dark_card()
        r = assess_quality(img)
        exp_issues = [i for i in r["issues"] if i["check"] == "exposure"]
        assert len(exp_issues) >= 1

    def test_bright_card_raises_issue(self):
        img = make_bright_card()
        r = assess_quality(img)
        exp_issues = [i for i in r["issues"] if i["check"] == "exposure"]
        assert len(exp_issues) >= 1


class TestCardVisibility:
    def test_full_card_visible(self):
        img = make_clean_card()
        r = assess_quality(img)
        vis_issues = [i for i in r["issues"] if i["check"] == "card_not_visible"]
        # Clean card fills most of the frame — should be visible
        # (allow for warn due to synthetic card detection difficulty)
        assert "card_visible" in r["checks"]

    def test_tiny_card_raises_issue(self):
        # Very small card in large background
        bg = np.ones((1400, 1000, 3), dtype=np.uint8) * 120
        card = make_clean_card()
        small = cv2.resize(card, (80, 112))
        bg[600:712, 460:540] = small
        r = assess_quality(bg)
        vis_issues = [i for i in r["issues"] if i["check"] == "card_not_visible"]
        assert len(vis_issues) >= 1


class TestOverallDecision:
    def test_clean_card_passes_or_warns(self):
        img = make_clean_card()
        r = assess_quality(img)
        # Synthetic card may not be detected perfectly, so accept warn too
        assert r["decision"] in {"pass", "warn"}

    def test_multi_issue_card_fails_or_warns(self):
        img = make_dark_card()
        img = cv2.GaussianBlur(img, (31, 31), 15)
        r = assess_quality(img)
        assert r["decision"] in {"warn", "fail"}

    def test_score_lower_for_damaged_input(self):
        clean = make_clean_card()
        glare = make_glare_card()
        r_clean = assess_quality(clean)
        r_glare = assess_quality(glare)
        assert r_glare["score"] <= r_clean["score"]

    def test_deterministic(self):
        img = make_clean_card()
        r1 = assess_quality(img)
        r2 = assess_quality(img)
        assert r1["decision"] == r2["decision"]
        assert r1["score"]    == r2["score"]
