"""Tests for batch grading utilities and artifact export."""

import csv
import io
import json
import os
import sys
import tempfile
import zipfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch_grade import _find_images, _find_front_back_pairs, _grade_one, _print_ranking
from grader.artifacts import (
    build_download_zip,
    get_csv_row,
    save_batch_csv,
    save_grading_artifacts,
    CSV_FIELDNAMES,
)
from grader import grade_card_full
from tests.conftest import make_clean_card, make_damaged_card

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _write_image(directory, name, img):
    path = os.path.join(directory, name)
    cv2.imwrite(path, img)
    return path


def _make_results(front=None, back=None):
    if front is None:
        front = make_clean_card()
    return grade_card_full(front, back, profile_name="tcg_generic",
                            run_quality_gate=False)


# ─── _find_images ─────────────────────────────────────────────────────────────

class TestFindImages:
    def test_returns_sorted_list(self, tmp_path):
        img = make_clean_card()
        _write_image(str(tmp_path), "card_b.jpg", img)
        _write_image(str(tmp_path), "card_a.jpg", img)
        found = _find_images(str(tmp_path))
        assert found == sorted(found)

    def test_finds_jpg_and_png(self, tmp_path):
        img = make_clean_card()
        _write_image(str(tmp_path), "card1.jpg", img)
        cv2.imwrite(str(tmp_path / "card2.png"), img)
        found = _find_images(str(tmp_path))
        assert len(found) == 2

    def test_ignores_non_image_files(self, tmp_path):
        img = make_clean_card()
        _write_image(str(tmp_path), "card.jpg", img)
        (tmp_path / "notes.txt").write_text("ignore me")
        found = _find_images(str(tmp_path))
        assert len(found) == 1
        assert found[0].endswith(".jpg")

    def test_empty_directory_returns_empty(self, tmp_path):
        assert _find_images(str(tmp_path)) == []

    def test_no_duplicates(self, tmp_path):
        img = make_clean_card()
        _write_image(str(tmp_path), "card.jpg", img)
        found = _find_images(str(tmp_path))
        assert len(found) == len(set(found))


# ─── _find_front_back_pairs ───────────────────────────────────────────────────

class TestFindFrontBackPairs:
    def test_detects_front_back_pair(self, tmp_path):
        img = make_clean_card()
        f = _write_image(str(tmp_path), "pikachu_front.jpg", img)
        b = _write_image(str(tmp_path), "pikachu_back.jpg", img)
        pairs = _find_front_back_pairs([f, b])
        assert "pikachu" in pairs
        assert pairs["pikachu"][0] == f
        assert pairs["pikachu"][1] == b

    def test_detects_f_b_suffix(self, tmp_path):
        img = make_clean_card()
        f = _write_image(str(tmp_path), "card001_f.jpg", img)
        b = _write_image(str(tmp_path), "card001_b.jpg", img)
        pairs = _find_front_back_pairs([f, b])
        assert "card001" in pairs
        assert pairs["card001"][1] is not None

    def test_single_image_treated_as_front_only(self, tmp_path):
        img = make_clean_card()
        p = _write_image(str(tmp_path), "lone_card.jpg", img)
        pairs = _find_front_back_pairs([p])
        assert "lone_card" in pairs
        assert pairs["lone_card"][1] is None

    def test_back_without_front_uses_back_as_front(self, tmp_path):
        img = make_clean_card()
        b = _write_image(str(tmp_path), "orphan_back.jpg", img)
        pairs = _find_front_back_pairs([b])
        assert "orphan" in pairs
        front_path, back_path = pairs["orphan"]
        assert front_path == b
        assert back_path is None

    def test_multiple_pairs(self, tmp_path):
        img = make_clean_card()
        files = []
        for name in ["a_front.jpg", "a_back.jpg", "b_front.jpg", "b_back.jpg"]:
            files.append(_write_image(str(tmp_path), name, img))
        pairs = _find_front_back_pairs(files)
        assert "a" in pairs
        assert "b" in pairs
        assert len(pairs) == 2

    def test_dash_prefix_detected(self, tmp_path):
        img = make_clean_card()
        f = _write_image(str(tmp_path), "card-front.jpg", img)
        b = _write_image(str(tmp_path), "card-back.jpg", img)
        pairs = _find_front_back_pairs([f, b])
        assert "card" in pairs
        assert pairs["card"][1] is not None


# ─── _grade_one ───────────────────────────────────────────────────────────────

class TestGradeOne:
    def test_returns_csv_row_dict(self, tmp_path):
        img = make_clean_card()
        front_path = _write_image(str(tmp_path), "card.jpg", img)
        row = _grade_one("test_card", front_path, None, "tcg_generic",
                          str(tmp_path), dry_run=True, verbose=False)
        assert isinstance(row, dict)
        assert "card_id" in row
        assert row["card_id"] == "test_card"

    def test_grade_field_in_valid_range(self, tmp_path):
        img = make_clean_card()
        front_path = _write_image(str(tmp_path), "card.jpg", img)
        row = _grade_one("test_card", front_path, None, "tcg_generic",
                          str(tmp_path), dry_run=True, verbose=False)
        assert 1.0 <= float(row["grade"]) <= 10.0

    def test_dry_run_no_files_created(self, tmp_path):
        img = make_clean_card()
        front_path = _write_image(str(tmp_path), "card.jpg", img)
        output_dir = str(tmp_path / "output")
        _grade_one("mycard", front_path, None, "tcg_generic",
                    output_dir, dry_run=True, verbose=False)
        assert not os.path.exists(output_dir)

    def test_non_dry_run_creates_artifacts(self, tmp_path):
        img = make_clean_card()
        front_path = _write_image(str(tmp_path), "card.jpg", img)
        output_dir = str(tmp_path / "output")
        _grade_one("mycard", front_path, None, "tcg_generic",
                    output_dir, dry_run=False, verbose=False)
        card_dir = os.path.join(output_dir, "mycard")
        assert os.path.isdir(card_dir)
        assert os.path.isfile(os.path.join(card_dir, "grade_result.json"))

    def test_bad_path_returns_none(self, tmp_path):
        result = _grade_one("missing", "/nonexistent/path/card.jpg", None,
                             "tcg_generic", str(tmp_path), dry_run=True, verbose=False)
        assert result is None

    def test_with_back_image(self, tmp_path):
        front_img = make_clean_card()
        back_img = make_damaged_card()
        front_path = _write_image(str(tmp_path), "card_front.jpg", front_img)
        back_path = _write_image(str(tmp_path), "card_back.jpg", back_img)
        row = _grade_one("card", front_path, back_path, "tcg_generic",
                          str(tmp_path), dry_run=True, verbose=False)
        assert row is not None
        assert 1.0 <= float(row["grade"]) <= 10.0


# ─── get_csv_row ─────────────────────────────────────────────────────────────

class TestGetCsvRow:
    def test_returns_dict_with_all_fields(self):
        results = _make_results()
        row = get_csv_row("test_001", results)
        for field in CSV_FIELDNAMES:
            assert field in row, f"CSV row missing field: {field}"

    def test_card_id_matches(self):
        results = _make_results()
        row = get_csv_row("my_card_123", results)
        assert row["card_id"] == "my_card_123"

    def test_grade_is_numeric(self):
        results = _make_results()
        row = get_csv_row("card", results)
        assert row["grade"] is not None
        assert 1.0 <= float(row["grade"]) <= 10.0

    def test_defect_counts_are_non_negative(self):
        results = _make_results()
        row = get_csv_row("card", results)
        assert row["defects_severe"] >= 0
        assert row["defects_moderate"] >= 0
        assert row["defects_minor"] >= 0

    def test_quality_not_run_when_skipped(self):
        results = _make_results()  # run_quality_gate=False
        row = get_csv_row("card", results)
        assert row["quality_decision"] == "not_run"

    def test_caps_triggered_is_string(self):
        results = _make_results()
        row = get_csv_row("card", results)
        assert isinstance(row["caps_triggered"], str)


# ─── save_batch_csv ───────────────────────────────────────────────────────────

class TestSaveBatchCsv:
    def test_creates_csv_file(self, tmp_path):
        results = _make_results()
        rows = [get_csv_row("card_a", results), get_csv_row("card_b", results)]
        path = str(tmp_path / "summary.csv")
        save_batch_csv(rows, path)
        assert os.path.isfile(path)

    def test_csv_has_header_row(self, tmp_path):
        results = _make_results()
        rows = [get_csv_row("card_a", results)]
        path = str(tmp_path / "summary.csv")
        save_batch_csv(rows, path)
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "card_id" in header
        assert "grade" in header

    def test_csv_sorted_best_grade_first(self, tmp_path):
        r1 = _make_results(make_clean_card())
        r2 = _make_results(make_damaged_card())
        row1 = get_csv_row("clean", r1)
        row2 = get_csv_row("damaged", r2)
        path = str(tmp_path / "summary.csv")
        save_batch_csv([row2, row1], path)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        grades = [float(r["grade"]) for r in rows if r["grade"]]
        # Should be sorted descending
        assert grades == sorted(grades, reverse=True)

    def test_csv_row_count_matches(self, tmp_path):
        results = _make_results()
        rows = [get_csv_row(f"card_{i}", results) for i in range(4)]
        path = str(tmp_path / "summary.csv")
        save_batch_csv(rows, path)
        with open(path, newline="") as f:
            data = list(csv.DictReader(f))
        assert len(data) == 4


# ─── save_grading_artifacts ───────────────────────────────────────────────────

class TestSaveGradingArtifacts:
    def test_creates_card_directory(self, tmp_path):
        front = make_clean_card()
        results = _make_results(front)
        card_dir = save_grading_artifacts("test_card", front, None, results,
                                           str(tmp_path))
        assert os.path.isdir(card_dir)

    def test_creates_json_file(self, tmp_path):
        front = make_clean_card()
        results = _make_results(front)
        card_dir = save_grading_artifacts("test_card", front, None, results,
                                           str(tmp_path))
        json_path = os.path.join(card_dir, "grade_result.json")
        assert os.path.isfile(json_path)

    def test_json_is_valid(self, tmp_path):
        front = make_clean_card()
        results = _make_results(front)
        card_dir = save_grading_artifacts("test_card", front, None, results,
                                           str(tmp_path))
        with open(os.path.join(card_dir, "grade_result.json")) as f:
            data = json.load(f)
        assert "grade_result" in data
        assert "card_id" in data

    def test_json_card_id_matches(self, tmp_path):
        front = make_clean_card()
        results = _make_results(front)
        card_dir = save_grading_artifacts("my_special_card", front, None,
                                           results, str(tmp_path))
        with open(os.path.join(card_dir, "grade_result.json")) as f:
            data = json.load(f)
        assert data["card_id"] == "my_special_card"

    def test_creates_front_normalized_image(self, tmp_path):
        front = make_clean_card()
        results = _make_results(front)
        card_dir = save_grading_artifacts("test_card", front, None, results,
                                           str(tmp_path))
        assert os.path.isfile(os.path.join(card_dir, "front_normalized.jpg"))

    def test_back_normalized_created_when_provided(self, tmp_path):
        front = make_clean_card()
        back = make_damaged_card()
        results = _make_results(front, back)
        card_dir = save_grading_artifacts("test_card", front, back, results,
                                           str(tmp_path))
        assert os.path.isfile(os.path.join(card_dir, "back_normalized.jpg"))

    def test_no_back_normalized_when_not_provided(self, tmp_path):
        front = make_clean_card()
        results = _make_results(front)
        card_dir = save_grading_artifacts("test_card", front, None, results,
                                           str(tmp_path))
        assert not os.path.isfile(os.path.join(card_dir, "back_normalized.jpg"))

    def test_overlay_images_saved(self, tmp_path):
        front = make_clean_card()
        results = _make_results(front)
        card_dir = save_grading_artifacts("test_card", front, None, results,
                                           str(tmp_path))
        for fname in ["overlay_centering.jpg", "overlay_edges.jpg",
                      "overlay_corners.jpg", "overlay_surface.jpg",
                      "overlay_full.jpg", "grade_summary.jpg"]:
            assert os.path.isfile(os.path.join(card_dir, fname)), \
                f"Expected artifact file missing: {fname}"


# ─── build_download_zip ───────────────────────────────────────────────────────

class TestBuildDownloadZip:
    def test_returns_bytes(self):
        front = make_clean_card()
        results = _make_results(front)
        data = build_download_zip("test_card", front, None, results)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_valid_zip_structure(self):
        front = make_clean_card()
        results = _make_results(front)
        data = build_download_zip("test_card", front, None, results)
        buf = io.BytesIO(data)
        assert zipfile.is_zipfile(buf)

    def test_zip_contains_json(self):
        front = make_clean_card()
        results = _make_results(front)
        data = build_download_zip("mycard", front, None, results)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
        assert "mycard/grade_result.json" in names

    def test_zip_contains_front_image(self):
        front = make_clean_card()
        results = _make_results(front)
        data = build_download_zip("mycard", front, None, results)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
        assert "mycard/front_normalized.jpg" in names

    def test_zip_json_is_valid(self):
        front = make_clean_card()
        results = _make_results(front)
        data = build_download_zip("mycard", front, None, results)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            json_bytes = zf.read("mycard/grade_result.json")
        parsed = json.loads(json_bytes)
        assert "grade_result" in parsed

    def test_zip_with_back_includes_back_image(self):
        front = make_clean_card()
        back = make_damaged_card()
        results = _make_results(front, back)
        data = build_download_zip("mycard", front, back, results)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
        assert "mycard/back_normalized.jpg" in names

    def test_zip_without_back_no_back_image(self):
        front = make_clean_card()
        results = _make_results(front)
        data = build_download_zip("mycard", front, None, results)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
        assert "mycard/back_normalized.jpg" not in names


# ─── _print_ranking ──────────────────────────────────────────────────────────

class TestPrintRanking:
    def test_runs_without_error(self, capsys):
        results = _make_results()
        rows = [get_csv_row(f"card_{i}", results) for i in range(3)]
        _print_ranking(rows)
        captured = capsys.readouterr()
        assert "Grade" in captured.out

    def test_handles_empty_list(self, capsys):
        _print_ranking([])
        captured = capsys.readouterr()
        assert len(captured.out) > 0  # Should still print header

    def test_handles_error_rows(self, capsys):
        rows = [{"card_id": "failed_card", "grade": None, "error": "load error"}]
        _print_ranking(rows)
        captured = capsys.readouterr()
        assert len(captured.out) > 0
