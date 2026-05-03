"""
Integration tests for the Streamlit dashboard module.

Tests cover:
- Loading results from file (happy path)
- Error handling for missing file
- Data parsing correctness (DataFrame construction)
- Coverage / metric computation helpers
- JSON Lines format validation on load
"""

import json
import os
from datetime import datetime

import pandas as pd
import pytest

from dashboard import (
    load_predictions,
    parse_to_dataframe,
    _compute_coverage,
    _compute_average_width,
    _compute_winkler_score,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_records():
    """Return a list of raw prediction dicts (as loaded from JSONL)."""
    return [
        {
            "timestamp": "2024-01-01T00:00:00",
            "actual_price": 50000.0,
            "lower_bound": 49000.0,
            "upper_bound": 51000.0,
        },
        {
            "timestamp": "2024-01-01T01:00:00",
            "actual_price": 52000.0,  # OUTSIDE → violation
            "lower_bound": 49000.0,
            "upper_bound": 51000.0,
        },
        {
            "timestamp": "2024-01-01T02:00:00",
            "actual_price": 50500.0,
            "lower_bound": 49500.0,
            "upper_bound": 51500.0,
        },
    ]


@pytest.fixture
def sample_jsonl_file(tmp_path, sample_records):
    """Write sample records to a temporary JSONL file and return its path."""
    path = str(tmp_path / "test_results.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for record in sample_records:
            f.write(json.dumps(record) + "\n")
    return path


@pytest.fixture
def sample_df(sample_records):
    """Return a parsed DataFrame from sample_records."""
    return parse_to_dataframe(sample_records)


# ---------------------------------------------------------------------------
# Tests for load_predictions (Requirement 13.1 / 10.4)
# ---------------------------------------------------------------------------

class TestLoadPredictions:
    """Tests for load_predictions function."""

    def test_loads_correct_number_of_records(self, sample_jsonl_file, sample_records):
        """Must load the same number of records as written."""
        records = load_predictions(sample_jsonl_file)
        assert len(records) == len(sample_records)

    def test_loaded_records_are_dicts(self, sample_jsonl_file):
        """Each loaded record must be a dictionary."""
        records = load_predictions(sample_jsonl_file)
        for rec in records:
            assert isinstance(rec, dict)

    def test_loaded_values_match_written(self, sample_jsonl_file, sample_records):
        """Loaded price values must match what was written."""
        records = load_predictions(sample_jsonl_file)
        for i, (loaded, original) in enumerate(zip(records, sample_records)):
            assert loaded["actual_price"] == pytest.approx(original["actual_price"]), \
                f"actual_price mismatch at index {i}"
            assert loaded["lower_bound"] == pytest.approx(original["lower_bound"])
            assert loaded["upper_bound"] == pytest.approx(original["upper_bound"])

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """Missing results file must raise FileNotFoundError."""
        missing = str(tmp_path / "does_not_exist.jsonl")
        with pytest.raises(FileNotFoundError):
            load_predictions(missing)

    def test_invalid_json_raises_value_error(self, tmp_path):
        """File with invalid JSON line must raise ValueError."""
        bad_file = str(tmp_path / "bad.jsonl")
        with open(bad_file, "w", encoding="utf-8") as f:
            f.write('{"valid": true}\n')
            f.write("NOT JSON AT ALL\n")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_predictions(bad_file)

    def test_skips_empty_lines(self, tmp_path, sample_records):
        """Empty lines must be ignored during loading."""
        path = str(tmp_path / "with_empties.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample_records[0]) + "\n")
            f.write("\n")   # blank line
            f.write(json.dumps(sample_records[1]) + "\n")
        records = load_predictions(path)
        assert len(records) == 2


# ---------------------------------------------------------------------------
# Tests for parse_to_dataframe (Requirement 10.8 / 13.1)
# ---------------------------------------------------------------------------

class TestParseToDataframe:
    """Tests for parse_to_dataframe function."""

    def test_returns_dataframe(self, sample_records):
        """Must return a pandas DataFrame."""
        df = parse_to_dataframe(sample_records)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, sample_records):
        """DataFrame must have all required columns including 'covered'."""
        df = parse_to_dataframe(sample_records)
        for col in ("timestamp", "actual_price", "lower_bound", "upper_bound", "covered"):
            assert col in df.columns, f"Column '{col}' missing from DataFrame"

    def test_timestamp_is_datetime(self, sample_records):
        """Timestamp column must be parsed as datetime, not string."""
        df = parse_to_dataframe(sample_records)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_prices_are_float(self, sample_records):
        """Price columns must be numeric floats."""
        df = parse_to_dataframe(sample_records)
        for col in ("actual_price", "lower_bound", "upper_bound"):
            assert pd.api.types.is_float_dtype(df[col]), \
                f"Column '{col}' is not float dtype"

    def test_covered_column_is_boolean(self, sample_records):
        """'covered' column must be boolean."""
        df = parse_to_dataframe(sample_records)
        assert df["covered"].dtype == bool

    def test_covered_logic_correct(self, sample_records):
        """'covered' must be True when actual_price is within bounds."""
        df = parse_to_dataframe(sample_records)
        # Record 0: 50000 inside [49000, 51000] → covered
        # Record 1: 52000 outside [49000, 51000] → NOT covered
        # Record 2: 50500 inside [49500, 51500] → covered
        assert df.iloc[0]["covered"] is True or df.iloc[0]["covered"] == True
        assert df.iloc[1]["covered"] is False or df.iloc[1]["covered"] == False
        assert df.iloc[2]["covered"] is True or df.iloc[2]["covered"] == True

    def test_sorted_by_timestamp(self, sample_records):
        """DataFrame must be sorted in ascending timestamp order."""
        # Reverse the input to test sorting
        reversed_records = list(reversed(sample_records))
        df = parse_to_dataframe(reversed_records)
        timestamps = df["timestamp"].tolist()
        assert timestamps == sorted(timestamps)

    def test_correct_row_count(self, sample_records):
        """DataFrame must have same number of rows as input records."""
        df = parse_to_dataframe(sample_records)
        assert len(df) == len(sample_records)


# ---------------------------------------------------------------------------
# Tests for metric computation helpers (Requirement 13.3)
# ---------------------------------------------------------------------------

class TestMetricHelpers:
    """Tests for dashboard metric computation functions."""

    def test_coverage_two_out_of_three(self, sample_df):
        """2 out of 3 records covered → coverage = 0.6667."""
        # sample_df has: record 0 covered, record 1 violated, record 2 covered
        coverage = _compute_coverage(sample_df)
        assert coverage == pytest.approx(2 / 3, abs=1e-4)

    def test_coverage_all_inside(self):
        """All records inside → coverage = 1.0."""
        df = pd.DataFrame([
            {"timestamp": "2024-01-01T00:00:00", "actual_price": 50000.0,
             "lower_bound": 49000.0, "upper_bound": 51000.0},
            {"timestamp": "2024-01-01T01:00:00", "actual_price": 50500.0,
             "lower_bound": 49500.0, "upper_bound": 51500.0},
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["covered"] = (df["lower_bound"] <= df["actual_price"]) & \
                        (df["actual_price"] <= df["upper_bound"])
        for col in ("actual_price", "lower_bound", "upper_bound"):
            df[col] = df[col].astype(float)

        assert _compute_coverage(df) == pytest.approx(1.0, abs=1e-4)

    def test_average_width_calculation(self, sample_df):
        """Average width must be mean of (upper - lower) across all rows."""
        expected = ((51000 - 49000) + (51000 - 49000) + (51500 - 49500)) / 3
        assert _compute_average_width(sample_df) == pytest.approx(expected, abs=0.01)

    def test_winkler_score_non_negative(self, sample_df):
        """Winkler score must be non-negative."""
        score = _compute_winkler_score(sample_df)
        assert score >= 0.0

    def test_winkler_score_all_inside(self):
        """When all actuals are inside, Winkler score equals average width."""
        df = pd.DataFrame([
            {"actual_price": 50000.0, "lower_bound": 49000.0, "upper_bound": 51000.0,
             "covered": True},
            {"actual_price": 50500.0, "lower_bound": 49500.0, "upper_bound": 51500.0,
             "covered": True},
        ])
        for col in ("actual_price", "lower_bound", "upper_bound"):
            df[col] = df[col].astype(float)
        score = _compute_winkler_score(df)
        avg_width = _compute_average_width(df)
        assert score == pytest.approx(avg_width, abs=0.01)

    def test_winkler_score_violation_adds_penalty(self, sample_df):
        """When there's a violation, Winkler score must exceed average width."""
        score = _compute_winkler_score(sample_df)
        avg_width = _compute_average_width(sample_df)
        # At least one violation → score > avg_width
        assert score > avg_width

    def test_coverage_has_four_decimal_places(self, sample_df):
        """Coverage must be rounded to 4 decimal places."""
        coverage = _compute_coverage(sample_df)
        assert coverage == round(coverage, 4)

    def test_average_width_has_two_decimal_places(self, sample_df):
        """Average width must be rounded to 2 decimal places."""
        avg_width = _compute_average_width(sample_df)
        assert avg_width == round(avg_width, 2)

    def test_winkler_score_has_two_decimal_places(self, sample_df):
        """Winkler score must be rounded to 2 decimal places."""
        score = _compute_winkler_score(sample_df)
        assert score == round(score, 2)
