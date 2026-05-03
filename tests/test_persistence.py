"""
Unit tests for the persistence module.

Tests cover:
- JSON Lines format correctness
- Timestamp formatting (ISO 8601)
- Price field formatting (float)
- Error handling on write failure
- File validation
- Loading results back from file
"""

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import patch, mock_open, MagicMock

import pytest

from bitcoin_forecasting.persistence import (
    save_results,
    load_results,
    validate_jsonl_file,
    _format_prediction_record,
    _validate_predictions_for_persistence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_predictions():
    """Return a small list of valid prediction dicts with datetime timestamps."""
    return [
        {
            "timestamp": datetime(2024, 1, 1, 0, 0, 0),
            "actual_price": 43250.50,
            "lower_bound": 42000.00,
            "upper_bound": 44500.00,
        },
        {
            "timestamp": datetime(2024, 1, 1, 1, 0, 0),
            "actual_price": 43800.75,
            "lower_bound": 42500.00,
            "upper_bound": 45000.00,
        },
        {
            "timestamp": datetime(2024, 1, 1, 2, 0, 0),
            "actual_price": 44100.25,
            "lower_bound": 43000.00,
            "upper_bound": 45200.00,
        },
    ]


@pytest.fixture
def predictions_with_extra_fields():
    """Return predictions with optional volatility and drift fields."""
    return [
        {
            "timestamp": datetime(2024, 1, 1, 0, 0, 0),
            "actual_price": 43250.50,
            "lower_bound": 42000.00,
            "upper_bound": 44500.00,
            "volatility": 0.023,
            "drift": 0.001,
        },
    ]


@pytest.fixture
def tmp_jsonl_file(tmp_path):
    """Return a temporary .jsonl file path."""
    return str(tmp_path / "test_results.jsonl")


# ---------------------------------------------------------------------------
# Tests for _format_prediction_record
# ---------------------------------------------------------------------------

class TestFormatPredictionRecord:
    """Tests for the internal _format_prediction_record helper."""

    def test_datetime_converted_to_iso_string(self):
        """Datetime timestamp must be serialised as ISO 8601 string."""
        pred = {
            "timestamp": datetime(2024, 3, 15, 12, 30, 45),
            "actual_price": 50000,
            "lower_bound": 49000,
            "upper_bound": 51000,
        }
        record = _format_prediction_record(pred)
        assert record["timestamp"] == "2024-03-15T12:30:45"

    def test_string_timestamp_preserved(self):
        """String timestamps should be kept as-is."""
        pred = {
            "timestamp": "2024-03-15T12:30:45",
            "actual_price": 50000,
            "lower_bound": 49000,
            "upper_bound": 51000,
        }
        record = _format_prediction_record(pred)
        assert record["timestamp"] == "2024-03-15T12:30:45"

    def test_prices_are_floats(self):
        """All price fields must be Python floats."""
        pred = {
            "timestamp": datetime(2024, 1, 1),
            "actual_price": 43250,      # int input
            "lower_bound": 42000,       # int input
            "upper_bound": 44500,       # int input
        }
        record = _format_prediction_record(pred)
        assert isinstance(record["actual_price"], float)
        assert isinstance(record["lower_bound"], float)
        assert isinstance(record["upper_bound"], float)

    def test_optional_fields_preserved(self):
        """Optional volatility and drift fields should be preserved."""
        pred = {
            "timestamp": datetime(2024, 1, 1),
            "actual_price": 50000.0,
            "lower_bound": 49000.0,
            "upper_bound": 51000.0,
            "volatility": 0.023,
            "drift": 0.001,
        }
        record = _format_prediction_record(pred)
        assert "volatility" in record
        assert "drift" in record
        assert record["volatility"] == pytest.approx(0.023)
        assert record["drift"] == pytest.approx(0.001)

    def test_required_fields_present(self):
        """Record must contain all four required fields."""
        pred = {
            "timestamp": datetime(2024, 1, 1),
            "actual_price": 50000.0,
            "lower_bound": 49000.0,
            "upper_bound": 51000.0,
        }
        record = _format_prediction_record(pred)
        assert "timestamp" in record
        assert "actual_price" in record
        assert "lower_bound" in record
        assert "upper_bound" in record


# ---------------------------------------------------------------------------
# Tests for save_results
# ---------------------------------------------------------------------------

class TestSaveResults:
    """Tests for save_results function."""

    def test_saves_correct_number_of_lines(self, sample_predictions, tmp_jsonl_file):
        """Each prediction must produce exactly one line in the output file."""
        save_results(sample_predictions, tmp_jsonl_file)
        with open(tmp_jsonl_file, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == len(sample_predictions)

    def test_each_line_is_valid_json(self, sample_predictions, tmp_jsonl_file):
        """Every line in the output file must be valid JSON."""
        save_results(sample_predictions, tmp_jsonl_file)
        with open(tmp_jsonl_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)  # must not raise
                    assert isinstance(obj, dict)

    def test_timestamp_is_iso_8601_string(self, sample_predictions, tmp_jsonl_file):
        """Timestamps in output file must be ISO 8601 strings."""
        save_results(sample_predictions, tmp_jsonl_file)
        with open(tmp_jsonl_file, encoding="utf-8") as f:
            first_record = json.loads(f.readline())
        timestamp = first_record["timestamp"]
        assert isinstance(timestamp, str)
        # Should be parsable back to datetime
        parsed = datetime.fromisoformat(timestamp)
        assert parsed == sample_predictions[0]["timestamp"]

    def test_prices_are_floats_in_file(self, sample_predictions, tmp_jsonl_file):
        """Price values in output file must be JSON numbers (floats)."""
        save_results(sample_predictions, tmp_jsonl_file)
        with open(tmp_jsonl_file, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert isinstance(record["actual_price"], float)
        assert isinstance(record["lower_bound"], float)
        assert isinstance(record["upper_bound"], float)

    def test_prices_match_input_values(self, sample_predictions, tmp_jsonl_file):
        """Written price values must match the input predictions."""
        save_results(sample_predictions, tmp_jsonl_file)
        records = []
        with open(tmp_jsonl_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        for i, pred in enumerate(sample_predictions):
            assert records[i]["actual_price"] == pytest.approx(pred["actual_price"])
            assert records[i]["lower_bound"] == pytest.approx(pred["lower_bound"])
            assert records[i]["upper_bound"] == pytest.approx(pred["upper_bound"])

    def test_optional_fields_written(self, predictions_with_extra_fields, tmp_jsonl_file):
        """Volatility and drift fields must be preserved in output."""
        save_results(predictions_with_extra_fields, tmp_jsonl_file)
        with open(tmp_jsonl_file, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert "volatility" in record
        assert "drift" in record

    def test_empty_predictions_raises_value_error(self, tmp_jsonl_file):
        """Empty predictions list must raise ValueError."""
        with pytest.raises(ValueError, match="predictions list is empty"):
            save_results([], tmp_jsonl_file)

    def test_missing_required_field_raises_value_error(self, tmp_jsonl_file):
        """Prediction missing required field must raise ValueError before writing."""
        predictions = [
            {
                "timestamp": datetime(2024, 1, 1),
                "actual_price": 50000.0,
                # missing lower_bound and upper_bound
            }
        ]
        with pytest.raises(ValueError, match="missing required fields"):
            save_results(predictions, tmp_jsonl_file)

    def test_write_failure_raises_os_error(self, sample_predictions):
        """IOError / PermissionError on write must be re-raised as OSError with context."""
        bad_path = "/nonexistent_dir/cannot_write_here.jsonl"
        with pytest.raises(OSError, match="Failed to write results"):
            save_results(sample_predictions, bad_path)

    def test_creates_file_on_disk(self, sample_predictions, tmp_jsonl_file):
        """Output file must exist on disk after save."""
        assert not os.path.exists(tmp_jsonl_file)
        save_results(sample_predictions, tmp_jsonl_file)
        assert os.path.exists(tmp_jsonl_file)

    def test_overwrites_existing_file(self, sample_predictions, tmp_jsonl_file):
        """Calling save_results twice must overwrite, not append."""
        save_results(sample_predictions, tmp_jsonl_file)
        save_results(sample_predictions[:1], tmp_jsonl_file)
        with open(tmp_jsonl_file, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        # Only 1 record from the second call
        assert len(lines) == 1

    def test_exact_720_records(self, tmp_jsonl_file):
        """Writing exactly 720 records must succeed without warning."""
        predictions = [
            {
                "timestamp": datetime(2024, 1, 1, i % 24, 0, 0),
                "actual_price": 50000.0 + i,
                "lower_bound": 49000.0,
                "upper_bound": 51000.0,
            }
            for i in range(720)
        ]
        save_results(predictions, tmp_jsonl_file)
        with open(tmp_jsonl_file, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 720


# ---------------------------------------------------------------------------
# Tests for load_results
# ---------------------------------------------------------------------------

class TestLoadResults:
    """Tests for load_results function."""

    def test_round_trip_save_and_load(self, sample_predictions, tmp_jsonl_file):
        """Data saved with save_results must be loadable with load_results."""
        save_results(sample_predictions, tmp_jsonl_file)
        loaded = load_results(tmp_jsonl_file)
        assert len(loaded) == len(sample_predictions)

    def test_loaded_fields_match_saved(self, sample_predictions, tmp_jsonl_file):
        """Loaded price values must match what was saved."""
        save_results(sample_predictions, tmp_jsonl_file)
        loaded = load_results(tmp_jsonl_file)
        for i, pred in enumerate(sample_predictions):
            assert loaded[i]["actual_price"] == pytest.approx(pred["actual_price"])
            assert loaded[i]["lower_bound"] == pytest.approx(pred["lower_bound"])
            assert loaded[i]["upper_bound"] == pytest.approx(pred["upper_bound"])

    def test_loaded_timestamp_is_string(self, sample_predictions, tmp_jsonl_file):
        """Loaded timestamps are ISO strings (not datetime objects)."""
        save_results(sample_predictions, tmp_jsonl_file)
        loaded = load_results(tmp_jsonl_file)
        assert isinstance(loaded[0]["timestamp"], str)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """Loading a non-existent file must raise FileNotFoundError."""
        non_existent = str(tmp_path / "does_not_exist.jsonl")
        with pytest.raises(FileNotFoundError, match="not found"):
            load_results(non_existent)

    def test_invalid_json_raises_value_error(self, tmp_jsonl_file):
        """File containing invalid JSON line must raise ValueError."""
        with open(tmp_jsonl_file, "w", encoding="utf-8") as f:
            f.write('{"valid": "json"}\n')
            f.write("THIS IS NOT JSON\n")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_results(tmp_jsonl_file)

    def test_skips_empty_lines(self, tmp_jsonl_file):
        """Empty lines in the file should be ignored."""
        with open(tmp_jsonl_file, "w", encoding="utf-8") as f:
            f.write('{"timestamp": "2024-01-01T00:00:00", "actual_price": 50000.0, '
                    '"lower_bound": 49000.0, "upper_bound": 51000.0}\n')
            f.write("\n")  # empty line
            f.write('{"timestamp": "2024-01-01T01:00:00", "actual_price": 50100.0, '
                    '"lower_bound": 49100.0, "upper_bound": 51100.0}\n')
        loaded = load_results(tmp_jsonl_file)
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# Tests for validate_jsonl_file
# ---------------------------------------------------------------------------

class TestValidateJsonlFile:
    """Tests for validate_jsonl_file function."""

    def test_valid_file_returns_true(self, sample_predictions, tmp_jsonl_file):
        """Well-formed JSON Lines file must return True."""
        save_results(sample_predictions, tmp_jsonl_file)
        result = validate_jsonl_file(tmp_jsonl_file)
        assert result is True

    def test_invalid_file_raises_value_error(self, tmp_jsonl_file):
        """File with invalid JSON line must raise ValueError."""
        with open(tmp_jsonl_file, "w", encoding="utf-8") as f:
            f.write("not valid json at all\n")
        with pytest.raises(ValueError, match="Invalid JSON"):
            validate_jsonl_file(tmp_jsonl_file)

    def test_non_existent_file_raises_file_not_found(self, tmp_path):
        """Non-existent file must raise FileNotFoundError."""
        path = str(tmp_path / "missing.jsonl")
        with pytest.raises(FileNotFoundError):
            validate_jsonl_file(path)

    def test_empty_file_returns_true(self, tmp_jsonl_file):
        """An empty file (no records) is valid JSON Lines (vacuously)."""
        with open(tmp_jsonl_file, "w", encoding="utf-8") as f:
            f.write("")
        result = validate_jsonl_file(tmp_jsonl_file)
        assert result is True
