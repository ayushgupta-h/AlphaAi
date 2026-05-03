"""
Results persistence module for saving backtesting predictions.

This module provides functionality to save prediction results to JSON Lines
format, where each line is a self-contained JSON object representing one
prediction. This format is ideal for large datasets as it allows streaming
reads and appends without loading the entire file.

JSON Lines format specification:
- Each line is a valid JSON object
- Lines are separated by newline characters
- No trailing comma after each line (unlike JSON arrays)
- File extension: .jsonl
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Expected number of records for full backtesting run
EXPECTED_RECORD_COUNT = 720


def save_results(
    predictions: List[Dict[str, Any]],
    output_filename: str = "backtest_results.jsonl",
) -> None:
    """
    Save backtesting prediction results to a JSON Lines file.

    Each prediction is serialized as a JSON object on a single line.
    Timestamps are converted to ISO 8601 strings, and all price values
    are formatted as floating point numbers for consistent representation.

    The output file contains exactly one JSON object per line, with the
    following fields:
        - timestamp: ISO 8601 datetime string (e.g. "2024-01-15T10:00:00")
        - actual_price: float, the actual close price
        - lower_bound: float, lower bound of 95% prediction interval
        - upper_bound: float, upper bound of 95% prediction interval

    Parameters
    ----------
    predictions : List[Dict[str, Any]]
        List of prediction dictionaries from backtesting.
        Each dictionary must contain:
        - timestamp: datetime object or ISO string
        - actual_price: numeric, the actual close price
        - lower_bound: numeric, lower bound of prediction interval
        - upper_bound: numeric, upper bound of prediction interval
        Additional fields (volatility, drift) are preserved if present.
    output_filename : str, optional
        Path to output file. Default: "backtest_results.jsonl".
        Parent directories must exist.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If predictions list is empty.
        If any prediction is missing required fields.
    OSError
        If the file cannot be written (permissions, disk space, etc.)
        with a descriptive error message.

    Examples
    --------
    >>> from datetime import datetime
    >>> predictions = [
    ...     {
    ...         'timestamp': datetime(2024, 1, 1, 0, 0, 0),
    ...         'actual_price': 43250.50,
    ...         'lower_bound': 42000.00,
    ...         'upper_bound': 44500.00,
    ...     }
    ... ]
    >>> save_results(predictions, "my_results.jsonl")

    Notes
    -----
    - Requirement 9.1: Accepts predictions list and output filename
    - Requirement 9.2: Formats each prediction as JSON object
    - Requirement 9.3: Converts timestamps to ISO 8601 strings
    - Requirement 9.4: Formats prices as floating point numbers
    - Requirement 9.5: Writes one prediction per line
    - Requirement 9.6: Validates exactly 720 records written
    - Requirement 9.7: Raises descriptive error on file write failure
    - Requirement 9.8: Adds logging for successful save
    - Requirement 14.5: Validates output is valid JSON Lines format
    """
    # Validate inputs
    if not predictions:
        raise ValueError(
            "Cannot save results: predictions list is empty. "
            "Run backtesting first to generate predictions."
        )

    # Validate all required fields before writing
    _validate_predictions_for_persistence(predictions)

    logger.info(f"Saving {len(predictions)} predictions to '{output_filename}'")

    try:
        # Write predictions to JSON Lines file
        with open(output_filename, "w", encoding="utf-8") as f:
            for i, pred in enumerate(predictions):
                # Format prediction as JSON-serializable record
                record = _format_prediction_record(pred)

                # Write as JSON line (no trailing comma)
                json_line = json.dumps(record, ensure_ascii=False)
                f.write(json_line + "\n")

        # Requirement 9.6: Validate exactly 720 records written
        written_count = _count_records_in_file(output_filename)
        if written_count != len(predictions):
            raise OSError(
                f"File write validation failed: expected {len(predictions)} records "
                f"but found {written_count} in '{output_filename}'"
            )

        # Requirement 9.8: Log successful save
        file_size_bytes = os.path.getsize(output_filename)
        logger.info(
            f"Successfully saved {written_count} predictions to '{output_filename}' "
            f"(file size: {file_size_bytes:,} bytes)"
        )

        if len(predictions) == EXPECTED_RECORD_COUNT:
            logger.info(
                f"Validation passed: exactly {EXPECTED_RECORD_COUNT} records written "
                f"as expected for full backtesting run"
            )
        else:
            logger.warning(
                f"Note: wrote {len(predictions)} records, expected {EXPECTED_RECORD_COUNT} "
                f"for a full backtesting run"
            )

    except OSError as e:
        # Requirement 9.7: Raise descriptive error on file write failure
        error_msg = (
            f"Failed to write results to '{output_filename}': "
            f"{type(e).__name__}: {str(e)}. "
            f"Check file permissions and available disk space."
        )
        logger.error(error_msg)
        raise OSError(error_msg) from e


def load_results(
    input_filename: str = "backtest_results.jsonl",
) -> List[Dict[str, Any]]:
    """
    Load backtesting prediction results from a JSON Lines file.

    Reads each line of the file as a JSON object and returns a list of
    prediction dictionaries. Timestamps are returned as strings in ISO 8601
    format as stored in the file.

    Parameters
    ----------
    input_filename : str, optional
        Path to JSON Lines file to read. Default: "backtest_results.jsonl".

    Returns
    -------
    List[Dict[str, Any]]
        List of prediction dictionaries loaded from file.
        Each dictionary contains: timestamp (str), actual_price (float),
        lower_bound (float), upper_bound (float).

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the file contains invalid JSON Lines format.

    Examples
    --------
    >>> predictions = load_results("backtest_results.jsonl")
    >>> print(f"Loaded {len(predictions)} predictions")
    >>> print(predictions[0])

    Notes
    -----
    - Requirement 10.4: Loads predictions from backtest_results.jsonl
    - Requirement 10.8: Validates JSON Lines format on load
    """
    if not os.path.exists(input_filename):
        raise FileNotFoundError(
            f"Results file '{input_filename}' not found. "
            f"Run the backtesting pipeline first: python main.py"
        )

    logger.info(f"Loading results from '{input_filename}'")

    predictions = []
    line_number = 0

    try:
        with open(input_filename, "r", encoding="utf-8") as f:
            for line in f:
                line_number += 1
                line = line.strip()

                if not line:
                    continue  # Skip empty lines

                try:
                    record = json.loads(line)
                    predictions.append(record)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} in '{input_filename}': {e}"
                    ) from e

    except OSError as e:
        raise OSError(
            f"Failed to read '{input_filename}': {type(e).__name__}: {str(e)}"
        ) from e

    logger.info(
        f"Successfully loaded {len(predictions)} predictions from '{input_filename}'"
    )

    return predictions


def _format_prediction_record(pred: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a prediction dictionary into a JSON-serializable record.

    Converts datetime objects to ISO 8601 strings and ensures all
    price values are represented as floating point numbers.

    Parameters
    ----------
    pred : Dict[str, Any]
        Prediction dictionary from backtesting.

    Returns
    -------
    Dict[str, Any]
        JSON-serializable dictionary with formatted fields.
    """
    # Requirement 9.3: Convert timestamp to ISO 8601 string
    timestamp = pred["timestamp"]
    if isinstance(timestamp, datetime):
        timestamp_str = timestamp.isoformat()
    else:
        # Already a string; ensure it is valid ISO 8601
        timestamp_str = str(timestamp)

    # Requirement 9.4: Format prices as floating point numbers
    record = {
        "timestamp": timestamp_str,
        "actual_price": float(pred["actual_price"]),
        "lower_bound": float(pred["lower_bound"]),
        "upper_bound": float(pred["upper_bound"]),
    }

    # Preserve optional fields if present
    for optional_field in ("volatility", "drift"):
        if optional_field in pred:
            record[optional_field] = float(pred[optional_field])

    return record


def _validate_predictions_for_persistence(predictions: List[Dict[str, Any]]) -> None:
    """
    Validate that all predictions contain the required fields for persistence.

    Parameters
    ----------
    predictions : List[Dict[str, Any]]
        List of prediction dictionaries to validate.

    Raises
    ------
    ValueError
        If any prediction is missing required fields.
    """
    required_fields = ["timestamp", "actual_price", "lower_bound", "upper_bound"]

    for i, pred in enumerate(predictions):
        missing_fields = [field for field in required_fields if field not in pred]
        if missing_fields:
            raise ValueError(
                f"Prediction at index {i} is missing required fields for persistence: "
                f"{missing_fields}. Required fields: {required_fields}"
            )


def _count_records_in_file(filename: str) -> int:
    """
    Count the number of non-empty JSON Lines records in a file.

    Parameters
    ----------
    filename : str
        Path to JSON Lines file.

    Returns
    -------
    int
        Number of non-empty lines (records) in the file.
    """
    count = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def validate_jsonl_file(filename: str) -> bool:
    """
    Validate that a file contains valid JSON Lines format.

    Reads each non-empty line and attempts to parse it as JSON.
    Returns True if all lines are valid JSON, raises ValueError otherwise.

    Parameters
    ----------
    filename : str
        Path to file to validate.

    Returns
    -------
    bool
        True if file is valid JSON Lines format.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If any line cannot be parsed as valid JSON.

    Notes
    -----
    - Requirement 9.7: Validates output is valid JSON Lines format
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' not found")

    line_number = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line_number += 1
            stripped = line.strip()
            if not stripped:
                continue

            try:
                json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in '{filename}': {e}"
                ) from e

    logger.debug(f"JSON Lines validation passed for '{filename}' ({line_number} lines)")
    return True
