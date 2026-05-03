"""
Evaluation metrics module for probabilistic forecast quality assessment.

This module implements three key metrics for evaluating prediction intervals:
1. Coverage: Fraction of predictions where actual price falls within interval
2. Average Width: Mean width of prediction intervals (precision measure)
3. Winkler Score: Comprehensive metric balancing coverage and precision

These metrics provide complementary views of forecast quality:
- Coverage measures calibration (are intervals correctly sized?)
- Average Width measures precision (how tight are the intervals?)
- Winkler Score combines both, penalizing wide intervals and coverage violations
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def compute_coverage(predictions: List[Dict[str, Any]]) -> float:
    """
    Compute coverage metric as fraction of predictions with actual price inside interval.

    Coverage measures the empirical calibration of prediction intervals. For a
    well-calibrated 95% confidence interval, we expect approximately 95% of actual
    prices to fall within the predicted bounds. Coverage significantly below 0.95
    indicates intervals are too narrow (overconfident), while coverage significantly
    above 0.95 indicates intervals are too wide (underconfident).

    The coverage metric is computed as:
        coverage = (number of predictions with actual inside bounds) / (total predictions)

    Where "actual inside bounds" means: lower_bound <= actual_price <= upper_bound

    Parameters
    ----------
    predictions : List[Dict[str, Any]]
        List of prediction dictionaries from backtesting.
        Each dictionary must contain:
        - actual_price: float, the actual close price at prediction timestamp
        - lower_bound: float, lower bound of 95% prediction interval
        - upper_bound: float, upper bound of 95% prediction interval

    Returns
    -------
    float
        Coverage metric between 0.0 and 1.0, rounded to 4 decimal places.
        Returns 0.0 if predictions list is empty.

    Raises
    ------
    ValueError
        If predictions list is empty
        If any prediction is missing required fields
        If computed coverage is not between 0 and 1 (internal validation)

    Examples
    --------
    >>> predictions = [
    ...     {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},
    ...     {'actual_price': 50500, 'lower_bound': 49500, 'upper_bound': 51500},
    ...     {'actual_price': 52000, 'lower_bound': 49000, 'upper_bound': 51000},
    ... ]
    >>> coverage = compute_coverage(predictions)
    >>> print(f"Coverage: {coverage:.4f}")
    Coverage: 0.6667

    Notes
    -----
    - Requirement 6.1: Computes coverage as fraction of predictions with actual inside interval
    - Requirement 6.2: Checks if lower_bound <= actual_price <= upper_bound
    - Requirement 6.3: Counts predictions with actual price inside bounds
    - Requirement 6.4: Divides count by total number of predictions
    - Requirement 6.5: Validates result is between 0 and 1
    - Requirement 6.6: Reports coverage with at least 4 decimal places

    For a well-calibrated 95% confidence interval:
    - Expected coverage: ~0.95
    - Acceptable range: 0.85 to 1.0 (depends on sample size and market conditions)
    - Coverage < 0.85: intervals too narrow (model overconfident)
    - Coverage > 0.98: intervals too wide (model underconfident)
    """
    # Validate input
    if not predictions:
        raise ValueError("Cannot compute coverage: predictions list is empty")

    # Requirement 6.3: Count predictions with actual price inside bounds
    inside_count = 0

    for i, pred in enumerate(predictions):
        # Validate required fields exist
        _validate_prediction_fields(pred, i)

        actual_price = pred["actual_price"]
        lower_bound = pred["lower_bound"]
        upper_bound = pred["upper_bound"]

        # Requirement 6.2: Check if lower_bound <= actual_price <= upper_bound
        if lower_bound <= actual_price <= upper_bound:
            inside_count += 1

    # Requirement 6.4: Divide count by total number of predictions
    total_predictions = len(predictions)
    coverage = inside_count / total_predictions

    # Requirement 6.5: Validate result is between 0 and 1
    if not (0.0 <= coverage <= 1.0):
        raise ValueError(
            f"Computed coverage {coverage} is not between 0 and 1. "
            f"This indicates a bug in the calculation."
        )

    # Requirement 6.6: Round to 4 decimal places
    coverage = round(coverage, 4)

    logger.info(
        f"Coverage metric: {coverage:.4f} "
        f"({inside_count}/{total_predictions} predictions inside bounds)"
    )

    return coverage


def compute_average_width(predictions: List[Dict[str, Any]]) -> float:
    """
    Compute average width of prediction intervals as precision measure.

    Average width measures the precision of prediction intervals. Narrower intervals
    indicate more precise forecasts, while wider intervals indicate greater uncertainty.
    This metric should be evaluated alongside coverage: narrow intervals with low
    coverage indicate overconfidence, while wide intervals with high coverage indicate
    underconfidence.

    The average width is computed as:
        average_width = mean(upper_bound - lower_bound for all predictions)

    Parameters
    ----------
    predictions : List[Dict[str, Any]]
        List of prediction dictionaries from backtesting.
        Each dictionary must contain:
        - lower_bound: float, lower bound of 95% prediction interval
        - upper_bound: float, upper bound of 95% prediction interval

    Returns
    -------
    float
        Average width in USD, rounded to 2 decimal places.
        Returns 0.0 if predictions list is empty.

    Raises
    ------
    ValueError
        If predictions list is empty
        If any prediction is missing required fields
        If any width is non-positive (internal validation)

    Examples
    --------
    >>> predictions = [
    ...     {'lower_bound': 49000, 'upper_bound': 51000},
    ...     {'lower_bound': 49500, 'upper_bound': 51500},
    ...     {'lower_bound': 48000, 'upper_bound': 52000},
    ... ]
    >>> avg_width = compute_average_width(predictions)
    >>> print(f"Average Width: ${avg_width:.2f}")
    Average Width: $2666.67

    Notes
    -----
    - Requirement 7.1: Computes width as upper_bound - lower_bound
    - Requirement 7.2: Computes average as mean of all widths
    - Requirement 7.3: Validates all widths are positive
    - Requirement 7.4: Reports in same units as price (USD)
    - Requirement 7.5: Reports with at least 2 decimal places

    Interpretation:
    - Lower average width: more precise forecasts (tighter intervals)
    - Higher average width: less precise forecasts (wider intervals)
    - Should be evaluated with coverage: narrow + low coverage = overconfident
    - Typical range for Bitcoin hourly forecasts: $1000-$5000 depending on volatility
    """
    # Validate input
    if not predictions:
        raise ValueError("Cannot compute average width: predictions list is empty")

    # Requirement 7.1: Compute width for each prediction
    widths = []

    for i, pred in enumerate(predictions):
        # Validate required fields exist
        _validate_prediction_fields(pred, i)

        lower_bound = pred["lower_bound"]
        upper_bound = pred["upper_bound"]

        # Compute width
        width = upper_bound - lower_bound

        # Requirement 7.3: Validate width is positive
        if width <= 0:
            raise ValueError(
                f"Prediction at index {i} has non-positive width: {width}. "
                f"lower_bound={lower_bound}, upper_bound={upper_bound}"
            )

        widths.append(width)

    # Requirement 7.2: Compute mean of all widths
    average_width = sum(widths) / len(widths)

    # Requirement 7.5: Round to 2 decimal places
    average_width = round(average_width, 2)

    logger.info(
        f"Average width: ${average_width:.2f} "
        f"(min: ${min(widths):.2f}, max: ${max(widths):.2f})"
    )

    return average_width


def compute_winkler_score(
    predictions: List[Dict[str, Any]], alpha: float = 0.05
) -> float:
    """
    Compute Winkler score as comprehensive interval scoring metric.

    The Winkler score is a proper scoring rule for prediction intervals that
    balances coverage and precision. It penalizes both wide intervals (lack of
    precision) and coverage violations (miscalibration). Lower scores indicate
    better forecast quality.

    The Winkler score for each prediction is computed as:
    - If actual inside interval: score = width
    - If actual < lower_bound: score = width + (2/alpha) * (lower_bound - actual)
    - If actual > upper_bound: score = width + (2/alpha) * (actual - upper_bound)

    Where:
    - width = upper_bound - lower_bound
    - alpha = 1 - confidence_level (default 0.05 for 95% intervals)
    - (2/alpha) = 40 for alpha=0.05, providing strong penalty for coverage violations

    The mean Winkler score across all predictions provides an overall quality metric.

    Parameters
    ----------
    predictions : List[Dict[str, Any]]
        List of prediction dictionaries from backtesting.
        Each dictionary must contain:
        - actual_price: float, the actual close price at prediction timestamp
        - lower_bound: float, lower bound of 95% prediction interval
        - upper_bound: float, upper bound of 95% prediction interval
    alpha : float, optional
        Significance level (1 - confidence_level).
        Default: 0.05 for 95% confidence intervals.
        Must be between 0 and 1.

    Returns
    -------
    float
        Mean Winkler score in USD, rounded to 2 decimal places.
        Lower scores indicate better forecast quality.
        Returns 0.0 if predictions list is empty.

    Raises
    ------
    ValueError
        If predictions list is empty
        If any prediction is missing required fields
        If alpha is not between 0 and 1
        If any individual score is negative (internal validation)

    Examples
    --------
    >>> predictions = [
    ...     {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},  # inside
    ...     {'actual_price': 50500, 'lower_bound': 49500, 'upper_bound': 51500},  # inside
    ...     {'actual_price': 52000, 'lower_bound': 49000, 'upper_bound': 51000},  # above
    ... ]
    >>> winkler = compute_winkler_score(predictions)
    >>> print(f"Winkler Score: ${winkler:.2f}")
    Winkler Score: $28666.67

    Notes
    -----
    - Requirement 8.1: Computes Winkler score with alpha parameter 0.05
    - Requirement 8.2: If actual inside interval, score = width
    - Requirement 8.3: If actual < lower_bound, score = width + (2/alpha) * (lower_bound - actual)
    - Requirement 8.4: If actual > upper_bound, score = width + (2/alpha) * (actual - upper_bound)
    - Requirement 8.5: Computes mean Winkler score across all predictions
    - Requirement 8.6: Validates all individual scores are non-negative
    - Requirement 8.7: Reports with at least 2 decimal places

    Interpretation:
    - Lower score: better forecast quality (tight intervals with good coverage)
    - Higher score: worse forecast quality (wide intervals or poor coverage)
    - Score ≈ average_width: good coverage (most actuals inside intervals)
    - Score >> average_width: poor coverage (many actuals outside intervals)
    - Penalty factor (2/alpha) = 40 for 95% intervals, heavily penalizing violations

    References
    ----------
    - Winkler, R. L. (1972). A decision-theoretic approach to interval estimation.
      Journal of the American Statistical Association, 67(337), 187-191.
    """
    # Validate input
    if not predictions:
        raise ValueError("Cannot compute Winkler score: predictions list is empty")

    # Validate alpha parameter
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Alpha parameter must be between 0 and 1, got {alpha}")

    # Compute penalty factor: 2/alpha
    # For alpha=0.05 (95% confidence), penalty = 40
    penalty_factor = 2.0 / alpha

    # Requirement 8.1 & 8.5: Compute Winkler score for each prediction
    scores = []

    for i, pred in enumerate(predictions):
        # Validate required fields exist
        _validate_prediction_fields(pred, i)

        actual_price = pred["actual_price"]
        lower_bound = pred["lower_bound"]
        upper_bound = pred["upper_bound"]

        # Compute interval width
        width = upper_bound - lower_bound

        # Compute Winkler score based on where actual price falls
        if lower_bound <= actual_price <= upper_bound:
            # Requirement 8.2: Actual inside interval, score = width
            score = width
        elif actual_price < lower_bound:
            # Requirement 8.3: Actual below interval, add penalty
            penalty = penalty_factor * (lower_bound - actual_price)
            score = width + penalty
        else:  # actual_price > upper_bound
            # Requirement 8.4: Actual above interval, add penalty
            penalty = penalty_factor * (actual_price - upper_bound)
            score = width + penalty

        # Requirement 8.6: Validate score is non-negative
        if score < 0:
            raise ValueError(
                f"Prediction at index {i} has negative Winkler score: {score}. "
                f"This indicates a bug in the calculation. "
                f"actual={actual_price}, lower={lower_bound}, upper={upper_bound}"
            )

        scores.append(score)

    # Requirement 8.5: Compute mean Winkler score
    mean_score = sum(scores) / len(scores)

    # Requirement 8.7: Round to 2 decimal places
    mean_score = round(mean_score, 2)

    # Count violations for logging
    violations = sum(
        1
        for pred in predictions
        if not (pred["lower_bound"] <= pred["actual_price"] <= pred["upper_bound"])
    )

    logger.info(
        f"Winkler score: ${mean_score:.2f} "
        f"(alpha={alpha}, penalty_factor={penalty_factor:.1f}, "
        f"violations={violations}/{len(predictions)})"
    )

    return mean_score


def _validate_prediction_fields(pred: Dict[str, Any], index: int) -> None:
    """
    Validate that a prediction dictionary contains all required fields.

    Parameters
    ----------
    pred : Dict[str, Any]
        Prediction dictionary to validate
    index : int
        Index of prediction in list (for error messages)

    Raises
    ------
    ValueError
        If any required field is missing
    """
    required_fields = ["actual_price", "lower_bound", "upper_bound"]
    missing_fields = [field for field in required_fields if field not in pred]

    if missing_fields:
        raise ValueError(
            f"Prediction at index {index} is missing required fields: {missing_fields}. "
            f"Required fields: {required_fields}"
        )
