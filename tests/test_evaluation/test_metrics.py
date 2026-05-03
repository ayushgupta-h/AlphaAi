"""
Unit tests for evaluation metrics module.

Tests cover:
- Coverage metric calculation with known data
- Average width metric calculation
- Winkler score calculation
- Validation of input data
- Edge cases and error handling
"""

import pytest
from hypothesis import given, strategies as st, settings
from bitcoin_forecasting.evaluation.metrics import (
    compute_coverage,
    compute_average_width,
    compute_winkler_score
)


class TestComputeCoverage:
    """Test suite for compute_coverage function."""
    
    def test_coverage_all_inside(self):
        """Test coverage when all predictions have actual price inside bounds."""
        predictions = [
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},
            {'actual_price': 50500, 'lower_bound': 49500, 'upper_bound': 51500},
            {'actual_price': 51000, 'lower_bound': 50000, 'upper_bound': 52000},
        ]
        
        coverage = compute_coverage(predictions)
        
        # All 3 predictions inside bounds, so coverage = 3/3 = 1.0
        assert coverage == 1.0, f"Expected coverage 1.0, got {coverage}"
    
    def test_coverage_none_inside(self):
        """Test coverage when no predictions have actual price inside bounds."""
        predictions = [
            {'actual_price': 52000, 'lower_bound': 49000, 'upper_bound': 51000},  # above
            {'actual_price': 48000, 'lower_bound': 49000, 'upper_bound': 51000},  # below
            {'actual_price': 53000, 'lower_bound': 50000, 'upper_bound': 52000},  # above
        ]
        
        coverage = compute_coverage(predictions)
        
        # 0 predictions inside bounds, so coverage = 0/3 = 0.0
        assert coverage == 0.0, f"Expected coverage 0.0, got {coverage}"
    
    def test_coverage_partial(self):
        """Test coverage with some predictions inside and some outside bounds."""
        predictions = [
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},  # inside
            {'actual_price': 52000, 'lower_bound': 49000, 'upper_bound': 51000},  # above
            {'actual_price': 50500, 'lower_bound': 49500, 'upper_bound': 51500},  # inside
            {'actual_price': 48000, 'lower_bound': 49000, 'upper_bound': 51000},  # below
            {'actual_price': 51000, 'lower_bound': 50000, 'upper_bound': 52000},  # inside
        ]
        
        coverage = compute_coverage(predictions)
        
        # 3 out of 5 inside bounds, so coverage = 3/5 = 0.6
        assert coverage == 0.6, f"Expected coverage 0.6, got {coverage}"
    
    def test_coverage_at_boundaries(self):
        """Test coverage when actual price equals lower or upper bound."""
        predictions = [
            {'actual_price': 49000, 'lower_bound': 49000, 'upper_bound': 51000},  # at lower
            {'actual_price': 51000, 'lower_bound': 49000, 'upper_bound': 51000},  # at upper
            {'actual_price': 50000, 'lower_bound': 50000, 'upper_bound': 50000},  # both equal
        ]
        
        coverage = compute_coverage(predictions)
        
        # All 3 should be counted as inside (inclusive bounds)
        assert coverage == 1.0, f"Expected coverage 1.0 for boundary cases, got {coverage}"
    
    def test_coverage_four_decimal_places(self):
        """Test that coverage is returned with 4 decimal places."""
        # Create predictions with 7 out of 9 inside (7/9 = 0.777777...)
        predictions = [
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},  # inside
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},  # inside
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},  # inside
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},  # inside
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},  # inside
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},  # inside
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},  # inside
            {'actual_price': 52000, 'lower_bound': 49000, 'upper_bound': 51000},  # outside
            {'actual_price': 48000, 'lower_bound': 49000, 'upper_bound': 51000},  # outside
        ]
        
        coverage = compute_coverage(predictions)
        
        # Should be rounded to 4 decimal places: 0.7778
        assert coverage == 0.7778, f"Expected coverage 0.7778, got {coverage}"
    
    def test_coverage_between_zero_and_one(self):
        """Test that coverage is always between 0 and 1."""
        test_cases = [
            # All inside
            [
                {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},
                {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},
            ],
            # All outside
            [
                {'actual_price': 52000, 'lower_bound': 49000, 'upper_bound': 51000},
                {'actual_price': 48000, 'lower_bound': 49000, 'upper_bound': 51000},
            ],
            # Mixed
            [
                {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},
                {'actual_price': 52000, 'lower_bound': 49000, 'upper_bound': 51000},
            ],
        ]
        
        for predictions in test_cases:
            coverage = compute_coverage(predictions)
            assert 0.0 <= coverage <= 1.0, \
                f"Coverage {coverage} is not between 0 and 1"
    
    def test_coverage_empty_list_raises_error(self):
        """Test that empty predictions list raises ValueError."""
        predictions = []
        
        with pytest.raises(ValueError, match="predictions list is empty"):
            compute_coverage(predictions)
    
    def test_coverage_missing_actual_price_raises_error(self):
        """Test that missing actual_price field raises ValueError."""
        predictions = [
            {'lower_bound': 49000, 'upper_bound': 51000},  # missing actual_price
        ]
        
        with pytest.raises(ValueError, match="missing required fields"):
            compute_coverage(predictions)
    
    def test_coverage_missing_lower_bound_raises_error(self):
        """Test that missing lower_bound field raises ValueError."""
        predictions = [
            {'actual_price': 50000, 'upper_bound': 51000},  # missing lower_bound
        ]
        
        with pytest.raises(ValueError, match="missing required fields"):
            compute_coverage(predictions)
    
    def test_coverage_missing_upper_bound_raises_error(self):
        """Test that missing upper_bound field raises ValueError."""
        predictions = [
            {'actual_price': 50000, 'lower_bound': 49000},  # missing upper_bound
        ]
        
        with pytest.raises(ValueError, match="missing required fields"):
            compute_coverage(predictions)
    
    def test_coverage_single_prediction(self):
        """Test coverage with single prediction."""
        # Inside
        predictions = [
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},
        ]
        coverage = compute_coverage(predictions)
        assert coverage == 1.0
        
        # Outside
        predictions = [
            {'actual_price': 52000, 'lower_bound': 49000, 'upper_bound': 51000},
        ]
        coverage = compute_coverage(predictions)
        assert coverage == 0.0
    
    def test_coverage_large_dataset(self):
        """Test coverage with large number of predictions."""
        # Create 1000 predictions, 950 inside (95% coverage)
        predictions = []
        for i in range(950):
            predictions.append({
                'actual_price': 50000,
                'lower_bound': 49000,
                'upper_bound': 51000
            })
        for i in range(50):
            predictions.append({
                'actual_price': 52000,
                'lower_bound': 49000,
                'upper_bound': 51000
            })
        
        coverage = compute_coverage(predictions)
        
        # Should be 950/1000 = 0.95
        assert coverage == 0.95, f"Expected coverage 0.95, got {coverage}"
    
    def test_coverage_realistic_bitcoin_prices(self):
        """Test coverage with realistic Bitcoin price ranges."""
        predictions = [
            {'actual_price': 43250.50, 'lower_bound': 42000.00, 'upper_bound': 44500.00},
            {'actual_price': 43800.75, 'lower_bound': 42500.00, 'upper_bound': 45000.00},
            {'actual_price': 44100.25, 'lower_bound': 43000.00, 'upper_bound': 45200.00},
            {'actual_price': 46000.00, 'lower_bound': 43500.00, 'upper_bound': 45500.00},  # outside
            {'actual_price': 43900.00, 'lower_bound': 42800.00, 'upper_bound': 45100.00},
        ]
        
        coverage = compute_coverage(predictions)
        
        # 4 out of 5 inside, so coverage = 4/5 = 0.8
        assert coverage == 0.8, f"Expected coverage 0.8, got {coverage}"
    
    def test_coverage_with_extra_fields(self):
        """Test that coverage works when predictions have extra fields."""
        predictions = [
            {
                'timestamp': '2024-01-01T00:00:00',
                'actual_price': 50000,
                'lower_bound': 49000,
                'upper_bound': 51000,
                'volatility': 0.5,
                'drift': 0.1
            },
            {
                'timestamp': '2024-01-01T01:00:00',
                'actual_price': 50500,
                'lower_bound': 49500,
                'upper_bound': 51500,
                'volatility': 0.6,
                'drift': 0.15
            },
        ]
        
        coverage = compute_coverage(predictions)
        
        # Both inside, so coverage = 2/2 = 1.0
        assert coverage == 1.0, f"Expected coverage 1.0, got {coverage}"


class TestComputeAverageWidth:
    """Test suite for compute_average_width function."""
    
    def test_average_width_uniform(self):
        """Test average width when all intervals have same width."""
        predictions = [
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},  # width = 2000
            {'actual_price': 50500, 'lower_bound': 49500, 'upper_bound': 51500},  # width = 2000
            {'actual_price': 51000, 'lower_bound': 50000, 'upper_bound': 52000},  # width = 2000
        ]
        
        avg_width = compute_average_width(predictions)
        
        assert avg_width == 2000.0, f"Expected average width 2000.0, got {avg_width}"
    
    def test_average_width_varying(self):
        """Test average width with varying interval widths."""
        predictions = [
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},  # width = 2000
            {'actual_price': 50500, 'lower_bound': 49000, 'upper_bound': 52000},  # width = 3000
            {'actual_price': 49500, 'lower_bound': 49000, 'upper_bound': 50000},  # width = 1000
        ]
        
        avg_width = compute_average_width(predictions)
        
        # Average = (2000 + 3000 + 1000) / 3 = 2000
        assert avg_width == 2000.0, f"Expected average width 2000.0, got {avg_width}"
    
    def test_average_width_two_decimal_places(self):
        """Test that average width is returned with 2 decimal places."""
        predictions = [
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},  # width = 2000
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51001},  # width = 2001
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51002},  # width = 2002
        ]
        
        avg_width = compute_average_width(predictions)
        
        # Average = (2000 + 2001 + 2002) / 3 = 2001.0
        assert avg_width == 2001.0, f"Expected average width 2001.0, got {avg_width}"
    
    def test_average_width_empty_list_raises_error(self):
        """Test that empty predictions list raises ValueError."""
        predictions = []
        
        with pytest.raises(ValueError, match="predictions list is empty"):
            compute_average_width(predictions)
    
    def test_average_width_missing_fields_raises_error(self):
        """Test that missing required fields raises ValueError."""
        predictions = [
            {'actual_price': 50000, 'lower_bound': 49000},  # missing upper_bound
        ]
        
        with pytest.raises(ValueError, match="missing required fields"):
            compute_average_width(predictions)
    
    def test_average_width_single_prediction(self):
        """Test average width with single prediction."""
        predictions = [
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},
        ]
        
        avg_width = compute_average_width(predictions)
        
        assert avg_width == 2000.0, f"Expected average width 2000.0, got {avg_width}"
    
    def test_average_width_all_positive(self):
        """Test that all widths are positive."""
        predictions = [
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},
            {'actual_price': 51000, 'lower_bound': 50000, 'upper_bound': 52000},
            {'actual_price': 50500, 'lower_bound': 48000, 'upper_bound': 53000},
        ]
        
        avg_width = compute_average_width(predictions)
        
        assert avg_width > 0, f"Average width must be positive, got {avg_width}"
    
    def test_average_width_invalid_bounds_raises_error(self):
        """Test that invalid bounds (lower >= upper) raises ValueError."""
        predictions = [
            {'actual_price': 50000, 'lower_bound': 51000, 'upper_bound': 49000},  # lower > upper
        ]
        
        with pytest.raises(ValueError, match="non-positive width"):
            compute_average_width(predictions)


class TestComputeWinklerScore:
    """Test suite for compute_winkler_score function."""
    
    def test_winkler_score_all_inside(self):
        """Test Winkler score when all predictions have actual inside bounds."""
        predictions = [
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},  # width = 2000
            {'actual_price': 50500, 'lower_bound': 49500, 'upper_bound': 51500},  # width = 2000
        ]
        
        score = compute_winkler_score(predictions)
        
        # All inside, so score = average width = 2000
        assert score == 2000.0, f"Expected Winkler score 2000.0, got {score}"
    
    def test_winkler_score_below_lower_bound(self):
        """Test Winkler score when actual is below lower bound."""
        predictions = [
            {'actual_price': 48000, 'lower_bound': 49000, 'upper_bound': 51000},
        ]
        
        score = compute_winkler_score(predictions, alpha=0.05)
        
        # width = 2000, penalty = 40 * (49000 - 48000) = 40000
        # score = 2000 + 40000 = 42000
        assert score == 42000.0, f"Expected Winkler score 42000.0, got {score}"
    
    def test_winkler_score_above_upper_bound(self):
        """Test Winkler score when actual is above upper bound."""
        predictions = [
            {'actual_price': 52000, 'lower_bound': 49000, 'upper_bound': 51000},
        ]
        
        score = compute_winkler_score(predictions, alpha=0.05)
        
        # width = 2000, penalty = 40 * (52000 - 51000) = 40000
        # score = 2000 + 40000 = 42000
        assert score == 42000.0, f"Expected Winkler score 42000.0, got {score}"
    
    def test_winkler_score_mixed(self):
        """Test Winkler score with mixed predictions."""
        predictions = [
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},  # inside: 2000
            {'actual_price': 52000, 'lower_bound': 49000, 'upper_bound': 51000},  # above: 2000 + 40*1000 = 42000
            {'actual_price': 48000, 'lower_bound': 49000, 'upper_bound': 51000},  # below: 2000 + 40*1000 = 42000
        ]
        
        score = compute_winkler_score(predictions, alpha=0.05)
        
        # Average = (2000 + 42000 + 42000) / 3 = 28666.67
        assert score == 28666.67, f"Expected Winkler score 28666.67, got {score}"
    
    def test_winkler_score_two_decimal_places(self):
        """Test that Winkler score is returned with 2 decimal places."""
        predictions = [
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},
            {'actual_price': 52000, 'lower_bound': 49000, 'upper_bound': 51000},
        ]
        
        score = compute_winkler_score(predictions, alpha=0.05)
        
        # Should be rounded to 2 decimal places
        assert isinstance(score, float)
        # Check it has at most 2 decimal places
        assert score == round(score, 2)
    
    def test_winkler_score_non_negative(self):
        """Test that Winkler score is always non-negative."""
        test_cases = [
            [{'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000}],
            [{'actual_price': 52000, 'lower_bound': 49000, 'upper_bound': 51000}],
            [{'actual_price': 48000, 'lower_bound': 49000, 'upper_bound': 51000}],
        ]
        
        for predictions in test_cases:
            score = compute_winkler_score(predictions)
            assert score >= 0, f"Winkler score must be non-negative, got {score}"
    
    def test_winkler_score_empty_list_raises_error(self):
        """Test that empty predictions list raises ValueError."""
        predictions = []
        
        with pytest.raises(ValueError, match="predictions list is empty"):
            compute_winkler_score(predictions)
    
    def test_winkler_score_invalid_alpha_raises_error(self):
        """Test that invalid alpha parameter raises ValueError."""
        predictions = [
            {'actual_price': 50000, 'lower_bound': 49000, 'upper_bound': 51000},
        ]
        
        # Alpha = 0
        with pytest.raises(ValueError, match="Alpha parameter must be between 0 and 1"):
            compute_winkler_score(predictions, alpha=0.0)
        
        # Alpha = 1
        with pytest.raises(ValueError, match="Alpha parameter must be between 0 and 1"):
            compute_winkler_score(predictions, alpha=1.0)
        
        # Alpha < 0
        with pytest.raises(ValueError, match="Alpha parameter must be between 0 and 1"):
            compute_winkler_score(predictions, alpha=-0.1)
        
        # Alpha > 1
        with pytest.raises(ValueError, match="Alpha parameter must be between 0 and 1"):
            compute_winkler_score(predictions, alpha=1.5)
    
    def test_winkler_score_default_alpha(self):
        """Test that default alpha is 0.05."""
        predictions = [
            {'actual_price': 48000, 'lower_bound': 49000, 'upper_bound': 51000},
        ]
        
        # Call without alpha parameter (should use default 0.05)
        score = compute_winkler_score(predictions)
        
        # With alpha=0.05, penalty factor = 40
        # width = 2000, penalty = 40 * 1000 = 40000
        # score = 42000
        assert score == 42000.0, f"Expected Winkler score 42000.0 with default alpha, got {score}"
    
    def test_winkler_score_different_alpha(self):
        """Test Winkler score with different alpha values."""
        predictions = [
            {'actual_price': 48000, 'lower_bound': 49000, 'upper_bound': 51000},
        ]
        
        # Alpha = 0.1, penalty factor = 2/0.1 = 20
        score = compute_winkler_score(predictions, alpha=0.1)
        # width = 2000, penalty = 20 * 1000 = 20000, score = 22000
        assert score == 22000.0, f"Expected Winkler score 22000.0 with alpha=0.1, got {score}"
    
    def test_winkler_score_at_boundaries(self):
        """Test Winkler score when actual equals bounds."""
        predictions = [
            {'actual_price': 49000, 'lower_bound': 49000, 'upper_bound': 51000},  # at lower
            {'actual_price': 51000, 'lower_bound': 49000, 'upper_bound': 51000},  # at upper
        ]
        
        score = compute_winkler_score(predictions)
        
        # Both at boundaries count as inside, so score = average width = 2000
        assert score == 2000.0, f"Expected Winkler score 2000.0 for boundary cases, got {score}"


# ============================================================================
# Property-Based Tests
# ============================================================================

class TestCoverageMetricProperties:
    """
    Property-based tests for coverage metric using Hypothesis.
    
    These tests verify universal correctness properties that should hold
    for all valid prediction sets, not just specific examples.
    """
    
    @given(
        predictions=st.lists(
            st.fixed_dictionaries({
                'actual_price': st.floats(min_value=1.0, max_value=1000000.0),
                'lower_bound': st.floats(min_value=1.0, max_value=1000000.0),
                'upper_bound': st.floats(min_value=1.0, max_value=1000000.0)
            }),
            min_size=1,
            max_size=100
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_coverage_between_zero_and_one(self, predictions):
        """
        **Property: Coverage must be between 0 and 1**
        **Validates: Requirements 6.5**
        
        This property test verifies that for ANY valid set of predictions,
        the computed coverage metric is ALWAYS between 0.0 and 1.0 (inclusive).
        
        This is a critical correctness property because:
        1. Coverage is a fraction/proportion, which must be in [0, 1]
        2. Coverage = 0 means no predictions had actual inside bounds
        3. Coverage = 1 means all predictions had actual inside bounds
        4. Values outside [0, 1] indicate a bug in the calculation
        
        The test generates random prediction sets with diverse combinations of:
        - Actual prices that may be inside, outside, or at boundaries
        - Lower and upper bounds that may be valid or inverted
        - Various prediction set sizes (1 to 100 predictions)
        
        Note: Some generated predictions may have invalid bounds (lower > upper),
        but the coverage function should still return a value in [0, 1] or raise
        an appropriate error.
        """
        # Filter out predictions with invalid bounds (lower_bound > upper_bound)
        # The coverage function expects valid intervals
        valid_predictions = []
        for pred in predictions:
            # Ensure lower_bound <= upper_bound by swapping if necessary
            lower = min(pred['lower_bound'], pred['upper_bound'])
            upper = max(pred['lower_bound'], pred['upper_bound'])
            valid_predictions.append({
                'actual_price': pred['actual_price'],
                'lower_bound': lower,
                'upper_bound': upper
            })
        
        # Skip if no valid predictions (shouldn't happen with min_size=1, but be safe)
        if not valid_predictions:
            return
        
        # Compute coverage
        coverage = compute_coverage(valid_predictions)
        
        # Property: Coverage must be between 0 and 1 (inclusive)
        assert 0.0 <= coverage <= 1.0, (
            f"Coverage must be between 0.0 and 1.0, but got {coverage}. "
            f"This indicates a bug in the coverage calculation. "
            f"Prediction set size: {len(valid_predictions)}"
        )
        
        # Additional check: coverage must be a valid float
        assert isinstance(coverage, float), (
            f"Coverage must be a float, but got {type(coverage).__name__}"
        )
        
        # Additional check: coverage must be finite
        assert coverage == coverage, (  # NaN check (NaN != NaN)
            f"Coverage must not be NaN. "
            f"Prediction set size: {len(valid_predictions)}"
        )


class TestWinklerScoreProperties:
    """
    Property-based tests for Winkler score using Hypothesis.
    
    These tests verify universal correctness properties that should hold
    for all valid prediction sets, not just specific examples.
    """
    
    @given(
        predictions=st.lists(
            st.fixed_dictionaries({
                'actual_price': st.floats(min_value=1.0, max_value=1000000.0),
                'lower_bound': st.floats(min_value=1.0, max_value=1000000.0),
                'upper_bound': st.floats(min_value=1.0, max_value=1000000.0)
            }),
            min_size=1,
            max_size=100
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_winkler_score_non_negative(self, predictions):
        """
        **Property: Winkler score must be non-negative**
        **Validates: Requirements 8.6**
        
        This property test verifies that for ANY valid set of predictions,
        the computed Winkler score is ALWAYS non-negative (>= 0).
        
        This is a critical correctness property because:
        1. Winkler score is composed of interval width (always positive) plus penalties
        2. Penalties are always non-negative (absolute differences * positive factor)
        3. A negative score would indicate a fundamental bug in the calculation
        4. The score represents a "cost" or "loss" which cannot be negative
        
        The test generates random prediction sets with diverse combinations of:
        - Actual prices that may be inside, outside, or at boundaries
        - Lower and upper bounds that may be valid or inverted
        - Various prediction set sizes (1 to 100 predictions)
        - Different relationships between actual price and interval bounds
        
        The Winkler score formula guarantees non-negativity:
        - If actual inside: score = width (positive)
        - If actual below: score = width + 40*(lower - actual) (both positive)
        - If actual above: score = width + 40*(actual - upper) (both positive)
        
        Any negative score indicates a bug in the implementation.
        """
        # Filter out predictions with invalid bounds (lower_bound > upper_bound)
        # The Winkler score function expects valid intervals
        valid_predictions = []
        for pred in predictions:
            # Ensure lower_bound <= upper_bound by swapping if necessary
            lower = min(pred['lower_bound'], pred['upper_bound'])
            upper = max(pred['lower_bound'], pred['upper_bound'])
            valid_predictions.append({
                'actual_price': pred['actual_price'],
                'lower_bound': lower,
                'upper_bound': upper
            })
        
        # Skip if no valid predictions (shouldn't happen with min_size=1, but be safe)
        if not valid_predictions:
            return
        
        # Compute Winkler score
        winkler_score = compute_winkler_score(valid_predictions)
        
        # Property: Winkler score must be non-negative
        assert winkler_score >= 0.0, (
            f"Winkler score must be non-negative, but got {winkler_score}. "
            f"This indicates a bug in the Winkler score calculation. "
            f"Prediction set size: {len(valid_predictions)}"
        )
        
        # Additional check: Winkler score must be a valid float
        assert isinstance(winkler_score, float), (
            f"Winkler score must be a float, but got {type(winkler_score).__name__}"
        )
        
        # Additional check: Winkler score must be finite
        assert winkler_score == winkler_score, (  # NaN check (NaN != NaN)
            f"Winkler score must not be NaN. "
            f"Prediction set size: {len(valid_predictions)}"
        )
        
        # Additional check: Winkler score must not be infinite
        assert winkler_score != float('inf'), (
            f"Winkler score must be finite, but got infinity. "
            f"Prediction set size: {len(valid_predictions)}"
        )
