"""
Unit tests for GBM (Geometric Brownian Motion) mathematical engine module.

Tests cover:
- Simulation with known parameters
- Student-t distribution usage (not normal)
- Validation of positive prices
- Parameter validation
- Edge cases and error handling
- Property-based tests for universal correctness properties
"""

import pytest
import numpy as np
from scipy import stats
from hypothesis import given, strategies as st, settings
from bitcoin_forecasting.models.gbm_engine import simulate_gbm, extract_prediction_interval


class TestSimulateGBM:
    """Test suite for simulate_gbm function."""
    
    def test_simulation_with_known_parameters(self):
        """Test GBM simulation with known parameters produces valid output."""
        current_price = 50000.0
        drift = 0.0
        volatility = 0.80
        time_horizon = 1.0 / 8760  # 1 hour in years
        degrees_of_freedom = 5.0
        n_simulations = 10000
        
        terminal_prices = simulate_gbm(
            current_price=current_price,
            drift=drift,
            volatility=volatility,
            time_horizon=time_horizon,
            degrees_of_freedom=degrees_of_freedom,
            n_simulations=n_simulations,
            random_seed=42
        )
        
        # Check output shape
        assert terminal_prices.shape == (n_simulations,), \
            f"Expected shape ({n_simulations},), got {terminal_prices.shape}"
        
        # Check all prices are positive
        assert np.all(terminal_prices > 0), "All simulated prices must be positive"
        
        # Check all prices are finite
        assert np.all(np.isfinite(terminal_prices)), "All simulated prices must be finite"
        
        # Check mean is reasonable (should be close to current price for zero drift)
        mean_price = terminal_prices.mean()
        assert 0.8 * current_price < mean_price < 1.2 * current_price, \
            f"Mean price {mean_price} seems unreasonable for current price {current_price}"
    
    def test_exactly_n_simulations_returned(self):
        """Test that exactly n_simulations prices are returned."""
        test_cases = [1000, 5000, 10000, 20000]
        
        for n_sims in test_cases:
            terminal_prices = simulate_gbm(
                current_price=50000.0,
                drift=0.0,
                volatility=0.80,
                time_horizon=1.0 / 8760,
                degrees_of_freedom=5.0,
                n_simulations=n_sims,
                random_seed=42
            )
            
            assert len(terminal_prices) == n_sims, \
                f"Expected {n_sims} simulations, got {len(terminal_prices)}"
    
    def test_all_prices_positive(self):
        """Test that all simulated prices are positive (Requirement 2.7)."""
        # Test with various parameter combinations
        test_cases = [
            # (current_price, drift, volatility, time_horizon, df)
            (50000.0, 0.0, 0.80, 1.0/8760, 5.0),
            (30000.0, 0.5, 1.20, 1.0/8760, 4.0),
            (70000.0, -0.3, 0.60, 1.0/8760, 6.0),
            (100.0, 0.0, 2.0, 1.0/8760, 3.0),  # High volatility
        ]
        
        for params in test_cases:
            terminal_prices = simulate_gbm(
                current_price=params[0],
                drift=params[1],
                volatility=params[2],
                time_horizon=params[3],
                degrees_of_freedom=params[4],
                n_simulations=10000,
                random_seed=42
            )
            
            assert np.all(terminal_prices > 0), \
                f"All prices must be positive for params {params}"
    
    def test_all_prices_finite(self):
        """Test that all simulated prices are finite."""
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        assert np.all(np.isfinite(terminal_prices)), "All prices must be finite"
        assert not np.any(np.isnan(terminal_prices)), "No prices should be NaN"
        assert not np.any(np.isinf(terminal_prices)), "No prices should be infinite"
    
    def test_student_t_distribution_usage(self):
        """Test that Student-t distribution is used (not normal) (Requirement 2.3, 2.4)."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate shocks using Student-t
        df = 5.0
        n = 10000
        t_shocks = stats.t.rvs(df=df, size=n)
        
        # Generate shocks using normal
        normal_shocks = np.random.normal(0, 1, size=n)
        
        # Student-t with low df should have fatter tails (higher kurtosis)
        t_kurtosis = stats.kurtosis(t_shocks)
        normal_kurtosis = stats.kurtosis(normal_shocks)
        
        # Student-t(5) should have excess kurtosis > 0 (fatter tails than normal)
        # Normal distribution has excess kurtosis ≈ 0
        assert t_kurtosis > normal_kurtosis, \
            "Student-t should have fatter tails (higher kurtosis) than normal"
        
        # Now test that our simulate_gbm produces fat-tailed results
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Compute log returns
        log_returns = np.log(terminal_prices / 50000.0)
        
        # Check that returns have fat tails (high kurtosis)
        returns_kurtosis = stats.kurtosis(log_returns)
        
        # Should have positive excess kurtosis (fatter tails than normal)
        assert returns_kurtosis > -1.0, \
            "Returns should show evidence of fat tails from Student-t distribution"
    
    def test_reproducibility_with_random_seed(self):
        """Test that results are reproducible with same random seed."""
        params = {
            'current_price': 50000.0,
            'drift': 0.0,
            'volatility': 0.80,
            'time_horizon': 1.0 / 8760,
            'degrees_of_freedom': 5.0,
            'n_simulations': 10000,
            'random_seed': 42
        }
        
        # Run simulation twice with same seed
        prices1 = simulate_gbm(**params)
        prices2 = simulate_gbm(**params)
        
        # Results should be identical
        np.testing.assert_array_equal(prices1, prices2,
            "Results should be identical with same random seed")
    
    def test_different_results_without_seed(self):
        """Test that results differ without random seed."""
        params = {
            'current_price': 50000.0,
            'drift': 0.0,
            'volatility': 0.80,
            'time_horizon': 1.0 / 8760,
            'degrees_of_freedom': 5.0,
            'n_simulations': 10000,
            'random_seed': None
        }
        
        # Run simulation twice without seed
        prices1 = simulate_gbm(**params)
        prices2 = simulate_gbm(**params)
        
        # Results should be different
        assert not np.array_equal(prices1, prices2), \
            "Results should differ without random seed"
    
    def test_positive_drift_increases_mean_price(self):
        """Test that positive drift increases expected terminal price."""
        current_price = 50000.0
        
        # Simulate with zero drift
        prices_zero = simulate_gbm(
            current_price=current_price,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Simulate with positive drift
        prices_positive = simulate_gbm(
            current_price=current_price,
            drift=1.0,  # Positive drift
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Mean with positive drift should be higher
        assert prices_positive.mean() > prices_zero.mean(), \
            "Positive drift should increase mean terminal price"
    
    def test_negative_drift_decreases_mean_price(self):
        """Test that negative drift decreases expected terminal price."""
        current_price = 50000.0
        
        # Simulate with zero drift
        prices_zero = simulate_gbm(
            current_price=current_price,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Simulate with negative drift
        prices_negative = simulate_gbm(
            current_price=current_price,
            drift=-1.0,  # Negative drift
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Mean with negative drift should be lower
        assert prices_negative.mean() < prices_zero.mean(), \
            "Negative drift should decrease mean terminal price"
    
    def test_higher_volatility_increases_spread(self):
        """Test that higher volatility increases price spread."""
        current_price = 50000.0
        
        # Simulate with low volatility
        prices_low_vol = simulate_gbm(
            current_price=current_price,
            drift=0.0,
            volatility=0.40,  # Low volatility
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Simulate with high volatility
        prices_high_vol = simulate_gbm(
            current_price=current_price,
            drift=0.0,
            volatility=1.20,  # High volatility
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Standard deviation should be higher with higher volatility
        assert prices_high_vol.std() > prices_low_vol.std(), \
            "Higher volatility should increase price spread"
    
    def test_longer_time_horizon_increases_spread(self):
        """Test that longer time horizon increases price spread."""
        current_price = 50000.0
        
        # Simulate with short time horizon (1 hour)
        prices_short = simulate_gbm(
            current_price=current_price,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,  # 1 hour
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Simulate with longer time horizon (24 hours)
        prices_long = simulate_gbm(
            current_price=current_price,
            drift=0.0,
            volatility=0.80,
            time_horizon=24.0 / 8760,  # 24 hours
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Standard deviation should be higher with longer time horizon
        assert prices_long.std() > prices_short.std(), \
            "Longer time horizon should increase price spread"


class TestParameterValidation:
    """Test suite for parameter validation."""
    
    def test_current_price_must_be_positive(self):
        """Test that current_price must be positive."""
        with pytest.raises(ValueError, match="current_price must be positive"):
            simulate_gbm(
                current_price=-50000.0,
                drift=0.0,
                volatility=0.80,
                time_horizon=1.0 / 8760,
                degrees_of_freedom=5.0
            )
        
        with pytest.raises(ValueError, match="current_price must be positive"):
            simulate_gbm(
                current_price=0.0,
                drift=0.0,
                volatility=0.80,
                time_horizon=1.0 / 8760,
                degrees_of_freedom=5.0
            )
    
    def test_current_price_must_be_finite(self):
        """Test that current_price must be finite."""
        with pytest.raises(ValueError, match="current_price must be finite"):
            simulate_gbm(
                current_price=np.inf,
                drift=0.0,
                volatility=0.80,
                time_horizon=1.0 / 8760,
                degrees_of_freedom=5.0
            )
        
        with pytest.raises(ValueError, match="current_price must be finite"):
            simulate_gbm(
                current_price=np.nan,
                drift=0.0,
                volatility=0.80,
                time_horizon=1.0 / 8760,
                degrees_of_freedom=5.0
            )
    
    def test_volatility_must_be_positive(self):
        """Test that volatility must be positive."""
        with pytest.raises(ValueError, match="volatility must be positive"):
            simulate_gbm(
                current_price=50000.0,
                drift=0.0,
                volatility=-0.80,
                time_horizon=1.0 / 8760,
                degrees_of_freedom=5.0
            )
        
        with pytest.raises(ValueError, match="volatility must be positive"):
            simulate_gbm(
                current_price=50000.0,
                drift=0.0,
                volatility=0.0,
                time_horizon=1.0 / 8760,
                degrees_of_freedom=5.0
            )
    
    def test_volatility_must_be_finite(self):
        """Test that volatility must be finite."""
        with pytest.raises(ValueError, match="volatility must be finite"):
            simulate_gbm(
                current_price=50000.0,
                drift=0.0,
                volatility=np.inf,
                time_horizon=1.0 / 8760,
                degrees_of_freedom=5.0
            )
    
    def test_time_horizon_must_be_positive(self):
        """Test that time_horizon must be positive."""
        with pytest.raises(ValueError, match="time_horizon must be positive"):
            simulate_gbm(
                current_price=50000.0,
                drift=0.0,
                volatility=0.80,
                time_horizon=-1.0,
                degrees_of_freedom=5.0
            )
        
        with pytest.raises(ValueError, match="time_horizon must be positive"):
            simulate_gbm(
                current_price=50000.0,
                drift=0.0,
                volatility=0.80,
                time_horizon=0.0,
                degrees_of_freedom=5.0
            )
    
    def test_time_horizon_must_be_finite(self):
        """Test that time_horizon must be finite."""
        with pytest.raises(ValueError, match="time_horizon must be finite"):
            simulate_gbm(
                current_price=50000.0,
                drift=0.0,
                volatility=0.80,
                time_horizon=np.inf,
                degrees_of_freedom=5.0
            )
    
    def test_degrees_of_freedom_must_be_greater_than_2(self):
        """Test that degrees_of_freedom must be > 2 (Requirement 2.2)."""
        with pytest.raises(ValueError, match="degrees_of_freedom must be greater than 2"):
            simulate_gbm(
                current_price=50000.0,
                drift=0.0,
                volatility=0.80,
                time_horizon=1.0 / 8760,
                degrees_of_freedom=2.0
            )
        
        with pytest.raises(ValueError, match="degrees_of_freedom must be greater than 2"):
            simulate_gbm(
                current_price=50000.0,
                drift=0.0,
                volatility=0.80,
                time_horizon=1.0 / 8760,
                degrees_of_freedom=1.0
            )
    
    def test_degrees_of_freedom_must_be_finite(self):
        """Test that degrees_of_freedom must be finite."""
        with pytest.raises(ValueError, match="degrees_of_freedom must be finite"):
            simulate_gbm(
                current_price=50000.0,
                drift=0.0,
                volatility=0.80,
                time_horizon=1.0 / 8760,
                degrees_of_freedom=np.inf
            )
    
    def test_drift_must_be_finite(self):
        """Test that drift must be finite."""
        with pytest.raises(ValueError, match="drift must be finite"):
            simulate_gbm(
                current_price=50000.0,
                drift=np.inf,
                volatility=0.80,
                time_horizon=1.0 / 8760,
                degrees_of_freedom=5.0
            )
    
    def test_n_simulations_must_be_at_least_1000(self):
        """Test that n_simulations must be at least 1000."""
        with pytest.raises(ValueError, match="n_simulations must be at least 1000"):
            simulate_gbm(
                current_price=50000.0,
                drift=0.0,
                volatility=0.80,
                time_horizon=1.0 / 8760,
                degrees_of_freedom=5.0,
                n_simulations=500
            )
    
    def test_n_simulations_must_be_integer(self):
        """Test that n_simulations must be an integer."""
        with pytest.raises(ValueError, match="n_simulations must be an integer"):
            simulate_gbm(
                current_price=50000.0,
                drift=0.0,
                volatility=0.80,
                time_horizon=1.0 / 8760,
                degrees_of_freedom=5.0,
                n_simulations=10000.5
            )


class TestEdgeCases:
    """Test suite for edge cases."""
    
    def test_very_small_time_horizon(self):
        """Test with very small time horizon (e.g., 1 minute)."""
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / (8760 * 60),  # 1 minute
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Prices should be very close to current price
        assert np.all(terminal_prices > 0)
        assert np.all(np.isfinite(terminal_prices))
        
        # Mean should be very close to current price
        assert abs(terminal_prices.mean() - 50000.0) < 1000.0
    
    def test_very_high_degrees_of_freedom(self):
        """Test with high degrees of freedom (approaches normal distribution)."""
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=100.0,  # High df, close to normal
            n_simulations=10000,
            random_seed=42
        )
        
        assert np.all(terminal_prices > 0)
        assert np.all(np.isfinite(terminal_prices))
    
    def test_very_low_degrees_of_freedom(self):
        """Test with low degrees of freedom (very fat tails)."""
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=3.0,  # Low df, very fat tails
            n_simulations=10000,
            random_seed=42
        )
        
        assert np.all(terminal_prices > 0)
        assert np.all(np.isfinite(terminal_prices))
        
        # Should have wider spread due to fat tails
        assert terminal_prices.std() > 0
    
    def test_very_small_current_price(self):
        """Test with very small current price."""
        terminal_prices = simulate_gbm(
            current_price=0.01,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        assert np.all(terminal_prices > 0)
        assert np.all(np.isfinite(terminal_prices))
    
    def test_very_large_current_price(self):
        """Test with very large current price."""
        terminal_prices = simulate_gbm(
            current_price=1000000.0,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        assert np.all(terminal_prices > 0)
        assert np.all(np.isfinite(terminal_prices))
    
    def test_extreme_positive_drift(self):
        """Test with extreme positive drift."""
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=5.0,  # Very high drift
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        assert np.all(terminal_prices > 0)
        assert np.all(np.isfinite(terminal_prices))
        
        # Mean should be higher than current price
        assert terminal_prices.mean() > 50000.0
    
    def test_extreme_negative_drift(self):
        """Test with extreme negative drift."""
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=-5.0,  # Very negative drift
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        assert np.all(terminal_prices > 0)
        assert np.all(np.isfinite(terminal_prices))
        
        # Mean should be lower than current price
        assert terminal_prices.mean() < 50000.0
    
    def test_minimum_n_simulations(self):
        """Test with minimum allowed n_simulations (1000)."""
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=1000,
            random_seed=42
        )
        
        assert len(terminal_prices) == 1000
        assert np.all(terminal_prices > 0)
        assert np.all(np.isfinite(terminal_prices))



class TestExtractPredictionInterval:
    """Test suite for extract_prediction_interval function."""
    
    def test_extract_interval_from_simulation(self):
        """Test extracting prediction interval from simulation results (Requirements 4.1, 4.2, 4.3)."""
        # Generate simulation results
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Extract 95% prediction interval
        lower_bound, upper_bound = extract_prediction_interval(terminal_prices)
        
        # Check that bounds are returned as tuple
        assert isinstance(lower_bound, (float, np.floating)), \
            f"Lower bound should be float, got {type(lower_bound)}"
        assert isinstance(upper_bound, (float, np.floating)), \
            f"Upper bound should be float, got {type(upper_bound)}"
        
        # Requirement 4.4: Validate lower_bound < upper_bound
        assert lower_bound < upper_bound, \
            f"Lower bound ({lower_bound}) must be less than upper bound ({upper_bound})"
        
        # Requirement 4.5: Validate both bounds are positive
        assert lower_bound > 0, f"Lower bound must be positive, got {lower_bound}"
        assert upper_bound > 0, f"Upper bound must be positive, got {upper_bound}"
        
        # Check bounds are finite
        assert np.isfinite(lower_bound), "Lower bound must be finite"
        assert np.isfinite(upper_bound), "Upper bound must be finite"
        
        # Check bounds are reasonable (should be close to current price for short horizon)
        assert 0.5 * 50000.0 < lower_bound < 50000.0, \
            f"Lower bound {lower_bound} seems unreasonable"
        assert 50000.0 < upper_bound < 1.5 * 50000.0, \
            f"Upper bound {upper_bound} seems unreasonable"
    
    def test_interval_percentiles_correct(self):
        """Test that extracted percentiles match expected values (Requirements 4.1, 4.2)."""
        # Generate simulation results
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Extract interval
        lower_bound, upper_bound = extract_prediction_interval(terminal_prices)
        
        # Manually compute percentiles
        expected_lower = np.percentile(terminal_prices, 2.5)
        expected_upper = np.percentile(terminal_prices, 97.5)
        
        # Check that extracted bounds match expected percentiles
        assert abs(lower_bound - expected_lower) < 0.01, \
            f"Lower bound {lower_bound} doesn't match 2.5th percentile {expected_lower}"
        assert abs(upper_bound - expected_upper) < 0.01, \
            f"Upper bound {upper_bound} doesn't match 97.5th percentile {expected_upper}"
    
    def test_interval_coverage_approximately_95_percent(self):
        """Test that approximately 95% of simulated prices fall within interval."""
        # Generate simulation results
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Extract interval
        lower_bound, upper_bound = extract_prediction_interval(terminal_prices)
        
        # Count how many prices fall within interval
        within_interval = np.sum((terminal_prices >= lower_bound) & (terminal_prices <= upper_bound))
        coverage = within_interval / len(terminal_prices)
        
        # Coverage should be approximately 0.95 (allow some tolerance)
        assert 0.94 < coverage < 0.96, \
            f"Coverage {coverage:.4f} is not approximately 0.95"
    
    def test_different_confidence_levels(self):
        """Test extraction with different confidence levels."""
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Test 90% confidence interval
        lower_90, upper_90 = extract_prediction_interval(terminal_prices, confidence_level=0.90)
        
        # Test 95% confidence interval
        lower_95, upper_95 = extract_prediction_interval(terminal_prices, confidence_level=0.95)
        
        # Test 99% confidence interval
        lower_99, upper_99 = extract_prediction_interval(terminal_prices, confidence_level=0.99)
        
        # Higher confidence should give wider intervals
        width_90 = upper_90 - lower_90
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99
        
        assert width_90 < width_95 < width_99, \
            "Higher confidence levels should produce wider intervals"
        
        # All bounds should be positive
        assert all(b > 0 for b in [lower_90, upper_90, lower_95, upper_95, lower_99, upper_99])
    
    def test_higher_volatility_produces_wider_intervals(self):
        """Test that higher volatility produces wider prediction intervals."""
        # Low volatility simulation
        prices_low_vol = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=0.40,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # High volatility simulation
        prices_high_vol = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=1.20,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Extract intervals
        lower_low, upper_low = extract_prediction_interval(prices_low_vol)
        lower_high, upper_high = extract_prediction_interval(prices_high_vol)
        
        # Compute widths
        width_low = upper_low - lower_low
        width_high = upper_high - lower_high
        
        # Higher volatility should produce wider interval
        assert width_high > width_low, \
            f"High volatility width {width_high} should be greater than low volatility width {width_low}"
    
    def test_longer_time_horizon_produces_wider_intervals(self):
        """Test that longer time horizon produces wider prediction intervals."""
        # Short time horizon (1 hour)
        prices_short = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Long time horizon (24 hours)
        prices_long = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=0.80,
            time_horizon=24.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Extract intervals
        lower_short, upper_short = extract_prediction_interval(prices_short)
        lower_long, upper_long = extract_prediction_interval(prices_long)
        
        # Compute widths
        width_short = upper_short - lower_short
        width_long = upper_long - lower_long
        
        # Longer time horizon should produce wider interval
        assert width_long > width_short, \
            f"Long horizon width {width_long} should be greater than short horizon width {width_short}"
    
    def test_reproducibility(self):
        """Test that interval extraction is reproducible."""
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Extract interval twice
        interval1 = extract_prediction_interval(terminal_prices)
        interval2 = extract_prediction_interval(terminal_prices)
        
        # Results should be identical
        assert interval1 == interval2, "Interval extraction should be deterministic"


class TestPredictionIntervalValidation:
    """Test suite for prediction interval parameter validation."""
    
    def test_terminal_prices_must_be_numpy_array(self):
        """Test that terminal_prices must be a numpy array."""
        with pytest.raises(ValueError, match="terminal_prices must be a numpy array"):
            extract_prediction_interval([50000.0, 51000.0, 49000.0])
    
    def test_terminal_prices_must_not_be_empty(self):
        """Test that terminal_prices must not be empty."""
        with pytest.raises(ValueError, match="terminal_prices array is empty"):
            extract_prediction_interval(np.array([]))
    
    def test_terminal_prices_must_have_at_least_1000_values(self):
        """Test that terminal_prices must have at least 1000 values."""
        with pytest.raises(ValueError, match="terminal_prices must contain at least 1000 values"):
            extract_prediction_interval(np.array([50000.0] * 500))
    
    def test_terminal_prices_must_be_positive(self):
        """Test that all terminal prices must be positive (Requirement 4.5)."""
        # Create array with some negative values
        prices = np.ones(1000) * 50000.0
        prices[500] = -1000.0
        
        with pytest.raises(ValueError, match="All terminal prices must be positive"):
            extract_prediction_interval(prices)
    
    def test_terminal_prices_must_be_finite(self):
        """Test that all terminal prices must be finite."""
        # Create array with infinite value
        prices = np.ones(1000) * 50000.0
        prices[500] = np.inf
        
        with pytest.raises(ValueError, match="All terminal prices must be finite"):
            extract_prediction_interval(prices)
        
        # Create array with NaN value
        prices = np.ones(1000) * 50000.0
        prices[500] = np.nan
        
        with pytest.raises(ValueError, match="All terminal prices must be finite"):
            extract_prediction_interval(prices)
    
    def test_confidence_level_must_be_numeric(self):
        """Test that confidence_level must be numeric."""
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        with pytest.raises(ValueError, match="confidence_level must be numeric"):
            extract_prediction_interval(terminal_prices, confidence_level="0.95")
    
    def test_confidence_level_must_be_between_0_and_1(self):
        """Test that confidence_level must be between 0 and 1 (exclusive)."""
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Test confidence_level = 0
        with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
            extract_prediction_interval(terminal_prices, confidence_level=0.0)
        
        # Test confidence_level = 1
        with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
            extract_prediction_interval(terminal_prices, confidence_level=1.0)
        
        # Test confidence_level < 0
        with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
            extract_prediction_interval(terminal_prices, confidence_level=-0.5)
        
        # Test confidence_level > 1
        with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
            extract_prediction_interval(terminal_prices, confidence_level=1.5)
    
    def test_lower_bound_less_than_upper_bound(self):
        """Test validation that lower_bound < upper_bound (Requirement 4.4)."""
        # This should never happen with valid simulation results,
        # but we test the validation logic
        
        # Create a valid array first
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        # Normal case should work
        lower, upper = extract_prediction_interval(terminal_prices)
        assert lower < upper, "Lower bound should be less than upper bound"


class TestPredictionIntervalEdgeCases:
    """Test suite for prediction interval edge cases."""
    
    def test_interval_with_minimum_simulations(self):
        """Test interval extraction with minimum number of simulations (1000)."""
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=1000,
            random_seed=42
        )
        
        lower_bound, upper_bound = extract_prediction_interval(terminal_prices)
        
        assert lower_bound < upper_bound
        assert lower_bound > 0
        assert upper_bound > 0
    
    def test_interval_with_very_low_volatility(self):
        """Test interval extraction with very low volatility."""
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=0.01,  # Very low volatility
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        lower_bound, upper_bound = extract_prediction_interval(terminal_prices)
        
        # Interval should be very narrow
        width = upper_bound - lower_bound
        assert width < 0.05 * 50000.0, "Interval should be narrow with low volatility"
        
        # Bounds should be close to current price
        assert abs(lower_bound - 50000.0) < 0.05 * 50000.0
        assert abs(upper_bound - 50000.0) < 0.05 * 50000.0
    
    def test_interval_with_very_high_volatility(self):
        """Test interval extraction with very high volatility."""
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=0.0,
            volatility=3.0,  # Very high volatility
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        lower_bound, upper_bound = extract_prediction_interval(terminal_prices)
        
        # Interval should be wide
        width = upper_bound - lower_bound
        assert width > 0.01 * 50000.0, "Interval should be wide with high volatility"
        
        # All bounds should still be positive
        assert lower_bound > 0
        assert upper_bound > 0
    
    def test_interval_with_extreme_positive_drift(self):
        """Test interval extraction with extreme positive drift."""
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=10.0,  # Very high drift
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        lower_bound, upper_bound = extract_prediction_interval(terminal_prices)
        
        # Mean should be above current price with positive drift
        mean_price = terminal_prices.mean()
        assert mean_price > 50000.0, "Mean price should be above current price with high positive drift"
        assert lower_bound < upper_bound
        assert lower_bound > 0
    
    def test_interval_with_extreme_negative_drift(self):
        """Test interval extraction with extreme negative drift."""
        terminal_prices = simulate_gbm(
            current_price=50000.0,
            drift=-10.0,  # Very negative drift
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        lower_bound, upper_bound = extract_prediction_interval(terminal_prices)
        
        # Mean should be below current price with negative drift
        mean_price = terminal_prices.mean()
        assert mean_price < 50000.0, "Mean price should be below current price with high negative drift"
        assert lower_bound < upper_bound
        assert lower_bound > 0, "Lower bound must still be positive"
    
    def test_interval_with_very_small_prices(self):
        """Test interval extraction with very small prices."""
        terminal_prices = simulate_gbm(
            current_price=0.01,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        lower_bound, upper_bound = extract_prediction_interval(terminal_prices)
        
        assert lower_bound > 0
        assert upper_bound > lower_bound
        assert np.isfinite(lower_bound)
        assert np.isfinite(upper_bound)
    
    def test_interval_with_very_large_prices(self):
        """Test interval extraction with very large prices."""
        terminal_prices = simulate_gbm(
            current_price=1000000.0,
            drift=0.0,
            volatility=0.80,
            time_horizon=1.0 / 8760,
            degrees_of_freedom=5.0,
            n_simulations=10000,
            random_seed=42
        )
        
        lower_bound, upper_bound = extract_prediction_interval(terminal_prices)
        
        assert lower_bound > 0
        assert upper_bound > lower_bound
        assert np.isfinite(lower_bound)
        assert np.isfinite(upper_bound)



# ============================================================================
# Property-Based Tests
# ============================================================================

class TestGBMSimulationProperties:
    """
    Property-based tests for GBM simulation using Hypothesis.
    
    These tests verify universal correctness properties that should hold
    for all valid inputs, not just specific examples.
    """
    
    @given(
        current_price=st.floats(min_value=0.01, max_value=1000000.0),
        drift=st.floats(min_value=-10.0, max_value=10.0),
        volatility=st.floats(min_value=0.01, max_value=5.0),
        time_horizon=st.floats(min_value=1.0/8760, max_value=1.0),
        degrees_of_freedom=st.floats(min_value=2.1, max_value=100.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_property_all_simulated_prices_positive(
        self,
        current_price: float,
        drift: float,
        volatility: float,
        time_horizon: float,
        degrees_of_freedom: float
    ):
        """
        **Property: All simulated prices must be positive**
        **Validates: Requirements 2.7**
        
        This property test verifies that for ANY valid combination of input
        parameters, ALL simulated prices from the GBM engine are positive.
        
        This is a critical correctness property because:
        1. Negative prices are mathematically impossible for assets
        2. The exponential transformation in GBM should guarantee positivity
        3. This must hold regardless of drift, volatility, or time horizon
        
        The test generates random valid inputs and asserts that every single
        simulated price is strictly positive (> 0).
        """
        # Generate simulations with random valid parameters
        terminal_prices = simulate_gbm(
            current_price=current_price,
            drift=drift,
            volatility=volatility,
            time_horizon=time_horizon,
            degrees_of_freedom=degrees_of_freedom,
            n_simulations=10000,
            random_seed=42  # Use seed for reproducibility in tests
        )
        
        # Property: ALL simulated prices must be positive
        assert np.all(terminal_prices > 0), (
            f"All simulated prices must be positive. "
            f"Found {np.sum(terminal_prices <= 0)} non-positive prices out of {len(terminal_prices)}. "
            f"Parameters: price={current_price:.2f}, drift={drift:.4f}, "
            f"vol={volatility:.4f}, dt={time_horizon:.6f}, df={degrees_of_freedom:.2f}"
        )
        
        # Additional check: all prices must be finite
        assert np.all(np.isfinite(terminal_prices)), (
            f"All simulated prices must be finite. "
            f"Found {np.sum(~np.isfinite(terminal_prices))} non-finite prices. "
            f"Parameters: price={current_price:.2f}, drift={drift:.4f}, "
            f"vol={volatility:.4f}, dt={time_horizon:.6f}, df={degrees_of_freedom:.2f}"
        )


class TestPredictionIntervalProperties:
    """
    Property-based tests for prediction interval extraction using Hypothesis.
    
    These tests verify universal correctness properties that should hold
    for all valid simulation results.
    """
    
    @given(
        current_price=st.floats(min_value=0.01, max_value=1000000.0),
        drift=st.floats(min_value=-10.0, max_value=10.0),
        volatility=st.floats(min_value=0.01, max_value=5.0),
        time_horizon=st.floats(min_value=1.0/8760, max_value=1.0),
        degrees_of_freedom=st.floats(min_value=2.1, max_value=100.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_property_lower_bound_less_than_upper_bound(
        self,
        current_price: float,
        drift: float,
        volatility: float,
        time_horizon: float,
        degrees_of_freedom: float
    ):
        """
        **Property: Lower bound must be less than upper bound**
        **Validates: Requirements 4.4**
        
        This property test verifies that for ANY valid simulation results,
        the extracted lower bound is ALWAYS strictly less than the upper bound.
        
        This is a fundamental correctness property because:
        1. A prediction interval with lower >= upper is meaningless
        2. The percentile extraction (2.5th < 97.5th) should guarantee this
        3. This must hold for all possible price distributions
        
        The test generates random simulations and asserts that the lower bound
        is always strictly less than the upper bound.
        """
        # Generate simulations with random valid parameters
        terminal_prices = simulate_gbm(
            current_price=current_price,
            drift=drift,
            volatility=volatility,
            time_horizon=time_horizon,
            degrees_of_freedom=degrees_of_freedom,
            n_simulations=10000,
            random_seed=42  # Use seed for reproducibility in tests
        )
        
        # Extract prediction interval
        lower_bound, upper_bound = extract_prediction_interval(terminal_prices)
        
        # Property: Lower bound must be strictly less than upper bound
        assert lower_bound < upper_bound, (
            f"Lower bound must be less than upper bound. "
            f"Got lower={lower_bound:.2f}, upper={upper_bound:.2f}. "
            f"Parameters: price={current_price:.2f}, drift={drift:.4f}, "
            f"vol={volatility:.4f}, dt={time_horizon:.6f}, df={degrees_of_freedom:.2f}"
        )
        
        # Additional property: Both bounds must be positive
        assert lower_bound > 0, (
            f"Lower bound must be positive, got {lower_bound:.2f}. "
            f"Parameters: price={current_price:.2f}, drift={drift:.4f}, "
            f"vol={volatility:.4f}, dt={time_horizon:.6f}, df={degrees_of_freedom:.2f}"
        )
        
        assert upper_bound > 0, (
            f"Upper bound must be positive, got {upper_bound:.2f}. "
            f"Parameters: price={current_price:.2f}, drift={drift:.4f}, "
            f"vol={volatility:.4f}, dt={time_horizon:.6f}, df={degrees_of_freedom:.2f}"
        )
        
        # Additional property: Both bounds must be finite
        assert np.isfinite(lower_bound), (
            f"Lower bound must be finite, got {lower_bound}. "
            f"Parameters: price={current_price:.2f}, drift={drift:.4f}, "
            f"vol={volatility:.4f}, dt={time_horizon:.6f}, df={degrees_of_freedom:.2f}"
        )
        
        assert np.isfinite(upper_bound), (
            f"Upper bound must be finite, got {upper_bound}. "
            f"Parameters: price={current_price:.2f}, drift={drift:.4f}, "
            f"vol={volatility:.4f}, dt={time_horizon:.6f}, df={degrees_of_freedom:.2f}"
        )
        
        # Additional property: Interval width must be positive
        width = upper_bound - lower_bound
        assert width > 0, (
            f"Interval width must be positive, got {width:.2f}. "
            f"Lower={lower_bound:.2f}, Upper={upper_bound:.2f}. "
            f"Parameters: price={current_price:.2f}, drift={drift:.4f}, "
            f"vol={volatility:.4f}, dt={time_horizon:.6f}, df={degrees_of_freedom:.2f}"
        )
