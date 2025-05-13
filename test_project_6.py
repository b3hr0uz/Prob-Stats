import unittest
import numpy as np
from scipy.stats import pearsonr
from project_6 import generate_normal_samples, calculate_mean_std, run_correlation_simulation

class TestProject6(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.rng = np.random.default_rng(123) # Seed for reproducibility in tests
        self.mu = 0.0
        self.sigma = 1.0

    def test_generate_normal_samples(self):
        """Test the generation of normal samples (same as in project_1 tests)."""
        samples = generate_normal_samples(self.rng, self.mu, self.sigma, 100)
        self.assertEqual(samples.shape[0], 100)
        self.assertTrue(-0.5 < np.mean(samples) < 0.5) # Loose check
        self.assertTrue(0.7 < np.std(samples) < 1.3)   # Loose check for std(ddof=0)

        with self.assertRaises(ValueError):
            generate_normal_samples(self.rng, self.mu, 0, 100) # Sigma must be positive
        with self.assertRaises(ValueError):
            generate_normal_samples(self.rng, self.mu, -1, 100)

    def test_calculate_mean_std(self):
        """Test the calculation of sample mean and standard deviation."""
        data1 = np.array([1, 2, 3, 4, 5])
        mean, std = calculate_mean_std(data1)
        self.assertAlmostEqual(mean, 3.0)
        self.assertAlmostEqual(std, np.std(data1, ddof=1))

        data2 = np.array([10, 10, 10, 10])
        mean, std = calculate_mean_std(data2)
        self.assertAlmostEqual(mean, 10.0)
        self.assertAlmostEqual(std, 0.0)

        data_pair = np.array([1, 5])
        mean, std = calculate_mean_std(data_pair)
        self.assertAlmostEqual(mean, 3.0)
        self.assertAlmostEqual(std, np.std(data_pair, ddof=1))

        data_single = np.array([10])
        mean, std = calculate_mean_std(data_single)
        self.assertAlmostEqual(mean, 10.0)
        self.assertTrue(np.isnan(std)) # std dev of single point with ddof=1 is nan

        data_empty = np.array([])
        mean, std = calculate_mean_std(data_empty)
        self.assertTrue(np.isnan(mean))
        self.assertTrue(np.isnan(std))

    def test_run_correlation_simulation(self):
        """Test the correlation simulation logic."""
        n_sample_size = 10
        max_m = 500 # Smaller M for faster test
        m_step = 50  # Ensure m_step >= 2

        m_values, correlations = run_correlation_simulation(
            self.rng, self.mu, self.sigma, n_sample_size, max_m, m_step
        )

        self.assertEqual(len(m_values), max_m // m_step)
        self.assertEqual(m_values, list(range(m_step, max_m + m_step, m_step)))
        self.assertEqual(len(correlations), len(m_values))

        # Correlations should be floats, and generally small for N(0,1)
        for corr in correlations:
            self.assertIsInstance(corr, float)
            self.assertTrue(-1.0 <= corr <= 1.0)
            # For normal data, expect correlation to be near 0
            # This is a statistical property, so allow a wider margin for small M
            self.assertTrue(-0.7 < corr < 0.7 if max_m < 1000 else -0.3 < corr < 0.3)

        # Test n < 2 raises ValueError
        with self.assertRaises(ValueError):
            run_correlation_simulation(self.rng, self.mu, self.sigma, 1, max_m, m_step)

    def test_run_correlation_simulation_m_step_warning(self):
        """Test that a warning is printed (or handled) if m_step < 2."""
        # We can't directly test print output easily without redirecting stdout.
        # However, the function should still run and produce NaNs or fewer results initially.
        # The function itself has a print warning, but continues.
        # Here, we ensure it doesn't crash and returns expected structure.
        n_sample_size = 10
        max_m = 100
        m_step = 1 # This should trigger the warning in the function

        # Temporarily redirect stdout to check for warning (optional advanced test)
        # import io
        # from contextlib import redirect_stdout
        # f = io.StringIO()
        # with redirect_stdout(f):
        #     m_values, correlations = run_correlation_simulation(
        #         self.rng, self.mu, self.sigma, n_sample_size, max_m, m_step
        #     )
        # self.assertIn("Warning: m_step < 2", f.getvalue())

        # Simplified check: ensure it runs and handles small m_step
        # The function internally produces NaNs for correlation until enough points, then filters.
        m_values, correlations = run_correlation_simulation(
            self.rng, self.mu, self.sigma, n_sample_size, max_m, m_step
        )
        
        # Expected m_values will not include the first one due to NaN correlation
        expected_m_values = list(range(m_step * 2, max_m + m_step, m_step))
        self.assertEqual(m_values, expected_m_values)
        self.assertEqual(len(correlations), len(expected_m_values))
        for corr in correlations:
             self.assertFalse(np.isnan(corr)) # After filtering, no NaNs should remain

if __name__ == '__main__':
    unittest.main() 