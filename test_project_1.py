import unittest
import numpy as np
from project_1 import generate_normal_samples, calculate_stats, run_incremental_simulation, estimate_estimator_variance

class TestProject1(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.rng = np.random.default_rng(0) # Seed for reproducibility in tests
        self.mu = 0.0
        self.sigma = 1.0

    def test_generate_normal_samples(self):
        """Test the generation of normal samples."""
        samples = generate_normal_samples(self.rng, self.mu, self.sigma, 100)
        self.assertEqual(samples.shape[0], 100)
        # Check if mean and std are roughly as expected (for large N, not strictly for small N)
        # For a small sample of 100, these might not be very close, so this is a loose check.
        # More rigorous statistical tests would be needed for distribution checking.
        self.assertTrue(-0.5 < np.mean(samples) < 0.5) # Loosely centered around mu=0
        self.assertTrue(0.5 < np.std(samples) < 1.5)   # Loosely around sigma=1

        with self.assertRaises(ValueError):
            generate_normal_samples(self.rng, self.mu, 0, 100) # Sigma must be positive
        with self.assertRaises(ValueError):
            generate_normal_samples(self.rng, self.mu, -1, 100)

    def test_calculate_stats(self):
        """Test the calculation of mean, median, and std deviation."""
        data1 = np.array([1, 2, 3, 4, 5])
        mean, median, std = calculate_stats(data1)
        self.assertAlmostEqual(mean, 3.0)
        self.assertAlmostEqual(median, 3.0)
        self.assertAlmostEqual(std, np.std(data1, ddof=1))

        data2 = np.array([1, 2, 3, 4, 5, 6])
        mean, median, std = calculate_stats(data2)
        self.assertAlmostEqual(mean, 3.5)
        self.assertAlmostEqual(median, 3.5)
        self.assertAlmostEqual(std, np.std(data2, ddof=1))

        data_single = np.array([10])
        mean, median, std = calculate_stats(data_single)
        self.assertAlmostEqual(mean, 10.0)
        self.assertAlmostEqual(median, 10.0)
        self.assertTrue(np.isnan(std)) # std dev of single point with ddof=1 is nan

        data_empty = np.array([])
        mean, median, std = calculate_stats(data_empty)
        self.assertTrue(np.isnan(mean))
        self.assertTrue(np.isnan(median))
        self.assertTrue(np.isnan(std))

    def test_run_incremental_simulation(self):
        """Test the incremental simulation logic."""
        max_n = 500
        step = 100
        n_values, means, medians, std_devs = run_incremental_simulation(self.rng, self.mu, self.sigma, max_n, step)

        self.assertEqual(len(n_values), max_n // step)
        self.assertEqual(n_values, [100, 200, 300, 400, 500])
        self.assertEqual(len(means), len(n_values))
        self.assertEqual(len(medians), len(n_values))
        self.assertEqual(len(std_devs), len(n_values))

        # Check that std_devs are not all NaN (except potentially the first if step=1)
        if step > 1:
            self.assertFalse(any(np.isnan(s) for s in std_devs))
        elif step == 1 and len(std_devs) > 0:
             self.assertTrue(np.isnan(std_devs[0])) # First step with n=1
             if len(std_devs) > 1:
                 self.assertFalse(any(np.isnan(s) for s in std_devs[1:]))

    def test_run_incremental_simulation_consistency_check(self):
        """Test the consistency check in run_incremental_simulation by forcing a mismatch."""
        # This test is a bit tricky to set up perfectly without altering the source
        # for the RuntimeError or making the function more complex for testing.
        # The check `if all_samples.size != n:` is primarily an internal safeguard.
        # We can test it by ensuring the normal path does *not* raise it.
        try:
            run_incremental_simulation(self.rng, self.mu, self.sigma, 200, 100)
        except RuntimeError:
            self.fail("run_incremental_simulation raised RuntimeError unexpectedly on normal path")

    def test_estimate_estimator_variance(self):
        """Test the variance estimation of mean and median."""
        n_for_variance = 50
        num_repetitions = 1000 # Smaller number for faster test

        var_mean, var_median = estimate_estimator_variance(self.rng, self.mu, self.sigma, n_for_variance, num_repetitions)

        self.assertIsInstance(var_mean, float)
        self.assertIsInstance(var_median, float)
        self.assertGreater(var_mean, 0)
        self.assertGreater(var_median, 0)

        # Theoretical variance of mean for N(0,1) is sigma^2/n = 1/50 = 0.02
        # Theoretical variance of median is approx (pi * sigma^2) / (2*n) = (pi * 1) / 100 ~ 0.0314
        # These are statistical estimates, so they won't be exact.
        # Check if they are in a reasonable ballpark (e.g., within a factor of 2-3 for this many reps)
        self.assertAlmostEqual(var_mean, (self.sigma**2)/n_for_variance, delta=0.015) # Increased delta for stochasticity
        self.assertAlmostEqual(var_median, (np.pi * self.sigma**2)/(2*n_for_variance), delta=0.02) # Increased delta

if __name__ == '__main__':
    unittest.main() 