import unittest
import numpy as np
import matplotlib.pyplot as plt # Not strictly needed for tests, but good for consistency if used elsewhere
from project_16 import generate_poisson_samples, calculate_stats, run_incremental_simulation, estimate_poisson_estimator_variance

class TestProject16(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.rng = np.random.default_rng(seed=777)
        self.lam = 10.0 # Example lambda

    def test_generate_poisson_samples(self):
        """Test the generation of Poisson samples."""
        samples = generate_poisson_samples(self.rng, self.lam, 100)
        self.assertEqual(samples.shape[0], 100)
        self.assertTrue(np.all(samples >= 0)) # Poisson values are non-negative
        self.assertTrue(np.issubdtype(samples.dtype, np.integer)) # Poisson values are integers
        # For large lambda, Poisson is bell-shaped around lambda
        self.assertTrue(self.lam * 0.5 < np.mean(samples) < self.lam * 1.5) # Loose check for mean

        with self.assertRaises(ValueError):
            generate_poisson_samples(self.rng, 0, 100) # Lambda must be positive
        with self.assertRaises(ValueError):
            generate_poisson_samples(self.rng, -1, 100)

    def test_calculate_stats_poisson(self):
        """Test calculation of mean, median, std for Poisson-like data."""
        data1 = np.array([8, 9, 10, 11, 12]) # Symmetric around 10
        mean, median, std = calculate_stats(data1)
        self.assertAlmostEqual(mean, 10.0)
        self.assertAlmostEqual(median, 10.0)
        self.assertAlmostEqual(std, np.std(data1, ddof=1))

        data2 = np.array([7, 8, 9, 10, 11, 12]) # Median is 9.5
        mean, median, std = calculate_stats(data2)
        self.assertAlmostEqual(mean, 9.5)
        self.assertAlmostEqual(median, 9.5)
        self.assertAlmostEqual(std, np.std(data2, ddof=1))

        data_single = np.array([5])
        mean, median, std = calculate_stats(data_single)
        self.assertAlmostEqual(mean, 5.0)
        self.assertAlmostEqual(median, 5.0)
        self.assertTrue(np.isnan(std))

    def test_run_incremental_simulation_poisson(self):
        """Test the incremental simulation logic for Poisson."""
        max_n = 500
        step = 100
        n_values, means, medians, std_devs = run_incremental_simulation(self.rng, self.lam, max_n, step)

        self.assertEqual(len(n_values), max_n // step)
        self.assertEqual(n_values, [100, 200, 300, 400, 500])
        self.assertEqual(len(means), len(n_values))
        self.assertEqual(len(medians), len(n_values))
        self.assertEqual(len(std_devs), len(n_values))
        if step > 1:
            self.assertFalse(any(np.isnan(s) for s in std_devs))

    def test_estimate_poisson_estimator_variance(self):
        """Test variance estimation for Poisson mean and median."""
        n_for_variance = 50
        num_repetitions = 1000 # Smaller for faster test

        var_mean, var_median = estimate_poisson_estimator_variance(self.rng, self.lam, n_for_variance, num_repetitions)

        self.assertIsInstance(var_mean, float)
        self.assertIsInstance(var_median, float)
        self.assertGreater(var_mean, 0)
        self.assertGreater(var_median, 0)

        # Theoretical variance of mean for Poisson(lam) is lam/n
        theoretical_var_mean = self.lam / n_for_variance
        # Check if estimated var_mean is in a reasonable ballpark
        self.assertAlmostEqual(var_mean, theoretical_var_mean, delta=theoretical_var_mean * 0.5) # Allow 50% diff due to stochasticity
        # For Poisson, Var(mean) is typically < Var(median)
        self.assertTrue(var_mean < var_median * 1.5) # var_median likely larger or similar

if __name__ == '__main__':
    unittest.main() 