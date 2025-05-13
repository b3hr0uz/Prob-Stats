import unittest
import numpy as np
from scipy import stats
# Assuming functions are defined in project_8.py (or a shared module)
from project_8 import generate_contaminated_sample, calculate_estimators, run_simulation

class TestProject8(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.rng = np.random.default_rng(666) # Seed for reproducibility
        # Parameters for Project 8
        self.n_normal = 50
        self.mu_normal = 5.0
        self.sigma_normal = 1.0
        self.n_uniform = 50
        self.low_uniform = -100.0
        self.high_uniform = 100.0
        self.total_trim_proportions = [0.1, 0.3, 0.6] # 10%, 30%, 60% TOTAL

    def test_generate_contaminated_sample_p8(self):
        """Test generation of a 50/50 contaminated sample."""
        sample = generate_contaminated_sample(
            self.rng, self.n_normal, self.mu_normal, self.sigma_normal,
            self.n_uniform, self.low_uniform, self.high_uniform
        )
        self.assertEqual(sample.size, self.n_normal + self.n_uniform)
        self.assertEqual(sample.size, 100)
        # Loose check: count potential uniform samples (outside likely normal range)
        potential_uniform_count = np.sum((sample < 0) | (sample > 10))
        # Expect roughly around n_uniform=50, maybe slightly less due to uniform range overlap
        self.assertTrue(potential_uniform_count > self.n_uniform * 0.7, f"Expected > {self.n_uniform * 0.7} potential outliers, found {potential_uniform_count}")

    def test_calculate_estimators_p8(self):
        """Test calculation of estimators with Project 8 trim proportions."""
        # Sample where calculations are easy to verify, n=10
        # Total trim: 10% (cut 0.05 each side), 30% (cut 0.15 each side), 60% (cut 0.3 each side)
        sample1 = np.array([-100, -90, 1, 4, 5, 6, 7, 90, 95, 100]) # n=10
        total_trim_props_test = [0.2, 0.4, 0.6] # Test 20%, 40%, 60% total
        # Corresponding proportions per side: 0.1, 0.2, 0.3
        estimators1 = calculate_estimators(sample1, total_trim_props_test)

        self.assertAlmostEqual(estimators1['mean'], 21.8) # Sum = 218
        self.assertAlmostEqual(estimators1['median'], 5.5) # (5+6)/2
        # trim_20% (cut 1 each end: -90, 1, 4, 5, 6, 7, 90, 95) -> mean = 26.0
        self.assertAlmostEqual(estimators1['trim_20%'], 26.0)
        # trim_40% (cut 2 each end: 1, 4, 5, 6, 7, 90) -> mean = 18.8333...
        self.assertAlmostEqual(estimators1['trim_40%'], np.mean([1, 4, 5, 6, 7, 90]))
        # trim_60% (cut 3 each end: 4, 5, 6, 7) -> mean = 5.5
        self.assertAlmostEqual(estimators1['trim_60%'], 5.5)

        # Test invalid total proportion (e.g., 1.0 or more)
        # The function now handles this by returning NaN and printing a warning
        estimators_invalid = calculate_estimators(sample1, [1.0])
        self.assertTrue(np.isnan(estimators_invalid['trim_100%']))
        estimators_invalid_neg = calculate_estimators(sample1, [-0.1])
        self.assertTrue(np.isnan(estimators_invalid_neg['trim_-10%']))


    def test_run_simulation_p8(self):
        """Test the main simulation loop for Project 8."""
        num_reps = 10 # Small number for a quick test
        results = run_simulation(self.rng, num_reps,
                                 self.n_normal, self.mu_normal, self.sigma_normal,
                                 self.n_uniform, self.low_uniform, self.high_uniform,
                                 self.total_trim_proportions)

        expected_keys = ['mean', 'median', 'trim_10%', 'trim_30%', 'trim_60%']
        self.assertEqual(set(results.keys()), set(expected_keys))

        for key in expected_keys:
            self.assertIsInstance(results[key], np.ndarray)
            self.assertEqual(results[key].shape, (num_reps,))
            self.assertFalse(np.isnan(results[key]).any()) # Expect valid numbers

        # Check if means are somewhat reasonable (very loose check for small run)
        # With 50/50 N(5,1) and U(-100,100), mean should be close to (5+0)/2 = 2.5
        # Median and trimmed means should be closer to 5.
        self.assertTrue(-5 < np.mean(results['mean']) < 10) # Mean around 2.5 expected
        self.assertTrue(3 < np.mean(results['median']) < 7)  # Median robust
        self.assertTrue(3 < np.mean(results['trim_10%']) < 7)
        self.assertTrue(3 < np.mean(results['trim_30%']) < 7)
        self.assertTrue(3 < np.mean(results['trim_60%']) < 7)

if __name__ == '__main__':
    unittest.main() 