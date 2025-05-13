import unittest
import numpy as np
from scipy import stats
from project_7 import generate_contaminated_sample, calculate_estimators, run_simulation

class TestProject7(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.rng = np.random.default_rng(555) # Seed for reproducibility
        self.n_normal = 80
        self.mu_normal = 5.0
        self.sigma_normal = 1.0
        self.n_uniform = 20
        self.low_uniform = -100.0
        self.high_uniform = 100.0
        self.trim_proportions = [0.1, 0.2, 0.3]

    def test_generate_contaminated_sample(self):
        """Test generation of a contaminated sample."""
        sample = generate_contaminated_sample(
            self.rng, self.n_normal, self.mu_normal, self.sigma_normal,
            self.n_uniform, self.low_uniform, self.high_uniform
        )
        self.assertEqual(sample.size, self.n_normal + self.n_uniform)
        # Check some properties - not strict, just sanity checks
        # Count how many are likely from uniform (e.g. > mu + 5*sigma or < mu - 5*sigma)
        # Normal part N(5,1), so 5*sigma = 5. Unlikely normal points outside [0, 10]
        # Uniform part U(-100,100)
        potential_uniform_count = np.sum((sample < 0) | (sample > 10))
        # This count should be related to n_uniform, but with randomness.
        # It's a very loose check.
        self.assertTrue(potential_uniform_count >= self.n_uniform * 0.5) # Expect at least half of uniforms to be far out

    def test_calculate_estimators(self):
        """Test calculation of mean, median, and trimmed means."""
        # Simple sample where calculations are easy to verify
        # Trim 10% (1 from each end), Trim 20% (2 from each end)
        sample1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # n=10
        trim_props1 = [0.1, 0.2]
        estimators1 = calculate_estimators(sample1, trim_props1)

        self.assertAlmostEqual(estimators1['mean'], 5.5)
        self.assertAlmostEqual(estimators1['median'], 5.5)
        self.assertAlmostEqual(estimators1['trim_10%'], np.mean([2,3,4,5,6,7,8,9])) # 5.5
        self.assertAlmostEqual(estimators1['trim_20%'], np.mean([3,4,5,6,7,8]))     # 5.5

        # Sample with outliers
        sample2 = np.array([-100, 1, 2, 3, 4, 5, 6, 7, 8, 100]) # n=10
        estimators2 = calculate_estimators(sample2, trim_props1)
        self.assertAlmostEqual(estimators2['mean'], 3.6) # Sum = 36, n = 10 -> mean = 3.6
        self.assertAlmostEqual(estimators2['median'], 4.5) # (4+5)/2
        self.assertAlmostEqual(estimators2['trim_10%'], np.mean([1,2,3,4,5,6,7,8])) # 4.5
        self.assertAlmostEqual(estimators2['trim_20%'], np.mean([2,3,4,5,6,7]))     # 4.5

        # Test empty sample
        empty_sample = np.array([])
        empty_estimators = calculate_estimators(empty_sample, self.trim_proportions)
        self.assertTrue(np.isnan(empty_estimators['mean']))
        self.assertTrue(np.isnan(empty_estimators['median']))
        self.assertTrue(np.isnan(empty_estimators['trim_10%']))

        # Test proportiontocut out of bounds
        with self.assertRaises(ValueError):
            calculate_estimators(sample1, [0.5])
        with self.assertRaises(ValueError):
            calculate_estimators(sample1, [-0.1])

    def test_run_simulation(self):
        """Test the main simulation loop for correct structure and types."""
        num_reps = 10 # Small number for a quick test
        results = run_simulation(self.rng, num_reps,
                                 self.n_normal, self.mu_normal, self.sigma_normal,
                                 self.n_uniform, self.low_uniform, self.high_uniform,
                                 self.trim_proportions)

        expected_keys = ['mean', 'median', 'trim_10%', 'trim_20%', 'trim_30%']
        self.assertEqual(set(results.keys()), set(expected_keys))

        for key in expected_keys:
            self.assertIsInstance(results[key], np.ndarray)
            self.assertEqual(results[key].shape, (num_reps,))
            self.assertFalse(np.isnan(results[key]).any()) # Expect valid numbers

        # Check if means are somewhat reasonable (very loose check for a small run)
        # For N(5,1) and U(-100,100), mean will be pulled away from 5.
        # Median and trimmed means should be closer to 5.
        self.assertTrue(0 < np.mean(results['mean']) < 15) # Mean likely between 0 and 10 for 80/20 mix with U(-100,100)
        self.assertTrue(3 < np.mean(results['median']) < 7) # Median should be robust
        self.assertTrue(3 < np.mean(results['trim_10%']) < 7)
        self.assertTrue(3 < np.mean(results['trim_20%']) < 7)
        self.assertTrue(3 < np.mean(results['trim_30%']) < 7)

if __name__ == '__main__':
    unittest.main() 