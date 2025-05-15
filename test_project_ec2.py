import unittest
import scipy.stats as stats
from project_ec2 import calculate_exact_probability, calculate_clt_approximation

class TestProjectEC2(unittest.TestCase):
    """Test cases for Project EC2 calculations."""

    def setUp(self):
        """Set up parameters for the tests."""
        self.num_variables_r = 30  # Number of successes for NB, also number of Geom variables summed
        self.prob_success_p = 1/6
        self.n_sum_target_trials_k = 170 # Target sum for S_30 (total trials)

        self.mean_s30 = self.num_variables_r / self.prob_success_p
        self.std_dev_s30 = (self.num_variables_r * (1 - self.prob_success_p) / (self.prob_success_p**2))**0.5

    def test_calculate_exact_probability(self):
        """Test P(S_30 > 170) using NB_failures parameterization for SciPy."""
        calculated_value = calculate_exact_probability(self.n_sum_target_trials_k, self.num_variables_r, self.prob_success_p)
        
        # Expected value based on P(F > 140) where F ~ NB_failures(r=30, p=1/6)
        # F = k_trials - r = 170 - 30 = 140
        expected_scipy_value = stats.nbinom.sf(140, self.num_variables_r, self.prob_success_p)
        
        self.assertAlmostEqual(calculated_value, expected_scipy_value, places=10,
                             msg="Exact probability calculation does not match direct scipy calculation for failures.")
        # This should now be approx 0.6031603187
        self.assertAlmostEqual(calculated_value, 0.6031603187, places=8)

    def test_calculate_clt_approximation(self):
        """Test the CLT approximation for P(S_30 > 170) with half-unit correction."""
        calculated_value = calculate_clt_approximation(self.n_sum_target_trials_k, self.num_variables_r, self.prob_success_p)
        
        corrected_target_for_clt = self.n_sum_target_trials_k + 0.5
        z_score = (corrected_target_for_clt - self.mean_s30) / self.std_dev_s30
        expected_scipy_value_clt = stats.norm.cdf(-z_score) # P(Z < -z_score)
        
        self.assertAlmostEqual(calculated_value, expected_scipy_value_clt, places=10,
                             msg="CLT approximation does not match direct scipy calculation based on derived Z-score.")
        # This should be approx 0.6242517279
        self.assertAlmostEqual(calculated_value, 0.6242517279060125, places=10)

if __name__ == '__main__':
    unittest.main() 