import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def generate_normal_samples(rng: np.random.Generator, mu: float, sigma: float, size: int) -> np.ndarray:
    """Generates samples from a normal distribution.

    :param rng: NumPy random number generator instance.
    :type rng: np.random.Generator
    :param mu: Mean of the normal distribution.
    :type mu: float
    :param sigma: Standard deviation of the normal distribution.
    :type sigma: float
    :param size: Number of samples to generate.
    :type size: int
    :return: Array of generated samples.
    :rtype: np.ndarray
    """
    if sigma <= 0:
        raise ValueError("Standard deviation (sigma) must be positive.")
    return rng.normal(loc=mu, scale=sigma, size=size)

def calculate_mean_std(data: np.ndarray) -> tuple[float, float]:
    """Calculates sample mean and sample standard deviation.

    :param data: Input data array.
    :type data: np.ndarray
    :return: A tuple containing (sample mean, sample standard deviation).
             Returns (nan, nan) if input data has length less than 2 for std dev calculation.
    :rtype: tuple[float, float]
    """
    n = data.size
    if n == 0:
        return np.nan, np.nan
    mean = np.mean(data)
    # Need at least 2 points for sample std dev (ddof=1)
    std_dev = np.std(data, ddof=1) if n >= 2 else np.nan
    return mean, std_dev

def run_correlation_simulation(rng: np.random.Generator, mu: float, sigma: float, n: int, max_m: int, m_step: int) -> tuple[list[int], list[float]]:
    """Runs the simulation to check correlation between sample mean and std dev.

    Performs experiments incrementally up to max_m repetitions.

    :param rng: NumPy random number generator instance.
    :type rng: np.random.Generator
    :param mu: Mean of the normal distribution.
    :type mu: float
    :param sigma: Standard deviation of the normal distribution.
    :type sigma: float
    :param n: Sample size for each individual experiment.
    :type n: int
    :param max_m: Maximum number of experiments (repetitions).
    :type max_m: int
    :param m_step: Increment in the number of experiments at each evaluation step.
    :type m_step: int
    :return: A tuple containing (list of M values, list of correlation coefficients).
    :rtype: tuple[list[int], list[float]]
    """
    if n < 2:
        raise ValueError("Sample size n must be at least 2 to calculate standard deviation.")
    if m_step < 2:
         print("Warning: m_step < 2. Correlation requires at least 2 data points. Adjusting m_step or first M value may be needed.")

    m_values = list(range(m_step, max_m + m_step, m_step))
    all_means = []
    all_stds = []
    correlations = []

    current_m = 0
    for target_m in m_values:
        num_new_experiments = target_m - current_m
        if num_new_experiments <= 0:
            # This shouldn't happen with range(m_step, ...)
            continue

        # Run the new experiments
        for _ in range(num_new_experiments):
            samples = generate_normal_samples(rng, mu, sigma, n)
            mean, std_dev = calculate_mean_std(samples)
            if not np.isnan(mean) and not np.isnan(std_dev):
                all_means.append(mean)
                all_stds.append(std_dev)
            else:
                # Should not happen if n >= 2
                print(f"Warning: NaN encountered for mean or std dev with n={n}. Skipping experiment result.")

        current_m = target_m

        # Calculate correlation for the current accumulated list (up to current_m)
        # Need at least 2 pairs to compute correlation
        if len(all_means) >= 2:
            # pearsonr returns (correlation, p-value)
            corr, _ = pearsonr(all_means, all_stds)
            # Handle potential NaN if variance is zero (though unlikely here)
            correlations.append(corr if not np.isnan(corr) else 0.0)
        else:
            # Not enough data points yet for correlation
            correlations.append(np.nan)

    # Filter out potential initial NaNs if m_step was too small initially
    valid_indices = [i for i, c in enumerate(correlations) if not np.isnan(c)]
    valid_m_values = [m_values[i] for i in valid_indices]
    valid_correlations = [correlations[i] for i in valid_indices]

    return valid_m_values, valid_correlations

def plot_correlation(m_values: list[int], correlations: list[float], n: int):
    """Plots the correlation coefficient against the number of experiments M.

    :param m_values: List of experiment counts (M).
    :type m_values: list[int]
    :param correlations: List of correlation coefficients corresponding to M values.
    :type correlations: list[float]
    :param n: Sample size used in each experiment.
    :type n: int
    """
    plt.figure(figsize=(10, 6))
    plt.plot(m_values, correlations, marker='.', linestyle='-')
    plt.axhline(0, color='r', linestyle='--', label='Expected Correlation (0)')
    plt.xlabel('Number of Experiments (M)')
    plt.ylabel('Correlation(Sample Mean, Sample Std Dev)')
    plt.title(f'Correlation Coefficient Convergence (n={n})')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.5, 0.5) # Zoom in y-axis as correlation should be small
    plt.tight_layout()
    plt.savefig(f"project_6_correlation_convergence_n{n}.png")
    print(f"Correlation plot saved to project_6_correlation_convergence_n{n}.png")
    # plt.show() # Comment out if running non-interactively

def main():
    """Main function to run Project 6 simulation."""
    # --- Parameters ---
    MU = 0.0
    SIGMA = 1.0
    # Use n=10 as suggested, can be changed to 20
    N_SAMPLE_SIZE = 10
    MAX_M = 10000 # Max number of experiments
    M_STEP = 100 # Increment step for M
    SEED = 43 # Use a different seed than project 1

    # Initialize random number generator
    rng = np.random.default_rng(SEED)

    print(f"Running correlation simulation (μ={MU}, σ={SIGMA}, n={N_SAMPLE_SIZE}) up to M={MAX_M}...")

    # --- Run Simulation ---
    m_values, correlations = run_correlation_simulation(rng, MU, SIGMA, N_SAMPLE_SIZE, MAX_M, M_STEP)

    if not m_values:
        print("No valid correlation results were generated. Check parameters (e.g., n, m_step, max_m).")
        return

    print("Plotting correlation convergence...")
    # --- Plot Results ---
    plot_correlation(m_values, correlations, N_SAMPLE_SIZE)

    print(f"\nFinal correlation coefficient at M={m_values[-1]}: {correlations[-1]:.6f}")
    print("\nConclusion: The simulation shows the correlation coefficient between the sample mean")
    print("and sample standard deviation converging towards 0, supporting the theoretical claim")
    print("of their independence for normally distributed data.")

if __name__ == "__main__":
    main() 