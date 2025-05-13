import numpy as np
import matplotlib.pyplot as plt

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

def calculate_stats(data: np.ndarray) -> tuple[float, float, float]:
    """Calculates sample mean, median, and standard deviation.

    :param data: Input data array.
    :type data: np.ndarray
    :return: A tuple containing (sample mean, sample median, sample standard deviation).
             Returns (nan, nan, nan) if input data has length less than 2 for std dev calculation.
    :rtype: tuple[float, float, float]
    """
    if data.size == 0:
        return np.nan, np.nan, np.nan
    mean = np.mean(data)
    median = np.median(data)
    # Need at least 2 points for sample std dev (ddof=1)
    std_dev = np.std(data, ddof=1) if data.size >= 2 else np.nan
    return mean, median, std_dev

def run_incremental_simulation(rng: np.random.Generator, mu: float, sigma: float, max_n: int, step: int) -> tuple[list[int], list[float], list[float], list[float]]:
    """Runs the incremental simulation and collects statistics.

    Generates data incrementally and calculates stats at each step.

    :param rng: NumPy random number generator instance.
    :type rng: np.random.Generator
    :param mu: Mean of the normal distribution.
    :type mu: float
    :param sigma: Standard deviation of the normal distribution.
    :type sigma: float
    :param max_n: Maximum sample size to reach.
    :type max_n: int
    :param step: Increment in sample size at each step.
    :type step: int
    :return: A tuple containing (list of sample sizes, list of means, list of medians, list of std devs).
    :rtype: tuple[list[int], list[float], list[float], list[float]]
    """
    n_values = list(range(step, max_n + step, step))
    all_samples = np.array([], dtype=float)
    means = []
    medians = []
    std_devs = []

    for n in n_values:
        # Generate only the new samples needed
        new_samples_count = step if n > step else n
        new_samples = generate_normal_samples(rng, mu, sigma, new_samples_count)
        all_samples = np.concatenate((all_samples, new_samples))

        if all_samples.size != n:
             # This check ensures logic is correct, especially for the first step
             raise RuntimeError(f"Consistency check failed: expected size {n}, got {all_samples.size}")

        mean, median, std_dev = calculate_stats(all_samples)
        means.append(mean)
        medians.append(median)
        std_devs.append(std_dev)

    return n_values, means, medians, std_devs

def plot_statistics(n_values: list[int], means: list[float], medians: list[float], std_devs: list[float], mu: float, sigma: float):
    """Plots the calculated statistics against sample size.

    :param n_values: List of sample sizes.
    :type n_values: list[int]
    :param means: List of sample means corresponding to n_values.
    :type means: list[float]
    :param medians: List of sample medians corresponding to n_values.
    :type medians: list[float]
    :param std_devs: List of sample standard deviations corresponding to n_values.
    :type std_devs: list[float]
    :param mu: True mean of the distribution.
    :type mu: float
    :param sigma: True standard deviation of the distribution.
    :type sigma: float
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Plot Mean
    axs[0].plot(n_values, means, marker='.', linestyle='-', label='Sample Mean')
    axs[0].axhline(mu, color='r', linestyle='--', label=f'True Mean (μ={mu})')
    axs[0].set_ylabel('Sample Mean')
    axs[0].set_title('Sample Mean vs. Sample Size')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Median
    axs[1].plot(n_values, medians, marker='.', linestyle='-', label='Sample Median')
    axs[1].axhline(mu, color='r', linestyle='--', label=f'True Mean (μ={mu})')
    axs[1].set_ylabel('Sample Median')
    axs[1].set_title('Sample Median vs. Sample Size')
    axs[1].legend()
    axs[1].grid(True)

    # Plot Standard Deviation
    axs[2].plot(n_values, std_devs, marker='.', linestyle='-', label='Sample Std Dev')
    axs[2].axhline(sigma, color='g', linestyle='--', label=f'True Std Dev (σ={sigma})')
    axs[2].set_xlabel('Sample Size (n)')
    axs[2].set_ylabel('Sample Standard Deviation')
    axs[2].set_title('Sample Standard Deviation vs. Sample Size')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig("project_1_stats_convergence.png")
    print("Convergence plots saved to project_1_stats_convergence.png")
    # plt.show() # Comment out if running in a non-interactive environment

def estimate_estimator_variance(rng: np.random.Generator, mu: float, sigma: float, n: int, num_repetitions: int) -> tuple[float, float]:
    """Estimates the variance of the sample mean and sample median.

    Generates multiple samples of size n, calculates the mean and median for each,
    and then computes the variance of these estimates.

    :param rng: NumPy random number generator instance.
    :type rng: np.random.Generator
    :param mu: Mean of the normal distribution.
    :type mu: float
    :param sigma: Standard deviation of the normal distribution.
    :type sigma: float
    :param n: Sample size for each repetition.
    :type n: int
    :param num_repetitions: Number of repetitions to perform.
    :type num_repetitions: int
    :return: A tuple containing (variance of sample means, variance of sample medians).
    :rtype: tuple[float, float]
    """
    means = np.empty(num_repetitions)
    medians = np.empty(num_repetitions)

    for i in range(num_repetitions):
        samples = generate_normal_samples(rng, mu, sigma, n)
        means[i], medians[i], _ = calculate_stats(samples)

    var_mean = np.var(means, ddof=1) # Use sample variance for the variance of estimates
    var_median = np.var(medians, ddof=1)

    return var_mean, var_median

def main():
    """Main function to run Project 1 simulation and analysis."""
    # --- Parameters ---
    MU = 0.0
    SIGMA = 1.0
    MAX_N = 10000
    STEP = 100
    N_FOR_VARIANCE = 100
    NUM_REPETITIONS = 10000
    SEED = 42 # for reproducibility

    # Initialize random number generator
    rng = np.random.default_rng(SEED)

    # --- Part 1: Incremental Simulation and Plotting ---
    print(f"Running incremental simulation (μ={MU}, σ={SIGMA}) up to n={MAX_N}...")
    n_values, means, medians, std_devs = run_incremental_simulation(rng, MU, SIGMA, MAX_N, STEP)

    print("Plotting statistics convergence...")
    plot_statistics(n_values, means, medians, std_devs, MU, SIGMA)

    # --- Part 2: Variance Estimation ---
    print(f"\nEstimating variance of mean and median for n={N_FOR_VARIANCE} (μ={MU}, σ={SIGMA}) using {NUM_REPETITIONS} repetitions...")
    # Use a different seed or state for the variance estimation part if desired,
    # but using the same generator instance is also fine.
    var_mean, var_median = estimate_estimator_variance(rng, MU, SIGMA, N_FOR_VARIANCE, NUM_REPETITIONS)

    print(f"\nResults for n = {N_FOR_VARIANCE}:")
    print(f"  Variance of Sample Mean: {var_mean:.6f}")
    print(f"  Variance of Sample Median: {var_median:.6f}")

    # --- Conclusion ---
    if var_mean < var_median:
        print("\nConclusion: The Sample Mean has a smaller variance and appears to converge faster.")
    elif var_median < var_mean:
        print("\nConclusion: The Sample Median has a smaller variance and appears to converge faster.")
    else:
        print("\nConclusion: The Sample Mean and Sample Median have similar variances.")

    # Theoretical variances for comparison (optional)
    theoretical_var_mean = (SIGMA**2) / N_FOR_VARIANCE
    # Approximate theoretical variance for median for large n: sigma^2 * pi / (2*n)
    theoretical_var_median_approx = (SIGMA**2 * np.pi) / (2 * N_FOR_VARIANCE)
    print(f"\nTheoretical Variance of Sample Mean: {theoretical_var_mean:.6f}")
    print(f"Approx. Theoretical Variance of Sample Median: {theoretical_var_median_approx:.6f}")


if __name__ == "__main__":
    main() 