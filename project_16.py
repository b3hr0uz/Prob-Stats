import numpy as np
import matplotlib.pyplot as plt
import time

def generate_poisson_samples(rng: np.random.Generator, lam: float, size: int) -> np.ndarray:
    """Generates samples from a Poisson distribution.

    :param rng: NumPy random number generator instance.
    :type rng: np.random.Generator
    :param lam: Lambda (rate) parameter of the Poisson distribution.
    :type lam: float
    :param size: Number of samples to generate.
    :type size: int
    :return: Array of generated samples.
    :rtype: np.ndarray
    """
    if lam <= 0:
        raise ValueError("Lambda (lam) must be positive.")
    return rng.poisson(lam=lam, size=size)

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
    std_dev = np.std(data, ddof=1) if data.size >= 2 else np.nan
    return mean, median, std_dev

def run_incremental_simulation(rng: np.random.Generator, lam: float, max_n: int, step: int) -> tuple[list[int], list[float], list[float], list[float]]:
    """Runs the incremental simulation for Poisson and collects statistics.

    :param rng: NumPy random number generator instance.
    :type rng: np.random.Generator
    :param lam: Lambda parameter of the Poisson distribution.
    :type lam: float
    :param max_n: Maximum sample size to reach.
    :type max_n: int
    :param step: Increment in sample size at each step.
    :type step: int
    :return: A tuple containing (list of sample sizes, list of means, list of medians, list of std devs).
    :rtype: tuple[list[int], list[float], list[float], list[float]]
    """
    n_values = list(range(step, max_n + step, step))
    all_samples = np.array([], dtype=int) # Poisson samples are integers
    means = []
    medians = []
    std_devs = []

    for n in n_values:
        new_samples_count = step if all_samples.size > 0 or n == step else n
        new_samples = generate_poisson_samples(rng, lam, new_samples_count)
        all_samples = np.concatenate((all_samples, new_samples))

        if all_samples.size != n:
             raise RuntimeError(f"Consistency check failed: expected size {n}, got {all_samples.size}")

        mean, median, std_dev = calculate_stats(all_samples)
        means.append(mean)
        medians.append(median)
        std_devs.append(std_dev)

    return n_values, means, medians, std_devs

def plot_poisson_statistics(n_values: list[int], means: list[float], medians: list[float], std_devs: list[float], lam: float):
    """Plots the calculated Poisson statistics against sample size.

    :param n_values: List of sample sizes.
    :type n_values: list[int]
    :param means: List of sample means.
    :type means: list[float]
    :param medians: List of sample medians.
    :type medians: list[float]
    :param std_devs: List of sample standard deviations.
    :type std_devs: list[float]
    :param lam: True lambda of the Poisson distribution.
    :type lam: float
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    true_mean = lam
    true_std_dev = np.sqrt(lam)
    # Median for Poisson is approx. floor(lambda + 1/3 - 0.02/lambda), or simply lam for large lambda
    true_median_approx = np.floor(lam + 1/3 - (0.02/lam if lam > 0.1 else 0)) if lam > 0 else 0


    axs[0].plot(n_values, means, marker='.', linestyle='-', label='Sample Mean')
    axs[0].axhline(true_mean, color='r', linestyle='--', label=f'True Mean (λ={true_mean:.2f})')
    axs[0].set_ylabel('Sample Mean')
    axs[0].set_title(f'Sample Mean vs. Sample Size (Poisson(λ={lam}))')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(n_values, medians, marker='.', linestyle='-', label='Sample Median')
    axs[1].axhline(true_mean, color='r', linestyle=':', label=f'True Mean (λ={true_mean:.2f})')
    axs[1].axhline(true_median_approx, color='purple', linestyle='--', label=f'Approx. True Median ({true_median_approx:.2f})')
    axs[1].set_ylabel('Sample Median')
    axs[1].set_title(f'Sample Median vs. Sample Size (Poisson(λ={lam}))')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(n_values, std_devs, marker='.', linestyle='-', label='Sample Std Dev')
    axs[2].axhline(true_std_dev, color='g', linestyle='--', label=f'True Std Dev (√λ={true_std_dev:.2f})')
    axs[2].set_xlabel('Sample Size (n)')
    axs[2].set_ylabel('Sample Standard Deviation')
    axs[2].set_title(f'Sample Standard Deviation vs. Sample Size (Poisson(λ={lam}))')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(f"project_16_poisson_stats_convergence_lam{lam}.png")
    print(f"Convergence plots saved to project_16_poisson_stats_convergence_lam{lam}.png")

def estimate_poisson_estimator_variance(rng: np.random.Generator, lam: float, n: int, num_repetitions: int) -> tuple[float, float]:
    """Estimates the variance of the sample mean and sample median for Poisson.

    :param rng: NumPy random number generator instance.
    :type rng: np.random.Generator
    :param lam: Lambda parameter of the Poisson distribution.
    :type lam: float
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
        samples = generate_poisson_samples(rng, lam, n)
        means[i], medians[i], _ = calculate_stats(samples)

    var_mean = np.var(means, ddof=1)
    var_median = np.var(medians, ddof=1)

    return var_mean, var_median

def main():
    """Main function to run Project 16 simulation and analysis."""
    LAMBDA = 10.0
    MAX_N = 10000
    STEP = 100
    N_FOR_VARIANCE = 100
    NUM_REPETITIONS_VARIANCE = 10000
    SEED = 46

    rng = np.random.default_rng(SEED)

    print(f"--- Project 16: Poisson Distribution (λ={LAMBDA}) Simulation ---")
    print(f"Running incremental simulation up to n={MAX_N}...")
    start_time = time.time()
    n_values, means, medians, std_devs = run_incremental_simulation(rng, LAMBDA, MAX_N, STEP)
    print(f"Incremental simulation finished in {time.time() - start_time:.2f} seconds.")

    print("Plotting statistics convergence...")
    plot_poisson_statistics(n_values, means, medians, std_devs, LAMBDA)

    print(f"\nEstimating variance of mean and median for n={N_FOR_VARIANCE} (λ={LAMBDA}) using {NUM_REPETITIONS_VARIANCE} repetitions...")
    start_time = time.time()
    var_mean, var_median = estimate_poisson_estimator_variance(rng, LAMBDA, N_FOR_VARIANCE, NUM_REPETITIONS_VARIANCE)
    print(f"Variance estimation finished in {time.time() - start_time:.2f} seconds.")

    print(f"\nResults for n = {N_FOR_VARIANCE}:")
    print(f"  Sample Mean converges to λ = {LAMBDA}")
    print(f"  Sample Median also converges towards λ = {LAMBDA} (approx. {np.floor(LAMBDA + 1/3 - (0.02/LAMBDA if LAMBDA > 0.1 else 0)):.2f})")
    print(f"  Sample Standard Deviation converges to sqrt(λ) = {np.sqrt(LAMBDA):.4f}")

    print(f"\n  Variance of Sample Mean: {var_mean:.6f}")
    print(f"  Variance of Sample Median: {var_median:.6f}")

    theoretical_var_mean = LAMBDA / N_FOR_VARIANCE
    print(f"  Theoretical Variance of Sample Mean (λ/n): {theoretical_var_mean:.6f}")

    if var_mean < var_median:
        print("\nConclusion: The Sample Mean has a smaller variance and is a better estimator for the central tendency of Poisson(λ) than the Sample Median.")
    else:
        print("\nConclusion: The Sample Median has a smaller or similar variance to the Sample Mean for Poisson(λ).")

if __name__ == "__main__":
    main() 