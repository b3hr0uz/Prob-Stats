import numpy as np
from scipy import stats
import time
import matplotlib.pyplot as plt

def generate_contaminated_sample(rng: np.random.Generator,
                                 n_normal: int, mu_normal: float, sigma_normal: float,
                                 n_uniform: int, low_uniform: float, high_uniform: float) -> np.ndarray:
    """Generates a single contaminated sample.

    Combines samples from a normal distribution and a uniform distribution.

    :param rng: NumPy random number generator instance.
    :type rng: np.random.Generator
    :param n_normal: Number of samples from the normal distribution.
    :type n_normal: int
    :param mu_normal: Mean of the normal distribution.
    :type mu_normal: float
    :param sigma_normal: Standard deviation of the normal distribution.
    :type sigma_normal: float
    :param n_uniform: Number of samples from the uniform distribution.
    :type n_uniform: int
    :param low_uniform: Lower bound of the uniform distribution.
    :type low_uniform: float
    :param high_uniform: Upper bound of the uniform distribution.
    :type high_uniform: float
    :return: The combined contaminated sample array.
    :rtype: np.ndarray
    """
    normal_samples = rng.normal(loc=mu_normal, scale=sigma_normal, size=n_normal)
    uniform_samples = rng.uniform(low=low_uniform, high=high_uniform, size=n_uniform)
    combined_sample = np.concatenate((normal_samples, uniform_samples))
    return combined_sample

def calculate_estimators(sample: np.ndarray, trim_proportions: list[float]) -> dict[str, float]:
    """Calculates mean, specified trimmed means, and median for a sample.

    :param sample: The input data sample.
    :type sample: np.ndarray
    :param trim_proportions: A list of proportions to cut for trimmed means (e.g., [0.1, 0.2, 0.3]).
    :type trim_proportions: list[float]
    :return: A dictionary containing the calculated estimator values.
             Keys are 'mean', 'median', and 'trim_X%' for each proportion.
    :rtype: dict[str, float]
    """
    estimators = {}
    if sample.size == 0:
        estimators['mean'] = np.nan
        estimators['median'] = np.nan
        for prop in trim_proportions:
            estimators[f'trim_{int(prop*100)}%'] = np.nan
        return estimators

    estimators['mean'] = np.mean(sample)
    estimators['median'] = np.median(sample)

    for prop in trim_proportions:
        if not (0 <= prop < 0.5):
            # trim_mean requires proportiontocut to be in [0, 0.5)
            raise ValueError(f"Trim proportion {prop} must be in the range [0, 0.5)")
        estimators[f'trim_{int(prop*100)}%'] = stats.trim_mean(sample, proportiontocut=prop)

    return estimators

def run_simulation(rng: np.random.Generator, num_repetitions: int,
                   n_normal: int, mu_normal: float, sigma_normal: float,
                   n_uniform: int, low_uniform: float, high_uniform: float,
                   trim_proportions: list[float]) -> dict[str, np.ndarray]:
    """Runs the full simulation for Project 7.

    Repeats the experiment num_repetitions times and collects estimator results.

    :param rng: NumPy random number generator instance.
    :type rng: np.random.Generator
    :param num_repetitions: Number of times to repeat the experiment.
    :type num_repetitions: int
    :param n_normal: Number of normal samples per experiment.
    :type n_normal: int
    :param mu_normal: Mean of the normal distribution.
    :type mu_normal: float
    :param sigma_normal: Standard deviation of the normal distribution.
    :type sigma_normal: float
    :param n_uniform: Number of uniform samples per experiment.
    :type n_uniform: int
    :param low_uniform: Lower bound of the uniform distribution.
    :type low_uniform: float
    :param high_uniform: Upper bound of the uniform distribution.
    :type high_uniform: float
    :param trim_proportions: List of proportions for trimmed means.
    :type trim_proportions: list[float]
    :return: A dictionary where keys are estimator names and values are arrays
             containing the results from all repetitions.
    :rtype: dict[str, np.ndarray]
    """
    estimator_names = ['mean', 'median'] + [f'trim_{int(p*100)}%' for p in trim_proportions]
    results = {name: np.empty(num_repetitions) for name in estimator_names}

    start_time = time.time()
    for i in range(num_repetitions):
        sample = generate_contaminated_sample(rng, n_normal, mu_normal, sigma_normal,
                                            n_uniform, low_uniform, high_uniform)
        estimators = calculate_estimators(sample, trim_proportions)
        for name in estimator_names:
            results[name][i] = estimators[name]

        if (i + 1) % (num_repetitions // 10) == 0:
             elapsed = time.time() - start_time
             print(f"  Completed {i+1}/{num_repetitions} repetitions in {elapsed:.2f} seconds.")

    print(f"Simulation finished in {time.time() - start_time:.2f} seconds.")
    return results

def analyze_results(results: dict[str, np.ndarray], target_mean: float, project_name: str = "project_7"):
    """Analyzes the simulation results and plots them.

    Calculates mean, bias, and standard deviation for each estimator.
    Plots these statistics for visual comparison.

    :param results: Dictionary of estimator results from the simulation.
    :type results: dict[str, np.ndarray]
    :param target_mean: The true mean of the non-contaminant distribution.
    :type target_mean: float
    :param project_name: Name of the project for saving plot files (e.g., "project_7", "project_8").
    :type project_name: str
    """
    print("\n--- Analysis Results ---")
    print(f"Target Mean (from Normal component): {target_mean}")

    estimator_names = list(results.keys())
    avg_values = []
    biases = []
    std_devs = []

    estimator_stats = {}
    for name in estimator_names:
        values = results[name]
        mean_val = np.mean(values)
        std_dev = np.std(values)
        bias = mean_val - target_mean

        avg_values.append(mean_val)
        biases.append(bias)
        std_devs.append(std_dev)

        estimator_stats[name] = {'mean': mean_val, 'std_dev': std_dev, 'bias': bias}
        print(f"  {name}:")
        print(f"    Average Value = {mean_val:.4f}")
        print(f"    Bias          = {bias:.4f}")
        print(f"    Std Deviation = {std_dev:.4f}")

    # Find most accurate (smallest absolute bias)
    most_accurate = min(estimator_stats.items(), key=lambda item: abs(item[1]['bias']))
    print(f"\nMost Accurate Estimator (closest average to {target_mean}): {most_accurate[0]} (Bias: {most_accurate[1]['bias']:.4f})")

    # Find most precise (smallest standard deviation)
    most_precise = min(estimator_stats.items(), key=lambda item: item[1]['std_dev'])
    print(f"Best (Most Precise) Estimator (smallest standard deviation): {most_precise[0]} (Std Dev: {most_precise[1]['std_dev']:.4f})")

    # --- Plotting ---
    x_pos = np.arange(len(estimator_names))

    # Plot 1: Average Values
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    ax1.bar(x_pos, avg_values, align='center', alpha=0.7, capsize=5)
    ax1.axhline(target_mean, color='r', linestyle='--', label=f'Target Mean ({target_mean})')
    ax1.set_ylabel('Average Estimated Value')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(estimator_names, rotation=45, ha="right")
    ax1.set_title('Average Value of Estimators Over Repetitions')
    ax1.legend()
    ax1.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{project_name}_estimator_averages.png")
    print(f"Plot of estimator averages saved to {project_name}_estimator_averages.png")

    # Plot 2: Biases
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    bars = ax2.bar(x_pos, biases, align='center', alpha=0.7, capsize=5)
    ax2.axhline(0, color='k', linestyle='--')
    ax2.set_ylabel('Bias (Average Value - Target Mean)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(estimator_names, rotation=45, ha="right")
    ax2.set_title('Bias of Estimators')
    ax2.grid(True, axis='y')
    # Add text labels for bias values
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom' if yval >=0 else 'top', ha='center')
    plt.tight_layout()
    plt.savefig(f"{project_name}_estimator_biases.png")
    print(f"Plot of estimator biases saved to {project_name}_estimator_biases.png")

    # Plot 3: Standard Deviations
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    ax3.bar(x_pos, std_devs, align='center', alpha=0.7, capsize=5, color='green')
    ax3.set_ylabel('Standard Deviation of Estimator')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(estimator_names, rotation=45, ha="right")
    ax3.set_title('Precision of Estimators (Standard Deviation)')
    ax3.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{project_name}_estimator_std_devs.png")
    print(f"Plot of estimator standard deviations saved to {project_name}_estimator_std_devs.png")
    # plt.show() # Uncomment to display plots interactively

def main():
    """Main function to run Project 7 simulation."""
    # --- Parameters ---
    N_NORMAL = 80
    N_UNIFORM = 20
    MU_NORMAL = 5.0
    SIGMA_NORMAL = 1.0
    LOW_UNIFORM = -100.0
    HIGH_UNIFORM = 100.0
    NUM_REPETITIONS = 10000
    TRIM_PROPORTIONS = [0.1, 0.2, 0.3] # Corresponds to 10%, 20%, 30%
    SEED = 44 # Yet another seed
    PROJECT_NAME = "project_7"

    print(f"--- Project 7: Robust Estimators (N(5,1) with U(-100,100) contamination) ---")
    print(f"Parameters: n_normal={N_NORMAL}, n_uniform={N_UNIFORM}, num_repetitions={NUM_REPETITIONS}")
    print(f"Trim proportions: {TRIM_PROPORTIONS}")

    # Initialize random number generator
    rng = np.random.default_rng(SEED)

    # --- Run Simulation ---
    print(f"\nStarting simulation for {NUM_REPETITIONS} repetitions...")
    results = run_simulation(rng, NUM_REPETITIONS,
                             N_NORMAL, MU_NORMAL, SIGMA_NORMAL,
                             N_UNIFORM, LOW_UNIFORM, HIGH_UNIFORM,
                             TRIM_PROPORTIONS)

    # --- Analyze Results ---
    analyze_results(results, MU_NORMAL, project_name=PROJECT_NAME)

if __name__ == "__main__":
    main() 