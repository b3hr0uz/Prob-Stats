import numpy as np
from scipy import stats
import time
import matplotlib.pyplot as plt

# Functions (generate_contaminated_sample, calculate_estimators, run_simulation, analyze_results)
# are identical to project_7.py. We will reuse them by potentially putting them
# in a shared utility module later if more projects need them.
# For now, we copy them here for a self-contained script.

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

def calculate_estimators(sample: np.ndarray, total_trim_proportions: list[float]) -> dict[str, float]:
    """Calculates mean, specified trimmed means, and median for a sample.

    :param sample: The input data sample.
    :type sample: np.ndarray
    :param total_trim_proportions: A list of *total* proportions to cut for trimmed means (e.g., [0.1, 0.3, 0.6]).
                                   Note: `scipy.stats.trim_mean` uses proportion from *each* end.
    :type total_trim_proportions: list[float]
    :return: A dictionary containing the calculated estimator values.
             Keys are 'mean', 'median', and 'trim_X%' for each proportion.
    :rtype: dict[str, float]
    """
    estimators = {}
    estimator_names = ['mean', 'median'] + [f'trim_{int(p*100)}%' for p in total_trim_proportions]

    if sample.size == 0:
        for name in estimator_names:
             estimators[name] = np.nan
        return estimators

    estimators['mean'] = np.mean(sample)
    estimators['median'] = np.median(sample)

    for prop_total in total_trim_proportions:
        # Convert total proportion to proportion for each end for scipy
        prop_each_end = prop_total / 2.0
        estimator_key = f'trim_{int(prop_total*100)}%'
        if not (0 <= prop_each_end < 0.5):
            # Ensure the proportion for each end is valid for trim_mean
            # Allow prop_each_end == 0 (which means prop_total == 0), equivalent to mean
             print(f"Warning: Total trim proportion {prop_total} results in invalid proportion per end ({prop_each_end}). Setting {estimator_key} to NaN.")
             estimators[estimator_key] = np.nan
             continue
            # raise ValueError(f"Total trim proportion {prop_total} results in invalid proportion per end ({prop_each_end}) for trim_mean")

        estimators[estimator_key] = stats.trim_mean(sample, proportiontocut=prop_each_end)

    return estimators

def run_simulation(rng: np.random.Generator, num_repetitions: int,
                   n_normal: int, mu_normal: float, sigma_normal: float,
                   n_uniform: int, low_uniform: float, high_uniform: float,
                   total_trim_proportions: list[float]) -> dict[str, np.ndarray]:
    """Runs the full simulation for Project 8.

    Repeats the experiment num_repetitions times and collects estimator results.
    Uses *total* trim proportions as input.

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
    :param total_trim_proportions: List of *total* proportions for trimmed means.
    :type total_trim_proportions: list[float]
    :return: A dictionary where keys are estimator names and values are arrays
             containing the results from all repetitions.
    :rtype: dict[str, np.ndarray]
    """
    estimator_names = ['mean', 'median'] + [f'trim_{int(p*100)}%' for p in total_trim_proportions]
    results = {name: np.empty(num_repetitions) for name in estimator_names}

    start_time = time.time()
    for i in range(num_repetitions):
        sample = generate_contaminated_sample(rng, n_normal, mu_normal, sigma_normal,
                                            n_uniform, low_uniform, high_uniform)
        # Pass the total proportions to calculate_estimators
        estimators = calculate_estimators(sample, total_trim_proportions)
        for name in estimator_names:
             # Check if the estimator calculation resulted in NaN (e.g., due to invalid trim)
            if name not in estimators or np.isnan(estimators[name]):
                 results[name][i] = np.nan # Store NaN if calculation failed
                 if i == 0: # Only warn once per estimator
                      print(f"Warning: Estimator '{name}' calculation failed for the first sample. Check parameters.")
            else:
                 results[name][i] = estimators[name]

        if (i + 1) % (num_repetitions // 10) == 0:
             elapsed = time.time() - start_time
             print(f"  Completed {i+1}/{num_repetitions} repetitions in {elapsed:.2f} seconds.")

    print(f"Simulation finished in {time.time() - start_time:.2f} seconds.")
    return results

def analyze_results(results: dict[str, np.ndarray], target_mean: float, project_name: str = "project_8"):
    """Analyzes the simulation results and plots them.

    Calculates mean, bias, and standard deviation for each estimator.
    Plots these statistics for visual comparison.
    Handles potential NaNs in results if an estimator couldn't be calculated.

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
    estimator_stats = {}
    valid_estimators_for_comparison = []

    for name in estimator_names:
        values = results[name]
        # Exclude NaNs from calculations if any occurred
        valid_values = values[~np.isnan(values)]
        if valid_values.size == 0:
            print(f"  {name}: No valid results found. Skipping analysis.")
            estimator_stats[name] = {'mean': np.nan, 'std_dev': np.nan, 'bias': np.nan}
            continue

        mean_val = np.mean(valid_values)
        std_dev = np.std(valid_values)
        bias = mean_val - target_mean

        estimator_stats[name] = {'mean': mean_val, 'std_dev': std_dev, 'bias': bias}
        valid_estimators_for_comparison.append(name)

        print(f"  {name}:")
        print(f"    Average Value = {mean_val:.4f}")
        print(f"    Bias          = {bias:.4f}")
        print(f"    Std Deviation = {std_dev:.4f}")
        if valid_values.size < values.size:
             print(f"    (Note: Based on {valid_values.size}/{values.size} valid results)")

    # Filter stats for valid estimators only for comparison/plotting
    valid_stats = {name: estimator_stats[name] for name in valid_estimators_for_comparison}
    if not valid_stats:
         print("\nNo estimators produced valid results for comparison.")
         return

    # Find most accurate (smallest absolute bias among valid estimators)
    most_accurate = min(valid_stats.items(), key=lambda item: abs(item[1]['bias']))
    print(f"\nMost Accurate Valid Estimator (closest average to {target_mean}): {most_accurate[0]} (Bias: {most_accurate[1]['bias']:.4f})")

    # Find most precise (smallest standard deviation among valid estimators)
    most_precise = min(valid_stats.items(), key=lambda item: item[1]['std_dev'])
    print(f"Best (Most Precise) Valid Estimator (smallest standard deviation): {most_precise[0]} (Std Dev: {most_precise[1]['std_dev']:.4f})")

    # --- Plotting (using only valid estimators) ---
    plot_names = list(valid_stats.keys())
    plot_avg_values = [valid_stats[name]['mean'] for name in plot_names]
    plot_biases = [valid_stats[name]['bias'] for name in plot_names]
    plot_std_devs = [valid_stats[name]['std_dev'] for name in plot_names]
    x_pos = np.arange(len(plot_names))

    # Plot 1: Average Values
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    ax1.bar(x_pos, plot_avg_values, align='center', alpha=0.7, capsize=5)
    ax1.axhline(target_mean, color='r', linestyle='--', label=f'Target Mean ({target_mean})')
    ax1.set_ylabel('Average Estimated Value')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(plot_names, rotation=45, ha="right")
    ax1.set_title('Average Value of Estimators Over Repetitions')
    ax1.legend()
    ax1.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{project_name}_estimator_averages.png")
    print(f"Plot of estimator averages saved to {project_name}_estimator_averages.png")

    # Plot 2: Biases
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    bars = ax2.bar(x_pos, plot_biases, align='center', alpha=0.7, capsize=5)
    ax2.axhline(0, color='k', linestyle='--')
    ax2.set_ylabel('Bias (Average Value - Target Mean)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(plot_names, rotation=45, ha="right")
    ax2.set_title('Bias of Estimators')
    ax2.grid(True, axis='y')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom' if yval >=0 else 'top', ha='center')
    plt.tight_layout()
    plt.savefig(f"{project_name}_estimator_biases.png")
    print(f"Plot of estimator biases saved to {project_name}_estimator_biases.png")

    # Plot 3: Standard Deviations
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    ax3.bar(x_pos, plot_std_devs, align='center', alpha=0.7, capsize=5, color='green')
    ax3.set_ylabel('Standard Deviation of Estimator')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(plot_names, rotation=45, ha="right")
    ax3.set_title('Precision of Estimators (Standard Deviation)')
    ax3.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{project_name}_estimator_std_devs.png")
    print(f"Plot of estimator standard deviations saved to {project_name}_estimator_std_devs.png")
    # plt.show() # Uncomment to display plots interactively

def main():
    """Main function to run Project 8 simulation."""
    # --- Parameters (Adjusted for Project 8) ---
    N_NORMAL = 50       # Changed
    N_UNIFORM = 50      # Changed
    MU_NORMAL = 5.0
    SIGMA_NORMAL = 1.0
    LOW_UNIFORM = -100.0
    HIGH_UNIFORM = 100.0
    NUM_REPETITIONS = 10000
    # Use TOTAL proportions discarded as per README interpretation and conversion in calculate_estimators
    TOTAL_TRIM_PROPORTIONS = [0.1, 0.3, 0.6] # Corresponds to 10%, 30%, 60% TOTAL
    SEED = 45 # Different seed again
    PROJECT_NAME = "project_8"

    print(f"--- Project 8: Robust Estimators (50/50 N(5,1) with U(-100,100) contamination) ---")
    print(f"Parameters: n_normal={N_NORMAL}, n_uniform={N_UNIFORM}, num_repetitions={NUM_REPETITIONS}")
    print(f"Total Trim proportions: {TOTAL_TRIM_PROPORTIONS} (interpreted as total % discarded)")

    # Initialize random number generator
    rng = np.random.default_rng(SEED)

    # --- Run Simulation ---
    print(f"\nStarting simulation for {NUM_REPETITIONS} repetitions...")
    results = run_simulation(rng, NUM_REPETITIONS,
                             N_NORMAL, MU_NORMAL, SIGMA_NORMAL,
                             N_UNIFORM, LOW_UNIFORM, HIGH_UNIFORM,
                             TOTAL_TRIM_PROPORTIONS)

    # --- Analyze Results ---
    analyze_results(results, MU_NORMAL, project_name=PROJECT_NAME)

if __name__ == "__main__":
    main() 