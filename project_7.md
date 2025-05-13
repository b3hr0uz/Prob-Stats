# Project 7: Robust Estimators with Contaminated Data (80/20)

## Problem Statement

Simulate a sample of `n = 100` random numbers where 80 are drawn from a normal distribution N(5, 1) (mean 5, standard deviation 1) and the other 20 are drawn from a uniform distribution U(-100, 100). The uniform samples represent "contamination" or "background noise". For this combined sample, calculate:

(a) The sample mean.
(b) The trimmed sample means, discarding 10%, 20%, and 30% of the data (from both ends).
(c) The sample median.

Which estimate appears to be the most accurate estimate of the central tendency of the main (normal) distribution (i.e., closest to μ=5)?

Repeat this entire experiment 10,000 times. Compute the standard deviation for each of these five estimates based on the results from the 10,000 repetitions. The estimate with the smallest standard deviation (variance) is considered the best (most statistically efficient) under these contamination conditions.

## Approach

1.  **Parameters**:
    *   `n`: Total sample size = 100.
    *   `n_normal`: Number of normal samples = 80.
    *   `n_uniform`: Number of uniform samples (contamination) = 20.
    *   Normal distribution: N(μ=5, σ=1).
    *   Uniform distribution: U(low=-100, high=100).
    *   `num_repetitions`: Number of times to repeat the experiment = 10,000.
    *   Trim proportions: 10% (0.1), 20% (0.2), 30% (0.3).

2.  **Single Experiment Simulation**:
    *   Define a function that performs one instance of the experiment:
        *   Generate `n_normal` samples from N(5, 1).
        *   Generate `n_uniform` samples from U(-100, 100).
        *   Combine these into a single sample of size `n=100`.
        *   Calculate the required statistics for this combined sample:
            *   Sample Mean: `numpy.mean()`.
            *   Trimmed Mean (10%): `scipy.stats.trim_mean(data, proportiontocut=0.1)`.
            *   Trimmed Mean (20%): `scipy.stats.trim_mean(data, proportiontocut=0.2)`.
            *   Trimmed Mean (30%): `scipy.stats.trim_mean(data, proportiontocut=0.3)`.
            *   Sample Median: `numpy.median()`.
        *   Return these five calculated values.

3.  **Repeated Experiments**:
    *   Create a main loop that calls the single experiment function `num_repetitions` (10,000) times.
    *   Store the five calculated statistics from each repetition in separate lists or arrays (e.g., a list of 10,000 sample means, a list of 10,000 10%-trimmed means, etc.).

4.  **Analysis**:
    *   **Accuracy**: Calculate the average value of each of the five estimators across the 10,000 repetitions. Compare these average values to the target mean of the non-contaminant distribution (μ=5). The estimator whose average is closest to 5 can be considered the most accurate (least biased by the contamination) in this scenario.
    *   **Precision (Variance/Standard Deviation)**: Calculate the standard deviation of each of the five lists of estimates. This measures the variability of each estimator across different random samples.
    *   **Best Estimator**: Identify the estimator with the smallest standard deviation. This estimator is the most precise or statistically efficient, meaning its value varies the least from sample to sample under these conditions.

5.  **Implementation Details**:
    *   Use `numpy` for random number generation (`default_rng`) and basic statistics.
    *   Use `scipy.stats.trim_mean` for calculating trimmed means.
    *   `project_7.py` will implement the simulation and analysis.
    *   `test_project_7.py` will contain unit tests.
    *   The "General remark" about incremental `n` does not apply here; we repeat the same fixed-size experiment multiple times. 