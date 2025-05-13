# Project 8: Robust Estimators with Heavy Contamination (50/50)

## Problem Statement

Simulate a sample of `n = 100` random numbers in which 50 are drawn from a normal distribution N(5, 1) and the other 50 are drawn from a uniform distribution U(-100, 100). This represents a heavier "contamination" or "background noise" scenario compared to Project 7.

For this sample, calculate:
(a) The sample mean.
(b) The trimmed sample means, discarding 10%, 30%, and **60%** of the data (from both ends).
(c) The sample median.

Which estimate appears to be the most accurate estimate of the central tendency of the main (normal) distribution (i.e., closest to μ=5)?

Repeat this entire experiment 10,000 times. Compute the standard deviation for each of these five estimates based on the results from the 10,000 repetitions. The estimate with the smallest standard deviation (variance) is considered the best (most statistically efficient) under these heavy contamination conditions.

## Approach

This project largely follows the methodology of Project 7, with adjustments to contamination levels and trimming percentages.

1.  **Parameters**:
    *   `n`: Total sample size = 100.
    *   `n_normal`: Number of normal samples = 50 (changed from 80).
    *   `n_uniform`: Number of uniform samples (contamination) = 50 (changed from 20).
    *   Normal distribution: N(μ=5, σ=1) (same as Project 7).
    *   Uniform distribution: U(low=-100, high=100) (same as Project 7).
    *   `num_repetitions`: Number of times to repeat the experiment = 10,000.
    *   Trim proportions: 10% (0.1), 30% (0.3), and **60%** (0.3 because `scipy.stats.trim_mean` takes proportion from each end, so 0.3 from each end is 60% total. Note: `proportiontocut` for `trim_mean` is for one side, but the problem asks for total. We must use 0.3 for `proportiontocut` to achieve 60% total discard if it means 30% from each end. If it means 60% from *each* end, that would discard 120% which is not possible. Assuming it means 30% from each end for a total of 60% discarded). Let's clarify the 60% trim. The `scipy.stats.trim_mean` parameter `proportiontocut` is the proportion of observations to cut from *each* end of the sorted array. So, a `proportiontocut` of 0.3 means 30% from the low end and 30% from the high end, totaling 60% of data discarded. This seems correct for the problem statement.
        *   Trim proportions for `scipy.stats.trim_mean`: 0.1 (for 10% total), 0.3 (for 30% total), and for 60% total we need to be careful. If `proportiontocut` means total, then 0.6. If it means from each end, then 0.3. The `scipy` doc says "Proportion of observations to cut from *each* end". So for 60% *total* discard, we need to discard 30% from each end. So, `proportiontocut` will be 0.3 for this case. This means the 30% and 60% trimmed means might be the same if interpreted this way or `proportiontocut` needs to be 0.05, 0.15, 0.3 for 10%, 30%, 60% total discarded. Let's re-read the README. "discarding 10%, 30%, and 60% of the data". This typically means total. `scipy.stats.trim_mean`'s `proportiontocut` is from *each* end. So, if we want to discard 10% *total*, `proportiontocut` should be 0.05. For 30% *total*, `proportiontocut` = 0.15. For 60% *total*, `proportiontocut` = 0.3.
        *   Corrected Trim proportions for `scipy.stats.trim_mean`: 0.05 (for 10% total), 0.15 (for 30% total), 0.3 (for 60% total).

2.  **Single Experiment Simulation (Identical to Project 7 logic)**:
    *   Generate `n_normal=50` samples from N(5, 1).
    *   Generate `n_uniform=50` samples from U(-100, 100).
    *   Combine to a single sample of `n=100`.
    *   Calculate statistics:
        *   Sample Mean.
        *   Trimmed Mean (10% total -> `proportiontocut=0.05`).
        *   Trimmed Mean (30% total -> `proportiontocut=0.15`).
        *   Trimmed Mean (60% total -> `proportiontocut=0.3`).
        *   Sample Median.

3.  **Repeated Experiments (Identical to Project 7 logic)**:
    *   Repeat `num_repetitions` (10,000) times.
    *   Store the five statistics from each repetition.

4.  **Analysis (Identical to Project 7 logic, including plotting)**:
    *   **Accuracy**: Calculate average of each estimator, compare to μ=5.
    *   **Precision**: Calculate standard deviation of each estimator.
    *   **Best Estimator**: Smallest standard deviation.
    *   **Visualization**: Generate and save bar charts for average values, biases, and standard deviations of the estimators, similar to Project 7, naming files like `project_8_estimator_averages.png`, etc.

5.  **Implementation Details**:
    *   The code from `project_7.py` can be largely reused, adjusting parameters (especially `N_NORMAL`, `N_UNIFORM`, and the list of `TRIM_PROPORTIONS` passed to `calculate_estimators` and used for naming in `run_simulation` and `analyze_results`).
    *   `project_8.py` will be the new script.
    *   `test_project_8.py` for unit tests. 