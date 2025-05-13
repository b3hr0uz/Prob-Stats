# Project 6: Independence of Sample Mean and Standard Deviation for Normal Data

## Problem Statement

Simulate `n` values of a normal random variable X = N(μ, σ²) (choose μ and σ > 0). Compute the sample mean (x̄) and sample standard deviation (s). The theory claims that these estimates are independent for normally distributed data. Check this claim experimentally. Repeat this experiment `M` times and compute the sample correlation coefficient between x̄ and s. Plot this correlation coefficient as a function of `M`. Check that it converges to zero as `M` → ∞ (going up to `M` = 10,000 would be enough). The value of `n` should be small, such as `n` = 10 or `n` = 20.

## Approach

1.  **Parameters**:
    *   μ (true mean): We will choose 0 for simplicity.
    *   σ (true standard deviation): We will choose 1 for simplicity.
    *   `n` (sample size for each experiment): We will use `n = 10` as suggested (a small value).
    *   `M_max` (maximum number of repetitions/experiments): We will go up to 10,000.
2.  **Simulation of Experiments**:
    *   We will perform `M` experiments, where `M` ranges incrementally from a small value (e.g., 100) up to `M_max` (e.g., 10,000), following the "General remark" to add new experiment results rather than re-generating all for each `M`.
    *   In each single experiment (indexed `i` from 1 to `M`):
        *   Generate `n` samples from N(μ, σ²).
        *   Calculate the sample mean (x̄<sub>i</sub>) using `numpy.mean()`.
        *   Calculate the sample standard deviation (s<sub>i</sub>) using `numpy.std(ddof=1)` (Bessel's correction).
3.  **Correlation Calculation (Incremental)**:
    *   We will maintain a list of all x̄ values and a list of all s values collected so far.
    *   As `M` increases, we will append the new x̄<sub>M</sub> and s<sub>M</sub> to these lists.
    *   For each value of `M` (e.g., 100, 200, ..., 10000), we will compute the Pearson correlation coefficient between the list of x̄'s (x̄<sub>1</sub>, ..., x̄<sub>M</sub>) and the list of s's (s<sub>1</sub>, ..., s<sub>M</sub>) accumulated up to that point. This can be done using `numpy.corrcoef()` or `scipy.stats.pearsonr()`.
    *   We need at least two pairs of (mean, std_dev) to calculate a correlation. The simulation will start with enough `M` to make this meaningful (e.g., `M_step >= 2`).
4.  **Plotting**:
    *   We will use `matplotlib` to plot the calculated sample correlation coefficient between x̄ and s as a function of `M` (number of experiments).
    *   We expect this plot to show the correlation coefficient converging towards 0 as `M` increases.
5.  **Implementation Details**:
    *   `project_6.py` will contain the main logic.
    *   `test_project_6.py` will contain unit tests for key functions.
    *   The "General remark" about incremental data generation applies to the number of experiments `M` here: as `M` increases, we add results from new experiments to the existing set. 