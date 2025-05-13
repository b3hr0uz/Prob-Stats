# Project 16: Poisson Distribution Simulation

## Problem Statement

Simulate `n` values of a Poisson random variable X = poisson(λ) (choose λ > 0). Compute the sample mean (x̄), sample median (m), and sample standard deviation (s). Plot these quantities as functions of `n` (following the "General remark" for incremental generation). Observe the limit values these statistics converge to as `n` → ∞. What are these limits? Compare with theory. Estimate the variance of x̄ and m for `n = 100` using 10,000 repetitions to determine which estimate is better (smaller variance).

## Approach

1.  **Parameters**:
    *   λ (lambda, rate parameter of Poisson): Choose a value, e.g., λ = 10.
    *   `max_n`: Maximum sample size for convergence plots, e.g., 10,000.
    *   `step`: Increment for `n` in convergence plots, e.g., 100.
    *   `n_variance`: Sample size for variance estimation, e.g., `n = 100`.
    *   `num_repetitions`: Number of repetitions for variance estimation, e.g., 10,000.

2.  **Simulation (Incremental)**:
    *   Similar to Project 1, generate Poisson samples incrementally.
    *   Start with `n = step` samples from poisson(λ).
    *   For subsequent `n` values (`2*step`, `3*step`, ..., `max_n`), add `step` new samples to the existing dataset.
    *   Use `numpy.random.Generator.poisson(lam=λ, size=...)`.

3.  **Statistics Calculation**:
    *   For each value of `n` in the incremental simulation, calculate:
        *   Sample Mean: `numpy.mean()`
        *   Sample Median: `numpy.median()` (Note: Poisson is discrete, median will be an integer or half-integer).
        *   Sample Standard Deviation: `numpy.std(ddof=1)`.

4.  **Plotting Convergence & Theoretical Limits**:
    *   Plot Sample Mean vs. `n`, Sample Median vs. `n`, Sample Standard Deviation vs. `n` on separate plots.
    *   **Theory**: For X ~ poisson(λ):
        *   E[X] (True Mean) = λ
        *   Var(X) (True Variance) = λ
        *   True Standard Deviation = sqrt(λ)
        *   The median of a Poisson distribution is approximately floor(λ + 1/3 - 0.02/λ) for large λ, typically close to λ.
    *   Add horizontal lines on the plots representing the theoretical limits (λ for mean/median, sqrt(λ) for std dev) to check convergence.

5.  **Convergence Comparison (Variance Estimation)**:
    *   Similar to Project 1, set `n = n_variance` (e.g., 100).
    *   Repeat `num_repetitions` (10,000) times:
        *   Generate a *new* sample of `n` values from poisson(λ).
        *   Calculate the sample mean (x̄) and sample median (m).
    *   Collect the 10,000 means and 10,000 medians.
    *   Compute the sample variance of the collected means and medians.
    *   Compare the variances. The estimator with the smaller variance is better (more efficient) for this `n`.
    *   **Theoretical Variance**: Var(x̄) = Var(X)/n = λ/n. There isn't a simple formula for Var(m) for Poisson, but we expect it to be larger than Var(x̄).

6.  **Implementation**: `project_16.py` and `test_project_16.py`. Include visualizations for the convergence plots. 