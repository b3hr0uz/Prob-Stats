# Project 1: Normal Distribution Simulation

## Problem Statement

Simulate n values of a normal random variable X = N(μ, σ²) (choose μ and σ > 0). Compute the sample mean (x̄), sample median (m), and sample standard deviation (s). Plot these quantities as functions of n. Check that x̄ and m converge to μ, and s converges to σ as n → ∞. Determine whether the sample mean or the sample median converges faster to μ by estimating their variances for n = 100 using 10,000 simulations.

## Approach

1.  **Parameters**: We will choose standard normal parameters for simplicity: μ = 0 and σ = 1.
2.  **Simulation (Incremental)**: Following the "General remark", we will simulate data incrementally. We start with n = 100 samples. Then, for subsequent steps (n = 200, 300, ..., 10000), we will add 100 new samples to the existing dataset rather than generating entirely new samples. This ensures smoother plots.
3.  **Statistics Calculation**: For each value of n, we will calculate:
    *   Sample Mean: `numpy.mean()`
    *   Sample Median: `numpy.median()`
    *   Sample Standard Deviation: `numpy.std(ddof=1)` (using Bessel's correction for an unbiased estimate of the population variance)
4.  **Plotting**: We will use `matplotlib` to generate three plots:
    *   Sample Mean vs. n
    *   Sample Median vs. n
    *   Sample Standard Deviation vs. n
    We expect the mean and median plots to converge towards μ=0 and the standard deviation plot towards σ=1.
5.  **Convergence Comparison (Variance Estimation)**:
    *   Set n = 100.
    *   Repeat 10,000 times:
        *   Generate a *new* sample of n=100 values from N(0, 1).
        *   Calculate the sample mean (x̄) and sample median (m) for this sample.
    *   Collect the 10,000 calculated means and 10,000 medians.
    *   Compute the sample variance of the collected means and the sample variance of the collected medians.
    *   The statistic (mean or median) with the smaller variance is considered to converge faster or be a more efficient estimator for this sample size.
6.  **Implementation**: The simulation, calculations, and plotting will be implemented in `project_1.py`. Unit tests will be in `test_project_1.py`. 