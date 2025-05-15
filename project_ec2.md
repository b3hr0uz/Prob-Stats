# Project EC2: Sum of Geometric Random Variables

## Problem Statement

Let \(X_1, X_2, \ldots, X_{30}\) denote a random sample of size 30 from a random variable \(X\) with the probability mass function (pmf):
\[ f(x) = \left(\frac{5}{6}\right)^{x-1} \frac{1}{6}, \quad x = 1, 2, \ldots \]

**(a)** Write an expression (use \(\Sigma\) notation) that calculates the probability \(P(X_1 + \ldots + X_{30} > 170)\). Then write a code evaluating this value.

**(b)** Find an approximation of the probability in (a) by the Central Limit Theorem and the half-unit correction.

## Solution

### Distribution of a Single Random Variable \(X_i\)

The given pmf is for a Geometric distribution, \(X \sim \text{Geom}(p=1/6)\), representing the number of trials needed to get the first success.
- Mean of \(X_i\): \(E[X_i] = \mu_X = 1/p = 1/(1/6) = 6\)
- Variance of \(X_i\): \(\text{Var}(X_i) = \sigma_X^2 = (1-p)/p^2 = (5/6)/(1/6)^2 = 30\)

### Part (a): Exact Probability for the Sum \(S_{30}\)

Let \(S_n = X_1 + X_2 + \ldots + X_n\). The sum of \(n=30\) i.i.d. Geometric(p) random variables follows a Negative Binomial distribution, where \(S_{30}\) represents the total number of trials required to achieve \(r=30\) successes. The pmf is \( P(S_{30}=k) = \binom{k-1}{r-1} p^r (1-p)^{k-r} \) for \(k = r, r+1, \ldots\).

**Mean and Variance of \(S_{30}\):**
- Mean: \(E[S_{30}] = \mu_S = r/p = 30/(1/6) = 180\)
- Variance: \(\text{Var}(S_{30}) = \sigma_S^2 = r(1-p)/p^2 = 30(5/6)/(1/6)^2 = 900\)
- Standard Deviation: \(\sigma_S = \sqrt{900} = 30\)

#### Calculating \(P(S_{30} > 170)\) using SciPy

We need to calculate \(P(S_{30} > 170)\). SciPy's `stats.nbinom` distribution is typically parameterized by the number of successes \(r\) and the probability of success \(p\), but its random variable represents the number of *failures* \(F\) that occur before achieving \(r\) successes.

The relationship between the total number of trials \(S_r\) and the number of failures \(F\) is \(S_r = F + r\).
Thus, the event \(S_{30} > 170\) is equivalent to \(F + 30 > 170\), which means \(F > 170 - 30 = 140\).

So, we need to calculate \(P(F > 140)\) where \(F \sim \text{NB_failures}(r=30, p=1/6)\).
Using Python's `scipy.stats.nbinom.sf(f, r, p)` function, which computes \(P(F > f)\) for the failure-parameterized Negative Binomial:
\[ P(S_{30} > 170) = P(F > 140) = \text{stats.nbinom.sf}(140, 30, 1/6) \approx 0.6031603187 \]

#### Note on Calculation Methods and SciPy Usage

It's important to understand the nuances of calculating probabilities, especially when using libraries like SciPy:

1.  **Equivalence of Approaches**: The probability \(P(S_{30} > 170)\) can be conceptualized in multiple equivalent ways:
    *   Directly as \(P(S_{30} > 170)\).
    *   As the complement: \(1 - P(S_{30} \le 170)\).
    *   When translated to SciPy's `nbinom` (which models failures \(F\)): \(P(F > 140)\).
    *   The complement in terms of failures: \(1 - P(F \le 140)\).
    All these lead to the same theoretical probability.

2.  **SciPy's `nbinom` and its Methods**: `scipy.stats.nbinom` is the object representing the Negative Binomial distribution. To get probabilities, we use its methods:
    *   `.sf(f, r, p)` calculates the Survival Function, \(P(F > f)\).
    *   `.cdf(f, r, p)` calculates the Cumulative Distribution Function, \(P(F \le f)\).
    *   `.pmf(f, r, p)` calculates the Probability Mass Function, \(P(F = f)\).
    Our choice of `stats.nbinom.sf(140, 30, 1/6)` directly computes \(P(F > 140)\).

3.  **Numerical Precision (`.sf` vs. `1 - .cdf`)**: For right-tail probabilities like \(P(F > f)\), using the survival function (`.sf()`) is generally preferred over `1 - .cdf(f)`. While mathematically equivalent, `.sf()` is often implemented to provide better numerical precision, especially when the tail probability is very small (though for \(P(F > 140) \approx 0.603\), the practical difference here might be negligible, using `.sf()` is good practice and more direct).

4.  **The Constant Probability `p`**: The probability of success \(p=1/6\) is a fundamental parameter of the underlying Bernoulli trials. This value remains constant whether we are analyzing the number of failures, total trials, or using different methods of calculation for the resulting Negative Binomial distribution.

### Part (b): Approximation using Central Limit Theorem (CLT)

The Central Limit Theorem (CLT) states that for a sufficiently large \(n\) (here \(n=30\)), the sum \(S_n\) can be approximated by a Normal distribution:
\[ S_{30} \approx N(\mu_S, \sigma_S^2) = N(180, 900) \]
To approximate \(P(S_{30} > 170)\), we use a half-unit correction because the Negative Binomial distribution is discrete.
\(P(S_{30} > 170)\) is equivalent to \(P(S_{30} \ge 171)\).
This is approximated by \(P(Y > 171 - 0.5) = P(Y > 170.5)\), where \(Y \sim N(180, 900)\).

To find this probability, we standardize \(Y\) to a Standard Normal Distribution \(Z \sim N(0,1)\):
\[ Z = \frac{Y - \mu_S}{\sigma_S} = \frac{Y - 180}{30} \]
So,
\[ P(Y > 170.5) = P\left(Z > \frac{170.5 - 180}{30}\right) = P\left(Z > \frac{-9.5}{30}\right) \approx P(Z > -0.31667) \]
Using the symmetry of the Normal distribution, \(P(Z > -0.31667) = P(Z < 0.31667)\).
This is \(\Phi(0.31667)\), where \(\Phi\) is the Cumulative Distribution Function (CDF) of the Standard Normal Distribution \(N(0,1)\).
\[ \Phi(0.31667) \approx 0.6247 \]

## Python Implementation and Visualization

The calculations and visualizations will be implemented in `project_ec2.py`.
- The exact probability will use `scipy.stats.nbinom.sf(140, 30, 1/6)`.
- The CLT approximation will calculate \(Z = (170.5 - 180)/30\) and then compute `scipy.stats.norm.cdf(-Z)`.
- Visualizations will show the PMF of \(S_{30} \sim NB(30, 1/6)\) (representing total trials) and the PDF of its Normal approximation \(N(180, 900)\), highlighting the calculated probabilities. 