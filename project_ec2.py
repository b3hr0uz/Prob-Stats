import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

def calculate_exact_probability(n_sum_target_trials: int, num_successes_r: int, prob_success_p: float) -> float:
    """Calculate the exact probability P(S_r > k_trials) for S_r ~ NB_trials(r, p).

    The sum S_r represents the total number of trials required to achieve r successes.
    This calculation uses SciPy's `nbinom.sf(f, r, p)` which expects `f` as the number of failures.
    The number of failures `f` is derived as `k_trials - r`.

    :param n_sum_target_trials: The target number of total trials (k_trials).
    :type n_sum_target_trials: int
    :param num_successes_r: The required number of successes (r).
    :type num_successes_r: int
    :param prob_success_p: The probability of success (p) on any given trial.
    :type prob_success_p: float
    :return: The exact probability P(S_r > n_sum_target_trials).
    :rtype: float
    """
    if n_sum_target_trials < num_successes_r:
        # If target trials is less than r, S_r is always greater by definition.
        return 1.0 
    # Number of failures f = k_trials - r
    num_failures_f = n_sum_target_trials - num_successes_r
    return stats.nbinom.sf(num_failures_f, num_successes_r, prob_success_p)

def calculate_clt_approximation(n_sum_target: int, num_variables: int, prob_success: float) -> float:
    """Approximate P(S_n > k) using the Central Limit Theorem with half-unit correction.

    S_n is the sum of n i.i.d. Geometric random variables. Its distribution is 
    approximated by a Normal distribution N(n*mu_X, n*sigma_X^2).
    The probability P(S_n > k) is approximated as P(Y > k + 0.5) where Y is the
    Normal approximation for S_n. This is calculated using the Z-score:
    Z = ( (k + 0.5) - E[S_n] ) / SD[S_n], then P(Z_std > Z_score).

    :param n_sum_target: The target sum (k) for P(S_n > k).
    :type n_sum_target: int
    :param num_variables: The number of i.i.d. Geometric variables summed (n).
    :type num_variables: int
    :param prob_success: The success probability (p) of the Geometric distribution.
    :type prob_success: float
    :return: The approximate probability P(S_n > n_sum_target) using CLT.
    :rtype: float
    """
    # Mean and variance of a single Geometric(p) variable X (number of trials for 1st success)
    mean_x = 1 / prob_success
    var_x = (1 - prob_success) / (prob_success**2)

    # Mean and variance of the sum S_n = X_1 + ... + X_n
    mean_s = num_variables * mean_x
    var_s = num_variables * var_x
    std_dev_s = var_s**0.5

    # Apply half-unit correction for P(S_n > k) -> P(NormalApprox > k + 0.5)
    corrected_target_for_clt = n_sum_target + 0.5

    # Calculate Z-score for the corrected target
    z_score = (corrected_target_for_clt - mean_s) / std_dev_s
    
    # P(Z_std > z_score) is equivalent to P(Z_std < -z_score) by symmetry, or 1 - P(Z_std < z_score)
    # Or directly use stats.norm.sf for P(X > x)
    return stats.norm.cdf(-z_score) # Using P(Z > z) = P(Z < -z) for N(0,1)

def plot_distributions(r_nb: int, p_nb: float, mean_norm: float, std_norm: float, 
                       k_highlight_nb: int, x_highlight_norm: float, filename: str):
    """Plot the Negative Binomial PMF, its Normal Approximation, and the Standard Normal PDF.

    Generates three subplots:
    1. PMF of NB(r, p) for total trials (S_r), highlighting P(S_r > k_highlight_nb).
    2. PDF of N(mu, sigma^2) approximating S_r, highlighting P(Y > x_highlight_norm).
    3. PDF of Standard Normal N(0,1), highlighting P(Z_std > z_score) where z_score is derived from the Normal approx.

    :param r_nb: Parameter `r` (successes) for Negative Binomial.
    :type r_nb: int
    :param p_nb: Parameter `p` (success probability) for Negative Binomial.
    :type p_nb: float
    :param mean_norm: Mean (mu) for the Normal approximation of S_r.
    :type mean_norm: float
    :param std_norm: Standard deviation (sigma) for the Normal approximation of S_r.
    :type std_norm: float
    :param k_highlight_nb: Value `k` for P(S_r > k) in the NB plot.
    :type k_highlight_nb: int
    :param x_highlight_norm: Value `x` (with half-unit correction) for P(Y > x) in the Normal approx plot.
    :type x_highlight_norm: float
    :param filename: Name of the file to save the plot.
    :type filename: str
    :return: None
    :rtype: None
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 18)) # Increased figure size for 3 plots

    # Subplot 1: Negative Binomial PMF (Distribution of S_r: total trials)
    lower_bound_sr = r_nb 
    upper_bound_sr = int(mean_norm + 4 * std_norm)
    k_values_sr = np.arange(lower_bound_sr, upper_bound_sr + 1)
    failures_values_nb = k_values_sr - r_nb 
    pmf_values_nb = stats.nbinom.pmf(failures_values_nb, r_nb, p_nb)
    axs[0].bar(k_values_sr, pmf_values_nb, label=f'NB_trials(r={r_nb}, p={p_nb:.4f}) PMF', alpha=0.7, color='skyblue')
    k_fill_sr = np.arange(k_highlight_nb + 1, upper_bound_sr + 1)
    if k_fill_sr.size > 0:
        failures_fill_nb = k_fill_sr - r_nb
        pmf_fill_nb = stats.nbinom.pmf(failures_fill_nb, r_nb, p_nb)
        axs[0].bar(k_fill_sr, pmf_fill_nb, color='steelblue', alpha=0.7, label=f'P(S_r > {k_highlight_nb})')
    axs[0].set_title(f'Negative Binomial PMF (Total Trials S_r={r_nb}, p={p_nb:.4f})')
    axs[0].set_xlabel('k (Total Number of Trials S_r)')
    axs[0].set_ylabel('Probability Mass P(S_r = k)')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.7)

    # Subplot 2: Normal Approximation PDF for S_r
    lower_bound_norm = mean_norm - 4 * std_norm
    upper_bound_norm = mean_norm + 4 * std_norm
    x_values_norm_approx = np.linspace(lower_bound_norm, upper_bound_norm, 500)
    pdf_values_norm_approx = stats.norm.pdf(x_values_norm_approx, mean_norm, std_norm)
    axs[1].plot(x_values_norm_approx, pdf_values_norm_approx, label=f'N(\mu={mean_norm:.1f}, \sigma^2={std_norm**2:.1f}) PDF', color='salmon')
    x_fill_norm_approx = np.linspace(x_highlight_norm, upper_bound_norm, 200)
    pdf_fill_norm_approx = stats.norm.pdf(x_fill_norm_approx, mean_norm, std_norm)
    axs[1].fill_between(x_fill_norm_approx, pdf_fill_norm_approx, color='orangered', alpha=0.5, label=f'P(Y > {x_highlight_norm:.1f})')
    axs[1].set_title(f'Normal Approximation PDF for S_r: Y ~ N({mean_norm:.1f}, {std_norm**2:.1f})')
    axs[1].set_xlabel('x (Approximation for Total Trials S_r)')
    axs[1].set_ylabel('Probability Density f(x)')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)

    # Subplot 3: Standard Normal PDF N(0,1)
    z_score_clt = (x_highlight_norm - mean_norm) / std_norm # This is the z_val from main
    x_values_std_norm = np.linspace(-4, 4, 500)
    pdf_values_std_norm = stats.norm.pdf(x_values_std_norm, 0, 1)
    axs[2].plot(x_values_std_norm, pdf_values_std_norm, label='Standard Normal N(0,1) PDF', color='lightgreen')
    
    # Highlight P(Z_std > z_score_clt)
    x_fill_std_norm = np.linspace(z_score_clt, 4, 200) # From z_score_clt to the right tail
    pdf_fill_std_norm = stats.norm.pdf(x_fill_std_norm, 0, 1)
    axs[2].fill_between(x_fill_std_norm, pdf_fill_std_norm, color='forestgreen', alpha=0.5, label=f'P(Z > {z_score_clt:.5f})')
    
    axs[2].set_title(f'Standard Normal PDF Z ~ N(0,1)')
    axs[2].set_xlabel('z (Standard Normal Variable)')
    axs[2].set_ylabel('Probability Density f(z)')
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.savefig(filename)
    print(f"\nPlot saved to {filename}")

def main():
    """Run calculations for Project EC2: Sum of Geometric Random Variables.
    
    This script calculates the exact probability P(S_30 > 170) for a sum of 30
    i.i.d. Geometric(1/6) random variables and its CLT approximation. It also
    generates and saves plots of the relevant distributions.

    :return: None
    :rtype: None
    """
    num_variables_r = 30  # Number of Geometric variables summed (r for NB)
    prob_success_p = 1/6  # Success probability for Geom(p)
    n_sum_target_k = 170  # Target sum for P(S_r > k_trials)
    project_name = "project_ec2"

    print(f"Project EC2: Sum of Geometric Random Variables")
    print(f"Calculating P(S_r > k) where S_r is the sum of r i.i.d. Geom(p) variables (total trials).")
    print(f"Parameters: r = {num_variables_r}, p = {prob_success_p:.4f}, k_trials = {n_sum_target_k}")

    exact_prob = calculate_exact_probability(n_sum_target_k, num_variables_r, prob_success_p)
    print(f"\nPart (a): Exact Probability P(S_{num_variables_r} > {n_sum_target_k})")
    # S_r (total trials) ~ NB_trials(r,p). E[S_r] = r/p. Var[S_r] = r(1-p)/p^2.
    # SciPy uses NB_failures(r,p) for F = S_r - r. P(S_r > k) = P(F > k-r)
    print(f"Calculated as P(F > {n_sum_target_k - num_variables_r}) for F ~ NB_failures(r={num_variables_r}, p={prob_success_p:.4f}).")
    print(f"P(S_{num_variables_r} > {n_sum_target_k}) = {exact_prob:.8f}")

    clt_approx_prob = calculate_clt_approximation(n_sum_target_k, num_variables_r, prob_success_p)
    mean_s_clt = num_variables_r * (1 / prob_success_p) # E[S_r]
    std_s_clt = (num_variables_r * (1 - prob_success_p) / (prob_success_p**2))**0.5 # SD[S_r]
    print(f"\nPart (b): CLT Approximation for P(S_{num_variables_r} > {n_sum_target_k}) with half-unit correction")
    print(f"S_{num_variables_r} approx. N(mean={mean_s_clt:.1f}, variance={std_s_clt**2:.1f}). Corrected target for S_r: {n_sum_target_k + 0.5}")
    z_val = (n_sum_target_k + 0.5 - mean_s_clt) / std_s_clt
    print(f"Z-score = ({n_sum_target_k + 0.5:.1f} - {mean_s_clt:.1f}) / {std_s_clt:.2f} = {z_val:.5f}")
    print(f"P(Z_std > {z_val:.5f}) = Phi({-z_val:.5f}) \approx {clt_approx_prob:.8f}") # P(Z > z) = P(Z < -z)

    plot_distributions(r_nb=num_variables_r, p_nb=prob_success_p, 
                       mean_norm=mean_s_clt, std_norm=std_s_clt, 
                       k_highlight_nb=n_sum_target_k, 
                       x_highlight_norm=n_sum_target_k + 0.5, 
                       filename=f"{project_name}_distributions.png")

if __name__ == "__main__":
    main() 