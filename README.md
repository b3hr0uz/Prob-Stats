# Prob-Stats: Numerical Simulations and Statistical Analysis

## Abstract

This repository contains a series of Python projects focused on exploring various concepts in probability and statistics through numerical simulation. Each project tackles a specific problem, demonstrating how to use libraries like NumPy, SciPy, and Matplotlib to generate data, calculate statistical measures, analyze convergence, and compare estimators. The projects emphasize clear problem statements, methodical approaches, and visual representation of results, along with unit tests to ensure correctness.

## Projects Overview

The following projects are included, each with a simulation script (`project_X.py`), a test script (`test_project_X.py`), and a markdown explanation (`project_X.md`).

### Project 1: Normal Distribution Simulation

**Summary:** This project simulates a normally distributed random variable. It calculates the sample mean, median, and standard deviation, plotting their convergence to theoretical values as the sample size increases. It also compares the efficiency of the sample mean versus the sample median by estimating their variances.

**Original Problem Statement:**
* 1. Simulate n values of a normal random variable X = N (μ, σ2) (choose μ and σ > 0 as you like), and compute the sample mean  ̄x, sample median m, sample standard deviation s. Plot these quantities as functions of n on three separate plots (see a general remark in the end). Check that  ̄x and m converge to μ, as n → ∞, and s converges to σ. Which one converges to μ faster, the sample mean or the sample median? To be sure, estimate the variance of both  ̄x and m for a particular value of n, such as n = 100 (by generating, say, 10000 different random samples of size n and computing the sample variance of the resulting estimates  ̄x and m. The estimate with the smaller variance is better).

### Project 6: Independence of Sample Mean and Standard Deviation (Normal Data)

**Summary:** This project experimentally verifies the theoretical claim that the sample mean and sample standard deviation are independent for normally distributed data. It involves M repetitions of generating a small sample, calculating these statistics, and then plotting the correlation coefficient between them as M increases, expecting convergence to zero.

**Original Problem Statement:**
* 6. Simulate n values of a normal random variable X = N (μ, σ2) (choose μ and σ > 0 as you like), and compute the sample mean  ̄x and sample standard deviation s. The theory claims that these estimates are independent. Check this claim experimentally. Repeat this experiment M times and compute the sample correlation coefficient between  ̄x and σ. Plot this correlation coefficient as functions of M (see a general remark in the end). Check that it converges to zero, as M → ∞ (going up to M = 10000 would be enough). The value of n should be small, such as n = 10 or n = 20.

### Project 7: Robust Estimators with Contaminated Data (80/20 Normal/Uniform)

**Summary:** This project investigates the robustness of various estimators (sample mean, trimmed means at 10%, 20%, 30%, and sample median) when data from a normal distribution is contaminated by a smaller portion of data from a wide uniform distribution. It assesses accuracy and precision (via standard deviation) over many repetitions.

**Original Problem Statement:**
* 7. Simulate a sample of n = 100 random numbers in which 80 are drawn from a normal distribution N (5, 1) and the other 20 are drawn from a uniform distribution U (−100, 100) (the latter represent "contamination", or "background noise"). For this sample calculate (a) the sample mean, (b) the trimmed sample means discarding 10%, 20%, and 30% of the data, (c) the sample median. Which estimate appears to be the most accurate? Repeat this experiment 10000 times and compute the standard deviation for each of these five estimates. The estimate with the smallest variance is best.

### Project 8: Robust Estimators with Heavy Contamination (50/50 Normal/Uniform)

**Summary:** Similar to Project 7, this project examines robust estimators but with a more significant contamination: 50% of the data comes from a normal distribution and 50% from a wide uniform distribution. It uses trimmed means discarding 10%, 30%, and 60% of the data, comparing them with the mean and median.

**Original Problem Statement:**
* 8. Simulate a sample of n = 100 random numbers in which 50 are drawn from a normal distribution N (5, 1) and the other 50 are drawn from a uniform distribution U (−100, 100) (the latter represent a "contamination", or a "background noise"). For this sample calculate (a) the sample mean, (b) the trimmed sample means discarding 10%, 30%, and 60% of the data, (c) the sample median. Which estimate appears to be the most accurate? Repeat this experiment 10000 times and compute the standard deviation for each of these five estimates. The estimate with the smallest variance is best.

### Project 16: Poisson Distribution Simulation

**Summary:** This project simulates a Poisson distributed random variable. It tracks the convergence of the sample mean, median, and standard deviation to their theoretical limits (λ, λ, and sqrt(λ) respectively) as sample size grows. It also compares the variance of the sample mean and median.

**Original Problem Statement:**
* 16. Simulate n values of a Poisson random variable X = poisson(λ) (choose λ > 0 as you like), and compute the sample mean  ̄x, sample median m, sample standard deviation s. Plot these quantities as functions of n on three separate plots (see a general remark in the end). Do these statistics converge to any limit values, as n → ∞? What are those limits? Do your conclusions agree with the theory? Estimate the variance of  ̄x and m for a particular value of n, such as n = 100 (by generating 10000 random samples of size n and computing the sample variance of the resulting estimates  ̄x and m). Which of these two estimates is better?

### Project EC2: Sum of Geometric Random Variables & CLT Approximation

**Summary:** This project focuses on a sum of 30 independent and identically distributed Geometric random variables (with p=1/6). It involves: 
1. Deriving the distribution of the sum (Negative Binomial) using Moment Generating Functions (MGFs).
2. Calculating the exact probability \(P(\sum X_i > 170)\) using the Negative Binomial distribution.
3. Approximating this probability using the Central Limit Theorem (CLT) with a half-unit correction.

**Original Problem Statement (from image provided for "Problem 1 (Extra Credit)")**
* Let \(X_1, \ldots, X_{30}\) denote a random sample of size 30 from a random variable with the pmf
  \[ f(x) = \left(\frac{5}{6}\right)^{x-1} \frac{1}{6}, x=1,2,\ldots \]
  (a) Write an expression (use \(\Sigma\) notation) that calculates the probability \(P(X_1 + \ldots + X_{30} > 170)\). Then write a code evaluating this value. (10)
  (b) Find an approximation of the probability in (a) by the Central Limit Theorem and the half-unit correction. (10)

### General Remark on Incremental Simulations

(The following applies to projects like 1, 6, and 16 where statistics are plotted as functions of increasing sample size `n` or number of experiments `M`.)

In many projects, you are supposed to generate a sample of n values of a certain random variable, compute some statistics and then plot their values as functions of n. Do this for certain values of n such as n = 100, 200, 300,. . .,10000. This gives you 100 different values of each statistic, well enough for a plot. Important: when increasing n from 100 to 200, then from 200 to 300, etc., do not generate a new sample for every new value of n. Instead, add 100 new values to the old sample (that would make your plots much smoother and nicer).

## Getting Started

Follow these instructions to set up the project environment and run the simulations.

### Prerequisites

*   **Python:** This project requires Python. If you don't have Python installed, download the latest version for your OS from [python.org/downloads](https://www.python.org/downloads/) or use a package manager. Python 3.8 or newer is recommended to ensure compatibility with the project dependencies.
*   `pip` (Python package installer, usually comes with Python)
*   `git` (version control system)

### Installation

#### 1. Install Git

*   **Linux (Debian/Ubuntu):**
    ```bash
    sudo apt update
    sudo apt install git
    ```
*   **Linux (Fedora):**
    ```bash
    sudo dnf install git
    ```
*   **macOS:**
    Git may already be installed. Try `git --version`. If not, it will prompt you to install Xcode Command Line Tools, which includes Git. Alternatively, you can install it via [Homebrew](https://brew.sh/):
    ```bash
    brew install git
    ```
*   **Windows:**
    Download and install Git for Windows from [git-scm.com](https://git-scm.com/download/win). During installation, you can choose to add Git to your PATH.

#### 2. Clone the Repository

Open your terminal or command prompt and navigate to the directory where you want to store the project. Then, clone the repository:
```bash
git clone https://github.com/b3hr0uz/Prob-Stats.git
cd Prob-Stats
```

#### 3. Set up Python Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies. This keeps your global Python installation clean.

*   **Linux/macOS:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    After activating, your terminal prompt should change to indicate you are in the `.venv` environment.

*   **Windows (Command Prompt):**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate.bat
    ```
*   **Windows (PowerShell):**
    ```bash
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    ```
    (You might need to adjust script execution policy on PowerShell: `Set-ExecutionPolicy Unrestricted -Scope Process`)

To deactivate the virtual environment later, simply type `deactivate`.

#### 4. Install Dependencies

Once the virtual environment is activated, install the required Python packages:
```bash
pip install -r requirements.txt
```

### Environment Tested With

The code and instructions have been tested with the following environment configuration:

*   OS: macOS Sonoma 14.4
*   Python: 3.10.17
*   numpy: 1.26.4
*   scipy: 1.13.0
*   matplotlib: 3.8.4

While the project might work with other versions, using this setup ensures compatibility.

## Running the Projects

Each project simulation can be run directly from the terminal:
```bash
python project_X.py
```
Replace `X` with the project number (e.g., `1`, `6`, `7`, `8`, `16`).
The scripts will typically print output to the console and save generated plots as PNG files in the root directory (e.g., `project_1_stats_convergence.png`).

## Running Tests

Unit tests are provided for each project to verify the correctness of the core functions. To run the tests for a specific project:
```bash
python test_project_X.py
```
Replace `X` with the project number.

To run all tests in the repository (if you have a test runner configured or run them individually):
```bash
# Example for running all tests if they are discoverable by the unittest module
# python -m unittest discover
# Or run each test file individually:
python test_project_1.py
python test_project_6.py
python test_project_7.py
python test_project_8.py
python test_project_16.py
python test_project_ec2.py
```

## Technology Stack and Concepts

### Core Libraries

*   **NumPy:** The fundamental package for numerical computation in Python. Used extensively for generating random numbers, array manipulations, and mathematical operations.
*   **SciPy:** Builds on NumPy and provides a large collection of scientific and technical computing functions. Used here for statistical functions like `trim_mean` and `pearsonr` (correlation coefficient).
*   **Matplotlib:** A comprehensive library for creating static, animated, and interactive visualizations in Python. Used to generate plots showing convergence of statistics, comparison of estimators, etc.

### Code Structure and Conventions

*   **Modular Design**: Each project is self-contained in its `project_X.py` script, with corresponding documentation in `project_X.md` and unit tests in `test_project_X.py`. This promotes clarity and ease of understanding for individual problems.
*   **Sphinx-compliant Docstrings**: Functions are documented using docstrings formatted in a way that is compatible with [Sphinx](https://www.sphinx-doc.org/), a popular Python documentation generator. This allows for easy generation of well-structured API documentation if desired. The docstrings explain the purpose of functions, their parameters, and what they return.
*   **Dunder Methods (`__...__`)**: The primary dunder (double underscore) method used is `if __name__ == "__main__":`. This is a standard Python idiom that allows a script to be run as the main program or imported as a module by another script without executing its main simulation logic automatically.
*   **Unit Testing (`unittest`)**: The built-in `unittest` module is used for writing and running tests. Each `test_project_X.py` file contains test cases for the core functionalities of the corresponding project script, ensuring reliability and correctness of the implemented algorithms and calculations.
*   **Iterative Development**: The creation of these projects often follows an iterative approach, where solutions are built up, tested, and refined. This is a common practice in software and data science project development.

## Repository Link

The source code and project files are hosted on GitHub:
[https://github.com/b3hr0uz/Prob-Stats](https://github.com/b3hr0uz/Prob-Stats)