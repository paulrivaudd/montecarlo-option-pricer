# Assignment — Monte Carlo Pricer

Goal: price a European option (call/put) by simulating Brownian trajectories, under the Black-Scholes model, in Python.

The code to write lives in `mc_pricer/`. Each file contains `TODO`s with the necessary theory in the docstring. The tests in `tests/` let you self-check at each step, independently of any solution — they use reference values computed once and for all.

Work step by step, in order. Run `pytest -v` regularly: the tests for the step you're working on should pass before moving on.

## Installation

```bash
cd assignment
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e .
```

## Step 1 — GBM simulation

**File: `mc_pricer/gbm.py`, class `GBMSimulator`**

The underlying follows, under the risk-neutral probability:

```
dS = r S dt + sigma S dW
```

Applying Ito's lemma to `ln(S)`, the drift term becomes constant and the equation integrates exactly (no need for an Euler scheme):

```
S_T = S_0 * exp[(r - sigma^2/2) T + sigma * sqrt(T) * Z],   Z ~ N(0,1)
```

To implement:
- `terminal(T, n_paths)`: draws `S_T` for `n_paths` trajectories, in a single vectorized normal draw (no Python loop).
- `paths(T, n_steps, n_paths)`: applies the same exact relation between consecutive dates (`dt = T / n_steps`) to get full trajectories. Useful later for path-dependent options.

Validation: `tests/test_gbm.py`.

## Step 2 — Payoff and discounting

**File: `mc_pricer/pricer.py`**

```
Price = E[e^{-rT} * payoff(S_T)]
```

estimated by the sample mean of the discounted payoff over the `N` trajectories simulated in step 1. By the central limit theorem, the standard error of this estimator is:

```
SE = std(discounted payoff) / sqrt(N)
```

which gives a confidence interval `mean ± z * SE` (z ≈ 1.96 for 95%).

To implement:
- `call_payoff(s_t, k)` and `put_payoff(s_t, k)`.
- `MonteCarloPricer.price(...)`: computes the price, standard error, and CI, returned in a `PricingResult`.

Validation: `tests/test_pricer.py`.

## Step 3 — Analytical Black-Scholes comparison

**File: `mc_pricer/black_scholes.py`, class `BlackScholes`**

Closed-form formula, derived from the same GBM dynamics:

```
d1 = [ln(S0/K) + (r + sigma^2/2) T] / (sigma sqrt(T))
d2 = d1 - sigma sqrt(T)
Call = S0 * N(d1) - K e^{-rT} N(d2)
Put  = K e^{-rT} N(-d2) - S0 * N(-d1)
```

where `N` is the standard normal CDF (`scipy.stats.norm.cdf`).

To implement:
- `_d1`, `_d2`, `call_price()`, `put_price()`.
- `compare_to_bs(mc_result, bs_price)` in `pricer.py`: absolute error, relative error, and whether the BS price falls within the Monte Carlo price's 95% CI.

Validation: `tests/test_black_scholes.py` (independently computed reference values), then `tests/test_pricer.py::test_call_matches_black_scholes` and `test_put_matches_black_scholes`.

## Step 4 — Convergence

**File: `mc_pricer/convergence.py`, function `convergence_curve`**

Resimulating at every N would be wasteful: simulate once at `N_max`, then compute the running mean and variance at a (log-spaced) grid of N values via `cumsum` / `cumsum²` — fully vectorized, no Python loop over N.

Reminder: for an array `x` whose cumulative mean is `running_mean` at index `n` (1-indexed), the running unbiased variance follows from `cumsum(x**2)` and `cumsum(x)` via:

```
Var = E[X^2] - E[X]^2   (adjust by n/(n-1) if you want the exact ddof=1 version)
```

`mc_pricer/visualization.py::plot_convergence` is already provided (plain matplotlib) — wire it to your `convergence_curve`.

To implement:
- `convergence_curve(s_t, k, r, T, option_type, n_points, confidence)`.

Validation: run `examples/run_pricer.py`; the plot should converge to the Black-Scholes line with a shrinking CI.

## Structure

```
assignment/
├── mc_pricer/
│   ├── gbm.py            # TODO step 1
│   ├── pricer.py         # TODO step 2 (+ compare_to_bs step 3)
│   ├── black_scholes.py  # TODO step 3
│   ├── convergence.py    # TODO step 4
│   └── visualization.py  # provided
├── examples/run_pricer.py
├── tests/
│   ├── test_gbm.py
│   ├── test_black_scholes.py
│   └── test_pricer.py
├── requirements.txt
└── pyproject.toml
```

A full solution exists in `../solution/` if you want to compare or get unstuck — try to do without it for as long as possible.
