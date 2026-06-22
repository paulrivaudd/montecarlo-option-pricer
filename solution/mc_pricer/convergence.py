"""Convergence diagnostics: running MC price and CI as N grows.

Resimulating at every N would waste the draws already made. Instead we
simulate once at N_max and read off the cumulative mean/variance at a
log-spaced grid of N values via cumsum, which is O(N_max) and fully
vectorized (no per-N Python loop, no resampling).
"""

import numpy as np
from scipy.stats import norm

from .pricer import PAYOFFS


def convergence_curve(s_t: np.ndarray, k: float, r: float, T: float,
                        option_type: str = "call", n_points: int = 40,
                        confidence: float = 0.95):
    payoff_fn = PAYOFFS[option_type]
    discounted = np.exp(-r * T) * payoff_fn(s_t, k)

    n_max = discounted.size
    ns = np.unique(np.logspace(1, np.log10(n_max), n_points).astype(int))

    cum_sum = np.cumsum(discounted)
    cum_sum_sq = np.cumsum(discounted ** 2)

    running_mean = cum_sum[ns - 1] / ns
    running_var = cum_sum_sq[ns - 1] / ns - running_mean ** 2
    running_se = np.sqrt(np.maximum(running_var, 0.0) / ns)

    z = norm.ppf(0.5 + confidence / 2)
    half_width = z * running_se

    return ns, running_mean, half_width
