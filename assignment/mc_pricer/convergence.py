"""Convergence diagnostics: running MC price and CI as N grows.

Resimulating at every N would waste the draws already made. Instead,
simulate once at N_max and read off the cumulative mean/variance at a
log-spaced grid of N values via cumsum — O(N_max), fully vectorized,
no per-N Python loop.
"""

import numpy as np
from scipy.stats import norm

from .pricer import PAYOFFS


def convergence_curve(s_t: np.ndarray, k: float, r: float, T: float,
                        option_type: str = "call", n_points: int = 40,
                        confidence: float = 0.95):
    """
    TODO step 4:
    1. discounted = exp(-r*T) * PAYOFFS[option_type](s_t, k).
    2. n_max = discounted.size ; build a grid `ns` of n_points log-spaced
       values between 10 and n_max (np.logspace + cast to int + np.unique).
    3. cum_sum = np.cumsum(discounted) ; cum_sum_sq = np.cumsum(discounted**2).
    4. For every n in `ns`: running_mean = cum_sum[n-1] / n.
       running_var = cum_sum_sq[n-1]/n - running_mean**2 (vectorized over all
       of `ns`, no Python loop).
    5. running_se = sqrt(max(running_var, 0) / n).
    6. z = norm.ppf(0.5 + confidence/2) ; half_width = z * running_se.
    7. Return (ns, running_mean, half_width).
    """
    raise NotImplementedError("Step 4: implement convergence_curve()")
