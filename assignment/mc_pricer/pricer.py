"""Payoffs and Monte Carlo pricing engine for European options.

Price = E[e^{-rT} * payoff(S_T)], estimated by the sample mean of the
discounted payoff over N simulated paths. By the CLT, the estimator's
standard error is std(discounted payoff) / sqrt(N), giving a confidence
interval mean +/- z * SE.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def call_payoff(s_t: np.ndarray, k: float) -> np.ndarray:
    """TODO step 2: max(S_T - K, 0), vectorized (np.maximum)."""
    raise NotImplementedError("Step 2: implement call_payoff()")


def put_payoff(s_t: np.ndarray, k: float) -> np.ndarray:
    """TODO step 2: max(K - S_T, 0), vectorized (np.maximum)."""
    raise NotImplementedError("Step 2: implement put_payoff()")


PAYOFFS = {"call": call_payoff, "put": put_payoff}


@dataclass
class PricingResult:
    price: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int
    confidence: float

    def __repr__(self) -> str:
        pct = int(self.confidence * 100)
        return (
            f"PricingResult(price={self.price:.4f}, "
            f"CI{pct}=[{self.ci_low:.4f}, {self.ci_high:.4f}], "
            f"std_error={self.std_error:.6f}, n_paths={self.n_paths})"
        )


class MonteCarloPricer:
    """Discounts and averages a payoff over simulated terminal prices."""

    def __init__(self, r: float, T: float):
        self.r = r
        self.T = T

    def price(self, s_t: np.ndarray, k: float, option_type: str = "call",
               confidence: float = 0.95) -> PricingResult:
        """
        TODO step 2:
        1. Look up the payoff function via PAYOFFS[option_type].
        2. Discount: discounted = exp(-r*T) * payoff(s_t, k).
        3. n = discounted.size, mean = discounted.mean().
        4. std_error = discounted.std(ddof=1) / sqrt(n).
        5. z = norm.ppf(0.5 + confidence/2) ; half_width = z * std_error.
        6. Return a PricingResult(price=mean, std_error=..., ci_low=mean-half_width,
           ci_high=mean+half_width, n_paths=n, confidence=confidence).
        """
        raise NotImplementedError("Step 2: implement MonteCarloPricer.price()")


def compare_to_bs(mc_result: PricingResult, bs_price: float) -> dict:
    """
    TODO step 3: absolute error, relative error, and whether bs_price falls
    within the MC result's confidence interval.

    Return a dict with keys:
    mc_price, bs_price, abs_error, rel_error, within_ci (bool).
    """
    raise NotImplementedError("Step 3: implement compare_to_bs()")
