"""Payoffs and Monte Carlo pricing engine for European options.

Price = E[e^{-rT} * payoff(S_T)], estimated by the sample mean of the
discounted payoff over N simulated paths. By the CLT, the estimator's
standard error is std(discounted payoff) / sqrt(N), which gives a
(asymptotic) confidence interval mean +/- z * SE.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def call_payoff(s_t: np.ndarray, k: float) -> np.ndarray:
    return np.maximum(s_t - k, 0.0)


def put_payoff(s_t: np.ndarray, k: float) -> np.ndarray:
    return np.maximum(k - s_t, 0.0)


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
            f"IC{pct}=[{self.ci_low:.4f}, {self.ci_high:.4f}], "
            f"std_error={self.std_error:.6f}, n_paths={self.n_paths})"
        )


class MonteCarloPricer:
    """Discounts and averages a payoff over simulated terminal prices."""

    def __init__(self, r: float, T: float):
        self.r = r
        self.T = T

    def price(self, s_t: np.ndarray, k: float, option_type: str = "call",
               confidence: float = 0.95) -> PricingResult:
        payoff_fn = PAYOFFS[option_type]
        discounted = np.exp(-self.r * self.T) * payoff_fn(s_t, k)

        n = discounted.size
        mean = discounted.mean()
        std_error = discounted.std(ddof=1) / np.sqrt(n)

        z = norm.ppf(0.5 + confidence / 2)
        half_width = z * std_error

        return PricingResult(
            price=mean,
            std_error=std_error,
            ci_low=mean - half_width,
            ci_high=mean + half_width,
            n_paths=n,
            confidence=confidence,
        )


def compare_to_bs(mc_result: PricingResult, bs_price: float) -> dict:
    """Absolute and relative gap between the MC estimate and the BS price."""
    abs_error = abs(mc_result.price - bs_price)
    rel_error = abs_error / bs_price if bs_price != 0 else float("nan")
    return {
        "mc_price": mc_result.price,
        "bs_price": bs_price,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "within_ci": mc_result.ci_low <= bs_price <= mc_result.ci_high,
    }
