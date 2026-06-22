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
    """TODO étape 2 : max(S_T - K, 0), vectorisé (np.maximum)."""
    raise NotImplementedError("Étape 2 : implémente call_payoff()")


def put_payoff(s_t: np.ndarray, k: float) -> np.ndarray:
    """TODO étape 2 : max(K - S_T, 0), vectorisé (np.maximum)."""
    raise NotImplementedError("Étape 2 : implémente put_payoff()")


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
        """
        TODO étape 2 :
        1. Récupère la fonction payoff via PAYOFFS[option_type].
        2. Actualise : discounted = exp(-r*T) * payoff(s_t, k).
        3. n = discounted.size, mean = discounted.mean().
        4. std_error = discounted.std(ddof=1) / sqrt(n).
        5. z = norm.ppf(0.5 + confidence/2) ; half_width = z * std_error.
        6. Retourne un PricingResult(price=mean, std_error=..., ci_low=mean-half_width,
           ci_high=mean+half_width, n_paths=n, confidence=confidence).
        """
        raise NotImplementedError("Étape 2 : implémente MonteCarloPricer.price()")


def compare_to_bs(mc_result: PricingResult, bs_price: float) -> dict:
    """
    TODO étape 3 : écart absolu, écart relatif, et test d'appartenance à l'IC.

    Retourne un dict avec les clés :
    mc_price, bs_price, abs_error, rel_error, within_ci (bool).
    """
    raise NotImplementedError("Étape 3 : implémente compare_to_bs()")
