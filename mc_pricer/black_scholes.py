"""Closed-form Black-Scholes price for European call/put options.

    d1 = [ln(S0/K) + (r + sigma^2/2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)
    Call = S0 * N(d1) - K e^{-rT} N(d2)
    Put  = K e^{-rT} N(-d2) - S0 * N(-d1)

This is the analytical solution of the same risk-neutral GBM dynamics used
by GBMSimulator, so it serves as the ground truth the Monte Carlo estimator
should converge to.
"""

import numpy as np
from scipy.stats import norm


class BlackScholes:
    def __init__(self, s0: float, k: float, r: float, sigma: float, T: float):
        self.s0 = s0
        self.k = k
        self.r = r
        self.sigma = sigma
        self.T = T

    @property
    def _d1(self) -> float:
        return (np.log(self.s0 / self.k) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )

    @property
    def _d2(self) -> float:
        return self._d1 - self.sigma * np.sqrt(self.T)

    def call_price(self) -> float:
        d1, d2 = self._d1, self._d2
        return self.s0 * norm.cdf(d1) - self.k * np.exp(-self.r * self.T) * norm.cdf(d2)

    def put_price(self) -> float:
        d1, d2 = self._d1, self._d2
        return self.k * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.s0 * norm.cdf(-d1)

    def price(self, option_type: str = "call") -> float:
        if option_type == "call":
            return self.call_price()
        if option_type == "put":
            return self.put_price()
        raise ValueError(f"Unknown option_type: {option_type}")
