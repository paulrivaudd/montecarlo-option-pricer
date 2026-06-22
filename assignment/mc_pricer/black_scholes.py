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
        """TODO step 3: see formula in the module docstring."""
        raise NotImplementedError("Step 3: implement _d1")

    @property
    def _d2(self) -> float:
        """TODO step 3: d2 = d1 - sigma * sqrt(T)."""
        raise NotImplementedError("Step 3: implement _d2")

    def call_price(self) -> float:
        """TODO step 3: S0*N(d1) - K*e^{-rT}*N(d2), using norm.cdf."""
        raise NotImplementedError("Step 3: implement call_price()")

    def put_price(self) -> float:
        """TODO step 3: K*e^{-rT}*N(-d2) - S0*N(-d1)."""
        raise NotImplementedError("Step 3: implement put_price()")

    def price(self, option_type: str = "call") -> float:
        if option_type == "call":
            return self.call_price()
        if option_type == "put":
            return self.put_price()
        raise ValueError(f"Unknown option_type: {option_type}")
