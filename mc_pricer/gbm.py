"""Geometric Brownian Motion simulation under the risk-neutral Black-Scholes model.

SDE: dS = r S dt + sigma S dW

Applying Ito's lemma to ln(S) removes the drift-correction term and gives an
SDE with constant coefficients, which integrates exactly:

    ln(S_t) = ln(S_0) + (r - sigma^2 / 2) t + sigma * W_t

so S_t is log-normal and can be drawn directly from a single standard normal
draw per path/step. This "exact scheme" has zero discretization bias,
unlike an Euler-Maruyama discretization of the original SDE.
"""

import numpy as np


class GBMSimulator:
    """Exact-scheme GBM path generator, fully vectorized over paths (and steps)."""

    def __init__(self, s0: float, r: float, sigma: float, seed: int | None = None):
        self.s0 = s0
        self.r = r
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def terminal(self, T: float, n_paths: int) -> np.ndarray:
        """Draw S_T directly from its exact log-normal law. Shape: (n_paths,)."""
        z = self.rng.standard_normal(n_paths)
        drift = (self.r - 0.5 * self.sigma ** 2) * T
        diffusion = self.sigma * np.sqrt(T) * z
        return self.s0 * np.exp(drift + diffusion)

    def paths(self, T: float, n_steps: int, n_paths: int) -> np.ndarray:
        """
        Simulate full trajectories on a uniform grid of n_steps + 1 dates
        (t=0 included), chaining the exact transition between consecutive
        dates. Needed for path-dependent payoffs (Asian, barrier, ...) in
        later projects. Shape: (n_paths, n_steps + 1).
        """
        dt = T / n_steps
        z = self.rng.standard_normal((n_paths, n_steps))
        increments = (self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * z
        log_s0 = np.log(self.s0)
        cum_log = log_s0 + np.cumsum(increments, axis=1)
        log_paths = np.hstack([np.full((n_paths, 1), log_s0), cum_log])
        return np.exp(log_paths)
