"""Geometric Brownian Motion simulation under the risk-neutral Black-Scholes model.

SDE: dS = r S dt + sigma S dW

Applying Ito's lemma to ln(S) removes the drift-correction term and gives an
SDE with constant coefficients, which integrates exactly:

    ln(S_t) = ln(S_0) + (r - sigma^2 / 2) t + sigma * W_t

so S_t is log-normal and can be drawn directly from a single standard normal
draw per path/step ("exact scheme", zero discretization bias).
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
        Z = self.rng.standard_normal(n_paths)
        s_t = self.s0 * np.exp((self.r - self.sigma**2 / 2) * T + self.sigma * np.sqrt(T) * Z)
        return s_t
    

    def paths(self, T: float, n_steps: int, n_paths: int) -> np.ndarray:
        """
        Simulate full trajectories on a uniform grid of n_steps + 1 dates
        (t=0 included). Shape: (n_paths, n_steps + 1). """
        
        dt = T / n_steps
        Z = self.rng.standard_normal((n_paths, n_steps))
        log_incr = (self.r - self.sigma**2 / 2) * dt + self.sigma * np.sqrt(dt) * Z
        log_paths = np.cumsum(log_incr, axis=1)
        log_paths = np.hstack((np.full((n_paths, 1), np.log(self.s0)), log_paths))
        s_paths = np.exp(log_paths)
        return s_paths  


# test Var(W_t) = t empirically to check that the standard normal draws are correct 

rng = np.random.default_rng(42)  # seed=42 for reproductibility 

t = 1.0          # time horizon 
n_trajectories = 10_000

Z = rng.standard_normal(n_trajectories)
W_t = np.sqrt(t) * Z

empirical_var = W_t.var()
print(f"Variance empirique : {empirical_var:.4f}")
print(f"Variance théorique : {t}")