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
        """
        Draw S_T directly from its exact log-normal law. Shape: (n_paths,).

        TODO:
        1. Tire `n_paths` réalisations Z ~ N(0,1) avec self.rng.standard_normal.
        2. Calcule S_T = s0 * exp[(r - sigma^2/2) * T + sigma * sqrt(T) * Z].
        3. Retourne le tableau (pas de boucle Python).
        """
        raise NotImplementedError("Étape 1 : implémente terminal()")

    def paths(self, T: float, n_steps: int, n_paths: int) -> np.ndarray:
        """
        Simulate full trajectories on a uniform grid of n_steps + 1 dates
        (t=0 included). Shape: (n_paths, n_steps + 1).

        TODO:
        1. dt = T / n_steps.
        2. Tire une matrice Z de forme (n_paths, n_steps).
        3. Calcule les incréments log-prix : (r - sigma^2/2)*dt + sigma*sqrt(dt)*Z.
        4. Cumule ces incréments (np.cumsum, axis=1) en partant de ln(s0).
        5. Concatène la colonne t=0 (toutes égales à ln(s0)) puis repasse en
           espace prix avec np.exp.
        """
        raise NotImplementedError("Étape 1 : implémente paths()")
