"""Plotting helpers for Monte Carlo diagnostics. Fourni — branche-le sur ta
convergence_curve() de l'étape 4, rien à coder ici."""

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(ns: np.ndarray, running_mean: np.ndarray, half_width: np.ndarray,
                       bs_price: float, option_type: str, savepath: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, running_mean, label="Prix Monte Carlo", color="#1f77b4")
    ax.fill_between(ns, running_mean - half_width, running_mean + half_width,
                     color="#1f77b4", alpha=0.2, label="IC 95%")
    ax.axhline(bs_price, color="#d62728", linestyle="--", label="Black-Scholes")
    ax.set_xscale("log")
    ax.set_xlabel("Nombre de simulations (N)")
    ax.set_ylabel(f"Prix {option_type}")
    ax.set_title("Convergence du prix Monte Carlo vers Black-Scholes")
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig
