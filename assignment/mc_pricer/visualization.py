"""Plotting helpers for Monte Carlo diagnostics. Provided — wire it to your
convergence_curve() from step 4, nothing to implement here."""

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(ns: np.ndarray, running_mean: np.ndarray, half_width: np.ndarray,
                       bs_price: float, option_type: str, savepath: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, running_mean, label="Monte Carlo price", color="#1f77b4")
    ax.fill_between(ns, running_mean - half_width, running_mean + half_width,
                     color="#1f77b4", alpha=0.2, label="95% CI")
    ax.axhline(bs_price, color="#d62728", linestyle="--", label="Black-Scholes")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulations (N)")
    ax.set_ylabel(f"{option_type} price")
    ax.set_title("Monte Carlo price convergence towards Black-Scholes")
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig
