"""Step 1 — checks GBMSimulator independently of the other steps."""

import numpy as np

from mc_pricer import GBMSimulator

S0, R, SIGMA, T = 100.0, 0.03, 0.20, 1.0


def test_terminal_shape():
    s_t = GBMSimulator(s0=S0, r=R, sigma=SIGMA, seed=0).terminal(T, n_paths=1000)
    assert s_t.shape == (1000,)
    assert np.all(s_t > 0)


def test_terminal_moments_match_lognormal_theory():
    # E[S_T] = S0 * e^{rT} under the risk-neutral measure.
    s_t = GBMSimulator(s0=S0, r=R, sigma=SIGMA, seed=1).terminal(T, n_paths=2_000_000)
    expected_mean = S0 * np.exp(R * T)
    assert abs(s_t.mean() - expected_mean) / expected_mean < 0.01


def test_paths_shape_and_initial_value():
    paths = GBMSimulator(s0=S0, r=R, sigma=SIGMA, seed=2).paths(T, n_steps=252, n_paths=10)
    assert paths.shape == (10, 253)
    assert np.allclose(paths[:, 0], S0)


def test_paths_terminal_consistent_with_terminal_method():
    # Both methods simulate the same law: means should be close on a large sample.
    sim_a = GBMSimulator(s0=S0, r=R, sigma=SIGMA, seed=3)
    sim_b = GBMSimulator(s0=S0, r=R, sigma=SIGMA, seed=4)
    a = sim_a.terminal(T, n_paths=500_000)
    b = sim_b.paths(T, n_steps=50, n_paths=500_000)[:, -1]
    assert abs(a.mean() - b.mean()) / a.mean() < 0.02
