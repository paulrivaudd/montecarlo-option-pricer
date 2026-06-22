import numpy as np

from mc_pricer import GBMSimulator, BlackScholes, MonteCarloPricer

S0, K, R, SIGMA, T = 100.0, 100.0, 0.03, 0.20, 1.0
N_PATHS = 500_000


def test_call_matches_black_scholes():
    s_t = GBMSimulator(s0=S0, r=R, sigma=SIGMA, seed=0).terminal(T, N_PATHS)
    result = MonteCarloPricer(r=R, T=T).price(s_t, K, option_type="call")
    bs_price = BlackScholes(S0, K, R, SIGMA, T).call_price()
    assert abs(result.price - bs_price) < 5 * result.std_error


def test_put_matches_black_scholes():
    s_t = GBMSimulator(s0=S0, r=R, sigma=SIGMA, seed=1).terminal(T, N_PATHS)
    result = MonteCarloPricer(r=R, T=T).price(s_t, K, option_type="put")
    bs_price = BlackScholes(S0, K, R, SIGMA, T).put_price()
    assert abs(result.price - bs_price) < 5 * result.std_error


def test_put_call_parity_on_bs():
    bs = BlackScholes(S0, K, R, SIGMA, T)
    lhs = bs.call_price() - bs.put_price()
    rhs = S0 - K * np.exp(-R * T)
    assert abs(lhs - rhs) < 1e-10


def test_paths_shape_and_initial_value():
    paths = GBMSimulator(s0=S0, r=R, sigma=SIGMA, seed=2).paths(T, n_steps=252, n_paths=10)
    assert paths.shape == (10, 253)
    assert np.allclose(paths[:, 0], S0)
