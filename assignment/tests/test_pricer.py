"""Step 2 (and step 3 for the comparison) — end-to-end Monte Carlo pricer."""

from mc_pricer import GBMSimulator, BlackScholes, MonteCarloPricer, compare_to_bs

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


def test_compare_to_bs_reports_within_ci():
    s_t = GBMSimulator(s0=S0, r=R, sigma=SIGMA, seed=2).terminal(T, N_PATHS)
    result = MonteCarloPricer(r=R, T=T).price(s_t, K, option_type="call")
    bs_price = BlackScholes(S0, K, R, SIGMA, T).call_price()
    comparison = compare_to_bs(result, bs_price)
    assert set(comparison) == {"mc_price", "bs_price", "abs_error", "rel_error", "within_ci"}
    assert comparison["within_ci"] is True
