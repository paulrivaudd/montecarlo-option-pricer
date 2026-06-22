"""Step 3 — reference values computed independently (scipy.stats.norm),
to check BlackScholes without depending on your own Monte Carlo simulation."""

from mc_pricer import BlackScholes

S0, K, R, SIGMA, T = 100.0, 100.0, 0.03, 0.20, 1.0
REF_CALL = 9.413403
REF_PUT = 6.457957


def test_call_price_reference_value():
    price = BlackScholes(S0, K, R, SIGMA, T).call_price()
    assert abs(price - REF_CALL) < 1e-4


def test_put_price_reference_value():
    price = BlackScholes(S0, K, R, SIGMA, T).put_price()
    assert abs(price - REF_PUT) < 1e-4


def test_put_call_parity():
    bs = BlackScholes(S0, K, R, SIGMA, T)
    import numpy as np
    lhs = bs.call_price() - bs.put_price()
    rhs = S0 - K * np.exp(-R * T)
    assert abs(lhs - rhs) < 1e-10
