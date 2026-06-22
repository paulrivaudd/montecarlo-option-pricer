"""End-to-end demo: price a European option by Monte Carlo and compare to Black-Scholes.
Run this once all 4 steps are implemented."""

from mc_pricer import (
    GBMSimulator,
    BlackScholes,
    MonteCarloPricer,
    compare_to_bs,
    convergence_curve,
    plot_convergence,
)

S0, K, R, SIGMA, T = 100.0, 100.0, 0.03, 0.20, 1.0
N_PATHS = 1_000_000
OPTION_TYPE = "call"

simulator = GBMSimulator(s0=S0, r=R, sigma=SIGMA, seed=42)
s_t = simulator.terminal(T=T, n_paths=N_PATHS)

mc_result = MonteCarloPricer(r=R, T=T).price(s_t, K, option_type=OPTION_TYPE)
bs_price = BlackScholes(s0=S0, k=K, r=R, sigma=SIGMA, T=T).price(OPTION_TYPE)
comparison = compare_to_bs(mc_result, bs_price)

print(f"Monte Carlo price : {mc_result.price:.4f}  (95% CI [{mc_result.ci_low:.4f}, {mc_result.ci_high:.4f}])")
print(f"Black-Scholes price: {bs_price:.4f}")
print(f"Absolute error    : {comparison['abs_error']:.6f}")
print(f"Relative error    : {comparison['rel_error']:.4%}")
print(f"BS within 95% CI  : {comparison['within_ci']}")

ns, running_mean, half_width = convergence_curve(s_t, K, R, T, option_type=OPTION_TYPE)
plot_convergence(ns, running_mean, half_width, bs_price, OPTION_TYPE, savepath="convergence.png")
print("Plot saved: convergence.png")
