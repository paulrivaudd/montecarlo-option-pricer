"""End-to-end demo: price a European option by Monte Carlo and compare to Black-Scholes.
À lancer une fois les 4 étapes codées."""

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

print(f"Prix Monte Carlo  : {mc_result.price:.4f}  (IC95% [{mc_result.ci_low:.4f}, {mc_result.ci_high:.4f}])")
print(f"Prix Black-Scholes: {bs_price:.4f}")
print(f"Ecart absolu      : {comparison['abs_error']:.6f}")
print(f"Ecart relatif     : {comparison['rel_error']:.4%}")
print(f"BS dans l'IC 95%  : {comparison['within_ci']}")

ns, running_mean, half_width = convergence_curve(s_t, K, R, T, option_type=OPTION_TYPE)
plot_convergence(ns, running_mean, half_width, bs_price, OPTION_TYPE, savepath="convergence.png")
print("Graphique sauvegarde : convergence.png")
