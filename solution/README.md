# Monte Carlo Pricer

Project 1/3 of the summer 2026 quant roadmap. Pricing of European options (call/put) by Monte Carlo simulation under the Black-Scholes model, with comparison to the closed-form formula and a convergence study.

## Model and method

The underlying follows a geometric Brownian motion under the risk-neutral probability measure:

```
dS = r S dt + sigma S dW
```

Applying Ito's lemma to ln(S) removes the non-constant drift term and gives an exact closed-form solution:

```
S_T = S_0 * exp[(r - sigma^2/2) T + sigma * sqrt(T) * Z],   Z ~ N(0,1)
```

This log-normal solution is used directly (an "exact scheme"), with no time discretization or Euler bias, to generate S_T in a single vectorized draw per path.

The option price is the discounted expected payoff:

```
Price = E[e^{-rT} * payoff(S_T)]
```

estimated by the sample mean over N paths. The central limit theorem gives the standard error `std(discounted payoff) / sqrt(N)` and the associated confidence interval, which shrinks at rate `1/sqrt(N)`.

The Monte Carlo price is compared to the closed-form Black-Scholes formula (same parameters, same dynamics), which serves as the exact reference.

## Project structure

```
montecarlo-option-pricer/
├── mc_pricer/
│   ├── gbm.py            # GBM simulation (exact scheme), NumPy-vectorized
│   ├── pricer.py         # Payoffs, discounted mean, standard error, CI
│   ├── black_scholes.py  # Closed-form Black-Scholes formula (call/put)
│   ├── convergence.py    # Convergence curve (running mean/CI via cumsum)
│   └── visualization.py  # Convergence plot (log scale)
├── examples/
│   └── run_pricer.py     # Full demo: MC price vs BS + plot
├── tests/
│   └── test_pricer.py    # MC / BS consistency and put-call parity tests
├── requirements.txt
└── pyproject.toml
```

Each module is an independent, reusable building block (GBM simulator, MC engine, analytical comparison, convergence diagnostics) designed to be the foundation for projects 2 and 3 of the roadmap (exotic, path-dependent options, etc.): `GBMSimulator.paths()` already generates full trajectories, not just the terminal value.

## Installation

```bash
git clone <repo_url>
cd montecarlo-option-pricer
pip install -e .
```

## Usage

```bash
python examples/run_pricer.py
```

```python
from mc_pricer import GBMSimulator, BlackScholes, MonteCarloPricer, compare_to_bs

simulator = GBMSimulator(s0=100, r=0.03, sigma=0.20, seed=42)
s_t = simulator.terminal(T=1.0, n_paths=1_000_000)

mc_result = MonteCarloPricer(r=0.03, T=1.0).price(s_t, k=100, option_type="call")
bs_price = BlackScholes(s0=100, k=100, r=0.03, sigma=0.20, T=1.0).call_price()

print(mc_result)
print(compare_to_bs(mc_result, bs_price))
```

## Tests

```bash
pytest
```

## Summer 2026 quant roadmap

1. **Monte Carlo Pricer** (this project) — European vanilla options, GBM, Black-Scholes comparison.
2. Path-dependent options (Asian, barrier) — reuses `GBMSimulator.paths()`.
3. TBD.
