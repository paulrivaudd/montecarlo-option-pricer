# Pricer Monte Carlo

Projet 1/3 de la roadmap quant été 2026. Pricing d'options européennes (call/put) par simulation Monte Carlo sous le modèle de Black-Scholes, avec comparaison à la formule fermée et étude de convergence.

## Modèle et méthode

Le sous-jacent suit un mouvement brownien géométrique sous la probabilité risque-neutre :

```
dS = r S dt + sigma S dW
```

En appliquant le lemme d'Itô à ln(S), on élimine le terme de dérive non constant et on obtient une solution fermée exacte :

```
S_T = S_0 * exp[(r - sigma^2/2) T + sigma * sqrt(T) * Z],   Z ~ N(0,1)
```

Cette solution log-normale est utilisée directement (schéma "exact"), sans discrétisation temporelle ni biais d'Euler, pour générer S_T en un seul tirage vectorisé par trajectoire.

Le prix de l'option est l'espérance actualisée du payoff :

```
Prix = E[e^{-rT} * payoff(S_T)]
```

estimée par la moyenne empirique sur N trajectoires. Le théorème central limite donne l'erreur standard `std(payoff actualisé) / sqrt(N)` et l'intervalle de confiance associé, qui se resserre en `1/sqrt(N)`.

Le prix Monte Carlo est comparé à la formule fermée de Black-Scholes (mêmes paramètres, même dynamique), qui sert de référence exacte.

## Structure du projet

```
montecarlo-option-pricer/
├── mc_pricer/
│   ├── gbm.py            # Simulation GBM (schéma exact), vectorisée NumPy
│   ├── pricer.py         # Payoffs, moyenne actualisée, erreur standard, IC
│   ├── black_scholes.py  # Formule fermée Black-Scholes (call/put)
│   ├── convergence.py    # Courbe de convergence (moyenne/IC glissants via cumsum)
│   └── visualization.py  # Graphique de convergence (échelle log)
├── examples/
│   └── run_pricer.py     # Démo complète : prix MC vs BS + graphique
├── tests/
│   └── test_pricer.py    # Tests de cohérence MC / BS / parité call-put
├── requirements.txt
└── pyproject.toml
```

Chaque module est une brique indépendante et réutilisable (simulateur GBM, moteur MC, comparaison analytique, diagnostics de convergence) pensée pour servir de socle aux projets 2 et 3 de la roadmap (options exotiques, path-dependent, etc.) : `GBMSimulator.paths()` génère déjà des trajectoires complètes, pas seulement la valeur terminale.

## Installation

```bash
git clone <repo_url>
cd montecarlo-option-pricer
pip install -e .
```

## Utilisation

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

## Roadmap quant été 2026

1. **Pricer Monte Carlo** (ce projet) — options vanille européennes, GBM, comparaison Black-Scholes.
2. Options path-dependent (Asian, barrière) — réutilise `GBMSimulator.paths()`.
3. À définir.
