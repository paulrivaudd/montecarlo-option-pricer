# Sujet — Pricer Monte Carlo

Objectif : pricer une option européenne (call/put) par simulation de trajectoires browniennes, sous le modèle de Black-Scholes, en Python.

Le code à écrire vit dans `mc_pricer/`. Chaque fichier contient des `TODO` avec la théorie nécessaire en docstring. Les tests dans `tests/` te permettent de t'auto-corriger à chaque étape, indépendamment d'une quelconque solution — ce sont des valeurs de référence calculées une fois pour toutes.

Travaille étape par étape, dans l'ordre. Lance `pytest -v` régulièrement : les tests de l'étape sur laquelle tu travailles doivent passer avant de continuer.

## Installation

```bash
cd sujet
python -m venv .venv
source .venv/bin/activate   # ou .venv\Scripts\activate sous Windows
pip install -e .
```

## Étape 1 — Simulation GBM

**Fichier : `mc_pricer/gbm.py`, classe `GBMSimulator`**

Le sous-jacent suit, sous la probabilité risque-neutre :

```
dS = r S dt + sigma S dW
```

En appliquant le lemme d'Itô à `ln(S)`, le terme de dérive devient constant et l'équation s'intègre exactement (pas besoin d'un schéma d'Euler) :

```
S_T = S_0 * exp[(r - sigma^2/2) T + sigma * sqrt(T) * Z],   Z ~ N(0,1)
```

À coder :
- `terminal(T, n_paths)` : tire `S_T` pour `n_paths` trajectoires, en un seul tirage normal vectorisé (pas de boucle Python).
- `paths(T, n_steps, n_paths)` : applique la même relation exacte entre dates consécutives (`dt = T / n_steps`), pour obtenir des trajectoires complètes. Utile plus tard pour des options path-dependent.

Validation : `tests/test_gbm.py`.

## Étape 2 — Payoff et actualisation

**Fichier : `mc_pricer/pricer.py`**

```
Prix = E[e^{-rT} * payoff(S_T)]
```

estimé par la moyenne empirique du payoff actualisé sur les `N` trajectoires simulées à l'étape 1. Par le théorème central limite, l'erreur standard de cet estimateur est :

```
SE = std(payoff actualisé) / sqrt(N)
```

ce qui donne un intervalle de confiance `moyenne ± z * SE` (z ≈ 1.96 pour 95%).

À coder :
- `call_payoff(s_t, k)` et `put_payoff(s_t, k)`.
- `MonteCarloPricer.price(...)` : calcule le prix, l'erreur standard et l'IC, retournés dans un `PricingResult`.

Validation : `tests/test_pricer.py`.

## Étape 3 — Comparaison Black-Scholes analytique

**Fichier : `mc_pricer/black_scholes.py`, classe `BlackScholes`**

Formule fermée, dérivée de la même dynamique GBM :

```
d1 = [ln(S0/K) + (r + sigma^2/2) T] / (sigma sqrt(T))
d2 = d1 - sigma sqrt(T)
Call = S0 * N(d1) - K e^{-rT} N(d2)
Put  = K e^{-rT} N(-d2) - S0 * N(-d1)
```

où `N` est la fonction de répartition de la loi normale standard (`scipy.stats.norm.cdf`).

À coder :
- `_d1`, `_d2`, `call_price()`, `put_price()`.
- `compare_to_bs(mc_result, bs_price)` dans `pricer.py` : écart absolu, écart relatif, et si le prix BS tombe dans l'IC à 95% du prix Monte Carlo.

Validation : `tests/test_black_scholes.py` (valeurs de référence calculées indépendamment) puis `tests/test_pricer.py::test_call_matches_black_scholes` et `test_put_matches_black_scholes`.

## Étape 4 — Convergence

**Fichier : `mc_pricer/convergence.py`, fonction `convergence_curve`**

Resimuler à chaque N serait inutile : simule une seule fois à `N_max`, puis calcule la moyenne et la variance glissantes à une grille de N (log-espacée) via `cumsum` / `cumsum²` — entièrement vectorisé, sans boucle Python sur N.

Rappel : pour un tableau `x` de moyenne cumulée `running_mean` à l'index `n` (1-indexé), la variance non biaisée glissante se déduit de `cumsum(x**2)` et `cumsum(x)` via :

```
Var = E[X^2] - E[X]^2   (puis ajuster par n/(n-1) si tu veux le ddof=1 exact)
```

`mc_pricer/visualization.py::plot_convergence` est déjà fourni (matplotlib pur) — branche-le sur ta `convergence_curve`.

À coder :
- `convergence_curve(s_t, k, r, T, option_type, n_points, confidence)`.

Validation : lance `examples/run_pricer.py`, le graphique doit converger vers la ligne Black-Scholes avec un IC qui se resserre.

## Structure

```
sujet/
├── mc_pricer/
│   ├── gbm.py            # TODO étape 1
│   ├── pricer.py         # TODO étape 2 (+ compare_to_bs étape 3)
│   ├── black_scholes.py  # TODO étape 3
│   ├── convergence.py    # TODO étape 4
│   └── visualization.py  # fourni
├── examples/run_pricer.py
├── tests/
│   ├── test_gbm.py
│   ├── test_black_scholes.py
│   └── test_pricer.py
├── requirements.txt
└── pyproject.toml
```

Un corrigé complet existe dans `../corrige/` si tu veux comparer ou te débloquer — essaie de t'en passer le plus longtemps possible.
