# Class Overview — HMM Source Code

> Focus: `src/api/v4/` (current production). Deprecated versions (v1–v3) are noted at the end.

---

## High-Level Architecture

The codebase follows a **component + composition** pattern. An `HMM` object wraps an `HMMParams` object, which is itself composed of a `transition` component and an `emission` component. Fitting is delegated to a `solver`, and inference (forward filtering) is handled by a separate algorithm class.

```
HMM
 └── HMMParams          (eqx.Module, BaseHMM)
      ├── transition     (BaseTransition)
      └── emission       (BaseEmission)

HMM.fit(solver=...) → BaseSolver subclass
HMM._compute_state_results() → ForwardAlgorithm → ForwardOutput
```

---

## Class Hierarchy

### Abstract Base Classes (`src/base/`)

| Class | Base | File |
|---|---|---|
| `BaseHMM` | `ABC, eqx.Module` | `base/base_hmm.py` |
| `BaseEmission` | `ABC, eqx.Module` | `base/base_emission.py` |
| `BaseTransition` | `ABC, eqx.Module` | `base/base_transition.py` |
| `BaseInference` | `ABC` | `base/base_inference.py` |
| `BaseSolver` | `ABC` | `base/base_solver.py` |
| `BaseOutput` | `ABC, eqx.Module` | `base/base_output.py` |

All base classes define abstract methods and a shared `update_param` / `__iter__` interface for JAX pytree compatibility via `equinox`.

---

### Concrete Classes (`src/api/v4/`)

#### HMM Model

| Class | Inherits | File | Role |
|---|---|---|---|
| `HMMParams` | `BaseHMM` | `hmm_models/hmm_params.py` | Holds transition + emission; delegates `density`, `transition_matrix`, `cdf` |
| `HMM` | — | `hmm_models/hmm.py` | User-facing wrapper; owns `HMMParams`, drives fitting and inference |
| `HMMResults` | `dataclass` | `hmm_models/results.py` | Frozen dataclass: convergence flag, log-likelihood, num_params |
| `StateResults` | `dataclass` | `hmm_models/results.py` | Frozen dataclass: filtered probs `utt`, predicted probs `ut`, pseudo-residuals |

#### Emission Models

| Class | Inherits | File | Key Parameters |
|---|---|---|---|
| `GaussEmission` | `BaseEmission` | `emissions/gauss_emission.py` | `mu0`, `log_mu_diff` (monotonic means), `log_sigma` |
| `AutoregressiveGaussEmission` | `BaseEmission` | `emissions/autoregressive_gauss_emission.py` | Same as above + `phi_tilde` (unconstrained AR coefficients) |

Both implement: `density()`, `cdf()`, `step()`, `mu()`, `sigma()`.  
Means are constrained to be **monotonically ordered** via cumulative exponentiation of `log_mu_diff`.  
AR coefficients are constrained to `(-1, 1)` via `phi_tilde → phi` transformation (stationarity).

#### Transition Models

| Class | Inherits | File | Key Parameters |
|---|---|---|---|
| `StaticTransition` | `BaseTransition` | `transitions/static_transition.py` | `transition_logits` shape `(K, K-1)` |
| `StaticTransitionHigherOrder` | `BaseTransition` | `transitions/static_transition_higher_order.py` | `transition_logits`, `order` (int); augments state space to `K^order` |

Both implement: `transition_matrix()`, `step()`.  
Logits are converted to a row-stochastic matrix via `logits_to_transition_matrix()` (from `base/utils.py`).

#### Inference Algorithm

| Class | Inherits | File | Role |
|---|---|---|---|
| `ForwardAlgorithm` | `BaseInference` | `algorithms/forward_algorithm.py` | Implements the forward (filtering) pass |
| `ForwardOutput` | `BaseOutput` | `algorithms/forward_outout.py` | Stores `ft` (scaling factors), `utt` (filtered), `ut` (predicted) |

`BaseInference.run()` uses `jax.lax.scan` over the time axis, calling `step()` at each `t`.  
`ForwardOutput.log_likelihood()` returns `sum(log(ft))`.

#### Solvers (Optimizers)

| Class | Inherits | File | Backend |
|---|---|---|---|
| `GradientSolver` | `BaseSolver` | `solvers/gradient_solver.py` | `optax` (default: Adam, lr=1e-3) |
| `LBFGSSolver` | `BaseSolver` | `solvers/lbfgs_solver.py` | `optax` L-BFGS |
| `Minimizer` | `BaseSolver` | `solvers/minimizer.py` | `jax.scipy.optimize.minimize` (default: BFGS) |

`BaseSolver` handles parameter partitioning (trainable vs frozen), builds the loss closure, and provides `_parse_frozen` / `_build_filter_spec` utilities. Each subclass implements `fit()`.

---

## Composition Relationships

```
HMM
├── params: HMMParams
│    ├── transition: BaseTransition  ←  StaticTransition
│    │                                  StaticTransitionHigherOrder
│    └── emission: BaseEmission      ←  GaussEmission
│                                       AutoregressiveGaussEmission
├── u_pre: jnp.ndarray               (initial state distribution, shape (1, K))
├── ll_fits: List[float]             (log-likelihood per iteration)
├── hmm_results: HMMResults | None
└── state_results: StateResults | None
```

---

## Data Flow: Fitting

```
HMM.fit(ys, xs, solver)
  └── BaseSolver.fit(hmm_params, ys, xs, u_pre, frozen, loss_fn)
       ├── _build_filter_spec()       partition trainable / frozen params
       ├── _build_loss_fn()           wraps negative_log_likelihood()
       │    └── ForwardAlgorithm.run(hmm_params, u_pre, ys, xs)
       │         └── jax.lax.scan over T timesteps
       │              └── ForwardAlgorithm.step(hmm_params, carry, t, ys, xs)
       │                   ├── hmm_params.transition_matrix(t, ys, xs)
       │                   └── hmm_params.density(t, ys, xs)
       └── optimizer loop (gradient steps or BFGS)
            └── returns updated HMMParams
```

---

## Data Flow: Inference (after fit)

```
HMM._compute_state_results(ys, xs)
  └── ForwardAlgorithm.run(hmm_params, u_pre, ys, xs)
       └── ForwardOutput  →  StateResults(utt, ut, pseudo_residuals)
```

---

## Utility Functions

| File | Functions |
|---|---|
| `base/utils.py` | `logits_to_transition_matrix`, `transition_matrix_to_logits` |
| `api/v4/utils.py` | `phi_to_phi_tilde`, `phi_tilde_to_phi`, `make_lag_matrix`, `load_y_data` |
| `api/v4/hmm_models/utils.py` | `AIC`, `BIC`, `LRT` |
| `api/v4/algorithms/likelihoods.py` | `negative_log_likelihood(output, hmm_params)` |
| `src/data.py` | `load_and_aggregate_data`, model save/load, init helpers |

---

## Parameter Constraints (Important for Theory Section)

| Parameter | Storage | Constraint Applied |
|---|---|---|
| `sigma` | `log_sigma` | `exp(log_sigma) > 0` |
| `mu` (ordered) | `mu0, log_mu_diff` | `mu[k] = mu[k-1] + exp(log_mu_diff[k])` |
| Transition matrix | `transition_logits` | Row-wise softmax → rows sum to 1 |
| AR coefficient `phi` | `phi_tilde` | `tanh(phi_tilde) ∈ (-1, 1)` → stationarity |

---

## Deprecated Code (`src/deprecated/`, `src/api/v1–v3/`)

The deprecated code evolved through four generations:

- **v1**: Monolithic `HMM` class (`deprecated/base/hmm.py`), no abstraction for emission/transition
- **v2**: Split into `Emission`/`Transition` base classes; log-parameterization introduced
- **v3**: Added `phi_tilde` constraint for AR models
- **v4** (current): Full redesign — `BaseHMM/BaseEmission/BaseTransition/BaseInference/BaseSolver` hierarchy, JAX `lax.scan` for inference, `equinox` pytrees throughout

The old `src/optim/` module (`BaseOptimizer`, `Minimizer`, `LBFGSOptimizer`) mirrors the v4 solver design but targets the deprecated HMM interface.

---

## Notes for Class Diagram

Recommended diagram scope (v4 only):

1. **Inheritance tree**: `BaseHMM → HMMParams`, `BaseEmission → {GaussEmission, AutoregressiveGaussEmission}`, `BaseTransition → {StaticTransition, StaticTransitionHigherOrder}`, `BaseInference → ForwardAlgorithm`, `BaseSolver → {GradientSolver, LBFGSSolver, Minimizer}`, `BaseOutput → ForwardOutput`
2. **Composition**: `HMM` ◆→ `HMMParams` ◆→ `BaseTransition` + `BaseEmission`; `HMM` ◆→ `HMMResults`, `StateResults`
3. **Uses (dependency)**: `HMM` --uses--> `BaseSolver`; `BaseSolver` --uses--> `ForwardAlgorithm`; `ForwardAlgorithm` --produces--> `ForwardOutput`
