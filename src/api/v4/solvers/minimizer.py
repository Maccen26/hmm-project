import equinox as eqx
from typing import Callable
from src.base.base_solver import BaseSolver


class Minimizer(BaseSolver):
    def __init__(self, method: str = "Nelder-Mead", n_iter: int = 1000):
        self.method = method
        self.n_iter = n_iter
        self.params = None
        self.opt_loss_val = None

    def fit(self, hmm_params, ys, xs=None, u_pre=None,
            frozen=None, loss_fn: Callable | None = None) -> None:
        from jaxopt import ScipyMinimize

        whole_frozen, element_frozen = self._parse_frozen(frozen)
        filter_spec = self._build_filter_spec(hmm_params, whole_frozen)
        trainable, static = eqx.partition(hmm_params, filter_spec)
        _loss_fn = self._build_loss_fn(static, u_pre, ys, xs, loss_fn=loss_fn,
                                       element_frozen=element_frozen,
                                       original_params=hmm_params)

        solver = ScipyMinimize(fun=_loss_fn, method=self.method,
                               maxiter=self.n_iter)
        result = solver.run(trainable)

        self.params = eqx.combine(result.params, static)
        self.params = self._restore_frozen_elements(
            self.params, element_frozen, hmm_params)
        self.opt_loss_val = float(result.state.fun_val)
