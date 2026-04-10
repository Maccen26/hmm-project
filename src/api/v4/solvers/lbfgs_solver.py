import equinox as eqx
from typing import Callable
from src.base.base_solver import BaseSolver


class LBFGSSolver(BaseSolver):
    def __init__(self, n_iter: int = 200):
        self.n_iter = n_iter
        self.params = None

    def fit(self, hmm_params, ys, xs=None, u_pre=None,
            frozen=None, loss_fn: Callable | None = None) -> None:
        from jaxopt import LBFGS

        filter_spec = self._build_filter_spec(hmm_params, frozen)
        trainable, static = eqx.partition(hmm_params, filter_spec)
        _loss_fn = self._build_loss_fn(static, u_pre, ys, xs, loss_fn=loss_fn)

        solver = LBFGS(fun=_loss_fn, maxiter=self.n_iter, implicit_diff=False)
        
        result = solver.run(trainable)

        self.params = eqx.combine(result.params, static)
