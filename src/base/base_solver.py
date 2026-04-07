from abc import ABC, abstractmethod
from typing import Callable
import equinox as eqx
import jax


class BaseSolver(ABC):
    """
    Stateless ABC for HMM solvers.
    After calling fit(), the fitted HMMParams are stored in self.params.
    """

    @abstractmethod
    def fit(self, hmm_params, ys, xs=None, u_pre=None,
            frozen=None, loss_fn: Callable | None = None) -> None:
        """Fit hmm_params to data. Result is stored in self.params."""
        ...

    def _build_filter_spec(self, params: eqx.Module, frozen: dict | None):
        """Returns a boolean pytree: True = trainable, False = frozen."""
        if frozen is None:
            return eqx.is_array

        def filter_fn(path, leaf):
            if path:
                last = path[-1]
                name = getattr(last, "name", getattr(last, "key", None))
                if name in frozen and frozen[name] == False:
                    return False
            return eqx.is_array(leaf)

        return jax.tree_util.tree_map_with_path(filter_fn, params)

    def _build_loss_fn(self, static, u_pre, ys, xs,
                       loss_fn: Callable | None = None) -> Callable:
        """Returns scalar loss(trainable) -> scalar."""
        from src.api.v4.likelihoods import negative_log_likelihood
        from src.api.v4.algorithms.forward_algorithm import ForwardAlgorithm

        _loss = loss_fn if loss_fn is not None else negative_log_likelihood

        def loss(trainable):
            full_params = eqx.combine(trainable, static)
            output = ForwardAlgorithm().run(full_params, u_pre, ys, xs)
            return _loss(output, full_params)

        return loss
