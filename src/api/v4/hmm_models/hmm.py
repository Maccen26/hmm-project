from src.api.v4.hmm_models.hmm_params import HMMParams
import jax.numpy as jnp
from typing import Callable
from src.api.v4.algorithms.forward_algorithm import ForwardAlgorithm

from src.base.base_inference import BaseInference
from src.base.base_emission import BaseEmission
from src.base.base_transition import BaseTransition


class HMM:
    def __init__(self, transition: BaseTransition, emission: BaseEmission,
                 inital_distribution=None):
        self.params = HMMParams(transition=transition, emission=emission)
        
        self.u_pre = self._set_initial_distribution(inital_distribution)

    def _set_initial_distribution(self, inital_distribution):
        if inital_distribution is not None:
            return inital_distribution
        return self._compute_stationary_distribution()

    def _compute_stationary_distribution(self):
        num_states = self.transition.transition_logits.shape[0]
        I = jnp.eye(num_states)
        E = jnp.ones((num_states, num_states))
        e = jnp.ones((num_states, 1))

        try:
            Gamma = self.transition.transition_matrix()
            delta = e.T @ jnp.linalg.inv(I - Gamma + E)
            return delta  # shape (1, num_states)
        except Exception as e:
            raise ValueError(
                f"Error computing inital state distribution. "
                f"Maybe the Stationary Transition matrix is not invertible? {e}"
            )

    @property
    def transition(self):
        return self.params.transition

    @property
    def emission(self):
        return self.params.emission

    def _set_inference_algorithm(self, inference: str) -> BaseInference:
        if inference == "forward":
            return ForwardAlgorithm()
        raise ValueError(f"Inference method {inference} could not be set")

    def fit(self, ys: jnp.ndarray, xs: jnp.ndarray | None = None,
            solver=None, frozen=None,
            loss_fn: Callable | None = None) -> None:
        if solver is None:
            from src.api.v4.solvers import GradientSolver
            solver = GradientSolver()
        solver.fit(self.params, ys, xs, u_pre=self.u_pre,
                   frozen=frozen, loss_fn=loss_fn)
        self.params = solver.params 

    

