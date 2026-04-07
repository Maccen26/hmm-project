from src.api.v4.hmm_models.hmm_params import HMMParams 
import jax.numpy as jnp 
from typing import Callable
from src.api.v4.algorithms.forward_algorithm import ForwardAlgorithm

from src.base.base_inference import BaseInference
from src.base.base_solver import BaseSolver
from src.base.base_emission import BaseEmission
from src.base.base_transition import BaseTransition


class HMM: 
    def __init__(self, transition: BaseTransition, emission: BaseEmission, inital_distribution=None): 

        self.params = HMMParams(transition=transition, emission=emission)
        self.u_pre = self._set_initial_distribution(inital_distribution)

    def _set_initial_distribution(self, inital_distribution): 
        if inital_distribution is not None: 
            return inital_distribution 
        else: 
            return self._compute_stationary_distribution() #Homogen Markov chain is assumed
        
    def _compute_stationary_distribution(self): 
        num_states = self.transition.transition_logits.shape[0]
        I = jnp.eye(num_states)
        E = jnp.ones((num_states, num_states))
        e = jnp.ones((num_states, 1))

        try:
            Gamma = self.transition.transition_matrix()
            delta = e.T @ jnp.linalg.inv(I - Gamma + E)  # (1, num_states)
            return delta.flatten()  # (num_states,)

        except Exception as e:
            raise ValueError(f"Error computing inital state distribution. Maybe the Stationary Transition matrix is not invertible? {e}")

    @property
    def transition(self):
        return self.params.transition
    
    @property
    def emission(self):
        return self.params.emission 
    
    
    def fit(self, 
            ys: jnp.ndarray, 
            xs: jnp.ndarray | None = None, 
            inference: str = "forward", 
            solver: str = "minimizer", 
            loss_fn: Callable | str = "nll"
            ):
        # Placeholder for future implementation of fitting procedure
        inference_alg = self._set_inference_algorithm(inference) 
        solver_obj   = self._set_solver(solver) 
        loss_fn       = self._set_loss_function(loss_fn)  

    def _set_inference_algorithm(self, inference: str) -> BaseInference: 
        if (inference == "forward"): 
            return ForwardAlgorithm(self.params) 
        raise ValueError(f"Inference method {inference} could not be set") 
    def _set_solver(self, solver: str) -> BaseSolver: 
        raise NotImplementedError("Solver setting not implemented yet")
    
    def _set_loss_function(self, loss_fn: Callable | str) -> Callable: 
        if (loss_fn == "nll"): 
            raise NotImplementedError("NLL loss function not implemented yet")
        
        raise ValueError(f"Loss function {loss_fn} could not be set")



        
        



        