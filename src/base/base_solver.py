from src.base.base_hmm import BaseHMM
import equinox as eqx
from abc import ABC, abstractmethod 
import jax.numpy as jnp
from typing import Callable, Any

class BaseSolver(ABC): 
    def __init__(self, hmm_params: BaseHMM, loss_fn: Callable, spec: Any = None): 
        self.hmm_params = hmm_params 
        self.loss_fn = loss_fn
        self.frozen_parameters = None 
        if (spec is not None): 
            self.trainaled_parameters, self.frozen_parameters = eqx.partition(self.hmm_params, spec)
        self.loss_fn = self._set_loss_function(loss_fn) 

    def _set_loss_function(self, loss_fn: Callable) -> Callable: 
        if (self.frozen_parameters is None): 
            return loss_fn
        return self.loss_wrapper(loss_fn) 
    
    def loss_wrapper(self, loss_fn: Callable) -> Callable: 
        

    @abstractmethod
    def run(self, ys, xs=None): 
        """Run the optimization procedure to fit the HMM parameters to the data."""
        ... 
