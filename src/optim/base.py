import jaxopt 
import equinox as eqx
import jax.numpy as jnp
from typing import Callable 

from src.base.hmm import HMM 


class BaseOptimizer:
    model: HMM
    loss_fn: Callable 

    def __init__(self, model: HMM, loss_fn: Callable):
        self.model = model 
        self.trainaled_parameters, self.frozen_parameters = eqx.partition(self.model, self.model.filter_spec())
        self.loss_fn = loss_fn 


    def run(self, y, x=None):
        raise NotImplementedError("Subclasses should implement this method.") 

