import jax.numpy as jnp 
from src.base.base_transition import BaseTransition
from src.base.base_emission import BaseEmission
from abc import ABC, abstractmethod
import equinox as eqx
from typing import Callable


class BaseHMM(ABC, eqx.Module):
    """
    Base class for a Hidden Markov Model (HMM) that combines a transition model and an emission model. 

    The HMM class should focus on combining the transition and emission models to data (y,x) and compute the different states
    troughout the sequence.
    """
    emission: BaseEmission
    transition: BaseTransition

    @abstractmethod
    def transition_matrix(self, t:int| None = None, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None) -> jnp.ndarray:  
        """
        Builds the transition matrix at time step t given the covariates at time step t.
        
        :param xt: covarites at time step t. 

        :return: transition matrix at time step t of dim (num_states, num_states) 
        """
        ... 
    
    @abstractmethod
    def density(self,  t:int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        y is the observation at time step t. 
        x is the covariates at time step t. 
        Returns the emission density p(y_t | z_t, x_t) at time step t with dimensions (num_states,).
        """
        ... 
    
    @abstractmethod
    def cdf(self, t:int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        y is the observation at time step t. 
        x is the covariates at time step t. 
        Returns the emission cdf P(Y_t <= y | z_t, x_t) at time step t with dimensions (num_states,).
        """
        ...  

    

    


