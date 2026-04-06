import equinox as eqx
from abc import ABC, abstractmethod 
import jax.numpy as jnp

class BaseEmission(eqx.Module, ABC): 
    """
    Base class for the emission component of an HMM.
    """

    @abstractmethod
    def density(self, yt, xt = None) -> jnp.ndarray:
        """
        y is the observation at time step t. 
        x is the covariates at time step t. 
        Returns the emission density p(y_t | z_t, x_t) at time step t with dimensions (num_states,).
        """
        ...  
    
    @abstractmethod
    def cdf(self, yt, xt=None) -> jnp.ndarray:
        """
        y is the observation at time step t. 
        x is the covariates at time step t. 
        Returns the emission cdf P(Y_t <= y | z_t, x_t) at time step t with dimensions (num_states,).
        """
        ...
