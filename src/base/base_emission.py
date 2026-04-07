import equinox as eqx
from abc import ABC, abstractmethod 
import jax.numpy as jnp
from typing import Any

class BaseEmission(eqx.Module, ABC): 
    """
    Base class for the emission component of an HMM.
    """

    @abstractmethod
    def density(self, t:int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        ys is the observations sequence. 
        xs is the covariates sequence.
        t is the time step. 
        Returns the emission density p(y_t | z_t, x_t) at time step t with dimensions (num_states,).
        """
        ...  

    @abstractmethod
    def step(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None) -> Any:
        """
        ys is the observations sequence. 
        xs is the covariates sequence.
        t is the time step. 

        Computes the parameters of the emission distribution at time step t given the covariates at time step t. 
        The output of this function will depend on the specific emission distribution used. 
        For example, for a Gaussian emission, this function could return the mean and standard deviation of the Gaussian distribution at time step t. 
        For a Poisson emission, this function could return the rate parameter of the Poisson distribution at time step t. 
        """
        ...
    
    @abstractmethod
    def cdf(self, t:int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        ys is the observations sequence. 
        xs is the covariates sequence.
        t is the time step. 

        Returns the emission cdf P(Y_t <= y | z_t, x_t) at time step t with dimensions (num_states,).
        """
        ...
