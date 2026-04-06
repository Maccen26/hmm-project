import jax.numpy as jnp
import equinox as eqx
from abc import ABC, abstractmethod


class BaseTransition(eqx.Module, ABC):
    """
    Base class for the transition component of an HMM.
    """

    @abstractmethod
    def transition_matrix(self, xt : jnp.ndarray | None = None) -> jnp.ndarray: 
        """
        Builds the transition matrix at time step t given the covariates at time step t.
        
        :param xt: covarites at time step t. 

        :return: transition matrix at time step t of dim (num_states, num_states) 
        """
        ... 

    @abstractmethod
    def u0(self) -> jnp.ndarray:
        """
        Generate the inital state distribution (u) at time step 0. 
        
        :param self: Description
        """
        ... 
    
    
    

