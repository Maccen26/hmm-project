import jax.numpy as jnp
import equinox as eqx
from abc import ABC, abstractmethod
from src.base.utils import logits_to_transition_matrix, transition_matrix_to_logits


class BaseTransition(eqx.Module, ABC):
    """
    Base class for the transition component of an HMM.
    """

    transition_logits: jnp.ndarray

    @classmethod
    def from_params(cls, transition_matrix):
        transition_logits =  transition_matrix_to_logits(transition_matrix)
        return cls(transition_logits)

    def transition_matrix(self, t:int| None = None, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None) -> jnp.ndarray: 
        """
        Builds the transition matrix at time step t given the covariates at time step t.
        
        :param xt: covarites at time step t. 

        :return: transition matrix at time step t of dim (num_states, num_states) 
        """
        logits = self.step(t, ys, xs)
        return logits_to_transition_matrix(logits)

    @abstractmethod
    def step(self, t: int | None, ys: jnp.ndarray | None, xs: jnp.ndarray | None) -> jnp.ndarray:
        """
        computes new transtions logits based on the covariates at time step t. 
        Return dim is (num_states, num_states - 1) and contains the off-diagonal elements of the transition matrix.

        :type t: int
        :param t: time step
        :type ys: jnp.ndarray
        :param ys: observation sequence
        :type xs: jnp.ndarray | None
        :param xs: covariate sequence (optional)
        :return: transition logits for time step t
        :rtype: jnp.ndarray
        """
        ...




    
    
    

