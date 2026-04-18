import jax
from src.base import BaseTransition 
import jax.numpy as jnp 
import equinox as eqx


def _make_2_state_transition_logits(logits): 
    """
    Converts a 2D array of transition logits of shape (num_states, num_states - 1) to a 3D array of shape (num_states, num_states, num_states) where the diagonal elements are set to a large negative value (e.g., -1000) to represent impossible self-transitions, and the off-diagonal elements are filled with the provided logits.

    :param logits: A 2D array of shape (num_states, num_states - 1) containing the transition logits for the off-diagonal elements.
    :return: A 3D array of shape (num_states, num_states, num_states) representing the full transition logits including self-transitions.
    """
    num_states = logits.shape[0]
    full_logits = jnp.full((num_states, num_states), -1000.0)  # Start with all transitions impossible
    given_indics = jnp.array([(0,0), (1,2), (2,2), (3,0)])

    zero_indics = jnp.array([(0,1), (1, 3), (2, 3), (3, 1)]) 

    rows = jnp.array([idx[0] for idx in given_indics])
    cols = jnp.array([idx[1] for idx in given_indics])
    full_logits = full_logits.at[rows, cols].set(logits.flatten())  
    rows_zero = jnp.array([idx[0] for idx in zero_indics])
    cols_zero = jnp.array([idx[1] for idx in zero_indics])
    full_logits = full_logits.at[rows_zero, cols_zero].set(0.0)  # Set the specified off-diagonal elements to 0.0 (log(1))
    return full_logits  


def logits_to_transition_matrix_higher_order(logits):
    """
    Expects a 2d build matrix with 0 entries
    """
    Gamma = jnp.exp(logits)
    return Gamma / Gamma.sum(axis=1, keepdims=True)  # Normalize rows to sum to 1


class StaticTransitionHigherOrder(BaseTransition):
    """
    Static transition model for an HMM. The transition matrix does not depend on the covariates at time step t. 

    transition_matrix_: jnp.ndarray is of dim (num_states, num_states - 1) and contains the off-diagonal elements of the transition matrix. 
    """
    order: int = eqx.field(static=True, default=2)  # Order of the Markov chain, static field since it doesn't change during training

    def __init__(self, transition_logits, order=2, num_states = 2):
        super().__init__(transition_logits)
        self.order = order 
        self._validate_transition_logits(num_states)  # Validate the shape of transition_logits based on the order

    def _validate_transition_logits(self, num_states):
        """
        Should validate that the number of possible transitions match the number of possible transition logits. 

        """
        expected_num_logits = num_states**(self.order + 1) - self.transition_logits.shape[0]  # Total transitions minus self-transitions
        actual_num_logits = self.transition_logits.size
        if actual_num_logits != expected_num_logits:
            raise ValueError(f"Expected {expected_num_logits} transition logits for order {self.order} and {num_states} states, but got {actual_num_logits}.")
        

    #Todo: Make it more geneeric. Right now we only implement for lags of 2 or 3 states.  

    def step(self, t: int | None, ys: jnp.ndarray | None, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        computes new transitiongs logits based on the order of the markov chain. 
        The goal is to make non possible transition have almost 0 probability. 

        
        :param self: Description
        :param xt: Description
        :return: Description
        :rtype: ndarray
        """

        return _make_2_state_transition_logits(self.transition_logits) 
    
    def transition_matrix(self, t:int| None = None, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None) -> jnp.ndarray: 
        """
        Builds the transition matrix at time step t given the covariates at time step t.
        
        :param xt: covarites at time step t. 

        :return: transition matrix at time step t of dim (num_states, num_states) 
        """
        logits = self.step(t, ys, xs)


        return logits_to_transition_matrix_higher_order(logits)
    

