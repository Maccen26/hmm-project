from src.base import BaseTransition 
import jax.numpy as jnp 


class StaticTransition(BaseTransition):
    """
    Static transition model for an HMM. The transition matrix does not depend on the covariates at time step t. 

    transition_matrix_: jnp.ndarray is of dim (num_states, num_states - 1) and contains the off-diagonal elements of the transition matrix. 
    """
    
    def step(self, t: int | None, ys: jnp.ndarray | None, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        computes new transtions logits based on the covariates at time step t. 

        
        :param self: Description
        :param xt: Description
        :return: Description
        :rtype: ndarray
        """
        return self.transition_logits 
    

