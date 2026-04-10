from src.base.base_hmm import BaseHMM

import jax.numpy as jnp

class HMMParams(BaseHMM):
    """
    HMM class that combines a transition model and an emission model. 
    Holds trainable parameters for both the transition and emission models.
    """

    def transition_matrix(self, t:int| None = None, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None) -> jnp.ndarray: 
        """
        Builds the transition matrix at time step t given the covariates at time step t.
        
        :param xt: covarites at time step t. 

        :return: transition matrix at time step t of dim (num_states, num_states) 
        """
        return self.transition.transition_matrix(t, ys, xs) 
    

    
    def density(self, t:int, ys: jnp.ndarray, xs: jnp.ndarray | None = None):
        """
        y is the observation at time step t. 
        x is the covariates at time step t. 
        Returns the emission density p(y_t | z_t, x_t) at time step t with dimensions (num_states,).
        """
        return self.emission.density(t, ys, xs) 
    
    def cdf(self, t:int, ys: jnp.ndarray, xs: jnp.ndarray | None = None):
        """
        y is the observation at time step t. 
        x is the covariates at time step t. 
        Returns the emission cdf P(Y_t <= y | z_t, x_t) at time step t with dimensions (num_states,).
        """
        return self.emission.cdf(t, ys, xs)  
    


        
    
    