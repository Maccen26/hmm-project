from base.base_hmm import BaseHMM
import jax.numpy as jnp

class HMM(BaseHMM):
    """
    HMM class that combines a transition model and an emission model. 

    The HMM class should focus on combining the transition and emission models to data (y,x) and compute the different states
    troughout the sequence.
    """


    def transition_matrix(self, xt : jnp.ndarray | None = None): 
        """
        Builds the transition matrix at time step t given the covariates at time step t.
        
        :param xt: covarites at time step t. 

        :return: transition matrix at time step t of dim (num_states, num_states) 
        """
        return self.transition.transition_matrix(xt) 
    
    def u0(self) -> jnp.ndarray:
        """
        Generate the inital state distribution (u) at time step 0. 
        
        :param self: Description
        """
        return self.transition.u0() 
    
    def density(self, yt, xt = None):
        """
        y is the observation at time step t. 
        x is the covariates at time step t. 
        Returns the emission density p(y_t | z_t, x_t) at time step t with dimensions (num_states,).
        """
        return self.emission.density(yt, xt) 
    
    def cdf(self, yt, xt = None):
        """
        y is the observation at time step t. 
        x is the covariates at time step t. 
        Returns the emission cdf P(Y_t <= y | z_t, x_t) at time step t with dimensions (num_states,).
        """
        return self.emission.cdf(yt, xt)