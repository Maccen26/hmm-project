import equinox as eqx
import jax.numpy as jnp
from jax.scipy.stats import norm 

class Emission(eqx.Module): 
    def __init__(self):
        pass 

    def step(self, xt = None):
        """
        xt is the covarites at time step t. 
        Returns the the emission parameters at time step t. 
        """
        raise ValueError("Emission step function not implemented. Please implement the step function to return the emission parameters at time step t given the covariates xt.") 
    
    def density(self, yt, xt = None):
        """
        y is the observation at time step t. 
        x is the covariates at time step t. 
        Returns the emission density p(y_t | z_t, x_t) at time step t with dimensions (num_states,).
        """
        raise ValueError("Emission density function not implemented. Please implement the density function to return the emission density p(y_t | z_t, x_t) at time step t given the observation y and covariates x.")  
    


class GaussianEmission(Emission):
    mu: jnp.ndarray
    log_sigma: jnp.ndarray

    def __init__(self, mu, log_sigma):
        self.mu = mu 
        self.log_sigma = log_sigma 

    def step(self, xt = None):
        """
        Compute the emission parameters at time step t given the covariates xt. 
        """
        sigma = jnp.exp(self.log_sigma)
        return self.mu, sigma

    def density(self, yt, xt = None):
        """
        Returns the emission density for a Gaussian Distribution one for each state given the observation yt and covariates xt.
        """

        mu, sigma = self.step(xt)

        return norm.pdf(yt, loc=mu, scale=sigma)


    
