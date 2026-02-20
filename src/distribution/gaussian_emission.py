import equinox as eqx
import jax
import jax.numpy as jnp

from utils import _gaussian_pdf 

class GaussianEmission(eqx.Module):
    mu: jax.Array
    log_sigma: jax.Array

def __init__(self, mu, log_sigma):
    self.mu = mu
    self.log_sigma = log_sigma #We need to ensure that sigma is positive, so we can parameterize it as log_sigma and then exponentiate it when we need the actual sigma value. 

    @property
    def log_sigma(self):
        return self._log_sigma 
    
    @property
    def sigma(self):
        return jnp.exp(self._log_sigma)   
    
    @property
    def mu(self):
        return self._mu
    
    def density(self, x):
        return _gaussian_pdf(x, self.mu, self.sigma) 

    
    

