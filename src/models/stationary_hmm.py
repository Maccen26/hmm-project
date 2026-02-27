import jax.numpy as jnp
from src.base.hmm import HMM
from src.base.transition import Transition
from src.base.emmision import Emission  
from jax.scipy.stats import norm 

class StationaryTransition(Transition):
    def __init__(self, transition_logits):
        super().__init__(transition_logits, initial_state_dist=None) 


class StationaryGaussianEmission(Emission):
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
    


class GaussianEmisionBackground(StationaryGaussianEmission):
    def __init__(self, mu, log_sigma):
        super().__init__(mu, log_sigma)
    
    def step(self, xt = None):
        mu = jnp.concatenate([jnp.array([400]), self.mu])
        return mu, jnp.exp(self.log_sigma)
    

class StationaryHMM(HMM):
    def __init__(self, transition_logits, mu, log_sigma):
        super().__init__(transition_logits, initial_state_dist=None) 
        self.transition = StationaryTransition(transition_logits)
        self.emission = GaussianEmisionBackground(mu, log_sigma) 
