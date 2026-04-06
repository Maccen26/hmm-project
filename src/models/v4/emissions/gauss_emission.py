from src.base.base_emission import BaseEmission 
import jax.numpy as jnp 
import jax.scipy.stats as stats 


class GaussEmission(BaseEmission):
    """
    Gaussian emission model for an HMM. The emission density is a Gaussian distribution with mean and variance that can depend on the covariates at time step t. 
    """ 
    log_mu_diff: jnp.ndarray
    mu0: float
    log_sigma: jnp.ndarray 

    @classmethod
    def from_params(cls, mu, sigma):
        log_mu_diff = jnp.log(jnp.diff(mu))  # Store log-differences to ensure monotonicity
        mu0 = float(mu[0])
        log_sigma = jnp.log(sigma)  # Store log of sigma to ensure positivity
        return cls(log_mu_diff, mu0, log_sigma)

    def density(self, yt, xt=None) -> jnp.ndarray:
        mu = self.mu(xt)
        sigma = self.sigma(xt) 
        return stats.norm.pdf(yt[:, None], loc=mu, scale=sigma) 
    
    def mu(self, xt=None):
        return self.mu0 + jnp.cumsum(jnp.exp(self.log_mu_diff)) 

    def sigma(self, xt=None):
        return jnp.exp(self.log_sigma) 
    