from src.base.base_emission import BaseEmission 
import jax.scipy.stats as stats 
import jax.numpy as jnp 
import jax 
import equinox as eqx

def phi_to_phi_tilde(phi):
    return jax.scipy.special.logit(phi)  # constrained → unconstrained

def phi_tilde_to_phi(phi_tilde):
    return jax.nn.sigmoid(phi_tilde)     # unconstrained → constrained (0, 1)

class AutoregressiveGaussEmission(BaseEmission):
    """
    Gaussian emission model for an HMM. The emission density is a Gaussian distribution with mean and variance that can depend on the covariates at time step t. 
    """ 
    log_mu_diff: jnp.ndarray
    mu0: jnp.ndarray
    log_sigma: jnp.ndarray 
    phi_tilde: jnp.ndarray 

    @classmethod
    def from_params(cls, mu, sigma, phi):
        log_mu_diff = jnp.log(jnp.diff(mu))  # Store log-differences to ensure monotonicity
        mu0 = mu[0]
        log_sigma = jnp.log(sigma)  # Store log of sigma to ensure positivity
        phi_tilde = phi_to_phi_tilde(phi)  # Store transformed phi values
        return cls(log_mu_diff, mu0, log_sigma, phi_tilde) 
    

    def density(self, t:int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        mu, sigma = self.step(t, ys, xs) 
        yt = ys[t]
        density = stats.norm.pdf(jnp.atleast_1d(yt)[:, None], loc=mu, scale=sigma) 
        k = self.phi_tilde.shape[0] #no of lags 
        return_val = jnp.where(t < k, jnp.ones_like(density), density)  #Density of 1 gives a log like of 0 
        return return_val
    
    def step(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None):
        return self.mu(t, ys, xs), self.sigma(t, ys, xs) 
    
    def mu(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None):
        base_mu = self.mu_vals(t, ys, xs)
        k = self.phi_tilde.shape[0]
        
        # Always produces shape (k,) — safe even when t < k
        lags = jax.lax.dynamic_slice(ys, (jnp.maximum(t - k, 0),), (k,))
        
        ar = jnp.sum(self.phi() * (lags[:, None] - base_mu[None, :]), axis=0)
        
        return jnp.where(t < k, base_mu, base_mu + ar)
    def mu_vals(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None):
        return jnp.concatenate([jnp.array([self.mu0]), self.mu0 + jnp.cumsum(jnp.exp(self.log_mu_diff))]) 

    def sigma(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None):
        return jnp.exp(self.log_sigma) 
    
    def phi(self):
        return phi_tilde_to_phi(self.phi_tilde)  # Transform phi_tilde back to (0, 1) range
    
    def cdf(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        mu = self.mu(t, ys, xs)
        sigma = self.sigma(t, ys, xs)
        return stats.norm.cdf(jnp.atleast_1d(ys[t])[:, None], loc=mu, scale=sigma)