import jax.numpy as jnp 
import jax


def _init_mu(y, num_states):
    """Initialize mu from evenly spaced quantiles of y, excluding 0 and 100."""
    quantiles = jnp.linspace(25, 75, num_states - 1)
    return jnp.percentile(y, quantiles)

def _init_log_sigma(y, num_states):
    """Initialize log_sigma as log variance of y, shared across states."""
    #return jnp.log(jnp.std(y)) * jnp.ones(num_states)
    return jnp.array([jnp.log(150)] * num_states)  
def _init_transition_logits(key, y, mu, num_states):

    # Diagonal: count observations below each mu, shape (num_states, 1)
    diag = jnp.array([jnp.sum(y < mu[i]) for i in range(num_states)], dtype=float)
    log_diag = jnp.log(diag + 1).reshape(num_states, 1)

    # Off-diagonal: random, shape (num_states, num_states - 2)
    off_diag = jax.random.normal(key, shape=(num_states, num_states - 2))

    # Concatenate along columns -> (num_states, num_states - 1)
    return jnp.concatenate([log_diag, off_diag], axis=1)

def random_init_hmm_params(key, y, num_states):
    """Initialize HMM parameters using data-driven heuristics."""
    mu = _init_mu(y, num_states)
    log_sigma = _init_log_sigma(y, num_states)
    gamma = _init_transition_logits(key, y, mu, num_states)
    return mu, log_sigma, gamma


def professor_init_3state_hmm_params():
    """
    Initialize 3-state HMM parameters exactly as the professor does in R:
        gamma.pars <- rep(-3, 6)
        pars <- c(gamma.pars, 800, 1000, rep(10, 3))

    Professor's sigma convention: sigma2 = exp(pars), sigma = sqrt(sigma2)
    Our convention:               sigma  = exp(log_sigma)
    -> log_sigma = log(sqrt(exp(10))) = 10 / 2 = 5.0
    """
    tgamma0    = jnp.full((3, 2), -3.0)
    mu0        = jnp.array([800.0, 1000.0])
    log_sigma0 = jnp.full(3, 5.0)
    return mu0, log_sigma0, tgamma0