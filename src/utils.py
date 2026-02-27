import jax.numpy as jnp 
import jax


def _init_mu(y, num_states):
    """Initialize mu from evenly spaced quantiles of y, excluding 0 and 100."""
    quantiles = jnp.linspace(10, 90, num_states)
    return jnp.percentile(y, quantiles)

def _init_log_sigma(y, num_states):
    """Initialize log_sigma as log variance of y, shared across states."""
    return jnp.log(jnp.var(y)) * jnp.ones(num_states)

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