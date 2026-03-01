import equinox as eqx
import jax.numpy as jnp
from src.base.hmm import HMM


def negative_log_likelihood(model: HMM, y: jnp.ndarray, x: jnp.ndarray | None = None):
    _, ft, _ = model.forward(y, x)
    log_likelihood = jnp.sum(jnp.log(ft))
    return -log_likelihood 

