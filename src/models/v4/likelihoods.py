from src.models.v4.algorithms import ForwardAlgorithm
import jax.numpy as jnp

def likelihood(forward_alg: ForwardAlgorithm, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> jnp.ndarray:
    """
    Compute the likelihood of the observed sequence given the model using the forward algorithm.
    
    :param forward_alg: An instance of ForwardAlgorithm initialized with the HMM
    :type forward_alg: ForwardAlgorithm
    :param ys: observation sequence
    :type ys: jnp.ndarray
    :param xs: covariate sequence (optional)
    :type xs: jnp.ndarray | None
    :return: likelihood of the observed sequence
    :rtype: jnp.ndarray
    """
    ut, ft = forward_alg.run(ys, xs)
    total_log_likelihood = jnp.sum(jnp.log(ft))
    return total_log_likelihood 


def negative_log_likelihood(forward_alg: ForwardAlgorithm, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> jnp.ndarray:
    """
    Compute the negative log-likelihood of the observed sequence given the model using the forward algorithm.
    
    :param forward_alg: An instance of ForwardAlgorithm initialized with the HMM
    :type forward_alg: ForwardAlgorithm
    :param ys: observation sequence
    :type ys: jnp.ndarray
    :param xs: covariate sequence (optional)
    :type xs: jnp.ndarray | None
    :return: negative log-likelihood of the observed sequence
    :rtype: jnp.ndarray
    """
    total_log_likelihood = likelihood(forward_alg, ys, xs)
    return -total_log_likelihood 

