from unittest import TestCase
from src.api.v4 import HMM, StaticTransition, GaussEmission
import jax.numpy as jnp
from src.api.v4.algorithms import ForwardAlgorithm
from src.api.v4.likelihoods import likelihood, negative_log_likelihood 

class TestLikelihoods(TestCase):
    def setUp(self) -> None:
        self.transition_logits = jnp.array([[-0.84729786, -1.94591015], 
                                            [-2.07944154, -1.09861229], 
                                            [-1.60943791, -1.09861229]]) 
        
        self.transition_matrix = StaticTransition(self.transition_logits)
        
        self.emission_mean = jnp.array([0.0, 1.0, 2.0]) 
        self.emission_sigma = jnp.array([1.0, 1.0, 1.0])
        self.emission = GaussEmission.from_params(self.emission_mean, self.emission_sigma) 

        self.hmm = HMM(transition=self.transition_matrix, emission=self.emission)

        self.forward_alg = ForwardAlgorithm(self.hmm)

    def test_likelihood_computation(self):
        ys = jnp.array([[0.5], [1.5], [2.5]]) 
        try:
            ll = likelihood(self.forward_alg, ys)
            nll = negative_log_likelihood(self.forward_alg, ys)
        except Exception as e:
            self.fail(f"Likelihood computation failed with error: {e}") 
    
    def test_likelihood_non_negative(self):
        ys = jnp.array([[0.5], [1.5], [2.5]]) 
        ll = likelihood(self.forward_alg, ys)
        nll = negative_log_likelihood(self.forward_alg, ys)
        self.assertTrue(ll <= 0.0)  # Log-likelihood should be non-positive
        self.assertTrue(nll >= 0.0)  # Negative log-likelihood should be non-negative