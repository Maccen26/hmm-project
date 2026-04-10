from unittest import TestCase
from src.api.v4 import StaticTransition, GaussEmission, HMMParams
import jax.numpy as jnp 

class TestHMMParams(TestCase):
    def setUp(self) -> None:
        self.transition_logits = jnp.array([[-0.84729786, -1.94591015], 
                                            [-2.07944154, -1.09861229], 
                                            [-1.60943791, -1.09861229]]) 
        
        self.transition_matrix = StaticTransition(self.transition_logits)
        
        self.emission_mean = jnp.array([0.0, 1.0, 2.0]) 
        self.emission_sigma = jnp.array([1.0, 1.0, 1.0])
        self.emission = GaussEmission.from_params(self.emission_mean, self.emission_sigma) 

    def test_build_hmm(self):
        try:
            hmm = HMMParams(transition=self.transition_matrix, emission=self.emission)
        except Exception as e:
            self.fail(f"Building HMM failed with error: {e}")  

    def test_get_params(self):
        hmm = HMMParams(transition=self.transition_matrix, emission=self.emission)
        try:
            transition = hmm.transition
            emission = hmm.emission
        except Exception as e:
            self.fail(f"Getting HMM parameters failed with error: {e}") 

        self.assertIsInstance(transition, StaticTransition)
        self.assertIsInstance(emission, GaussEmission) 

    
    def test_get_transition_params(self):
        hmm = HMMParams(transition=self.transition_matrix, emission=self.emission)
        try:
            transition_matrix = hmm.transition_matrix()
        except Exception as e:
            self.fail(f"Getting transition matrix from HMM failed with error: {e}") 

        self.assertTrue(jnp.allclose(transition_matrix, self.transition_matrix.transition_matrix()))


    def test_get_emission_params(self):
        hmm = HMMParams(transition=self.transition_matrix, emission=self.emission)
        ys = jnp.array([[0.0], [1.0], [2.0]])  # 3 timesteps, 1-dimensional obs
        xs = None  # No covariates
        t = 0
        try:
            density = hmm.density(t=t, ys=ys[0], xs=xs)
            cdf = hmm.cdf(t=t, ys=ys[0], xs=xs)
        except Exception as e:
            self.fail(f"Getting emission parameters from HMM failed with error: {e}")

        self.assertTrue(density.shape == (1, 3))  # Should return a vector of length num_states
        self.assertTrue(cdf.shape == (1, 3))

    def test_density_values_are_non_negative(self):
        hmm = HMMParams(transition=self.transition_matrix, emission=self.emission)
        ys = jnp.array([[0.0], [1.0], [2.0]])
        xs = None
        density = hmm.density(t=0, ys=ys[0], xs=xs)
        self.assertTrue(jnp.all(density >= 0.0))
