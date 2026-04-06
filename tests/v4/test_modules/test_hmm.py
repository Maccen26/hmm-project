from unittest import TestCase
from src.models.v4 import StaticTransition, GaussEmission, HMM
import jax.numpy as jnp 

class TestHMM(TestCase):
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
            hmm = HMM(transition=self.transition_matrix, emission=self.emission)
        except Exception as e:
            self.fail(f"Building HMM failed with error: {e}") 
