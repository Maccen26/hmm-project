from unittest import TestCase
from src.api.v4 import HMMParams, StaticTransition, GaussEmission
import jax.numpy as jnp

class TestHMMOptimizers(TestCase):
    def setUp(self) -> None:
        self.transition_logits = jnp.array([[-0.84729786, -1.94591015], 
                                            [-2.07944154, -1.09861229], 
                                            [-1.60943791, -1.09861229]]) 
        
        self.transition_matrix = StaticTransition(self.transition_logits)
        
        self.emission_mean = jnp.array([0.0, 1.0, 2.0]) 
        self.emission_sigma = jnp.array([1.0, 1.0, 1.0])
        self.emission = GaussEmission.from_params(self.emission_mean, self.emission_sigma) 
        self.hmm_params = HMMParams(transition=self.transition_matrix, emission=self.emission) 

    def test_optimizer_initialization(self):
        pass 