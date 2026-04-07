from unittest import TestCase
from src.api.v4 import StaticTransition 
import jax.numpy as jnp 
from src.base.utils import logits_to_transition_matrix, transition_matrix_to_logits


class TestStaticTransition(TestCase):
    """
    Should test build and computation of transition matrix and step function.
    """
    
    def setUp(self) -> None:
        self.transition_logits = jnp.array([[-0.84729786, -1.94591015], 
                                            [-2.07944154, -1.09861229], 
                                            [-1.60943791, -1.09861229]]) 
        
        self.transition_matrix = logits_to_transition_matrix(self.transition_logits)
        
        self.dim = self.transition_logits.shape[0]

    def test_build_from_params(self): 
        try:
            transition = StaticTransition.from_params(self.transition_matrix)
        except Exception as e:
            self.fail(f"Building StaticTransition from params failed with error: {e}") 

    def test_build_from_logits(self):
        try:
            transition = StaticTransition(self.transition_logits)
        except Exception as e:
            self.fail(f"Building StaticTransition from logits failed with error: {e}")

    def test_build_gives_same_transition_matrix(self):
        transition_from_logits = StaticTransition(self.transition_logits)

        transition_from_matrix = StaticTransition.from_params(self.transition_matrix)

        matrix_from_logits = transition_from_logits.transition_matrix()
        matrix_from_matrix = transition_from_matrix.transition_matrix()

        self.assertTrue(jnp.allclose(matrix_from_logits, self.transition_matrix))
        self.assertTrue(jnp.allclose(matrix_from_matrix, self.transition_matrix))

    def test_mapping_logits_to_matrix_and_back(self):
        matrix_from_logits = logits_to_transition_matrix(self.transition_logits) 
        self.assertEqual(matrix_from_logits.shape, (self.dim, self.dim)) 
        self.assertTrue(jnp.allclose(transition_matrix_to_logits(matrix_from_logits), self.transition_logits))

    def test_transtion_matrix_is_probability_matrix(self):
        transition = StaticTransition.from_params(self.transition_matrix)
        computed_transition_matrix = transition.transition_matrix()
        self.assertTrue(jnp.allclose(computed_transition_matrix.sum(axis=1), 1.0))
        self.assertTrue(jnp.all(computed_transition_matrix >= 0.0))

    def test_step_returns_same_logits_for_any_time(self):
        transition = StaticTransition(self.transition_logits)
        ys = jnp.array([[0.0], [1.0]])
        logits_t0 = transition.step(t=0, ys=ys, xs=None)
        logits_t5 = transition.step(t=5, ys=ys, xs=None)
        self.assertTrue(jnp.allclose(logits_t0, logits_t5))

    def test_transition_matrix_has_correct_shape(self):
        transition = StaticTransition(self.transition_logits)
        matrix = transition.transition_matrix()
        self.assertEqual(matrix.shape, (self.dim, self.dim))
        
