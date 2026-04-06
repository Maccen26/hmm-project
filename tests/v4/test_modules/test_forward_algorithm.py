from unittest import TestCase 
from src.models.v4 import ForwardAlgorithm, StaticTransition, GaussEmission, HMM 
import jax.numpy as jnp

class TestForwardAlgorithm(TestCase):
    def setUp(self) -> None:
        self.transition_logits = jnp.array([[-0.84729786, -1.94591015], 
                                            [-2.07944154, -1.09861229], 
                                            [-1.60943791, -1.09861229]]) 
        
        self.transition_matrix = StaticTransition(self.transition_logits)
        
        self.emission_mean = jnp.array([0.0, 1.0, 2.0]) 
        self.emission_sigma = jnp.array([1.0, 1.0, 1.0])
        self.emission = GaussEmission.from_params(self.emission_mean, self.emission_sigma) 

        self.hmm = HMM(transition=self.transition_matrix, emission=self.emission)

    def test_build_forward_algorithm(self):
        try:
            forward_alg = ForwardAlgorithm(self.hmm)
        except Exception as e:
            self.fail(f"Building ForwardAlgorithm failed with error: {e}") 

    def test_forward_algorithm_step(self):
        forward_alg = ForwardAlgorithm(self.hmm)
        ys = jnp.array([[0.0], [1.0], [2.0]])  # 3 timesteps, 1-dimensional obs
        xs = None  # No covariates

        carry_0 = forward_alg.initialize(ys, xs)
        try:
            carry_1, output_1 = forward_alg.step(carry_0, t=0, ys=ys, xs=xs)
            carry_2, output_2 = forward_alg.step(carry_1, t=1, ys=ys, xs=xs)
            carry_3, output_3 = forward_alg.step(carry_2, t=2, ys=ys, xs=xs)
        except Exception as e:
            self.fail(f"Forward algorithm step failed with error: {e}")

        # Check output shapes
        self.assertTrue(carry_1.shape == (1, 3))  # Should be a vector of length num_states
        self.assertTrue(carry_2.shape == (1, 3))
        self.assertTrue(carry_3.shape == (1, 3))
        self.assertTrue(isinstance(output_1, float) or isinstance(output_1, tuple))  # f_t should be a scalar
        self.assertTrue(isinstance(output_2, float) or isinstance(output_2, tuple))
        self.assertTrue(isinstance(output_3, float) or isinstance(output_3, tuple))

    def test_forward_run(self):
        forward_alg = ForwardAlgorithm(self.hmm)
        ys = jnp.array([[0.0], [1.0], [2.0]])  # 3 timesteps, 1-dimensional obs
        xs = None  # No covariates

        try:
            carry, outputs = forward_alg.run(ys, xs)
        except Exception as e:
            self.fail(f"Forward algorithm run failed with error: {e}")

        # Check final carry shape and outputs length
        #self.assertTrue(carry.shape == (1, 3))  # Final forward variable should be a vector of length num_states
    def test_forward_run_output_length(self):
        forward_alg = ForwardAlgorithm(self.hmm)
        ys = jnp.array([[0.0], [1.0], [2.0]])  # 3 timesteps, 1-dimensional obs
        xs = None  # No covariates

        carry, outputs = forward_alg.run(ys, xs)

        self.assertTrue(len(outputs) == ys.shape[0])  # Should have one output per timestep
