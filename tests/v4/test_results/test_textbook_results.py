from unittest import TestCase 
import jax.numpy as jnp
from src.api.v4 import HMM, StaticTransition, GaussEmission

class TestTextbookResults(TestCase):

    def _make_static_hmm(self, result_type: int) -> HMM:  
        """
        Page 329 to 332 in Jans Book
        """
        if (result_type == 4): 
            mean = jnp.array([400, 536, 890, 1307]) 
            sd = jnp.array([22, 77, 111, 197])  
            u0 = jnp.array([0.19, 0.35, 0.15, 0.32])
            Gamma = jnp.array([
            [8.5e-01, 1.5e-01, 0.0028, 9.7e-09],
            [8.3e-02, 8.5e-01, 0.0679, 6.8e-09],
            [4.3e-04, 1.6e-01, 0.6842, 1.5e-01],
            [2.6e-08, 6.1e-09, 0.0723, 9.3e-01],]
            )

        elif (result_type == 3):
            mean = jnp.array([400, 547, 1188]) 
            sd = jnp.array([24, 85, 252])
            u0 = jnp.array([0.2, 0.35, 0.45])

            Gamma = jnp.array([
                [8.5e-01, 0.15,    4.8e-51],
                [8.2e-02, 0.86,    6.3e-02],
                [3.4e-09, 0.05,    9.5e-01],])
            
        emission = GaussEmission.from_params(mean, sd)
        transition = StaticTransition.from_params(Gamma) 
        hmm = HMM(transition=transition, emission=emission, inital_distribution=u0) 
        return hmm 
    
    def test_textbook_results_3_states(self):
        hmm = self._make_static_hmm(result_type=3)
        self.assertTrue(jnp.allclose(hmm.u_pre, jnp.array([0.2, 0.35, 0.45])))
        self.assertTrue(jnp.allclose(hmm.transition.transition_logits, jnp.log(jnp.array([
                [8.5e-01, 0.15,    4.8e-51],
                [8.2e-02, 0.86,    6.3e-02],
                [3.4e-09, 0.05,    9.5e-01],]))))
        self.assertTrue(jnp.allclose(hmm.mu, jnp.array([400, 547, 1188])))
        self.assertTrue(jnp.allclose(hmm.sigma, jnp.array([24, 85, 252])))
            

        

