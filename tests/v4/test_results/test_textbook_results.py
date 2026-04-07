from unittest import TestCase 
import jax.numpy as jnp
from src.api.v4 import HMM, StaticTransition, GaussEmission
from src.api.v4.utils import load_y_data


class TestTextbookResults(TestCase):

    def setUp(self) -> None:
        self.results = {
            "static": {
                "3_state": 
                    -10602.0,
                "4_state": 
                    -10389.0
            }
        }

        self.ys = load_y_data() 

        self.ERROR_TOLERANCE = 1.0  # Allowable error tolerance for log-likelihood comparisons




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
    
    def _make_autoregressive_hmm(self, result_type: int) -> HMM:
        if (result_type == 1): 
            mean = jnp.array([400, 453, 760, 1590]) 
            sd = jnp.array([13, 42, 108, 72]) 
            u0 = jnp.array([0.29, 0.21, 0.17, 0.34]) 
            Gamma = jnp.array([
            [8.2e-01, 0.0946, 0.087, 9.3e-09],
            [1.8e-01, 0.6875, 0.136, 5.5e-09],
            [9.6e-02, 0.2104, 0.414, 2.8e-01],
            [2.3e-08, 0.0049, 0.133, 8.6e-01],]
            )




    def test_setup(self):
        try:
            hmm = self._make_static_hmm(result_type=3) 
            hmm = self._make_static_hmm(result_type=4)
        except Exception as e:
            self.fail(f"Setting up HMM for textbook results failed with error: {e}") 

    def test_3_state_hmm(self):
        hmm = self._make_static_hmm(result_type=3)  
        self.assertAlmostEqual(hmm.log_likelihood(self.ys), float(self.results["static"]["3_state"]), delta=self.ERROR_TOLERANCE) 

    def test_4_state_hmm(self):
        hmm = self._make_static_hmm(result_type=4)  
        self.assertAlmostEqual(hmm.log_likelihood(self.ys), float(self.results["static"]["4_state"]), delta=self.ERROR_TOLERANCE)


    def test_4_state_autoregressive_hmm(self):
        # This is a placeholder for a future test that would check the 4-state autoregressive HMM results from Jans book
        pass
            

        

