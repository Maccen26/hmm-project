from unittest import TestCase

from scipy import stats 
from src.models.v4 import GaussEmission 
import jax.numpy as jnp


class TestGaussEmission(TestCase):
    """
    Should test build and computation of emission parameters, density and cdf.
    """
    
    def setUp(self) -> None:
        self.mean = jnp.array([0.0, 1.0, 2.0]) 
        self.sigma = jnp.array([1.0, 1.0, 1.0])

        self.mu0 = jnp.array(self.mean[0])
        self.log_mu_diff = jnp.log(jnp.array([self.mean[i] - self.mean[i-1] for i in range(1, len(self.mean))]))
        self.log_sigma = jnp.log(self.sigma)

    def test_build(self):
        emission = GaussEmission(mu0=self.mu0, log_mu_diff=self.log_mu_diff, log_sigma=self.log_sigma)

        self.assertTrue(jnp.allclose(emission.mu0, self.mu0))
        self.assertTrue(jnp.allclose(emission.log_mu_diff, self.log_mu_diff))
        self.assertTrue(jnp.allclose(emission.log_sigma, self.log_sigma))

    def test_build_from_params(self):
        emission = GaussEmission.from_params(self.mean, self.sigma)

        self.assertTrue(jnp.allclose(emission.mu0, self.mu0))
        self.assertTrue(jnp.allclose(emission.log_mu_diff, self.log_mu_diff))
        self.assertTrue(jnp.allclose(emission.log_sigma, self.log_sigma))

    def test_density_success(self):
        emission = GaussEmission.from_params(self.mean, self.sigma)
        try: 
            yt = jnp.array([0.0])
            density = emission.density(yt)
            self.assertTrue(density.shape == (1, len(self.mean))) 
        except Exception as e:
            self.fail(f"Density computation failed with error: {e}") 

    def test_density_correctness(self):
        emission = GaussEmission.from_params(self.mean, self.sigma)
        yt = jnp.array([0.0])
        density = emission.density(yt)
        expected_density = stats.norm.pdf(yt, loc=self.mean, scale=self.sigma)
        self.assertTrue(jnp.allclose(density, expected_density))

    def test_density_dim(self): 
        y = jnp.array([0.0])
        emission = GaussEmission.from_params(self.mean, self.sigma)
        density = emission.density(y)
        self.assertTrue(density.shape == (1, len(self.mean))) #Every row is a new obs  

    def test_cdf_success(self):
        emission = GaussEmission.from_params(self.mean, self.sigma)
        try: 
            yt = jnp.array([0.0])
            cdf = emission.cdf(yt)
            self.assertTrue(cdf.shape == (1, len(self.mean))) #Every row is a new obs  
            self.assertTrue(jnp.isclose(cdf[0, 0], 0.5)) # CDF at mean should be 0.5
        except Exception as e:
            self.fail(f"CDF computation failed with error: {e}")





