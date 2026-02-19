from src.distribution.base_dist import BaseDistribution
import jax.numpy as jnp
import jax.scipy.stats.norm as norm 
import jax


@jax.jit
def _gaussian_pdf(x, mean, std):
    """
    x    : (T,) vector of observations
    mean : (num_states,) vector
    std  : (num_states,) vector
    returns: (num_states, T) matrix
    """
    return norm.pdf(x[:, None], loc=mean, scale=std).T  # (T, num_states) -> (num_states, T)


class Gaussian(BaseDistribution): 
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.mean = mean
        self.std = std

    def density(self, x): 
        """Compute Gaussian density for numerical stability."""
        return _gaussian_pdf(x, self.mean, self.std)
    



    

if __name__ == "__main__":
        # Example usage
    
    gaussian = Gaussian(mean=0, std=1)
    x = jnp.arange(0, 100, 0.1)  # Or jnp.linspace(0, 5, 50)

    print("Density:", gaussian.density(x))

