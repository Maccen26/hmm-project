from hmm.stationary_hmm import StationaryHMM 

import jax.numpy as jnp
import jaxopt 
import jax 


@jax.jit
def likelihood_grad(model: StationaryHMM, y : jnp.ndarray):
    
    def loss(model, y):
        return -model.log_likelihood(y)  # We negate because we want to maximize likelihood, but optimizers typically minimize loss.

    solver = jaxopt.LBFGS(fun=loss, maxiter=100)
    (opt_model, log_likelihood) = solver.run(model, y)  
    return opt_model, log_likelihood 


