import jax.numpy as jnp
from src.hmm.utils import recursive_filter 
from src.distribution.gaussian_emission import GaussianEmission 
import equinox as eqx 
from jax.nn import softmax
import jax

@jax.jit
def _to_transition_matrix(logits):
    return softmax(logits, axis=-1)



class StationaryTransition(eqx.Module):
    transition_logits: jnp.ndarray

    def __init__(self, transition_logits):
        self.transition_logits = transition_logits 

    
    @property
    def transition_matrix(self):
        return _to_transition_matrix(self.transition_logits) 
    
    @property
    def transition_logits(self):
        return self._transition_logits
    
    @property
    def num_states(self):
        return self.transition_logits.shape[0] 

    def init_stationary_distribution(self):
        I = jnp.eye(self.num_states)
        E = jnp.ones((self.num_states, self.num_states))
        e = jnp.ones((self.num_states, 1))
        delta = e.T @ jnp.linalg.inv(I - self.transition_matrix + E)  # (1, num_states)
        return delta.flatten()  # (num_states,) 
    



class StationaryHMM(eqx.Module):
    transition: StationaryTransition
    emission_distributions: GaussianEmission 

    def __init__(self, transition_logits, mu, log_sigma):
        self.transition = StationaryTransition(transition_logits)
        self.emission_distributions = GaussianEmission(mu, log_sigma) 

    def forward(self, y : jnp.ndarray):
        # --- t=0: initialize ---
        g = self.emission_distributions.density(y)  # (T, num_states) 

        ut0   = self.transition.init_stationary_distribution()         
        g0    =  g[0]
        ft0   = jnp.sum(ut0 * g0)
        utt0  = ut0 * g0 / ft0

        # --- t=1..T-1: scan ---
        # Transpose so scan iterates over time: (T-1, num_states)
        g_rest = g[1:]

        Ut, ft_rest, Utt = recursive_filter(utt0, g_rest, self.transition.transition_matrix)

        # --- Concatenate t=0 with t=1..T-1 ---
        ut     = jnp.concatenate([ut0[None, :],  Ut],  axis=0)  # (T, num_states)
        u_norm = jnp.concatenate([utt0[None, :], Utt], axis=0)  # (T, num_states)
        ft     = jnp.concatenate([ft0[None],     ft_rest], axis=0)  # (T,) 

        return ut, u_norm, ft
 
    def log_likelihood(self, y):
        _, _, ft = self.forward(y)
        return jnp.sum(jnp.log(ft))  
    


    



