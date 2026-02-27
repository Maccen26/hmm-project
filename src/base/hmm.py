import jax.numpy as jnp 
from jax import lax
import equinox as eqx


from src.base.transition import Transition
from src.base.emmision import Emission
import jax

class HMM(eqx.Module):
    transition: Transition
    emission: Emission

    def __init__(self, transition_logits: jnp.ndarray, initial_state_dist: jnp.ndarray | None = None, ):
        self._validate_transition_logits(transition_logits) 

        self.transition = Transition(transition_logits, initial_state_dist)
        self.emission = Emission()  


    
    def _validate_transition_logits(self, transition_logits):
        m, n = transition_logits.shape 
        if (n != (m - 1)):
            raise ValueError(f"Expected transition_logits to have shape (m, m-1), but got {transition_logits.shape}") 
        

    def forward(self, y, x=None):
        """
        implements the forward algorithm for HMMs. 
        x is of dim (T, covariestes_dim) 
        y is of dim (T, obs_dim) 

        where T is the sequence length.
        """

        ut0 = self.transition.u0()

        _, (Ut, ft, Utt) = lax.scan(self.step, ut0, y, x) 
        return Ut, ft, Utt 
    
    def step(self, ut_prev, yt, xt = None): 
        Gamma = self.transition.transition_matrix(xt) 
        u_t = ut_prev @ Gamma 

        g_t = self.emission.density(yt, xt) 

        f_t = jnp.sum(u_t * g_t) 
        u_tt = u_t * g_t / f_t

        return u_tt, (u_tt, f_t, u_t) 
    
    def filter_spec(self): 
        return jax.tree_util.tree_map(eqx.is_inexact_array, self)

    






















