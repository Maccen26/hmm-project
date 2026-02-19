
import jax.numpy as jnp
from jax import lax
import jax 


@jax.jit
def recursive_filter(Utt0, g, Gamma):
    def scan_fn(Utt_prev, g_i):
        Ut_i = Utt_prev @ Gamma
        v = Ut_i * g_i
        ft_i = jnp.sum(v)
        Utt_i = v / ft_i
        return Utt_i, (Ut_i, ft_i, Utt_i)  # (carry, stacked outputs)

    _, (Ut_i, ft_i, Utt_i)  = lax.scan(scan_fn, Utt0, g[1:])

    return Ut_i, ft_i, Utt_i


class StationaryHMM:
    def __init__(self, transition_matrix: jnp.ndarray, emission_distributions : jnp.ndarray):
        self.transition_matrix = transition_matrix
        self.emission_distributions = emission_distributions

        self.num_states = transition_matrix.shape[0] 
        self.num_obs = emission_distributions.shape[1] 
        
        #init of statedistr
        self.ut = jnp.zeros((self.num_obs, self.num_states))
        self.u_norm = jnp.zeros((self.num_obs, self.num_states)) 
        self.ft = jnp.zeros(self.num_obs)
    
    def forward(self, method : str = "filter"):
        if method == "filter":
            return self._forward_filter()
        else:
            raise ValueError("Invalid method. Must be 'filter'.") 
        

    def _forward_filter(self): 
        self.ut[:, 0] = self.init_stationary_distribution()  

        self.ft[0] = jnp.sum(self.ut[:, 0] * self.emission_distributions[:, 0]) 

        self.u_norm[:, 0] = self.ut[:, 0] * self.emission_distributions[:, 0] / self.ft[0] 

        # Recursive filtering for t > 0
        _, (Ut, ft_rest, Utt) = recursive_filter(
        self.u_norm[:, 0], self.emission_distributions[:, 1:], self.transition_matrix
    )
        
            # --- Step 3: Concatenate t=0 with t=1..T-1 ---
        self.ut     = jnp.concatenate([self.ut[None, :],  Ut],  axis=0)  # row 0, all states
        self.u_norm = jnp.concatenate([self.u_norm[None, :], Utt], axis=0)
        self.ft     = jnp.concatenate([self.ft[0][None],     ft_rest], axis=0)




    def init_stationary_distribution(self):
        # Assuming a stationary distribution over states for initialization
        # This assumes the transition matrix is already in stationary form
        I = jnp.eye(self.num_states) 
        E = jnp.ones((self.num_states, self.num_states))
        e = jnp.ones((self.num_states, 1)) 
        delta = e@jnp.linalg.inv(I - self.transition_matrix + E)
        return delta.flatten()  # Return as a 1D array for state distribution 