from src.base import BaseTransition 
import jax.numpy as jnp 
import jax

def logits_to_transition_matrix(logits: jnp.ndarray) -> jnp.ndarray:
    """
    Inverse of transition_matrix_to_logits.
    
    Places exp(pars) in off-diagonal positions of an identity matrix
    (diagonal = 1 = exp(0), the reference), then row-normalizes.
    This is just softmax per row.
    """
    m = logits.shape[0]
    exp_pars = jnp.exp(logits.flatten())
    Gamma = jnp.eye(m)
    mask = ~jnp.eye(m, dtype=bool)
    Gamma = Gamma.at[mask].set(exp_pars)
    return Gamma / Gamma.sum(axis=1, keepdims=True)

def transition_matrix_to_logits(Gamma: jnp.ndarray) -> jnp.ndarray:
    """
    Direct translation of the R Markov.link function.
    
    Maps a transition matrix to unconstrained logit parameters
    by computing log(gamma_ij / gamma_ii) for off-diagonal entries.
    """
    m = Gamma.shape[0]
    # Zero diagonal — rowSums then gives 1 - gamma_ii
    Gamma = Gamma.at[jnp.diag_indices(m)].set(0.0)
    # 1 - rowSums recovers the original diagonal
    diag_vals = 1.0 - Gamma.sum(axis=1)
    # log-ratio: each entry divided by its row's diagonal value
    beta = jnp.log(Gamma / diag_vals[:, None])
    # Extract off-diagonal (R does transpose then extract —
    # transpose changes extraction order to column-major)
    mask = ~jnp.eye(m, dtype=bool)
    return beta[mask].reshape(m, m - 1)

class StaticTransition(BaseTransition):
    """
    Static transition model for an HMM. The transition matrix does not depend on the covariates at time step t. 

    transition_matrix_: jnp.ndarray is of dim (num_states, num_states - 1) and contains the off-diagonal elements of the transition matrix. 
    """
    transition_logits: jnp.ndarray

    @classmethod
    def from_params(cls, transition_matrix):
        transition_logits =  transition_matrix_to_logits(transition_matrix)
        return cls(transition_logits) 
    

    def transition_matrix(self, xt=None) -> jnp.ndarray:
        logits = self.step(xt)
        return jax.nn.softmax(logits, axis=1)
    
    def step(self, xt = None): 
        """
        xt is the covarites at time step t. 
        Returns the the transtion logits matrix at time step t of dim (num_states, num_states) 
        """
        num_states = self.transition_logits.shape[0] 
        tGamma = jnp.zeros((num_states, num_states))
        tGamma = tGamma.at[jnp.diag_indices(num_states)].set(0.0)
        rows, cols = jnp.where(~jnp.eye(num_states, dtype=bool), size=num_states * (num_states - 1))
        tGamma = tGamma.at[rows, cols].set(self.transition_logits.flatten())
        return tGamma  

    def u0(self):
        num_states = self.transition_logits.shape[0]
        I = jnp.eye(num_states)
        E = jnp.ones((num_states, num_states))
        e = jnp.ones((num_states, 1))

        try:
            Gamma = self.transition_matrix()
            delta = e.T @ jnp.linalg.inv(I - Gamma + E)  # (1, num_states)
            return delta.flatten()  # (num_states,)

        except Exception as e:
            raise ValueError(f"Error computing inital state distribution. Maybe the Stationary Transition matrix is not invertible? {e}")
