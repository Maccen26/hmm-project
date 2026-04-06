from src.base import BaseTransition 
import jax.numpy as jnp 
import jax

def transition_matrix_to_logits(T: jnp.ndarray) -> jnp.ndarray:
    """
    Inverse of the logits → softmax transition matrix construction.
    
    Since the diagonal of the logit matrix is fixed at 0,
    softmax is shift-invariant, so:
        logits[i, j] = log(T[i, j]) - log(T[i, i])
    
    Then extract only the off-diagonal elements.
    """
    log_T = jnp.log(T)
    # Subtract diagonal (per row) to pin diagonal logits at 0
    logits = log_T - jnp.diag(log_T)[:, None]

    # Extract off-diagonal elements
    n = T.shape[0]
    mask = ~jnp.eye(n, dtype=bool)
    return logits[mask]

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
