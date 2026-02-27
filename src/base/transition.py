import functools
import jax.numpy as jnp
import equinox as eqx
import jax



class Transition(eqx.Module):
    transition_logits: jnp.ndarray

    num_states : int = eqx.field(static=True)
    initial_state_dist: jnp.ndarray = eqx.field(static=True)

    def __init__(self, transition_logits, initial_state_dist = None):
        self.transition_logits = transition_logits

        #Static fields
        self.num_states = transition_logits.shape[0] 
        self.initial_state_dist = initial_state_dist if initial_state_dist is not None else self._compute_inital_state_distribution()

    def transition_matrix(self, xt = None): #x here should replect the covariates for the current time step
        transition_logits_t = self.step(xt = xt) 
        return self._to_transition_matrix(transition_logits_t)

    def step(self, xt = None): 
        """
        xt is the covarites at time step t. 
        Returns the the transtion logits matrix at time step t of dim (num_states, num_states) 
        """
        tGamma = jnp.zeros((self.num_states, self.num_states))
        tGamma = tGamma.at[jnp.diag_indices(self.num_states)].set(1.0)
        rows, cols = jnp.where(~jnp.eye(self.num_states, dtype=bool), size=self.num_states * (self.num_states - 1))
        tGamma = tGamma.at[rows, cols].set(self.transition_logits.flatten())
        return tGamma  
    
    def _to_transition_matrix(self, transition_logits_t):
        """
        Computes the transition matrix where each row is a probability distribution over the next state given the current state. 

        
        :param self: Description
        :param transition_logits_t: transition logits at time step t of dim (num_states, num_states)
        :return: transition matrix at time step t of dim (num_states, num_states)
        """
        return jax.nn.softmax(transition_logits_t, axis=1)
    
    def _compute_inital_state_distribution(self):
        I = jnp.eye(self.num_states)
        E = jnp.ones((self.num_states, self.num_states))
        e = jnp.ones((self.num_states, 1))

        try: 
            Gamma = self._inital_transition_matrix() 
            delta = e.T @ jnp.linalg.inv(I - Gamma + E)  # (1, num_states)
            return delta.flatten()  # (num_states,)
        
        except Exception as e:
            raise ValueError("Error computing inital state distribution. Maybe the Stationary Transition matrix is not invertible? {e}")
    

        
    def _inital_transition_matrix(self):
        try: 
            Gamma = self.transition_matrix() 
            return Gamma
        except Exception as e:
            raise ValueError(f"Error computing transition matrix for initial state distribution. Did you step function include covaries xt?:\n {e}")
        
    

