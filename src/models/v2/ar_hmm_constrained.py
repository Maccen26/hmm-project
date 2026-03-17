import jax
import jax.numpy as jnp
import equinox as eqx

from src.base.hmm import HMM
from src.models.v2.stationary_hmm import StationaryTransition
from src.models.v2.stationary_hmm import StationaryGaussianEmission


class ArGaussianEmisionBackgroundConstrained(StationaryGaussianEmission):
    phi_tilde: jnp.ndarray  # unconstrained AR(1) parameter per state, shape (num_states,)

    def __init__(self, mu, log_sigma, phi_tilde):
        super().__init__(mu, log_sigma)
        self.phi_tilde = phi_tilde

    def step(self, xt=None):
        phi = (2 * jnp.exp(self.phi_tilde)) / (1 + jnp.exp(self.phi_tilde)) - 1  # maps to (-1, 1)
        mu = jnp.concatenate([jnp.array([400.0]), self._compute_mu()])  # shape (num_states,)
        mu = mu + phi * (xt - mu)  # xt = y_{t-1}
        return mu, jnp.exp(self.log_sigma)


class ArHMMConstrained(HMM):
    def __init__(self, transition_logits, mu, log_sigma, phi_tilde):
        super().__init__(transition_logits, initial_state_dist=None)
        self.transition = StationaryTransition(transition_logits)
        self.emission = ArGaussianEmisionBackgroundConstrained(mu, log_sigma, phi_tilde)

