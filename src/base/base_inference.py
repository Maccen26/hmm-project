import equinox as eqx
from abc import abstractmethod
from typing import Any
import jax.numpy as jnp
import jax

class BaseInference(eqx.Module):
    """
    Base class for inference algorithms for HMMs.
    
    Subclasses implement `step` (single iteration) and `run` (full sequence).
    The HMM is stored as a regular field so gradients flow through it.
    """
    hmm: Any  # Should be an instance of HMM, but we avoid circular imports heres
    def __init__(self, hmm):
        self.hmm = hmm

    @abstractmethod
    def step(self, carry: Any, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> Any:
        """
        Single iteration of the algorithm.
        
        Args:
            carry: Algorithm-specific state from previous step
            t: Current time index
            ys: Full observation sequence (indexed by t inside)
            xs: Optional full covariate sequence
            
        Returns:
            (new_carry, output) tuple compatible with jax.lax.scan
        """
        ...

    def initialize(self, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> Any:
        """
        Compute initial carry (e.g., log_alpha_0 for forward algorithm).
        """
        return self.hmm.u0()

    def run(self, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> Any:
        """
        Run the full algorithm over a sequence using jax.lax.scan.
        """

        carry_0 = self.initialize(ys, xs)

        def scan_fn(carry, t):
            return self.step(carry, t, ys, xs)

        carry_final, outputs = jax.lax.scan(scan_fn, carry_0, jnp.arange(1, len(ys)))
        return self.postprocess(carry_0, carry_final, outputs)

    def postprocess(self, carry_0, carry_final, outputs):
        """
        Optional method to post-process scan outputs (e.g., compute log-likelihood from final carry).
        """
        return outputs