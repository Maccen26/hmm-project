from src.base import BaseInference
from src.models.v4 import HMM
import jax.numpy as jnp
import jax
from typing import Any


class ForwardAlgorithm(BaseInference):

    hmm: HMM

    def __init__(self, hmm: HMM):
        super().__init__(hmm)

    def initialize(self, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> Any:
        return self.hmm.u0()

    def step(self, carry: Any, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> Any:
        ut_prev = carry
        yt = ys[t]
        xt = xs[t] if xs is not None else None

        Gamma = self.hmm.transition_matrix(xt)
        u_t = ut_prev @ Gamma
        g_t = self.hmm.density(yt, xt)
        f_t = jnp.sum(u_t * g_t)
        u_tt = u_t * g_t / f_t
        return u_tt, (u_tt, f_t)

    def run(self, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> Any:
        # t=0: apply emission directly to initial distribution (no transition)
        u0 = self.hmm.u0()
        xt_0 = xs[0] if xs is not None else None
        g_0 = self.hmm.density(ys[0], xt_0)
        f_0 = jnp.sum(u0 * g_0)
        u_00 = u0 * g_0 / f_0  # shape (1, num_states) due to broadcasting

        def scan_fn(carry, t):
            return self.step(carry, t, ys, xs)

        _, outputs = jax.lax.scan(scan_fn, u_00, jnp.arange(1, len(ys)))
        u_tts, f_ts = outputs

        all_u = jnp.concatenate([u_00[None], u_tts], axis=0)
        all_f = jnp.concatenate([jnp.array([f_0]), f_ts])
        return all_u, all_f
