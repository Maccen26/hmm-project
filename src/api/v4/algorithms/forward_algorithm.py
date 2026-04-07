from src.base import BaseInference
from src.api.v4.hmm_models.hmm_params import HMMParams 
import jax.numpy as jnp
import jax
from typing import Any


class ForwardAlgorithm(BaseInference):

    def __init__(self, hmm_params: HMMParams):
        super().__init__(hmm_params)


    def step(self, carry: Any, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> Any:
        ut_prev = carry
        yt = ys[t]
        xt = xs[t] if xs is not None else None

        Gamma = self.hmm_params.transition_matrix(t, ys, xs)  # shape (num_states, num_states)
        u_t = ut_prev @ Gamma
        g_t = self.hmm_params.density(t, yt, xt)  # shape (1, num_states)
        f_t = jnp.sum(u_t * g_t)
        u_tt = u_t * g_t / f_t
        return u_tt, (u_tt, f_t) 
    

    def postprocess(self, carry_0, carry_final, outputs) -> jnp.ndarray:
        _, ft = outputs
        return outputs
    
