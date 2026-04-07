from src.base import BaseInference
from src.api.v4.hmm_models.hmm_params import HMMParams 
import jax.numpy as jnp
from src.api.v4.algorithms.forward_outout import ForwardOutput
from typing import Any


class ForwardAlgorithm(BaseInference):



    def step(self, hmm_params: Any, carry: Any, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> Any:
        ut_prev = carry
        yt = ys[t]
        xt = xs[t] if xs is not None else None

        Gamma = hmm_params.transition_matrix(t, ys, xs)  # shape (num_states, num_states)
        u_t = ut_prev @ Gamma
        g_t = hmm_params.density(t, ys, xs)  # shape (1, num_states)
        f_t = jnp.sum(u_t * g_t)
        u_tt = u_t * g_t / f_t
        return u_tt, (u_tt, f_t) 
    

    def postprocess(self, carry_0, carry_final, outputs) -> ForwardOutput:
        utt, ft = outputs
        return ForwardOutput(ft=ft, utt=utt)
    
