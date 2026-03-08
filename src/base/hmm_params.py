from src.base.hmm import HMM
import jax 
import jax.numpy as jnp
import dataclasses
import equinox as eqx


class HMMParams: 
    def __init__(self, model: HMM): 
        for param in jax.tree_util.tree_leaves(model):
            if not isinstance(param, jnp.ndarray):
                raise ValueError(f"Expected all parameters to be jnp.ndarray, but got {type(param)}")
            

        self.params = flat_trainable_params(model)





def get_trainable_params_by_class(model):
    """Return: {ClassName: [ {field_name: array, ...}, ... ]}"""
    out = {}

    def walk(node):
        if isinstance(node, eqx.Module):
            cls = type(node).__name__
            instance_params = {}

            for f in dataclasses.fields(node):
                v = getattr(node, f.name)

                if eqx.is_inexact_array(v):  # float/complex JAX arrays
                    instance_params[f.name] = v
                else:
                    walk(v)

            if instance_params:
                out.setdefault(cls, []).append(instance_params)

        elif isinstance(node, (list, tuple)):
            for x in node:
                walk(x)
        elif isinstance(node, dict):
            for x in node.values():
                walk(x)

    walk(model)
    return out


def flat_trainable_params(model):
    out = {}

    def walk(node):
        if isinstance(node, eqx.Module):
            for f in dataclasses.fields(node):
                v = getattr(node, f.name)

                if eqx.is_inexact_array(v):
                    key = f.name.lstrip("_")   # "_mu" -> "mu", "_log_sigma" -> "log_sigma"
                    out[key] = v
                elif isinstance(v, eqx.Module):
                    walk(v)
                elif isinstance(v, (list, tuple)):
                    for x in v:
                        walk(x)
                elif isinstance(v, dict):
                    for x in v.values():
                        walk(x)

    walk(model)
    return out


if __name__ == "__main__":
    from src.models.stationary_hmm import StationaryHMM
    
    transition_logits = jnp.array([[ 0.0], [ 0.0]])
    mu = jnp.array([1.0, 2.0])
    
    log_sigma = jnp.array([2.0, 2.0])

    model = StationaryHMM(transition_logits, mu, log_sigma)
    params = HMMParams(model)

    print(params.params)