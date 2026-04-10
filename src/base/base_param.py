import equinox as eqx
import jax.numpy as jnp

class BaseParam(eqx.Module):
    """
    Interface for defining A working param. 
    """

    val: jnp.ndarray 

    def __init__(self, val: jnp.ndarray):
        if (not isinstance(val, jnp.ndarray)):
            raise ValueError(f"Expected val to be a jnp array, but got {type(val)}") 
        val = val.flatten()
        if (val.shape != (1,)):
            raise ValueError(f"Expected val to be a scalar with shape (1,), but got shape {val.shape}")
        
        self.val = val
    
    def update_val(self, new_val: jnp.ndarray) -> "BaseParam":
        """
        Returns a new instance of the param with the updated value. 
        """
        return self.__class__(val=new_val) 


