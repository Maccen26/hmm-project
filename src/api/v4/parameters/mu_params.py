from src.base import BaseParam, BaseWorkingParamSet 
import jax.numpy as jnp
import jax
from typing import Tuple
import equinox as eqx
class MuParameter(BaseParam):
    """
    Class for the working parameter mu. 
    """


class MuWorkingParamSet(BaseWorkingParamSet):
    """
    Class for the set of working parameters for the mu vals. 
    """

    mu_list: list[MuParameter]  
    name : str = eqx.field(static=True, default="mu")

    def __init__(self, working_params: jax.Array):

        self.mu_list = [MuParameter(val=working_param) for working_param in working_params]

    def from_natural_params(self, natural_params: jnp.ndarray) -> "MuWorkingParamSet":
        """
        Creates a new instance of the working param set from the natural params. 
        """
        natural_params = natural_params.flatten() # Ensure it's 1D
        mu0 = natural_params[0] 
        mu_rest = jnp.log(jnp.diff(natural_params)) 
        working_params = jnp.concatenate([jnp.asarray([mu0]), mu_rest])
        return MuWorkingParamSet(working_params=working_params)
    
    def to_jnp_array(self) -> jnp.ndarray:
        """
        Converts the working param set to a jnp array. 
        """
        return jnp.atleast_2d(jnp.array([mu_param.val for mu_param in self.mu_list])).T #Ensure (1, num states) dim

    def get_all_working_params(self) -> jax.Array:
        """
        Returns the working params of the mu vals. 
        """
        return self.to_jnp_array()
    
    def get_all_natural_params(self) -> jax.Array:
        """
        Returns the natural params of the mu vals. 
        """
        working_params = self.get_all_working_params() 
        mu0 = working_params[0, 0] 
        mu_rest = working_params[0, 1:]
        natural_params = jnp.concatenate([jnp.asarray([mu0]), mu0 + jnp.cumsum(jnp.exp(mu_rest))]) 
        natural_params = jnp.atleast_2d(natural_params) #Ensure (1, num states) dim
        return natural_params
    

    def get_working_param(self, index: Tuple[int, int]) -> jax.Array:
        """
        Returns a working param in the set of working params. 
        """
        wp = self.get_all_working_params() 
        row, col = index 
        return wp[row, col]
    
    def update_working_param(self, index: Tuple[int, int], new_value: jnp.ndarray) -> "MuWorkingParamSet":
        """
        Returns a new instance of the working param set with the updated param.
        """
        _, col = index
        new_param = MuParameter(val=new_value)
        return eqx.tree_at(lambda ps: ps.mu_list[col], self, new_param)
        
    
    def get_natural_param(self, index: Tuple[int, int]) -> jax.Array:
        """
        Returns the natural param corresponding to the working param. 
        """
        np = self.get_all_natural_params() 
        row, col = index 
        return np[row, col]
    
    def update_natural_param(self, index: Tuple[int, int], new_value: jnp.ndarray) -> "MuWorkingParamSet":
        """
        Returns a new instance of the working param set with the updated natural param.
        """
        _, col = index
        natural_params = self.get_all_natural_params().flatten() # Ensure it's 1D
        updated_natural = natural_params.at[col].set(new_value.squeeze())
        return self.from_natural_params(updated_natural)
