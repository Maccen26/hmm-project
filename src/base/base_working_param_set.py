import equinox as eqx 
import jax.numpy as jnp
from abc import ABC, abstractmethod 
import jax 
from typing import Tuple 

class BaseWorkingParamSet(eqx.Module, ABC):
    """
    Interface for decopuling parameters for jnp.arrays data structure. 
    You use it for example for the mu vals. They are coupled as a set of working params 

    The class should expose all parametrs as jax.arrays
    """

    @abstractmethod
    def from_natural_params(self, natural_params: jnp.ndarray) -> "BaseWorkingParamSet":
        """
        Creates a new instance of the working param set from the natural params. 
        """
        ...

    @abstractmethod
    def to_jnp_array(self) -> jax.Array:
        """
        Converts the working param set to a jnp array. 
        """
        ...


    @abstractmethod
    def get_all_working_params(self) -> jax.Array:
        """
        Returns the working params of the 
        
        :param self: Description
        :return: Description
        :rtype: Any
        """
        ... 

    @abstractmethod 
    def get_all_natural_params(self) -> jax.Array:
        """
        Returns the natural params of the working params. 
        """
        ...

    @abstractmethod
    def get_working_param(self, index: Tuple[int, int]) -> jax.Array:
        """
        Returns a working param in the set of working params. 
        """
        ...

    @abstractmethod
    def update_working_param(self, index: Tuple[int, int], new_value: jnp.ndarray) -> "BaseWorkingParamSet":
        """
        Returns a new instance of the working param set with the updated param.
        """
        ...
        
    @abstractmethod
    def get_natural_param(self, index: Tuple[int, int]) -> jax.Array:
        """
        Returns the natural param corresponding to the working param. 
        """
        ...
    
    @abstractmethod
    def update_natural_param(self, index: Tuple[int, int], new_value: jnp.ndarray) -> "BaseWorkingParamSet":
        """
        Returns a new instance of the working param set with the updated natural param.
        """
        ...
