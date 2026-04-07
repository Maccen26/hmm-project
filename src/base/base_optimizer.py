import equinox as eqx 
from abc import ABC, abstractmethod 

class BaseOptimizer(eqx.Module, ABC): 
    def __init__(self): 
        pass 