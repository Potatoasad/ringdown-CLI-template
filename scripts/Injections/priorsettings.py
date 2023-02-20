#######################################################
### Prior Settings
import copy
from dataclasses import dataclass
import numpy as np

@dataclass
class PriorSettings:
    A_scale: float
    M_min: float
    M_max: float
    flat_A: bool = True
    
    @property
    def inputs(self):
        new_outs = copy.deepcopy(self.__dict__)
        return {k:v for k,v in new_outs.items() if v is not None}
    
#PS = PriorSettings(A_scale=1e-21, M_min=30, M_max=170)
#PS.inputs
#######################################################