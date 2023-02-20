#######################################################
### Preconditioning
import copy
from dataclasses import dataclass
from typing import Union

@dataclass
class Preconditioning:
    ds: int
    duration: float = 0.1
    digital_filter: bool = True
    flow: Union[float, None] = None
    fhigh: Union[float, None] = None
    trim: Union[float, None] = None
    
    @property
    def inputs(self):
        new_outs = copy.deepcopy(self.__dict__)
        return {k:v for k,v in new_outs.items() if v is not None}
    
#PC = Preconditioning(ds=4, duration=0.1)
#######################################################