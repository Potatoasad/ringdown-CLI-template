######################################################
import ringdown
import numpy as np
from dataclasses import dataclass
from typing import Union
from .mode import *

### Abstract QNM function model
@dataclass
class QnmModel:
    modes: list
    coefficients: Union[dict,None] = None
    model: Union[str,None] = None
    
    @property
    def inputs(self):
        return {'modes': self.modes, 'coefficients': self.coefficents}

class KerrQnms(QnmModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = "mchi"
        for mode in self.modes:
            if not self.mode_test(mode):
                raise ValueError(f"This QnmModel is not defined for {mode}")
        
    def mode_test(self, mode):
        if (mode.s != -2): raise ValueError("Only s = -2 modes are ready for this")
        if (mode.p != -1): raise ValueError("Only prograde modes are allowed p = -1")
        return True
        
    def generate_ftau(self, M, chi): # returns [f0,f1],[tau0,tau1]
        modes = self.modes
        if not np.all([self.mode_test(mode) for mode in modes]):
            raise ValueError("These modes are not supported yet")
        fs = []; taus=[];
        for mode in modes:
            fs_i, tau_i = ringdown.qnms.get_ftau(M=M,chi=chi,n=mode.n, l=mode.l)
            fs.append(fs_i); taus.append(tau_i)
        return fs, taus
    
class ChargedQnms(QnmModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = "mchiq_exact"
        for mode in self.modes:
            if not self.mode_test(mode):
                raise ValueError(f"This QnmModel is not defined for {mode}")
        
    def mode_test(self, mode):
        if (mode.s != -2): raise ValueError("Only s = -2 modes are ready for this")
        if (mode.p != -1): raise ValueError("Only prograde modes are allowed p = -1")
        if (mode.l !=  2): raise ValueError("Only l=2 are possible right now")
        if (mode.n not in [0,1,2]): raise ValueError("Only n <= 2 are possible right now")
        return True
        
    def generate_ftau(self, M, chi, Q): # returns [f0,f1],[tau0,tau1]
        return ringdown.coefficients.charged_coefficients.get_charged_ftau(M=M,chi=chi,Q=Q)

#######################################################
