########################################################
### Abstract Injection Model
from typing import Union
from dataclasses import dataclass
import numpy as np
import scipy
import ringdown
import matplotlib.pyplot as plt
from .timing import *
from .qnm_model import *
from .detector_noise import *

@dataclass
class Injection:
    params: dict
    modes: list
    polarizations: list
    timing: Union[DetectorTiming,None] = None
    noise: Union[DetectorNoise,None] = None
    noise_scaling : float = 1
    qnm_model: QnmModel = None
    detectors: tuple = ("H1", "L1")
    strain: Union[dict, None] = None
    noise_realization: Union[dict, None] = None
    signal: Union[dict, None] = None
    
    def plot_strain(self):
        fig, axes = plt.subplots(nrows=2, figsize= (10,6))
        time_delay_dict = self.timing.target.time_delay_dict
        t_ref = self.timing.target.t_geo
        for i,ifo in enumerate(self.detectors):
            self.strain[ifo].plot(ax=axes[i], label=ifo)
            axes[i].set_title(ifo + " strain")
            axes[i].set_xlim((t_ref + time_delay_dict[ifo] - 0.03, t_ref + time_delay_dict[ifo] + 0.05))
            axes[i].axvline(t_ref + time_delay_dict[ifo], color='r', linestyle='--', label="start time")
            axes[i].legend()

        plt.tight_layout()
        
    def truncated_strain(self,t_initial, t_final=None, duration=None):
        if t_final is None:
            if duration is None:
                raise ValueError("Need one of t_final or duration")
            else:
                t_final = t_initial + duration
        
        delays = self.timing.target.time_delay_dict
        the_truncated_strain = {ifo: data[(t_initial + delays[ifo]):(t_final + delays[ifo])] for ifo, data in self.strain.items()}
        return the_truncated_strain
    
    def truncated_signal(self,t_initial, t_final=None, duration=None):
        if t_final is None:
            if duration is None:
                raise ValueError("Need one of t_final or duration")
            else:
                t_final = t_initial + duration
        
        delays = self.timing.target.time_delay_dict
        the_truncated_signal = {ifo: data[(t_initial + delays[ifo]):(t_final + delays[ifo])] for ifo, data in self.signal.items()}
        return the_truncated_signal
    
    def rescale_signal(self, rescale):
        pols = self.polarizations
        self.polarizations = [Polarization(A=pol.A*rescale, theta=pol.theta, ellip=pol.ellip, phi=pol.phi) for pol in pols]
    
    def SNR(self, t_initial=None, t_final=None, duration=None):
        if t_initial is None:
            t_initial = self.timing.target.t_geo
        
        truncated_strain = self.truncated_strain(t_initial, t_final, duration)
        truncated_signal = self.truncated_signal(t_initial, t_final, duration)
        whitened_strain = self.noise.whiten(truncated_strain, noise_scaling=self.noise_scaling)
        whitened_signal = self.noise.whiten(truncated_signal, noise_scaling=self.noise_scaling)
        
        SNR = {}
        for ifo in self.strain.keys():
            #numerator = np.dot(whitened_strain[ifo].values,whitened_signal[ifo].values)
            denomenator = np.sqrt(np.dot(whitened_signal[ifo].values,whitened_signal[ifo].values))
            SNR[ifo] = denomenator
            
        SNR['total'] = np.sqrt(np.sum([s**2 for s in SNR.values()]))
        return SNR
    
    def plot_whitened(self, t_initial=None, t_final=None, duration=0.1):
        if t_initial is None:
            t_initial = self.timing.target.t_geo
        
        truncated_strain = self.truncated_strain(t_initial, t_final, duration)
        truncated_signal = self.truncated_signal(t_initial, t_final, duration)
        whitened_strain = self.noise.whiten(truncated_strain, noise_scaling=self.noise_scaling)
        whitened_signal = self.noise.whiten(truncated_signal, noise_scaling=self.noise_scaling)
        
        fig, axes = plt.subplots(nrows=2, figsize= (10,6))
        time_delay_dict = self.timing.target.time_delay_dict
        t_ref = self.timing.target.t_geo
        for i,ifo in enumerate(self.detectors):
            whitened_strain[ifo].plot(ax=axes[i], label=f"{ifo} whitened strain", alpha=0.2)
            whitened_strain[ifo].plot(ax=axes[i], label=f"{ifo} whitened signal", alpha=0.5)
            axes[i].set_title(ifo + " strain")
            axes[i].set_xlim((t_ref + time_delay_dict[ifo] - 0.005, t_ref + time_delay_dict[ifo] + 0.05))
            axes[i].axvline(t_ref + time_delay_dict[ifo], color='r', linestyle='--', label="start time")
            axes[i].legend(loc="lower right")

        plt.tight_layout()
        
    @property
    def t_geo(self):
        return self.timing.target.t_geo
        
    def generate(self, **noise_kwargs):
        target = self.timing.target
        ra = target.ra; dec = target.dec; psi = target.psi
        
        ## Input parsing
        time = self.timing.geocent_times
        param_input = self.signal_params
        signal = ringdown.Ringdown.from_parameters(time=time, t0=self.timing.target.t_geo, **param_input)
        
        ## Projection
        signal_strain = {}
        for ifo in self.detectors:
            signal_strain[ifo] = signal.project(ifo=ifo,t0=target.t_geo, delay=target.time_delay_dict[ifo],
                                         ra=ra,dec=dec, psi=psi)
            
        noise_strain = {}
        if self.noise is not None:
            new_noise = self.noise.generate(**noise_kwargs)
            for ifo in self.detectors:
                # What we are about to do is completely fine if the sample rates of the noise and signal are the same
                initial_index = 0
                noise_strain[ifo] = new_noise[ifo].iloc[initial_index:(initial_index + len(signal_strain[ifo]))]
                noise_strain[ifo].index = signal_strain[ifo].index
                # NEW EDIT: DON'T MAKE IT WHITE EDIT: Make the noise white for now
                #noise_strain[ifo] = (1e-20)*ringdown.Data(np.random.normal(0,1,len(noise_strain[ifo].index)), 
                #                                  index=noise_strain[ifo].index
        for ifo in self.detectors:
            self.signal[ifo] = signal_strain[ifo]
            noise_default = signal_strain[ifo].copy()
            noise_default[:] = 0.0
            self.noise_realization[ifo] = noise_strain.get(ifo, noise_default)
            self.strain[ifo] = self.signal[ifo] + self.noise_scaling*self.noise_realization[ifo]

        return self.strain



#########################################
# KERR BLACK HOLES
########################################
### Specific model that can generate
class KerrInjection(Injection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strain = {}
        self.noise_realization = {}
        self.signal = {}
        #self.params["model"] = 'mchiq_exact'
        if self.qnm_model is None:
            self.qnm_model = KerrQnms(modes=self.modes)
            
        if (self.noise is not None) and (self.timing is None):
            self.timing = self.noise.timing
            
        if (self.noise is not None) and not self.noise.timing.timings_are_compatible(self.timing):
            raise ValueError("The time arrays for the noise generator and the input timing are not compatible")
            
        self.generate()
            
    @property
    def signal_params(self):
        param_input = {k:v for k,v in self.params.items() if k not in ['M','chi']};
        f,tau = self.qnm_model.generate_ftau(**{k:v for k,v in self.params.items() if k in ['M','chi']})
        param_input.update({'f': f, 'tau': tau})
        pol_params = lambda k : [getattr(p, k) for p in self.polarizations]
        param_input.update({k.lower() : pol_params(k) for k in ['A','theta','ellip', 'phi']})
        return param_input    

    
#########################################
# CHARGED BLACK HOLES
########################################
### Specific model that can generate
class ChargedInjection(Injection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strain = {}
        self.noise_realization = {}
        self.signal = {}
        #self.params["model"] = 'mchiq_exact'
        if self.qnm_model is None:
            self.qnm_model = ChargedQnms(modes=self.modes)
            
        if (self.noise is not None) and (self.timing is None):
            self.timing = self.noise.timing
            
        if (self.noise is not None) and not self.noise.timing.timings_are_compatible(self.timing):
            raise ValueError("The time arrays for the noise generator and the input timing are not compatible")
            
        self.generate()
            
    @property
    def signal_params(self):
        param_input = {k:v for k,v in self.params.items() if k not in ['M','chi']};
        f,tau = self.qnm_model.generate_ftau(**{k:v for k,v in self.params.items() if k in ['M','chi','Q']})
        param_input.update({'f': f, 'tau': tau})
        pol_params = lambda k : [getattr(p, k) for p in self.polarizations]
        param_input.update({k.lower() : pol_params(k) for k in ['A','theta','ellip', 'phi']})
        return param_input
    
    
        
#CI = ChargedInjection(
#    params    = {'M': 100, 'chi': 0.4, 'Q': 0.3},
#    modes = [Mode(n=0), Mode(n=1)],
#    polarizations = [Polarization(A=1e-21, theta=0.1, ellip=0.1, phi=0.4),
#                     Polarization(A=1e-21, theta=0.1, ellip=0.1, phi=0.4)],
#    noise = DetectorNoise(
#                     target = Target(ra=1.95, dec=-1.27, psi=0.82, t_H1=1126259462.423), 
#                     strain = event.strain(), seed=20)
#)

#data = CI.generate()
#######################################################