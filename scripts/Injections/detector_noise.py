########################################################
import scipy
import numpy as np
import ringdown
from dataclasses import dataclass
from .timing import *

def generate_strain_from_psd(psd, t_start):
    acf = psd.to_acf()
    timearray = acf.index
    N = len(timearray)
    timearray = timearray[0:(N)]
    dt = timearray[1] - timearray[0]
    
    yf = np.copy(timearray)
    asd = np.sqrt(psd.values)
    #yf[0] = np.sqrt(np.sum(asd**2))
    yf[0:((N//2)+1)] = asd
    yf[N//2] = 1
    yf[((N//2)+1):] = np.conjugate(np.flip(yf[1:(N//2)]))
    
    yf = np.sqrt(psd.values)
    phase = (1+0.0j)*np.copy(timearray)

    phase[0] = 1
    phase[1:(N//2)] = np.exp( 1j*2*np.pi*np.random.rand(len(yf[1:N//2])) )
    phase[N//2] = 1
    phase[((N//2)+1):] = np.conjugate(np.flip(phase[1:(N//2)]))

    new_noise = scipy.fft.ifft(phase*yf)

    new_noise = ringdown.Data(new_noise.real, index=x)
    
    return new_noise

def make_new_noise(noise, seed=None):
    N = len(noise)
    dt = noise.index[1] - noise.index[0]

    x = noise.index
    y = noise.values

    yf = scipy.fft.fft(y)
    phase = np.copy(yf)
    
    if seed is not None:
        rng = np.random.default_rng(seed)
        rand_phases = rng.random(len(yf[1:N//2]))
    else:
        rand_phases = np.random.random(len(yf[1:N//2]))

    phase[0] = 1
    phase[1:(N//2)] = np.exp( 1j*2*np.pi*rand_phases )
    phase[N//2] = 1
    phase[((N//2)+1):] = np.conjugate(np.flip(phase[1:(N//2)]))

    new_noise = scipy.fft.ifft(phase*yf)

    new_noise = ringdown.Data(new_noise.real, index=x)
    
    return new_noise

def get_noise_from_strain(strain, seed=None):
    noise = {}
    for ifo, event_strain in strain.items():
        noise[ifo] = make_new_noise(event_strain, seed=seed)
    return noise
    

@dataclass
class DetectorNoise:
    def __init__(self, strain=None, psd=None, timing=None, target=None, seed=None):
        self.strain = strain
        self.psd = psd
        if psd is not None:
            self.acf = {k:v.to_acf() for k,v in self.psd.items()}
        else:
            self.acf = None
        self.timing = timing
        self.target = target
        self.seed = seed
        
        if (strain is not None) and (timing is not None):
            raise ValueError("Strain dictates the timing (samplerate etc.), you can set the target using target=")
        
        if timing is None:
            if strain is None:
                raise ValueError("Need the DetectorTiming object to fix the time array")
            else:
                if (target is None):
                    raise ValueError("Need the DetectorTiming object to fix the time array")
                else:
                    self.timing = DetectorTiming.from_strain(target=self.target, strains=self.strain)
                    if not (self.strain["H1"].index[0]  < (self.target.t_H1) < self.strain["H1"].index[-1]):
                        raise ValueError("The reference time is not in the noise array")
            
        if self.strain is None:
            raise NotImplementedError("At the moment generating noise strain from PSD is not implemented")
            self.strain = generate_strain_from_psd(psd, timing)
        elif self.psd is None:
            self.psd = {k: v.get_psd() for k,v in self.strain.items()}
            self.acf = {k:v.to_acf() for k,v in self.psd.items()}
    
    def get_noise_from_strain(self,seed=None):
        if seed is None:
            seed = self.seed
        noise = {}
        for ifo, event_strain in self.strain.items():
            noise[ifo] = make_new_noise(event_strain, seed=seed)
        return noise
    
    def generate(self,seed=None):
        return self.get_noise_from_strain(seed=seed)
    
    def whiten(self, data, noise_scaling = 1):
        return {ifo: (self.acf[ifo]*noise_scaling**2).whiten(data[ifo]) for ifo in data.keys()}
    
    def __repr__(self):
        return_str = f"""DetectorNoise(
        H1: t_start = {self.timing.times["H1"][0]} , t_end = {self.timing.times["H1"][-1]}
        L1: t_start = {self.timing.times["L1"][0]} , t_end = {self.timing.times["L1"][-1]}
        detectors: {self.timing.target.detectors}, 
        seed: {self.seed}
)
        """
        return return_str
       
        
class NoNoise(DetectorNoise):
    def __init__(self):
        pass
    
#timing = DetectorTiming(Target(ra=1.95, dec=-1.27, psi=0.82, t_H1=1126259462.423), fsamp=2**14, duration=0.1)
#DN = DetectorNoise(strain=event.strain(), target=Target(ra=1.95, dec=-1.27, psi=0.82, t_H1=1126259462.423))
#DN
#######################################################
