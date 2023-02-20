#########################################################
import lal
import ringdown
import numpy as np

class Target:
    def __init__(self, ra, dec, psi, t_geo=None, t_H1=None, t_L1=None,detectors=("H1","L1")):
        self.ra = ra
        self.dec = dec
        self.psi = psi
        self.t_geo = t_geo
        self.t_H1 = t_H1
        self.t_L1 = t_L1
        self.detectors = detectors
        
        ref_time = self.t_geo or self.t_H1 or self.t_L1
        
        time_delay_dict = {}
        for ifo in self.detectors:
            time_delay_dict[ifo] = lal.TimeDelayFromEarthCenter(lal.cached_detector_by_prefix[ifo].location,
                                                                self.ra, self.dec, 
                                                                ref_time)
        if self.t_geo is not None:
            self.t_H1 = self.t_geo + time_delay_dict["H1"]
            self.t_L1 = self.t_geo + time_delay_dict["L1"]
        elif self.t_H1 is not None:
            self.t_geo = self.t_H1 - time_delay_dict["H1"]
            self.t_L1 = self.t_geo + time_delay_dict["L1"]
        elif self.t_L1 is not None:
            self.t_geo = self.t_L1 - time_delay_dict["L1"]
            self.t_H1 = self.t_geo + time_delay_dict["H1"]
            
    @property
    def inputs(self):
        return dict(ra=self.ra, dec=self.dec, psi=self.psi, t_geo=self.t_geo, detectors=self.detectors)
    
    @property
    def time_delay_dict(self):
        time_delay_dict = {}
        for ifo in self.detectors:
            time_delay_dict[ifo] = lal.TimeDelayFromEarthCenter(lal.cached_detector_by_prefix[ifo].location,
                                                                self.ra, self.dec, 
                                                                self.t_geo)
        return time_delay_dict
    
    def __repr__(self):
        my_repr = f"""Target(
        Reference Geocenter Time = {self.t_geo} s GPS Time
        Reference H1 Time        = {self.t_H1} s GPS Time
        Reference L1 Time        = {self.t_L1} s GPS Time
        ra = {self.ra}
        dec = {self.dec}
        psi = {self.psi}
        detectors = {self.detectors}
        )"""
        return my_repr
    

def generate_the_times(t0, t_start, t_end, fsamp):
    # Preference will be given to include t0 and have a fixed fsamp:
    times = np.linspace(t_start, t_end, num=int(fsamp*(t_end-t_start)))
    return times - times[np.argmin(np.abs(times-t0))] + t0

class DetectorTiming:
    def __init__(self, target : Target, fsamp, duration=None, t_start=None, t_end=None):
        self.target = target
        self.fsamp = int(fsamp)
        self.t_start = t_start
        self.t_end = t_end
        self.duration = duration
        self._times = {}
        if (self.t_start is not None) and (self.t_end is not None):
            self.duration = self.t_start - self.t_end 
        else:
            self.t_start = self.target.t_geo - self.duration/2
            self.t_end = self.target.t_geo + self.duration/2
            
        self.geocent_times = generate_the_times(self.target.t_geo, self.t_start, self.t_end, self.fsamp)
        
    @property
    def times(self):
        time_delay_dict = self.target.time_delay_dict
        self._times = {};
        
        for ifo, delay in time_delay_dict.items():
            self._times[ifo] = self.geocent_times + delay
        return self._times
    
    @classmethod
    def from_strain(cls, target : Target, strains):
        v = list(strains.values())[0]
        time_delays = target.time_delay_dict
        fsamp = int(1/(v.index[1] - v.index[0]))
        ks = list(time_delays.keys())[0]
        geocent_time = v.index - time_delays[ks]
        t_end = geocent_time[-1]
        t_start = geocent_time[0]
        return cls(target=target, fsamp=fsamp, t_start=t_start, t_end=t_end)
        
    
    def __repr__(self):
        my_repr = f"""DetectorTiming(
        Sample Rate = {self.fsamp} Hz,
        Reference Geocenter Time = {self.target.t_geo} s GPS Time
        Reference H1 Time        = {self.target.t_H1} s GPS Time
        Reference L1 Time        = {self.target.t_L1} s GPS Time
        ra = {self.target.ra}
        dec = {self.target.dec}
        psi = {self.target.psi}
        detectors = {self.target.detectors}
        )"""
        return my_repr
    
    def timings_are_compatible(self, other):
        """Overrides the default implementation"""
        if isinstance(other, DetectorTiming):
            return np.all([np.all(np.isclose(self.times[ifo],other.times[ifo])) for ifo in self.times.keys()])
            
        return False

                
#DT = DetectorTiming(Target(ra=1.95, dec=-1.27, psi=0.82, t_H1=1126259462.423), fsamp=2**14, duration=0.1)
#DT
#########################################################