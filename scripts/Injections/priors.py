import pymc as pm
import aesara.tensor as at
from dataclasses import dataclass
import numpy as np
import arviz as az
from .polarizations import *

@dataclass
class AmplitudePrior:
    modes : list
    trace = None
    
    def generate(self):
        model = self.model
        # Model should either be a pymc model or a dict of computed values
        if isinstance(model,pm.model.Model):
            with self.model:
                trace = pm.sample()
            self.trace = trace
        elif isinstance(model, dict):
            self.trace = model
        else:
            raise NotImplemented("Generating from this model is not implemented yet")
    
    def sample(self):
        if self.trace is None:
            self.generate()
        if isinstance(self.trace, dict):
            N = len(list(self.trace.values())[0])
            draw = np.random.randint(N)
            return {key: value[draw] for key,value in self.trace.items()}
        elif isinstance(self.trace, az.data.inference_data.InferenceData):
            n_chains, n_draws, n_modes = self.trace.posterior.Apx.values.shape
            chain = np.random.randint(n_chains)
            draw = np.random.randint(n_draws)
            line = []
            for mode in range(n_modes):
                line.append(Polarization(**{key: getattr(self.trace.posterior,key).values[chain,draw,mode] for key in ['Apx', 'Apy', 'Acx', 'Acy']}))
            return line
        else:
            raise NotImplemented("Don't recognize this trace object")

@dataclass
class FlatAPrior(AmplitudePrior):
    A_scale : float = 2.5e-21
    flat_A : int = True
    
    @property
    def model(self):
        coords = {'mode': self.modes}
        unif_lower = [-1]*len(self.modes)
        unif_upper = [1]*len(self.modes)
        A_scale = self.A_scale
        
        def a_from_quadratures(Apx, Apy, Acx, Acy):
            A = 0.5*(at.sqrt(at.square(Acy + Apx) + at.square(Acx - Apy)) +
                     at.sqrt(at.square(Acy - Apx) + at.square(Acx + Apy)))
            return A
        
        with pm.Model(coords=coords) as model_A:
            Apx_unit = pm.Uniform("Apx_unit",lower=unif_lower, upper=unif_upper, dims=['mode'])
            Apy_unit = pm.Uniform("Apy_unit",lower=unif_lower, upper=unif_upper,dims=['mode'])
            Acx_unit = pm.Uniform("Acx_unit",lower=unif_lower, upper=unif_upper, dims=['mode'])
            Acy_unit = pm.Uniform("Acy_unit",lower=unif_lower, upper=unif_upper, dims=['mode'])

            Apx = pm.Deterministic("Apx", A_scale*Apx_unit, dims=['mode'])
            Apy = pm.Deterministic("Apy", A_scale*Apy_unit, dims=['mode'])
            Acx = pm.Deterministic("Acx", A_scale*Acx_unit, dims=['mode'])
            Acy = pm.Deterministic("Acy", A_scale*Acy_unit, dims=['mode'])
            
            A = pm.Deterministic("A", a_from_quadratures(Apx, Apy, Acx, Acy), dims=['mode'])
            
            if self.flat_A:
                pm.Potential("flat_A_prior", -3*at.sum(at.log(A)))
            
        return model_A
    
@dataclass
class GaussianAPrior(AmplitudePrior):
    A_scale : float = 2.5e-21
    flat_A : int = False
    
    @property
    def model(self):
        coords = {'mode': np.arange(len(self.modes))}
        unif_lower = [-1]*len(self.modes)
        unif_upper = [1]*len(self.modes)
        A_scale = self.A_scale
        
        def a_from_quadratures(Apx, Apy, Acx, Acy):
            A = 0.5*(at.sqrt(at.square(Acy + Apx) + at.square(Acx - Apy)) +
                     at.sqrt(at.square(Acy - Apx) + at.square(Acx + Apy)))
            return A
        
        with pm.Model(coords=coords) as model_A:
            Apx_unit = pm.Normal("Apx_unit", dims=['mode'])
            Apy_unit = pm.Normal("Apy_unit", dims=['mode'])
            Acx_unit = pm.Normal("Acx_unit", dims=['mode'])
            Acy_unit = pm.Normal("Acy_unit", dims=['mode'])

            Apx = pm.Deterministic("Apx", A_scale*Apx_unit, dims=['mode'])
            Apy = pm.Deterministic("Apy", A_scale*Apy_unit, dims=['mode'])
            Acx = pm.Deterministic("Acx", A_scale*Acx_unit, dims=['mode'])
            Acy = pm.Deterministic("Acy", A_scale*Acy_unit, dims=['mode'])
            
            A = pm.Deterministic("A", a_from_quadratures(Apx, Apy, Acx, Acy), dims=['mode'])
            
            if self.flat_A:
                pm.Potential("flat_A_prior", -3*at.sum(at.log(A)))
            
        return model_A

#AP = FlatAPrior(modes=[Mode(n=0),Mode(n=1)], A_scale=5e-21)
#AP.generate()

import pymc as pm
import aesara.tensor as at
import arviz as az

@dataclass
class BlackHoleParametersPrior:
    trace = None
    variables = None
    N_samples : int = 8000
    
    def generate(self):
        model = self.model
        # Model should either be a pymc model or a dict of computed values
        if isinstance(model,pm.model.Model):
            with self.model:
                trace = pm.sample()
            self.trace = trace
        elif isinstance(model, dict):
            self.trace = model
        else:
            raise NotImplemented("Generating from this model is not implemented yet")
        
    def sample(self):
        if self.trace is None:
            self.generate()
        if isinstance(self.trace, dict):
            N = len(list(self.trace.values())[0])
            draw = np.random.randint(N)
            return {key: value[draw] for key,value in self.trace.items()}
        elif isinstance(self.trace, az.data.inference_data.InferenceData):
            n_chains, n_draws = self.trace.posterior.M.values.shape
            chain = np.random.randint(n_chains)
            draw = np.random.randint(n_draws)
            line = {key: getattr(self.trace.posterior,key).values[chain,draw] for key in self.variables}
            return line
        else:
            raise NotImplemented("Don't recognize this trace object")

@dataclass
class KerrBlackHolePrior(BlackHoleParametersPrior):
    M_min : float = 40
    M_max : float = 200
    chi_min : float = 0.0
    chi_max : float = 0.99
    
    @property
    def model(self):
        M = self.M_min + (self.M_max - self.M_min) * np.random.rand(self.N_samples)
        chi = self.chi_min + (self.chi_max - self.chi_min) * np.random.rand(self.N_samples)
        return dict(M = M, chi = chi)
    
@dataclass
class ChargedBlackHolePrior(BlackHoleParametersPrior):
    M_min : float = 40
    M_max : float = 200
    r2_qchi_min : float = 0.0
    r2_qchi_max : float = 0.99
    theta_qchi_min : float = 0.0
    theta_qchi_max : float = np.pi/2
    
    @property
    def model(self):
        M = self.M_min + (self.M_max - self.M_min) * np.random.rand(self.N_samples)
        r2_qchi = self.r2_qchi_min + (self.r2_qchi_max - self.r2_qchi_min) * np.random.rand(self.N_samples)
        theta_qchi = self.theta_qchi_min + (self.theta_qchi_max - self.theta_qchi_min) * np.random.rand(self.N_samples)
        Q = np.sqrt(r2_qchi)*np.sin(theta_qchi)
        chi = np.sqrt(r2_qchi)*np.cos(theta_qchi)
        return dict(M = M, chi = chi, Q=Q)
 
        

#AP = FlatAPrior(modes=[Mode(n=0),Mode(n=1)], A_scale=5e-21)
#AP.generate()

#KP = ChargedBlackHolePrior()
#KP.sample()





