#######################################################
### Inference Object
import os
from typing import Union
import numpy as np
from numpy.linalg import inv
import ringdown
import pymc as pm
import aesara.tensor as at
import scipy
from .injection import *
from .preconditioning import *
from .priorsettings import *
from .configfile import *
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns
from .priormodels import *

@dataclass
class InferenceObject:
    injection: Injection
    preconditioning: Preconditioning
    prior_settings: PriorSettings
    target: Union[Target,None] = None
    modes: Union[modes, None] = None
    #qnm_model: Union[QnmModel,None] = None
    model: Union[str, None] = None
    fit: Union[ringdown.Fit, None] = None
    detectors: Union[list, None] = None
    _prior_fit: Union[ringdown.Fit, None] = None
    
    def __post_init__(self):
        ## detectors: inherit from injection if none
        if self.detectors is None:
            self.detectors = self.injection.detectors
        
        ## modes: inherit from injection if none
        if self.modes is None:
            self.modes = self.injection.modes
        
        ## model: inherit from injection if none
        if self.model is None:
            self.model = self.injection.qnm_model.model
            
        ## qnm_model: inherit from injection if none
        #if self.qnm_model is None:
        #    self.qnm_model = self.injection.qnm_model
            
        ## target: inherit from injection if none
        if self.target is None:
            self.target = self.injection.timing.target
        
        fit = ringdown.Fit(model=self.model, modes=[m.to_tuple() for m in self.modes])
        
        strains = self.injection.generate(seed=3)
        
        for ifo in self.detectors:
            fit.add_data(self.injection.strain[ifo], ifo=ifo)
            
        PC_dict = {k:v for k,v in self.preconditioning.__dict__.items() if ((v is not None) and (k != 'duration'))}
            
        fit.set_target(t0=self.target.t_geo, ra=self.target.ra, 
                       dec=self.target.dec, psi=self.target.psi, 
                       duration=self.preconditioning.duration)
        
        fit.condition_data(**PC_dict)
        fit.compute_acfs()
        
        fit.update_prior(**self.prior_settings.__dict__)
        
        self.config = None
        
        self.fit = fit
        
    def injection_SNR(self):
        duration = self.preconditioning.duration
        t_geo = self.target.t_geo
        return self.injection.SNR(t_initial=t_geo, duration=duration)
    
    def to_config(self, filepath):
        IO = self
        cf = ConfigFile(filepath)
        cf.read()
        cf.set_val("Injection", "model", IO.injection.qnm_model.model, as_str=True)
        cf.set_val("Injection", "modes", IO.injection.modes)
        cf.set_val("Injection", "polarizations", IO.injection.polarizations)
        cf.set_val("Injection", "target", IO.injection.timing.target)
        cf.set_val("Injection", "params", IO.injection.params)

        cf.set_val("Noise", "seed", IO.injection.noise.seed)
        cf.set_val("Noise", "strain", """db.event("GW150914").strain()""")

        cf.set_val("Analysis", "model",  IO.model, as_str=True)
        cf.set_val("Analysis", "modes",  IO.modes)
        cf.set_val("Analysis", "target",  IO.target)

        cf.set_val("Preconditioning", "preconditioning", IO.preconditioning)

        cf.set_val("Prior Settings", "priorsettings", IO.prior_settings)

        cf.write()
        return cf
        
    @classmethod
    def from_config(cls, filepath):
        cf = ConfigFile(filepath)
        cf.read()
        constructor = {"mchi" : KerrInjection, "mchiq_exact" : ChargedInjection}
        CI2 = constructor[cf["Injection","model"]](params = cf["Injection", "params"],
                      modes = cf["Injection","modes"],
                      polarizations = cf["Injection", "polarizations"],
                      noise = DetectorNoise(target = cf["Injection","target"], 
                                            strain = cf["Noise","strain"], 
                                            seed=cf["Noise","seed"])
                      )

        available_models = {'mchi': KerrQnms, 'mchiq_exact': ChargedQnms}
        
        return cls(
            injection = CI2,
            modes = cf["Analysis","modes"],
            target = cf["Analysis","target"],
            model = cf["Analysis","model"],
            preconditioning = cf["Preconditioning","preconditioning"],
            prior_settings = cf["Prior Settings","priorsettings"]
        )
    
    @classmethod
    def from_folder(cls, folderpath):
        filepath = os.path.join(folderpath, "config.ini")
        cf = ConfigFile(filepath)
        cf.read()
        constructor = {"mchi" : KerrInjection, "mchiq_exact" : ChargedInjection}
        CI2 = constructor[cf["Injection","model"]](params = cf["Injection", "params"],
                      modes = cf["Injection","modes"],
                      polarizations = cf["Injection", "polarizations"],
                      noise = DetectorNoise(target = cf["Injection","target"], 
                                            strain = cf["Noise","strain"], 
                                            seed=cf["Noise","seed"])
                      )

        available_models = {'mchi': KerrQnms, 'mchiq_exact': ChargedQnms}
        
        result = cls(
            injection = CI2,
            modes = cf["Analysis","modes"],
            target = cf["Analysis","target"],
            model = cf["Analysis","model"],
            preconditioning = cf["Preconditioning","preconditioning"],
            prior_settings = cf["Prior Settings","priorsettings"]
        )
        if os.path.exists(os.path.join(folderpath, "samples.nc")):
            result.fit.result = az.from_netcdf(os.path.join(folderpath, "samples.nc"))
        return result
    
    def save_run(self, filepath):
        self.fit.result.to_netcdf(filepath)
        
    def to_folder(self, folderpath):
        if not os.path.exists(folderpath):
            os.mkdir(folderpath)
        
        config_location = os.path.join(folderpath, "config.ini")
        self.to_config(config_location)
        
        if self.fit.result is not None:
            samples_location = os.path.join(folderpath, "samples.nc")
            self.save_run(samples_location)
            
    def plot_trace(self, var_names=None):
        IO = self
        if var_names is None:
            a_rep = lambda x : 'A' if x in ['a'] else x
            var_names = list([a_rep(a) for a in IO.injection.signal_params]) + list(IO.injection.params.keys())
        theplot = az.plot_trace(IO.fit.result, var_names=var_names)
        n_chains = len(IO.fit.result.posterior.chain)
        n_modes = len(IO.fit.result.posterior.mode)

        for param_plot in theplot:
            param = param_plot[0].title.get_text()
            if param == 'A':
                param_new = 'a'
            else:
                param_new = param

            if param_new in IO.injection.signal_params.keys():
                values = IO.injection.signal_params[param_new]
                for mode in range(n_modes):
                    param_plot[0].axvline(values[mode], c=theplot[0][0].lines[0 + mode*n_chains].get_c())
            elif param_new in IO.injection.params.keys():
                param_plot[0].axvline(IO.injection.params[param_new], c=theplot[0][0].lines[0].get_c())

        plt.show()
        
    def compute_h_det(self):
        the_posterior = self.fit.result.posterior
        t0s = self.fit.model_input['t0']
        ts = self.fit.model_input['times']
        Fps = self.fit.model_input['Fps']
        Fcs = self.fit.model_input['Fcs']

        ndet = len(self.detectors)
        nmode = len(self.modes)
        nsamp = ts[0].shape[0]
        ndraws = len(the_posterior.draw)
        nchains = len(the_posterior.chain)

        t0s = at.as_tensor_variable(t0s).reshape((1, 1, ndet, 1, 1))
        ts = at.as_tensor_variable(ts).reshape((1, 1, ndet, 1, nsamp))
        Fps = at.as_tensor_variable(Fps).reshape((1, 1, ndet, 1, 1))
        Fcs = at.as_tensor_variable(Fcs).reshape((1, 1, ndet, 1, 1))
        fs = at.as_tensor_variable(the_posterior.f.values).reshape((nchains, ndraws, 1, nmode, 1))
        gammas = at.as_tensor_variable(the_posterior.gamma.values).reshape((nchains, ndraws, 1, nmode, 1))
        Apxs = at.as_tensor_variable(the_posterior.Apx.values).reshape((nchains, ndraws, 1, nmode, 1))
        Apys = at.as_tensor_variable(the_posterior.Apy.values).reshape((nchains, ndraws, 1, nmode, 1))
        Acxs = at.as_tensor_variable(the_posterior.Acx.values).reshape((nchains, ndraws, 1, nmode, 1))
        Acys = at.as_tensor_variable(the_posterior.Acy.values).reshape((nchains, ndraws, 1, nmode, 1))

        h_det_mode = ringdown.model.rd(ts - t0s, fs, gammas, Apxs, Apys, Acxs, Acys, Fps, Fcs)
        h_det = at.sum(h_det_mode, axis=3)

        h_det_ev = h_det.eval()
        h_det_mode_ev = h_det_mode.eval()
        self.fit.result.posterior['h_det'] = (('chain', 'draw', 'ifo', 'time_index'), h_det_ev)
        self.fit.result.posterior['h_det_mode'] = (('chain', 'draw', 'ifo', 'mode', 'time_index'), h_det_mode_ev)
        
    def compute_prior(self):
        prior_mappings = {'mchi' : make_mchi_model_prior, 
                          'mchiq': make_mchiq_model_prior,
                          'mchiq_exact':make_mchiq_exact_model_prior}
        
        if self.fit.model in prior_mappings.keys():
            prior_func = prior_mappings[self.fit.model]
        else:
            raise NotImplementedError(f"The model '{self.fit.model}' is not implemented for priors")
            
        prior_model = prior_func(**self.fit.model_input)
        with prior_model:
            prior_result = pm.sample(cores=1)
            prior_trace = az.convert_to_inference_data(prior_result)
        
        self._prior_fit = self.fit.copy()
        self._prior_fit.result = prior_trace
        
    @property
    def prior_fit(self):
        if self._prior_fit is None:
            self.compute_prior()
            
        return self._prior_fit
    
    def plot_whitened_data(self, figsize=(10,8), dpi=200):
        saved_vars = list(self.fit.result.posterior.variables.keys())
        if ('h_det' not in saved_vars):
            self.compute_h_det()
        
        prior_fit = self.prior_fit
        times = self.fit.model_input['times']
        signals_post = {}
        ds = self.preconditioning.ds
        for i,ifo in enumerate(self.detectors):
            signals_post[ifo] = self.injection.signal[ifo].condition(t0 = getattr(self.target, f"t_{ifo}"), ds=ds)[times[i]]

        upper_prior = {}
        lower_prior = {}
        whitened = prior_fit.whitened_templates
        for i,ifo in enumerate(self.detectors):
            lower_prior[ifo] = np.array([np.percentile(whitened[i,j,:],5) for j in range(whitened.shape[1])])
            upper_prior[ifo] = np.array([np.percentile(whitened[i,j,:],95) for j in range(whitened.shape[1])])

        upper_posterior = {}
        lower_posterior = {}
        whitened = self.fit.whitened_templates
        for i,ifo in enumerate(self.detectors):
            lower_posterior[ifo] = np.array([np.percentile(whitened[i,j,:],5) for j in range(whitened.shape[1])])
            upper_posterior[ifo] = np.array([np.percentile(whitened[i,j,:],95) for j in range(whitened.shape[1])])
        
        fig, axes = plt.subplots(nrows=2, figsize=figsize, dpi=dpi)
        for i,ifo in enumerate(self.detectors):
            axes[i].plot((inv(self.fit.model_input['Ls'][i]) @ signals_post[ifo].values), label='whitened injection', color='k')
            axes[i].plot((inv(self.fit.model_input['Ls'][i]) @ self.fit.analysis_data[ifo].values), label='whitened strain', alpha=0.7)
            #axes[i].plot((inv(IO.fit.model_input['Ls'][i]) @ MAP_post[ifo].values), label='MAP estimate')
            axes[i].fill_between(np.arange(len(signals_post[ifo].values)), lower_prior[ifo], upper_prior[ifo], label="90% signal prior range", alpha=0.3)
            axes[i].fill_between(np.arange(len(signals_post[ifo].values)), lower_posterior[ifo], upper_posterior[ifo], label="90% signal posterior range", alpha=0.3, color='r')
            axes[i].set_title(ifo)
            axes[i].grid(alpha=0.2)
            axes[i].legend()
        return fig, axes
        
    def pair_plot(self, x='M', y='chi'):
        x_d = getattr(self.prior_fit.result.posterior,x).values.flatten()
        y_d = getattr(self.prior_fit.result.posterior,y).values.flatten()
        df = pd.DataFrame({x:x_d, y:y_d, 'dist': 'prior'})

        inj_dict = self.injection.signal_params.copy()
        inj_dict.update(self.injection.params)
        def rep_list(x):
            if isinstance(x, list):
                return x[0]
            return x
        inj_dict = {k:rep_list(v) for k,v in inj_dict.items()}

        x_d2 = getattr(self.fit.result.posterior,x).values.flatten()
        y_d2 = getattr(self.fit.result.posterior,y).values.flatten()
        df2 = pd.DataFrame({x:x_d2, y:y_d2, 'dist': 'posterior'})

        df_total = pd.concat([df,df2])
        the_plot = sns.jointplot(df_total,x=x,y=y, hue='dist', kind='kde', fill=True, alpha=1, common_norm=False)
        if (x in inj_dict.keys()) and (y in inj_dict.keys()):
            the_plot.ax_joint.scatter([inj_dict[x]],[inj_dict[y]],marker='+')
        plt.show()

    def plot_whitened_posteriors(self):
        IO = self
        fig, ax = plt.subplots(nrows=3, ncols=2, sharex='col', figsize=(20, 10))

        ms, mf, mo, mi = {}, {}, {}, {}

        for i, ifo in enumerate(self.fit.ifos):

            # mean reconstructed signal at each detector
            ms[ifo] = self.fit.result.posterior.h_det.mean(axis=(0,1)).values[i,:]

            # mean reconstructed fundamental mode at each detector
            mf[ifo] = self.fit.result.posterior.h_det_mode.mean(axis=(0,1)).values[i,0,:]

            # mean reconstructed overtone at each detector
            mo[ifo] = self.fit.result.posterior.h_det_mode.mean(axis=(0,1)).values[i,1,:]
            
            # mean reconstructed overtone at each detector
            mi[ifo] = self.injection.signal[ifo][self.fit.analysis_data[ifo].time].values
            
        for ts_dict in [ms, mf, mo, mi]:
            for i, d in ts_dict.items():
                ts_dict[i] = ringdown.Data(d, ifo=i, index=self.fit.analysis_data[i].time)

        wd = self.fit.whiten(self.fit.analysis_data)
        ws = self.fit.whiten(ms)
        wf = self.fit.whiten(mf)
        wo = self.fit.whiten(mo)
        wi = self.fit.whiten(mi)

        for i, ifo in enumerate(self.fit.ifos):
            ax[0,i].set_title(ifo)
            t = self.fit.analysis_data[ifo].time
            longest_tau = self.injection.signal_params['tau'][0]
            m = abs(t - t[0]) < 5*longest_tau
            c = sns.color_palette()[0]
            ax[0,i].plot(t[m], IO.fit.result.posterior.h_det.mean(axis=(0,1)).values[i,:][m], label=r'Total', color=c)
            ax[0,i].fill_between(t[m], np.quantile(self.fit.result.posterior.h_det.values[:,:,i,:], 0.84, axis=(0,1))[m], 
                                       np.quantile(self.fit.result.posterior.h_det.values[:,:,i,:], 0.16, axis=(0,1))[m], color=c, alpha=0.25)
            ax[0,i].plot(t[m], mi[ifo][m], label=r'Injection', color='r')
            #for j in range(2):
            #    c = sns.color_palette()[j+1]
            #    ax[0,i].plot(t[m], self.fit.result.posterior.h_det_mode.mean(axis=(0,1)).values[i,j,:][m], label=r'$n={:d}$'.format(j), color=c)
            #    ax[0,i].fill_between(t[m], np.quantile(self.fit.result.posterior.h_det_mode.values[:,:,i,j,:], 0.84, axis=(0,1))[m], 
            #                               np.quantile(self.fit.result.posterior.h_det_mode.values[:,:,i,j,:], 0.16, axis=(0,1))[m], color=c, alpha=0.25)
            #    ax[0,i].legend()
            ax[0,i].legend()

            ax[1,i].errorbar(t[m], wd[ifo][m], np.ones_like(t[m]), color='k', fmt='.')
            ax[1,i].plot(t[m], ws[ifo][m], color=sns.color_palette()[0], label='Total')
            #ax[1,i].plot(t[m], wf[ifo][m], color=sns.color_palette()[1], label='n=0')
            #ax[1,i].plot(t[m], wo[ifo][m], color=sns.color_palette()[2], label='n=1')
            ax[1,i].plot(t[m], wi[ifo][m], color='r', label='Injection')
            ax[1,i].legend()

            ax[2,i].errorbar(t[m], wd[ifo][m]-ws[ifo][m], np.ones_like(t[m]), fmt='.', color='k', label='Full Residual')
            #ax[2,i].errorbar(t[m], wd[ifo][m]-wf[ifo][m], np.ones_like(t[m]), fmt='.', color=sns.color_palette()[2], alpha=0.5, label=r'Only $n = 0$')
            #ax[2,i].errorbar(t[m], wd[ifo][m]-wo[ifo][m], np.ones_like(t[m]), fmt='.', color=sns.color_palette()[1], alpha=0.5, label=r'Only $n = 1$')
            ax[2,i].errorbar(t[m], wd[ifo][m]-wi[ifo][m], np.ones_like(t[m]), fmt='.', color='r', alpha=0.5, label=r'Injected')
            ax[2,i].legend()
#######################################################
