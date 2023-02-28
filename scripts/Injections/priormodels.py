import numpy as np
import aesara.tensor as at
import pymc as pm

from ringdown.model import ellip_from_quadratures, a_from_quadratures, chi_factors
from ringdown.model import phiR_from_quadratures, phiL_from_quadratures, compute_h_det_mode, flat_A_quadratures_prior
FREF = 2985.668287014743
MREF = 68.0

def make_mchi_model_prior(t0, times, strains, Ls, Fps, Fcs, f_coeffs, g_coeffs,
                    **kwargs):
    M_min = kwargs.pop("M_min")
    M_max = kwargs.pop("M_max")
    chi_min = kwargs.pop("chi_min")
    chi_max = kwargs.pop("chi_max")
    A_scale = kwargs.pop("A_scale")
    #df_max = kwargs.pop("df_max")
    #dtau_max = kwargs.pop("dtau_max")
    #perturb_f = kwargs.pop("perturb_f", 0)
    #perturb_tau = kwargs.pop("perturb_tau", 0)
    flat_A = kwargs.pop("flat_A", True)
    flat_A_ellip = kwargs.pop("flat_A_ellip", False)

    if flat_A and flat_A_ellip:
        raise ValueError("at most one of `flat_A` and `flat_A_ellip` can be "
                         "`True`")
    if (chi_min < 0) or (chi_max > 1):
        raise ValueError("chi boundaries must be contained in [0, 1)")

    ndet = len(t0)
    nt = len(times[0])
    nmode = f_coeffs.shape[0]

    ifos = kwargs.pop('ifos', np.arange(ndet))
    modes = kwargs.pop('modes', np.arange(nmode))

    coords = {
        'ifo': ifos,
        'mode': modes,
        'time_index': np.arange(nt)
    }

    with pm.Model(coords=coords) as model:
        pm.ConstantData('times', times, dims=['ifo', 'time_index'])
        pm.ConstantData('t0', t0, dims=['ifo'])
        pm.ConstantData('L', Ls, dims=['ifo', 'time_index', 'time_index'])

        M = pm.Uniform("M", M_min, M_max)
        chi = pm.Uniform("chi", chi_min, chi_max)

        Apx_unit = pm.Normal("Apx_unit", dims=['mode'])
        Apy_unit = pm.Normal("Apy_unit", dims=['mode'])
        Acx_unit = pm.Normal("Acx_unit", dims=['mode'])
        Acy_unit = pm.Normal("Acy_unit", dims=['mode'])

        #df = pm.Uniform("df", -df_max, df_max, dims=['mode'])
        #dtau = pm.Uniform("dtau", -dtau_max, dtau_max, dims=['mode'])

        Apx = pm.Deterministic("Apx", A_scale*Apx_unit, dims=['mode'])
        Apy = pm.Deterministic("Apy", A_scale*Apy_unit, dims=['mode'])
        Acx = pm.Deterministic("Acx", A_scale*Acx_unit, dims=['mode'])
        Acy = pm.Deterministic("Acy", A_scale*Acy_unit, dims=['mode'])

        A = pm.Deterministic("A", a_from_quadratures(Apx, Apy, Acx, Acy),
                             dims=['mode'])
        ellip = pm.Deterministic("ellip",
            ellip_from_quadratures(Apx, Apy, Acx, Acy),
            dims=['mode'])

        f0 = FREF*MREF/M
        f = pm.Deterministic("f",
            f0*chi_factors(chi, f_coeffs),
            dims=['mode'])
        gamma = pm.Deterministic("gamma",
            f0*chi_factors(chi, g_coeffs),
            dims=['mode'])
        tau = pm.Deterministic("tau", 1/gamma, dims=['mode'])
        Q = pm.Deterministic("Q", np.pi * f * tau, dims=['mode'])
        phiR = pm.Deterministic("phiR",
             phiR_from_quadratures(Apx, Apy, Acx, Acy),
             dims=['mode'])
        phiL = pm.Deterministic("phiL",
             phiL_from_quadratures(Apx, Apy, Acx, Acy),
             dims=['mode'])
        theta = pm.Deterministic("theta", -0.5*(phiR + phiL), dims=['mode'])
        phi = pm.Deterministic("phi", 0.5*(phiR - phiL), dims=['mode'])

        h_det_mode = pm.Deterministic("h_det_mode",
                compute_h_det_mode(t0, times, Fps, Fcs, f, gamma,
                                   Apx, Apy, Acx, Acy),
                dims=['ifo', 'mode', 'time_index'])
        h_det = pm.Deterministic("h_det", at.sum(h_det_mode, axis=1),
                                 dims=['ifo', 'time_index'])

        # Priors:

        # Flat in M-chi already

        # Amplitude prior
        if flat_A:
            # bring us back to flat-in-quadratures
            pm.Potential("flat_A_quadratures_prior",
                         flat_A_quadratures_prior(Apx_unit, Apy_unit,
                                                  Acx_unit, Acy_unit))
            # bring us to flat-in-A prior
            pm.Potential("flat_A_prior", -3*at.sum(at.log(A)))
        elif flat_A_ellip:
            # bring us back to flat-in-quadratures
            pm.Potential("flat_A_quadratures_prior",
                         flat_A_quadratures_prior(Apx_unit, Apy_unit,
                                                  Acx_unit, Acy_unit))
            # bring us to flat-in-A and flat-in-ellip prior
            pm.Potential("flat_A_ellip_prior", 
                         at.sum(-3*at.log(A) - at.log1m(at.square(ellip))))

        # Flat prior on the delta-fs and delta-taus
        
        return model

    
def make_mchiq_model_prior(t0, times, strains, Ls, Fps, Fcs, f_coeffs, g_coeffs, df_coeffs, dg_coeffs,
                    **kwargs):
    M_min = kwargs.pop("M_min")
    M_max = kwargs.pop("M_max")
    A_scale = kwargs.pop("A_scale")
    r2_qchi_min = kwargs.pop("r2_qchi_min")
    r2_qchi_max = kwargs.pop("r2_qchi_max")
    theta_qchi_min = kwargs.pop("theta_qchi_min")
    theta_qchi_max = kwargs.pop("theta_qchi_max")
    chi_min = np.sqrt(r2_qchi_min)*np.cos(theta_qchi_max)
    chi_max = np.sqrt(r2_qchi_max)*np.cos(theta_qchi_min)
    flat_A = kwargs.pop("flat_A", True)
    flat_A_ellip = kwargs.pop("flat_A_ellip", False)

    if flat_A and flat_A_ellip:
        raise ValueError("at most one of `flat_A` and `flat_A_ellip` can be `True`")
    if (chi_min < 0) or (chi_max > 1):
        raise ValueError("chi boundaries must be contained in [0, 1)")

    ndet = len(t0)
    nt = len(times[0])
    nmode = f_coeffs.shape[0]

    ifos = kwargs.pop('ifos', np.arange(ndet))
    modes = kwargs.pop('modes', np.arange(nmode))

    coords = {
        'ifo': ifos,
        'mode': modes,
        'time_index': np.arange(nt)
    }


    with pm.Model(coords=coords) as model:
        pm.ConstantData('times', times, dims=['ifo', 'time_index'])
        pm.ConstantData('t0', t0, dims=['ifo'])
        pm.ConstantData('L', Ls, dims=['ifo', 'time_index', 'time_index'])

        M = pm.Uniform("M", M_min, M_max)
        r2_qchi = pm.Uniform("r2_qchi", r2_qchi_min, r2_qchi_max)
        theta_qchi = pm.Uniform("theta_qchi", theta_qchi_min, theta_qchi_max)

        q        = pm.Deterministic("q",        r2_qchi*(at.sin(theta_qchi)**2))
        Q_charge = pm.Deterministic("Q_charge", at.sqrt(q))
        chi      = pm.Deterministic("chi",      at.sqrt(r2_qchi)*(at.cos(theta_qchi)))

        Apx_unit = pm.Normal("Apx_unit", dims=['mode'])
        Apy_unit = pm.Normal("Apy_unit", dims=['mode'])
        Acx_unit = pm.Normal("Acx_unit", dims=['mode'])
        Acy_unit = pm.Normal("Acy_unit", dims=['mode'])

        Apx = pm.Deterministic("Apx", A_scale*Apx_unit, dims=['mode'])
        Apy = pm.Deterministic("Apy", A_scale*Apy_unit, dims=['mode'])
        Acx = pm.Deterministic("Acx", A_scale*Acx_unit, dims=['mode'])
        Acy = pm.Deterministic("Acy", A_scale*Acy_unit, dims=['mode'])

        A = pm.Deterministic("A", a_from_quadratures(Apx, Apy, Acx, Acy),
                             dims=['mode'])
        ellip = pm.Deterministic("ellip",
            ellip_from_quadratures(Apx, Apy, Acx, Acy),
            dims=['mode'])

        f0 = FREF*MREF/M
        f = pm.Deterministic("f", f0*(chi_factors(chi, f_coeffs) + q*chi_factors(chi, df_coeffs)), dims=['mode'])
        gamma = pm.Deterministic("gamma", f0*(chi_factors(chi, g_coeffs) + q*chi_factors(chi, dg_coeffs)), dims=['mode'])
        tau = pm.Deterministic("tau", 1/gamma, dims=['mode'])
        Q = pm.Deterministic("Q", np.pi * f * tau, dims=['mode'])
        phiR = pm.Deterministic("phiR",
             phiR_from_quadratures(Apx, Apy, Acx, Acy),
             dims=['mode'])
        phiL = pm.Deterministic("phiL",
             phiL_from_quadratures(Apx, Apy, Acx, Acy),
             dims=['mode'])
        theta = pm.Deterministic("theta", -0.5*(phiR + phiL), dims=['mode'])
        phi = pm.Deterministic("phi", 0.5*(phiR - phiL), dims=['mode'])

        h_det_mode = pm.Deterministic("h_det_mode",
                compute_h_det_mode(t0, times, Fps, Fcs, f, gamma,
                                   Apx, Apy, Acx, Acy),
                dims=['ifo', 'mode', 'time_index'])
        h_det = pm.Deterministic("h_det", at.sum(h_det_mode, axis=1),
                                 dims=['ifo', 'time_index'])


        # Priors:

        # Flat in M-chi already

        # Amplitude prior
        if flat_A:
            # bring us back to flat-in-quadratures
            pm.Potential("flat_A_quadratures_prior",
                         flat_A_quadratures_prior(Apx_unit, Apy_unit,
                                                  Acx_unit, Acy_unit))
            # bring us to flat-in-A prior
            pm.Potential("flat_A_prior", -3*at.sum(at.log(A)))
        elif flat_A_ellip:
            # bring us back to flat-in-quadratures
            pm.Potential("flat_A_quadratures_prior",
                         flat_A_quadratures_prior(Apx_unit, Apy_unit,
                                                  Acx_unit, Acy_unit))
            # bring us to flat-in-A and flat-in-ellip prior
            pm.Potential("flat_A_ellip_prior", 
                         at.sum(-3*at.log(A) - at.log1m(at.square(ellip))))

        # Flat prior on the delta-fs and delta-taus

        # Likelihood:
        #for i in range(ndet):
        #    key = ifos[i]
        #    if isinstance(key, bytes):
        #        # Don't want byte strings in our names!
        #        key = key.decode('utf-8')
        #    _ = pm.MvNormal(f"strain_{key}", mu=h_det[i,:], chol=Ls[i],
        #                    observed=strains[i], dims=['time_index'])
        
        return model


def make_mchiq_exact_model_prior(t0, times, strains, Ls, Fps, Fcs, f_coeffs, g_coeffs, b_omega, c_omega, b_gamma, c_gamma, Y0_omega, Y0_gamma,
                    **kwargs):
    M_min = kwargs.pop("M_min")
    M_max = kwargs.pop("M_max")
    A_scale = kwargs.pop("A_scale")
    r2_qchi_min = kwargs.pop("r2_qchi_min")
    r2_qchi_max = kwargs.pop("r2_qchi_max")
    theta_qchi_min = kwargs.pop("theta_qchi_min")
    theta_qchi_max = kwargs.pop("theta_qchi_max")
    chi_min = np.sqrt(r2_qchi_min)*np.cos(theta_qchi_max)
    chi_max = np.sqrt(r2_qchi_max)*np.cos(theta_qchi_min)
    flat_A = kwargs.pop("flat_A", True)
    flat_A_ellip = kwargs.pop("flat_A_ellip", False)

    if flat_A and flat_A_ellip:
        raise ValueError("at most one of `flat_A` and `flat_A_ellip` can be `True`")
    if (chi_min < 0) or (chi_max > 1):
        raise ValueError("chi boundaries must be contained in [0, 1)")

    ndet = len(t0)
    nt = len(times[0])
    nmode = f_coeffs.shape[0]

    ifos = kwargs.pop('ifos', np.arange(ndet))
    modes = kwargs.pop('modes', np.arange(nmode))

    coords = {
        'ifo': ifos,
        'mode': modes,
        'time_index': np.arange(nt)
    }

    Y0_bij_omega = at.as_tensor_variable([Y0_omega[mode]*np.array(b_omega[mode]) for mode in range(len(b_omega))])
    cij_omega = at.as_tensor_variable(c_omega)

    Y0_bij_gamma = at.as_tensor_variable([Y0_gamma[mode]*np.array(b_gamma[mode]) for mode in range(len(b_gamma))])
    cij_gamma = at.as_tensor_variable(c_gamma)


    with pm.Model(coords=coords) as model:
        pm.ConstantData('times', times)
        pm.ConstantData('t0', t0)
        pm.ConstantData('L', Ls)

        M = pm.Uniform("M", M_min, M_max)
        r2_qchi = pm.Uniform("r2_qchi", r2_qchi_min, r2_qchi_max)
        theta_qchi = pm.Uniform("theta_qchi", theta_qchi_min, theta_qchi_max)

        q        = pm.Deterministic("q",        r2_qchi*(at.sin(theta_qchi)**2))
        Q_charge = pm.Deterministic("Q_charge", at.sqrt(q))
        chi      = pm.Deterministic("chi",      at.sqrt(r2_qchi)*(at.cos(theta_qchi)))

        Apx_unit = pm.Normal("Apx_unit", dims=['mode'])
        Apy_unit = pm.Normal("Apy_unit", dims=['mode'])
        Acx_unit = pm.Normal("Acx_unit", dims=['mode'])
        Acy_unit = pm.Normal("Acy_unit", dims=['mode'])

        Apx = pm.Deterministic("Apx", A_scale*Apx_unit, dims=['mode'])
        Apy = pm.Deterministic("Apy", A_scale*Apy_unit, dims=['mode'])
        Acx = pm.Deterministic("Acx", A_scale*Acx_unit, dims=['mode'])
        Acy = pm.Deterministic("Acy", A_scale*Acy_unit, dims=['mode'])

        A = pm.Deterministic("A", a_from_quadratures(Apx, Apy, Acx, Acy),
                             dims=['mode'])
        ellip = pm.Deterministic("ellip",
            ellip_from_quadratures(Apx, Apy, Acx, Acy),
            dims=['mode'])

        f0 = FREF*MREF/M
        f = pm.Deterministic("f", (f0/(2*np.pi))*(chiq_exact_factors(chi, Q_charge, Y0_bij_omega, cij_omega)), dims=['mode'])
        gamma = pm.Deterministic("gamma", f0*(chiq_exact_factors(chi, Q_charge, Y0_bij_gamma, cij_gamma)), dims=['mode'])
        tau = pm.Deterministic("tau", 1/gamma, dims=['mode'])
        Q = pm.Deterministic("Q", np.pi * f * tau, dims=['mode'])
        phiR = pm.Deterministic("phiR",
             phiR_from_quadratures(Apx, Apy, Acx, Acy),
             dims=['mode'])
        phiL = pm.Deterministic("phiL",
             phiL_from_quadratures(Apx, Apy, Acx, Acy),
             dims=['mode'])
        theta = pm.Deterministic("theta", -0.5*(phiR + phiL), dims=['mode'])
        phi = pm.Deterministic("phi", 0.5*(phiR - phiL), dims=['mode'])

        h_det_mode = pm.Deterministic("h_det_mode",
                compute_h_det_mode(t0, times, Fps, Fcs, f, gamma,
                                   Apx, Apy, Acx, Acy),
                dims=['ifo', 'mode', 'time_index'])
        h_det = pm.Deterministic("h_det", at.sum(h_det_mode, axis=1),
                                 dims=['ifo', 'time_index'])

        # Priors:

        # Flat in M-chi already

        # Amplitude prior
        if flat_A:
            # bring us back to flat-in-quadratures
            pm.Potential("flat_A_quadratures_prior",
                         flat_A_quadratures_prior(Apx_unit, Apy_unit,
                                                  Acx_unit, Acy_unit))
            # bring us to flat-in-A prior
            pm.Potential("flat_A_prior", -3*at.sum(at.log(A)))
        elif flat_A_ellip:
            # bring us back to flat-in-quadratures
            pm.Potential("flat_A_quadratures_prior",
                         flat_A_quadratures_prior(Apx_unit, Apy_unit,
                                                  Acx_unit, Acy_unit))
            # bring us to flat-in-A and flat-in-ellip prior
            pm.Potential("flat_A_ellip_prior", 
                         at.sum(-3*at.log(A) - at.log1m(at.square(ellip))))

        # Flat prior on the delta-fs and delta-taus

        # Likelihood:
        #for i in range(ndet):
        #    key = ifos[i]
        #    if isinstance(key, bytes):
        #        # Don't want byte strings in our names!
        #        key = key.decode('utf-8')
        #    _ = pm.MvNormal(f"strain_{key}", mu=h_det[i,:], chol=Ls[i],
        #                    observed=strains[i], dims=['time_index'])
        
        return model
    
    
def make_ftau_model(t0, times, strains, Ls, **kwargs):
    f_min = kwargs.pop("f_min")
    f_max = kwargs.pop("f_max")
    gamma_min = kwargs.pop("gamma_min")
    gamma_max = kwargs.pop("gamma_max")
    A_scale = kwargs.pop("A_scale")
    flat_A = kwargs.pop("flat_A", True)
    nmode = kwargs.pop("nmode", 1)

    ndet = len(t0)
    nt = len(times[0])

    ifos = kwargs.pop('ifos', np.arange(ndet))
    modes = kwargs.pop('modes', np.arange(nmode))

    coords = {
        'ifo': ifos,
        'mode': modes,
        'time_index': np.arange(nt)
    }

    with pm.Model(coords=coords) as model:
        pm.ConstantData('times', times, dims=['ifo', 'time_index'])
        pm.ConstantData('t0', t0, dims=['ifo'])
        pm.ConstantData('L', Ls, dims=['ifo', 'time_index', 'time_index'])

        f = pm.Uniform("f", f_min, f_max, dims=['mode'])
        gamma = pm.Uniform('gamma', gamma_min, gamma_max, dims=['mode'],
                           transform=pm.distributions.transforms.ordered)

        Ax_unit = pm.Normal("Ax_unit", dims=['mode'])
        Ay_unit = pm.Normal("Ay_unit", dims=['mode'])

        A = pm.Deterministic("A",
            A_scale*at.sqrt(at.square(Ax_unit)+at.square(Ay_unit)),
            dims=['mode'])
        phi = pm.Deterministic("phi", at.arctan2(Ay_unit, Ax_unit),
                               dims=['mode'])

        tau = pm.Deterministic('tau', 1/gamma, dims=['mode'])
        Q = pm.Deterministic('Q', np.pi*f*tau, dims=['mode'])

        Apx = A*at.cos(phi)
        Apy = A*at.sin(phi)

        h_det_mode = pm.Deterministic("h_det_mode",
            compute_h_det_mode(t0, times, np.ones(ndet), np.zeros(ndet),
                               f, gamma, Apx, Apy, np.zeros(nmode),
                               np.zeros(nmode)),
            dims=['ifo', 'mode', 'time_index'])
        h_det = pm.Deterministic("h_det", at.sum(h_det_mode, axis=1),
                                 dims=['ifo', 'time_index'])

        # Priors:

        # Flat in M-chi already

        # Amplitude prior
        if flat_A:
            # first bring us to flat in quadratures
            pm.Potential("flat_A_quadratures_prior",
                         0.5*at.sum(at.square(Ax_unit) + at.square(Ay_unit)))
            pm.Potential("flat_A_prior", -at.sum(at.log(A)))

        # Flat prior on the delta-fs and delta-taus
        return model