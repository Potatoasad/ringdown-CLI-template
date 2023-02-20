#########################################################
import numpy as np

### Abstract Polarization Model
class Polarization:
    def __init__(self, **kwargs):
        ## First polarization variables
        A = kwargs.pop("A", None)
        theta = kwargs.pop("theta", None)
        ellip = kwargs.pop("ellip", None)
        phi = kwargs.pop("phi", None)
        
        ## Second Polarization variables
        Apx = kwargs.pop("Apx", None)
        Apy = kwargs.pop("Apy", None)
        Acx = kwargs.pop("Acx", None)
        Acy = kwargs.pop("Acy", None)
        
        self.A = A; self.theta = theta; self.phi = phi; self.ellip = ellip;
        self.Apx = Apx; self.Apy = Apy; self.Acx = Acx; self.Acy = Acy;
        
        ## Check which polarization variables were entered and fill the other one accordingly
        first_pol_vars_available = not [x for x in (A, theta, ellip, phi) if x is None]
        second_pol_vars_available = not [x for x in (Apx, Apy, Acx, Acy) if x is None]
        if first_pol_vars_available:
            self.Apx, self.Apy, self.Acx, self.Acy = self.first_pol_to_second_pol(A,theta,ellip, phi)
        elif second_pol_vars_available:
            self.A, self.theta,self.ellip, self.phi = self.second_pol_to_first_pol(Apx, Apy, Acx, Acy)
        else:
            raise ValueError("Not enough information to construct a polarization vector")
    
    @staticmethod
    def a_from_quadratures(Apx, Apy, Acx, Acy):
        A = 0.5*(np.sqrt(np.square(Acy + Apx) + np.square(Acx - Apy)) +
                 np.sqrt(np.square(Acy - Apx) + np.square(Acx + Apy)))
        return A

    def ellip_from_quadratures(self,Apx, Apy, Acx, Acy):
        A = self.a_from_quadratures(Apx, Apy, Acx, Acy)
        e = 0.5*(np.sqrt(np.square(Acy + Apx) + np.square(Acx - Apy)) -
                 np.sqrt(np.square(Acy - Apx) + np.square(Acx + Apy))) / A
        return e

    @staticmethod
    def phiR_from_quadratures(Apx, Apy, Acx, Acy):
        return np.arctan2(-Acx + Apy, Acy + Apx)

    @staticmethod
    def phiL_from_quadratures(Apx, Apy, Acx, Acy):
        return np.arctan2(-Acx - Apy, -Acy + Apx)
    
    @staticmethod
    def first_pol_to_second_pol(A,theta,ellip, phi):
        cth = np.cos(theta); sth = np.sin(theta);
        cph = np.cos(phi); sph = np.sin(phi);

        Apx = A*(cth*cph + ellip*sth*sph)
        Apy = A*(cth*sph - ellip*sth*cph)
        Acx = A*(sth*cph - ellip*cth*sph)
        Acy = A*(sth*sph + ellip*cth*cph)

        return Apx, Apy, Acx, Acy

    def second_pol_to_first_pol(self, Apx, Apy, Acx, Acy):
        phiR = self.phiR_from_quadratures(Apx, Apy, Acx, Acy)
        phiL = self.phiL_from_quadratures(Apx, Apy, Acx, Acy)
        ellip = self.ellip_from_quadratures(Apx, Apy, Acx, Acy)
        A = self.a_from_quadratures(Apx, Apy, Acx, Acy)
        theta = -0.5*(phiR + phiL)
        phi = 0.5*(phiR - phiL)
        return A, theta, ellip, phi
    
    @property
    def quadratures(self):
        return {'Apx': self.Apx, 'Apy': self.Apy, 'Acx': self.Acx, 'Acy': self.Acy}
    
    @property
    def parameters(self):
        return {'A': self.A, 'theta': self.theta, 'ellip': self.ellip, 'phi': self.phi}
    
    @property
    def inputs(self):
        return self.parameters
    
    def __repr__(self):
        the_repr = f"Polarization({ ', '.join([f'{k} = {v}' for k,v in self.inputs.items()])  })"
        return the_repr
    
    
### Test:
#def my_polarization_test():
#    input_params = dict(A=1.0, theta=0.1, ellip=0.1, phi=0.1)
#    P1 = Polarization(**input_params)
#    output_params = Polarization(**P1.quadratures).parameters
#    return np.all([np.isclose(input_params[k],output_params[k]) for k in input_params.keys()])
    
#assert my_polarization_test()
#########################################################