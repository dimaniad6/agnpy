import numpy as np
from scipy.interpolate import CubicSpline
from agnpy.spectra import PowerLaw, ExpCutoffPowerLaw, BrokenPowerLaw
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h, hbar, k_B
from astropy.coordinates import Distance
from matplotlib import pyplot as plt

def interpolator(e,fph):

    log10_f = CubicSpline(np.log10(e), np.log10(fph.value))
    f = np.power(10, log10_f(np.log10(e)))
    return f

class InterpolatedDistribution:

    def __init__(self, gamma, n):

        self.gamma = gamma
        self.gamma_max = np.max(self.gamma)
        self.gamma_min = np.min(self.gamma)
        self.n = n
        # call make the interpolation
        self.log10_f = self.log10_interpolation()

    def log10_interpolation(self):

        interpolator = CubicSpline(
            np.log10(self.gamma), np.log10(self.n.value))

        return interpolator

    def evaluate(self, gamma, gamma_min, gamma_max):
        log10_gamma = np.log10(gamma)
        values = np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max),
            np.power(10, self.log10_f(log10_gamma)),
            0,
        )
        return values * u.Unit("cm-3")

    def __call__(self, gamma):
        return self.evaluate(gamma, self.gamma_min, self.gamma_max)


def soft_synch(epsilon):

    nu = epsilon * mec2 / h.to('eV s')
    f = esynch.sed_flux(nu)
    fph = (3/4) * (3 * (distPKS.to('cm')**2) * (f / (nu*h.to('eV s')))) / (c.to('cm s-1') * (R**2) * (doppler_s**4) * (epsilon) )

    return fph.to('cm-3')


mec2 = (m_e * c ** 2).to('eV')
mpc2 = (m_p * c ** 2).to('eV')

# DEFINING ALL PARAMETERS #######################################
B = 80 * u.G
redshift = 0.117
distPKS = Distance(z=redshift) # Already inside blob definition
doppler_s = 30
Gamma_bulk = 16
R = 5.2e14 * u.cm #radius of the blob
vol = (4. / 3) * np.pi * R ** 3
#electron distribution parameters
p1 = 2.
p2 = 4.32
k2 = 600 * u.Unit('cm-3')
e_dist = BrokenPowerLaw(k2,p1,p2,4e3,1,6e4)
blob = Blob(R_b=R,
        z=redshift,
        delta_D=doppler_s,
        Gamma=Gamma_bulk,
        B=B,
        n_e = e_dist,
        )
###################################################################

esynch = Synchrotron(blob, ssa = 'True')
epsilon = np.logspace(-7,0,1000)
function = soft_synch(epsilon)

f_ph2 = InterpolatedDistribution(epsilon, function)
f_ph = f_ph2(epsilon)

plt.loglog(epsilon,f_ph, '.')
plt.loglog(epsilon, function)
plt.show()
