import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h, hbar, k_B
from astropy.table import Table, Column
from astropy.coordinates import Distance
from scipy.integrate import quad, dblquad, nquad, simps, trapz
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import timeit
import re
from agnpy.photomeson import PhotoHadronicInteraction_Reference
from agnpy.photomeson import PhotoHadronicInteraction_Reference2
from agnpy.utils.plot import plot_sed
from agnpy.spectra import ExpCutoffPowerLaw as ECPL
from agnpy.emission_regions import Blob

plt.style.use('proton_synchrotron')

def epsilon_equivalency(nu, m = m_e):
    if m == m_e:
        epsilon_equivalency = h.to('eV s') * nu / mec2

    elif m == m_p:
        epsilon_equivalency = h.to('eV s')* nu / mpc2

    return epsilon_equivalency


def BlackBody(epsilon):
    T = 2.7 *u.K
    kT = (k_B * T).to('eV').value
    c1 = c.to('cm s-1').value
    h1 = h.to('eV s').value
    norm = 8*np.pi/(h1**3*c1**3)
    num = (mpc2.value *epsilon) ** 2
    denom = np.exp(mpc2.value * epsilon / kT) - 1
    return norm * (num / denom) * u.Unit('cm-3')

mec2 = (m_e * c ** 2).to('eV')
Ec = 3*1e20 * u.eV
mpc2 = (m_p * c ** 2).to('eV')
gamma_cut = Ec / mpc2
p = 2.
A = (0.24153*1e11)/(mpc2.value**2) * u.Unit('cm-3')
p_dist = ECPL(A, p, gamma_cut, 1e3, 1e20)

nu = np.logspace(27,37,3)*u.Hz
E  = nu * h.to('eV s')
proton_gamma = PhotoHadronicInteraction_Reference(p_dist, BlackBody)
proton_gamma3 = PhotoHadronicInteraction_Reference2(p_dist, BlackBody)

spec = proton_gamma.spectrum(nu, 'photon')
spec2= proton_gamma3.spectrum(nu, 'photon')

plt.loglog((E), (spec  * E ), color='orange', label = 'Upper limit: $10^{21}$ eV')
plt.loglog((E), (spec2 * E ), color='blue', label = 'Upper limit: $10^{21}$ eV')
plt.legend('upper left')
plt.show()
