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
from agnpy.photomeson import PhotoHadronicInteraction
from agnpy.spectra import ExpCutoffPowerLaw as ECPL
from agnpy.emission_regions import Blob
from ..utils.conversion import nu_to_epsilon_prime, B_to_cgs


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

start = timeit.default_timer()

# Define source parameters
B = 80 * u.G
redshift = 0.117
distPKS = Distance(z=redshift) # Already inside blob definition
doppler_s = 30
Gamma_bulk = 16
R = 5.2e14 * u.cm #radius of the blob
vol = (4. / 3) * np.pi * R ** 3

mec2 = (m_e * c ** 2).to('eV')

Ec = 3*1e20 * u.eV
mpc2 = (m_p * c ** 2).to('eV')
gamma_cut = Ec / mpc2
p = 2.
A = (0.24153*1e11)/(mpc2.value**2) * u.Unit('cm-3')

p_dist = ECPL(A, p, gamma_cut, 1e3, 1e20)

blob = Blob(R_b=R,
        z=redshift,
        delta_D=doppler_s,
        Gamma=Gamma_bulk,
        B=B,
        n_p= p_dist
)

nu = np.logspace(25,37,50)*u.Hz
gamma = epsilon_equivalency(nu)
energies = gamma * mec2

proton_gamma = PhotoHadronicInteraction(blob, BlackBody)
spec = proton_gamma.spectrum(nu_aha, 'photon')
# spec_ele = proton_gamma.spectrum(gammas, 'electron')
# spec_posi = proton_gamma.spectrum_electron(gammas, 'positron')
# spec_nu_muon = proton_gamma.spectrum(nu3, 'nu_muon')
# spec_antinu_muon = proton_gamma.spectrum(nu, 'antinu_muon')
# spec_nu_electron = proton_gamma.spectrum(nu, 'nu_electron')



sed = proton_gamma.sed_flux(nu, 'photon')

plt.loglog((energies), (sed), lw=2.2, ls='-', color='blue',label = 'agnpy')

plt.show()

stop = timeit.default_timer()
print("Elapsed time for computation = {} secs".format(stop - start))
