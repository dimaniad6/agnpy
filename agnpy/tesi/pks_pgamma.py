import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h, hbar, k_B
from astropy.table import Table, Column
from astropy.coordinates import Distance
from scipy.integrate import quad, dblquad, nquad, simps, trapz
from agnpy.synchrotron import Synchrotron
import numpy as np
import matplotlib.pyplot as plt
import timeit
import re
from agnpy.photomeson import PhotoHadronicInteraction
from agnpy.spectra import PowerLaw, ExpCutoffPowerLaw, BrokenPowerLaw
from agnpy.emission_regions import Blob

#plt.style.use('proton_synchrotron')
start = timeit.default_timer()
lognu, lognuFnu= np.genfromtxt('data/Cerruti/second_email/test_pss.dat',  dtype = 'float', comments = '#', usecols = (0,4), unpack = True)
nu_data = 10**lognu
nuFnu_data = 10**lognuFnu

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

def soft_synch(epsilon):
    nu = epsilon * mec2 / h.to('eV s')
    f = esynch.sed_flux(nu)
    fph = (3/2) * (3 * (distPKS.to('cm')**2) * (f / (epsilon*mec2))) / (c.to('cm s-1') * (R**2) * (doppler_s**4) * (epsilon) )
    # print (fph)
    return fph.to('cm-3')



# Define source parameters
B = 80 * u.G
redshift = 0.117
distPKS = Distance(z=redshift) # Already inside blob definition
doppler_s = 30
Gamma_bulk = 16
R = 5.2e14 * u.cm #radius of the blob
vol = (4. / 3) * np.pi * R ** 3

mec2 = (m_e * c ** 2).to('eV')
mpc2 = (m_p * c ** 2).to('eV')

#proton distribution parameters
p = 2.
k1 = 120000 * u.Unit('cm-3')
#electron distribution parameters
p1 = 2.
p2 = 4.32
k2 = 600 * u.Unit('cm-3')


p_dist = PowerLaw(k1, p, 1, 1e9)
e_dist = BrokenPowerLaw(k2,p1,p2,4e3,1,6e4)

blob = Blob(R_b=R,
        z=redshift,
        delta_D=doppler_s,
        Gamma=Gamma_bulk,
        B=B,
        n_e = e_dist,
        n_p= p_dist,
)


esynch = Synchrotron(blob, soft_synch)

nu = np.logspace(16,30,10)*u.Hz
gamma = epsilon_equivalency(nu)
energies = gamma * mec2


proton_gamma = PhotoHadronicInteraction(blob, BlackBody)

sed = proton_gamma.sed_flux(nu, 'photon')

plt.loglog(nu, sed, lw=2.2, ls='-', color='blue',label = 'agnpy')
plt.loglog(nu_data, nuFnu_data, 'o', label ='M. Cerruti 2012')
plt.show()

stop = timeit.default_timer()
print("Elapsed time for computation = {} secs".format(stop - start))

# spec_ele = proton_gamma.spectrum(gammas, 'electron')
# spec_posi = proton_gamma.spectrum_electron(gammas, 'positron')
# spec_nu_muon = proton_gamma.spectrum(nu3, 'nu_muon')
# spec_antinu_muon = proton_gamma.spectrum(nu, 'antinu_muon')
# spec_nu_electron = proton_gamma.spectrum(nu, 'nu_electron')
