import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.constants import m_e, m_p
from astropy.coordinates import Distance
from astropy.cosmology import default_cosmology, FlatLambdaCDM, Planck15

from agnpy.spectra import ExpCutoffBrokenPowerLaw
#from agnpy.emission_regions import Blob
from blob_cosmo import Blob
from agnpy.synchrotron import ProtonSynchrotron
from utils_pytes import make_comparison_plot
from agnpy.utils.plot import load_mpl_rc

load_mpl_rc()  # adopt agnpy plotting style
agnpy_dir = Path(__file__).parent.parent.parent  # go to the agnpy root

# Set some plot parameters
plt.rcParams['figure.constrained_layout.use'] = False
plt.rcParams['savefig.bbox'] = 'tight'
#plt.rcParams['savefig.pad_inches'] = 0.5 #0.07
plt.rcParams["figure.autolayout"] = False

# Redshifts for Cerruti models
z1 = 0.32
z2 = 0.044

# Compute luminosity distance with default astropy cosmology
dL1a = Distance(z=z1)
dL2a = Distance(z=z2)

# Compute luminosity distance with M. Cerruti cosmology (for now I just change the Hubble constant)
cosmo_cerruti = FlatLambdaCDM(H0 = 70 * u.km / u.s / u.Mpc, Om0 = 0.307, Tcmb0=2.725 * u.K, Neff=3.05, m_nu=[0., 0., 0.06] * u.eV, Ob0=0.0486)
dL1c = Distance(z=z1, cosmology=cosmo_cerruti)
dL2c = Distance(z=z2, cosmology=cosmo_cerruti)

print('dL1 agnpy   = ', dL1a, ' and dL2 agnpy   = ', dL2a)
print('dL1 Cerruti = ', dL1c, ' and dL2 Cerruti = ', dL2c)
print(' ')
print('####################### Parameters of the default cosmology #######################')
print('The default cosmology is ', default_cosmology.get())
print('Omega_r = ', Planck15.Ogamma0)
print('Omega_Lambda = ', Planck15.Ode0)
print('Omega_k = ', Planck15.Ok0)
print(' ')
print('####################### Parameters of Cerruti`s cosmology #######################')
print('Omega_r = ', cosmo_cerruti.Ogamma0)
print('Omega_Lambda = ', cosmo_cerruti.Ode0)
print('Omega_k = ', cosmo_cerruti.Ok0)
print(' ')

# Cerruti first model ExpCutoffBPL
# Source parameters
B1 = 10 * u.G
redshift1 = 0.32
doppler_s1 = 30
R1 = 1e16 * u.cm #radius of the blob
# Extract data of particle distribution and synchrotron spectrum from M. Cerruti
gamma2, dndg2 = np.genfromtxt('./data/Cerruti/first_email/test_ps.dat',  dtype = 'float', comments = '#', usecols = (2,3), unpack = True)
lognu2, lognuFnu2= np.genfromtxt('./data/Cerruti/first_email/test_pss.dat',  dtype = 'float', comments = '#', usecols = (0,4), unpack = True)
nu_data2 = 10**lognu2 * u.Unit('Hz')
nuFnu_data2 = 10**lognuFnu2 * u.Unit("erg cm-2 s-1")
dndg2 = dndg2 * u.Unit('cm-3')
# Distribution and SED
n_p2 = ExpCutoffBrokenPowerLaw(k=12e4*(2.5e+09)**(-2.2) / u.Unit('cm3'),
            p1 = 2.2,
            p2 = 3.2,
            gamma_c= 2.5e+09,
            gamma_b = 2.5e+09,
            gamma_min= 1,
            gamma_max=1e20,
            mass=m_p
        )
#n2 = n_p2(gamma2)
blob2 = Blob(R_b=R1,
        z=redshift1,
        delta_D=doppler_s1,
        B=B1,
        n_p=n_p2
        )   

# print(blob2.d_L, '   ', blob2.d_L.to('Mpc'))
# print(dL1c.to('cm'), '   ', dL1c)
psynch2 = ProtonSynchrotron(blob2, ssa = True)
psed2 = psynch2.sed_flux(nu_data2)

dL_ratio = (dL1a ** 2) / (dL1c ** 2)
psedc_rescaled = dL_ratio * psed2
print(dL_ratio)

# sed comparison plot
nu_range = [1e10, 1e28] * u.Hz
make_comparison_plot(
        nu_data2,
        psedc_rescaled,
        nuFnu_data2,
        "Cerruti, rescaled",
        "Cerruti",
        "Proton synchrotron from ExpCutoffBrokenPowerLaw",
        f"{agnpy_dir}/agnpy/tesi/comparison_sync_cerruti/figures/agnpy_vs_cerruti_psync_cosmology_H070_Om03.png",
        "sed",
        # y_range=[1e-16, 1e-8],
        comparison_range=nu_range.to_value("Hz"),
)