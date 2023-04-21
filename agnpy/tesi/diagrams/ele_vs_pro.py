import numpy as np
import astropy.units as u
from agnpy.spectra import PowerLaw, BrokenPowerLaw, LogParabola, ExpCutoffPowerLaw, ExpCutoffBrokenPowerLaw
from agnpy.utils.conversion import mpc2, mec2
from agnpy.utils.plot import plot_sed
from agnpy.emission_regions import Blob
from agnpy.utils.plot import load_mpl_rc
from agnpy.synchrotron import Synchrotron, ProtonSynchrotron
import matplotlib.pyplot as plt
from astropy.constants import m_p, h, m_e,m_p
from astropy.coordinates import Distance

load_mpl_rc()  # adopt agnpy plotting style
#plt.style.use('proton_synchrotron')

# Define source parameters PKS2155
B = 1 * u.G
redshift = 0.117
distPKS = Distance(z=redshift) # Already inside blob definition
doppler_s = 30
Gamma_bulk = 16
R = 5.2e14 * u.cm #radius of the blob
vol = (4. / 3) * np.pi * R ** 3



norm_p2 = 12e3 / u.Unit('cm3')
u_p = 3.7e2 * u.Unit('erg cm-3')

# define the emission region and the radiative process
# PL_e = PowerLaw(k=10* u.Unit("cm-3"),
#         p=2.0,
#         gamma_min=1e3,
#         gamma_max=1e5,
#         mass=m_p,
# )
#
# PL = PowerLaw(k=13000 * u.Unit("cm-3"),
#         p=2.0,
#         gamma_min=1e3,
#         gamma_max=2e8,
#         mass=m_p,
# )

PL = PowerLaw(k=1 * u.Unit("cm-3"),
        p=2.0,
        gamma_min=1e2,
        gamma_max=1e6,
        mass=m_e,
)

# compute the SED over an array of frequencies
nu = np.logspace(10,30, 100) * u.Hz

blob = Blob(R_b=R,
        z=redshift,
        delta_D=doppler_s,
        Gamma=Gamma_bulk,
        B=B,
        n_p = PL,
        n_e = PL
)

esynch = Synchrotron(blob)
psynch= ProtonSynchrotron(blob)
esed = esynch.sed_flux(nu)
psed = psynch.sed_flux(nu)

# plt.loglog(nu, esed, label = 'Electron Synchrotron ')
# plt.loglog(nu, psed,  label = 'Proton Synctrotron')
plot_sed(nu,  esed, linestyle = '-', label = 'Electron Synctrotron')
plot_sed(nu,  psed, linestyle = '-.', label = 'Proton Synctrotron')

plt.ylim(1e-17, 1e-12)
plt.xlim(1e10, 1e26) # For frequencies
plt.show()
