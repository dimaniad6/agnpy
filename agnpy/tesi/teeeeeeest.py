import numpy as np
import astropy.units as u
from agnpy.spectra import PowerLaw, BrokenPowerLaw, LogParabola, ExpCutoffPowerLaw, ExpCutoffBrokenPowerLaw
from agnpy.utils.conversion import mpc2
from agnpy.utils.plot import plot_sed
from agnpy.emission_regions import Blob
from agnpy.utils.plot import load_mpl_rc
from agnpy.synchrotron import Synchrotron, ProtonSynchrotron
import matplotlib.pyplot as plt
from astropy.constants import m_p,m_e, h
from astropy.coordinates import Distance

load_mpl_rc()  # adopt agnpy plotting style
#plt.style.use('proton_synchrotron')

# Define source parameters PKS2155
B = 1 * u.G
redshift = 0.1
distPKS = Distance(z=redshift) # Already inside blob definition
doppler_s = 30
Gamma_bulk = 16
R = 1e16 * u.cm #radius of the blob
vol = (4. / 3) * np.pi * R ** 3

norm_p2 = 12e3 / u.Unit('cm3')
u_p = 3.7e2 * u.Unit('erg cm-3')


n_p = PowerLaw(k=3000 * u.Unit("cm-3"),
        p=2.0,
        gamma_min=1e4,
        gamma_max=1e8,
        mass=m_p,
)

n_e = PowerLaw(k= 3000 * u.Unit("cm-3"),
        p=2.0,
        gamma_min=1e4,
        gamma_max=1e8,
        mass=m_e,
)

# compute the SED over an array of frequencies
nu = np.logspace(6,30, 200) * u.Hz


blob = Blob(R_b=R,
    z=redshift,
    delta_D=doppler_s,
    Gamma=Gamma_bulk,
    B=B,
    n_e = n_e,
    n_p= n_p
)


synch = Synchrotron(blob)
psynch= ProtonSynchrotron(blob, ssa = 'False')

sed = synch.sed_flux(nu)
psed=psynch.sed_flux(nu)
plot_sed(nu,  sed, label = 'Electron Synchrotron',linestyle = '--')
plot_sed(nu, psed,  label = 'Proton Synctrotron')

plt.ylim(1e-28, 1e-6)
plt.xlim(1e7, 1e30) # For frequencies

plt.show()
