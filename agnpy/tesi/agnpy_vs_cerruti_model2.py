import numpy as np
import astropy.units as u
from agnpy.spectra import BrokenPowerLaw, ExpCutoffPowerLaw, ExpCutoffBrokenPowerLaw
from agnpy.emission_regions import Blob
#from agnpy.synchrotron import Synchrotron
#from synchrotron_new import Synchrotron
from agnpy.synchrotron import ProtonSynchrotron, Synchrotron
#from proton_synchrotron import ProtonSynchrotron

from agnpy.utils.plot import plot_sed
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc
#from astropy.constants import e, h, c, m_e, m_p, sigma_T
from astropy.constants import m_p, m_e
from astropy.coordinates import Distance
from agnpy.absorption import EBL
from agnpy.utils.conversion import mec2, mpc2
#import matplotlib.style
from agnpy.compton import SynchrotronSelfCompton

from pathlib import Path
from utils_pytes import make_comparison_plot

load_mpl_rc()  # adopt agnpy plotting style

agnpy_dir = Path(__file__).parent.parent.parent  # go to the agnpy root

# Set some plot parameters
plt.rcParams['figure.constrained_layout.use'] = False
plt.rcParams['savefig.bbox'] = 'tight'
#plt.rcParams['savefig.pad_inches'] = 0.5 #0.07
plt.rcParams["figure.autolayout"] = False

# Define source parameters
B = 62.8936 * u.G
redshift = 0.044
doppler_s = 30
R = 9.06234e+14 * u.cm #radius of the blob

##################################################################
#    Third email: ExpCutoffBrokenPowerLaw, protons
##################################################################

# Extract data of proton distribution and synchrotron spectrum from M. Cerruti
gamma, dndg = np.genfromtxt('./data/Cerruti/third_email/new175_30_1147_ps.dat',  dtype = 'float', comments = '#', usecols = (2,3), unpack = True)
lognu, lognuFnu = np.genfromtxt(f"{agnpy_dir}/agnpy/tesi/data/Cerruti/third_email/new175_30_1147_pss.dat",  dtype = 'float', comments = '#', usecols = (0,4), unpack = True)
nu_ref = 10**lognu * u.Unit('Hz')
sed_ref = 10**lognuFnu *u.Unit("erg cm-2 s-1")
dndg = dndg * u.Unit('cm-3')

# Define particle distribution in agnpy and compute synchrotron spectrum
n_p = ExpCutoffBrokenPowerLaw(k=7.726e-4*(3.64227e+09)**(-1.5) / u.Unit('cm3'), #7.726e-4 / u.Unit('cm3'), 3.5e-18
            p1 = 1.5 ,
            p2 = 2.5,
            gamma_c= 3.64227e+09,
            gamma_b = 3.64227e+09,
            gamma_min= 1,
            gamma_max=1e20,
            mass=m_p
        )
n = n_p(gamma)
blob = Blob(R_b=R,
        z=redshift,
        delta_D=doppler_s,
        B=B,
        n_p=n_p
        )
psynch = ProtonSynchrotron(blob, ssa = False)
sed_agnpy = psynch.sed_flux(nu_ref)

# Distribution comparison plot
gamma_range = [gamma[0], gamma[-1]]
fig, ax = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05}, figsize=(8, 6))
ax[0].loglog(gamma, dndg, marker=".", ls="-", color="k", lw=1.5, label='Cerruti')
ax[0].loglog(gamma, n, marker=".", ls="--", color="crimson", lw=1.5, label='agnpy')
ax[0].set_ylabel(r'Particle distribution [cm$^{-3}$]')
ax[0].set_title('ExpCutoffBrokenPowerLaw')
ax[0].legend(loc="best")
#ax[0].set_ylim(1e-30, 1e-10)
ax[0].axvline(gamma_range[0], ls="--", color="dodgerblue")
ax[0].axvline(gamma_range[1], ls="--", color="dodgerblue")
ax[0].grid(ls=":")
# plot the deviation in the bottom panel
deviation = n / dndg - 1
ax[1].axhline(0, ls="-", color="darkgray")
ax[1].axhline(0.2, ls="--", color="darkgray")
ax[1].axhline(-0.2, ls="--", color="darkgray")
ax[1].axhline(0.3, ls=":", color="darkgray")
ax[1].axhline(-0.3, ls=":", color="darkgray")
ax[1].set_ylim([-0.5, 0.5])
ax[1].semilogx(gamma, deviation, marker=".", ls="--", color="crimson", lw=1.5, label=r'$n(\gamma)_{agnpy}/n(\gamma)_{reference}-1$')
ax[1].set_xlabel(r'$\gamma$')
ax[1].legend(loc="best")
ax[1].axvline(gamma_range[0], ls="--", color="dodgerblue")
ax[1].axvline(gamma_range[1], ls="--", color="dodgerblue")
fig.savefig('./comparison_sync_cerruti/figures/agnpy_vs_cerruti_distr_ECBPL_resid_model2.png')


# sed comparison plot
nu_range = [1e10, 1e28] * u.Hz
make_comparison_plot(
        nu_ref,
        sed_agnpy,
        sed_ref,
        "agnpy",
        "Cerruti",
        "Proton synchrotron from ExpCutoffBrokenPowerLaw",
        f"{agnpy_dir}/agnpy/tesi/comparison_sync_cerruti/figures/agnpy_vs_cerruti_psync_ExpCutoffBPL_model2.png",
        "sed",
        # y_range=[1e-16, 1e-8],
        comparison_range=nu_range.to_value("Hz"),
)



##################################################################
#    Third email: ExpCutoffBrokenPowerLaw, electrons
##################################################################

# Extract data of electron distribution and synchrotron spectrum from M. Cerruti
gamma2, dndg2 = np.genfromtxt('./data/Cerruti/third_email/new175_30_1147_es.dat',  dtype = 'float', comments = '#', usecols = (2,3), unpack = True)
lognu2, lognuFnu2 = np.genfromtxt(f"{agnpy_dir}/agnpy/tesi/data/Cerruti/third_email/new175_30_1147_ss.dat",  dtype = 'float', comments = '#', usecols = (0,2), unpack = True)
nu_ref2 = 10**lognu2 * u.Unit('Hz')
sed_ref2 = 10**lognuFnu2 *u.Unit("erg cm-2 s-1")
dndg2 = dndg2 * u.Unit('cm-3')

# Define particle distribution in agnpy and compute synchrotron spectrum
n_e = ExpCutoffBrokenPowerLaw(k=1.25336*(200)**(-1.5) / u.Unit('cm3'), #4.8e-4 1.25336
        p1 = 1.5,
        p2 = 2.5,
        gamma_c= 26441.5, # 26441.5,
        gamma_b = 200,
        gamma_min= 200,
        gamma_max=1e15,
        mass=m_e
        )
ne = n_e(gamma2)
blob2 = Blob(R_b=R,
        z=redshift,
        delta_D=doppler_s,
        B=B,
        n_e=n_e
        )
esynch = Synchrotron(blob2, ssa = True)
esed_agnpy = esynch.sed_flux(nu_ref2)

# Distribution comparison plot
gamma_range2 = [gamma2[0], gamma2[-1]]
fig3, ax3 = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05}, figsize=(8, 6))
ax3[0].loglog(gamma2, dndg2, marker=".", ls="-", color="k", lw=1.5, label='Cerruti')
ax3[0].loglog(gamma2, ne, marker=".", ls="--", color="crimson", lw=1.5, label='agnpy')
ax3[0].set_ylabel(r'Particle distribution [cm$^{-3}$]')
ax3[0].set_title('ExpCutoffBrokenPowerLaw')
ax3[0].legend(loc="best")
#ax3[0].set_ylim(1e-30, 1e-10)
ax3[0].axvline(gamma_range2[0], ls="--", color="dodgerblue")
ax3[0].axvline(gamma_range2[1], ls="--", color="dodgerblue")
ax3[0].grid(ls=":")
# plot the deviation in the bottom panel
deviation2 = ne / dndg2 - 1
ax3[1].axhline(0, ls="-", color="darkgray")
ax3[1].axhline(0.2, ls="--", color="darkgray")
ax3[1].axhline(-0.2, ls="--", color="darkgray")
ax3[1].axhline(0.3, ls=":", color="darkgray")
ax3[1].axhline(-0.3, ls=":", color="darkgray")
ax3[1].set_ylim([-0.5, 0.5])
ax3[1].semilogx(gamma2, deviation2, marker=".", ls="--", color="crimson", lw=1.5, label=r'$n(\gamma)_{agnpy}/n(\gamma)_{reference}-1$')
ax3[1].set_xlabel(r'$\gamma$')
ax3[1].legend(loc="best")
ax3[1].axvline(gamma_range2[0], ls="--", color="dodgerblue")
ax3[1].axvline(gamma_range2[1], ls="--", color="dodgerblue")
fig3.savefig('./comparison_sync_cerruti/figures/agnpy_vs_cerruti_distr_ECBPL_resid_model2_electrons.png')


# sed comparison plot
nu_range = [1e10, 1e23] * u.Hz
make_comparison_plot(
        nu_ref2,
        esed_agnpy,
        sed_ref2,
        "agnpy",
        "Cerruti",
        "Electron synchrotron from ExpCutoffBrokenPowerLaw",
        f"{agnpy_dir}/agnpy/tesi/comparison_sync_cerruti/figures/agnpy_vs_cerruti_esync_ExpCutoffBPL_model2.png",
        "sed",
        y_range=[1e-20, 1e-9],
        comparison_range=nu_range.to_value("Hz"),
)