import numpy as np
import astropy.units as u
from agnpy.spectra import BrokenPowerLaw, ExpCutoffPowerLaw, ExpCutoffBrokenPowerLaw
from agnpy.emission_regions import Blob
#from agnpy.synchrotron import Synchrotron
#from synchrotron_new import Synchrotron
from agnpy.synchrotron import ProtonSynchrotron
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
#plt.rcParams['savefig.pad_inches'] = 0.07
plt.rcParams["figure.autolayout"] = False

# Define source parameters
B = 10 * u.G
redshift = 0.32
doppler_s = 30
R = 1e16 * u.cm #radius of the blob

##################################################################
#    First email: ExpCutoffBrokenPowerLaw
##################################################################

# Extract data of particle distribution and synchrotron spectrum from M. Cerruti
gamma2, dndg2 = np.genfromtxt('./data/Cerruti/first_email/test_ps.dat',  dtype = 'float', comments = '#', usecols = (2,3), unpack = True)
lognu2, lognuFnu2= np.genfromtxt('./data/Cerruti/first_email/test_pss.dat',  dtype = 'float', comments = '#', usecols = (0,4), unpack = True)
nu_data2 = 10**lognu2 * u.Unit('Hz')
nuFnu_data2 = 10**lognuFnu2 * u.Unit("erg cm-2 s-1")
dndg2 = dndg2 * u.Unit('cm-3')
# Define particle distribution in agnpy and compute synchrotron spectrum
n_p2 = ExpCutoffBrokenPowerLaw(k=12e4*(2.5e+09)**(-2.2) / u.Unit('cm3'),
            p1 = 2.2,
            p2 = 3.2,
            gamma_c= 2.5e+09,
            gamma_b = 2.5e+09,
            gamma_min= 1,
            gamma_max=1e20,
            mass=m_p
        )
n2 = n_p2(gamma2)
blob2 = Blob(R_b=R,
        z=redshift,
        delta_D=doppler_s,
        B=B,
        n_p=n_p2
)
psynch2 = ProtonSynchrotron(blob2, ssa = True)
psed2 = psynch2.sed_flux(nu_data2)
# Compare distributions
plt.figure(figsize = (6.92, 4.29))
plt.title('ExpCutoffBrokenPowerLaw')
plt.scatter(gamma2,n2, marker='*', label='agnpy', s = 30)
plt.scatter(gamma2,dndg2, marker='.', label='Cerruti')
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.savefig('./comparison_sync_cerruti/figures/agnpy_vs_cerruti_distr_ExpCutoffBPL')
# Compare synchrotron spectrum
plt.figure(figsize = (6.92, 4.29))
plt.title('Proton synchrotron from ExpCutoffBrokenPowerLaw')
plt.scatter(nu_data2,psed2, marker='*', label='agnpy', s = 30)
plt.scatter(nu_data2,nuFnu_data2, marker='.', label='Cerruti')
plt.yscale("log")
plt.xscale("log")
plt.legend()


# Distribution comparison plot
gamma_range2 = [gamma2[0], gamma2[-1]]
fig2, ax2 = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05}, figsize=(8, 6))
# plot the distribution in the upper panel
# plot the reference distr with a continuous line and agnpy distr with a dashed one
ax2[0].loglog(gamma2, dndg2, marker=".", ls="-", color="k", lw=1.5, label='Cerruti')
ax2[0].loglog(gamma2, n2, marker=".", ls="--", color="crimson", lw=1.5, label='agnpy')
ax2[0].set_ylabel(r'Particle distribution [cm$^{-3}$]')
ax2[0].set_title('ExpCutoffBrokenPowerLaw')
ax2[0].legend(loc="best")
#ax2[0].set_ylim(1e-30, 1e-10)
ax2[0].axvline(gamma_range2[0], ls="--", color="dodgerblue")
ax2[0].axvline(gamma_range2[1], ls="--", color="dodgerblue")
ax2[0].grid(ls=":")
# plot the deviation in the bottom panel
deviation2 = n2 / dndg2 - 1
ax2[1].axhline(0, ls="-", color="darkgray")
ax2[1].axhline(0.2, ls="--", color="darkgray")
ax2[1].axhline(-0.2, ls="--", color="darkgray")
ax2[1].axhline(0.3, ls=":", color="darkgray")
ax2[1].axhline(-0.3, ls=":", color="darkgray")
ax2[1].set_ylim([-0.5, 0.5])
ax2[1].semilogx(gamma2, deviation2, marker=".", ls="--", color="crimson", lw=1.5, label=r'$n(\gamma)_{agnpy}/n(\gamma)_{reference}-1$')
ax2[1].set_xlabel(r'$\gamma$')
ax2[1].legend(loc="best")
ax2[1].axvline(gamma_range2[0], ls="--", color="dodgerblue")
ax2[1].axvline(gamma_range2[1], ls="--", color="dodgerblue")
fig2.savefig('./comparison_sync_cerruti/figures/agnpy_vs_cerruti_distr_ECBPL_resid.png')


# sed comparison plot
nu_range = [1e10, 1e28] * u.Hz
make_comparison_plot(
        nu_data2,
        psed2,
        nuFnu_data2,
        "agnpy",
        "Cerruti",
        "Proton synchrotron from ExpCutoffBrokenPowerLaw",
        f"{agnpy_dir}/agnpy/tesi/comparison_sync_cerruti/figures/agnpy_vs_cerruti_psync_ExpCutoffBPL.png",
        "sed",
        # y_range=[1e-16, 1e-8],
        comparison_range=nu_range.to_value("Hz"),
)


##################################################################
#    Second email: ExpCutoffPowerLaw
##################################################################

# Extract data of particle distribution and synchrotron spectrum from M. Cerruti
gamma, dndg = np.genfromtxt('./data/Cerruti/second_email/test_ps.dat',  dtype = 'float', comments = '#', usecols = (2,3), unpack = True)
lognu, lognuFnu= np.genfromtxt('./data/Cerruti/second_email/test_pss.dat',  dtype = 'float', comments = '#', usecols = (0,4), unpack = True)
nu_data = 10**lognu * u.Unit('Hz')
nuFnu_data = 10**lognuFnu * u.Unit("erg cm-2 s-1")
dndg = dndg * u.Unit('cm-3')
# Define particle distribution in agnpy and compute synchrotron spectrum
n_p = ExpCutoffPowerLaw(k=12e4 * u.Unit('cm-3'),
        p = 2.2,
        gamma_c= 2.5e9,
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
psynch = ProtonSynchrotron(blob, ssa = True)
# # compute the SED over an array of frequencies
# nu = np.logspace(9, 29, 100) * u.Hz
psed = psynch.sed_flux(nu_data)
# Compare distributions
plt.figure(figsize = (6.92, 4.29))
plt.title('ExpCutoffPowerLaw')
plt.scatter(gamma,n, marker='*', label='agnpy', s = 30)
plt.scatter(gamma,dndg, marker='.', label='Cerruti')
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.savefig('./comparison_sync_cerruti/figures/agnpy_vs_cerruti_distr_ExpCutoffPL')
# Compare synchrotron spectrum
plt.figure(figsize = (6.92, 4.29))
plt.title('Proton synchrotron from ExpCutoffPowerLaw')
plt.scatter(nu_data,psed, marker='*', label='agnpy', s = 30)
plt.scatter(nu_data,nuFnu_data, marker='.', label='Cerruti')
plt.yscale("log")
plt.xscale("log")
#plt.ylim(1e-22,1e-9)
plt.legend()

# Distribution comparison plot
gamma_range = [gamma[0], gamma[-1]]
fig, ax = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05}, figsize=(8, 6))
# plot the distribution in the upper panel
# plot the reference sed with a continuous line and agnpy sed with a dashed one
ax[0].loglog(gamma, dndg, marker=".", ls="-", color="k", lw=1.5, label='Cerruti')
ax[0].loglog(gamma, n, marker=".", ls="--", color="crimson", lw=1.5, label='agnpy')
ax[0].set_ylabel(r'Particle distribution [cm$^{-3}$]')
ax[0].set_title('ExpCutoffPowerLaw')
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
fig.savefig('./comparison_sync_cerruti/figures/agnpy_vs_cerruti_distr_ECPL_resid.png')


# sed comparison plot
nu_range = [1e10, 1e28] * u.Hz
make_comparison_plot(
        nu_data,
        psed,
        nuFnu_data,
        "agnpy",
        "Cerruti",
        "Proton synchrotron from ExpCutoffPowerLaw",
        f"{agnpy_dir}/agnpy/tesi/comparison_sync_cerruti/figures/agnpy_vs_cerruti_psync_ExpCutoffPL.png",
        "sed",
        # y_range=[1e-16, 1e-8],
        comparison_range=nu_range.to_value("Hz"),
)



##################################################################
#    Second email: ExpCutoffPowerLaw, change of normalization
##################################################################

# Data are the same as before (second email)
# Define particle distribution in agnpy and compute synchrotron spectrum
n_p3 = ExpCutoffPowerLaw(k=13e4 * u.Unit('cm-3'),
        p = 2.2,
        gamma_c= 2.5e9,
        gamma_min= 1,
        gamma_max=1e20,
        mass=m_p
)
n3 = n_p3(gamma)
blob3 = Blob(R_b=R,
        z=redshift,
        delta_D=doppler_s,
        B=B,
        n_p=n_p3
)
psynch3 = ProtonSynchrotron(blob3, ssa = True)
psed3 = psynch3.sed_flux(nu_data)
# Compare distributions
plt.figure(figsize = (6.92, 4.29))
plt.title('ExpCutoffPowerLaw')
plt.scatter(gamma,n3, marker='*', label='agnpy', s = 30)
plt.scatter(gamma,dndg, marker='.', label='Cerruti')
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.savefig('./comparison_sync_cerruti/figures/agnpy_vs_cerruti_distr_ExpCutoffPL_kchanged')
# Compare synchrotron spectrum
plt.figure(figsize = (6.92, 4.29))
plt.title('Proton synchrotron from ExpCutoffPowerLaw')
plt.scatter(nu_data,psed3, marker='*', label='agnpy', s = 30)
plt.scatter(nu_data,nuFnu_data, marker='.', label='Cerruti')
plt.yscale("log")
plt.xscale("log")
#plt.ylim(1e-22,1e-9)
plt.legend()
#plt.show()

# Distribution comparison plot
gamma_range = [gamma[0], gamma[-1]]
fig3, ax3 = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05}, figsize=(8, 6))
# plot the distribution in the upper panel
# plot the reference sed with a continuous line and agnpy sed with a dashed one
ax3[0].loglog(gamma, dndg, marker=".", ls="-", color="k", lw=1.5, label='Cerruti')
ax3[0].loglog(gamma, n3, marker=".", ls="--", color="crimson", lw=1.5, label='agnpy')
ax3[0].set_ylabel(r'Particle distribution [cm$^{-3}$]')
ax3[0].set_title('ExpCutoffPowerLaw')
ax3[0].legend(loc="best")
#ax3[0].set_ylim(1e-30, 1e-10)
ax3[0].axvline(gamma_range[0], ls="--", color="dodgerblue")
ax3[0].axvline(gamma_range[1], ls="--", color="dodgerblue")
ax3[0].grid(ls=":")
# plot the deviation in the bottom panel
deviation3 = n3 / dndg - 1
ax3[1].axhline(0, ls="-", color="darkgray")
ax3[1].axhline(0.2, ls="--", color="darkgray")
ax3[1].axhline(-0.2, ls="--", color="darkgray")
ax3[1].axhline(0.3, ls=":", color="darkgray")
ax3[1].axhline(-0.3, ls=":", color="darkgray")
ax3[1].set_ylim([-0.5, 0.5])
ax3[1].semilogx(gamma, deviation3, marker=".", ls="--", color="crimson", lw=1.5, label=r'$n(\gamma)_{agnpy}/n(\gamma)_{reference}-1$')
ax3[1].set_xlabel(r'$\gamma$')
ax3[1].legend(loc="best")
ax3[1].axvline(gamma_range[0], ls="--", color="dodgerblue")
ax3[1].axvline(gamma_range[1], ls="--", color="dodgerblue")
fig3.savefig('./comparison_sync_cerruti/figures/agnpy_vs_cerruti_distr_ECPL_kchanged_resid.png')


# sed comparison plot
nu_range = [1e10, 1e28] * u.Hz
make_comparison_plot(
        nu_data,
        psed3,
        nuFnu_data,
        "agnpy",
        "Cerruti",
        "Proton synchrotron from ExpCutoffPowerLaw",
        f"{agnpy_dir}/agnpy/tesi/comparison_sync_cerruti/figures/agnpy_vs_cerruti_psync_ExpCutoffPL_kchanged.png",
        "sed",
        # y_range=[1e-16, 1e-8],
        comparison_range=nu_range.to_value("Hz"),
)
