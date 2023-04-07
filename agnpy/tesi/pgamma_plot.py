import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h, k_B
from astropy.coordinates import Distance
from scipy.integrate import quad, dblquad, nquad, simps, trapz
import numpy as np
from agnpy.spectra import ExpCutoffPowerLaw as ECPL
from agnpy.spectra import PowerLaw as PL
import matplotlib.pyplot as plt
from agnpy.photomeson import PhotoHadronicInteraction_kelner_log as PhotoHadronicInteraction_Reference3
from agnpy.photomeson import PhotoHadronicInteraction_log as photomeson
from pathlib import Path
from agnpy.emission_regions import Blob

plt.style.use('proton_synchrotron')
B = 80 * u.G
redshift = 0.117
distPKS = Distance(z=redshift) # Already inside blob definition
doppler_s = 30
Gamma_bulk = 16
R = 5.2e14 * u.cm #radius of the blob
vol = (4. / 3) * np.pi * R ** 3

def BlackBody(gamma):
    """ BB radiation for target photons """
    T = 2.7 *u.K
    kT = (k_B * T).to('eV').value
    c1 = c.to('cm s-1').value
    h1 = h.to('eV s').value
    norm = 8*np.pi/(h1**3*c1**3)
    num = (mpc2.value * gamma) ** 2
    denom = np.exp(mpc2.value * gamma / kT) - 1
    return norm * (num / denom)*u.Unit("cm-3")

mpc2 = (m_p * c ** 2).to('eV')
mec2 = (m_e * c ** 2).to('eV')

agnpy_dir = Path(__file__).parent.parent.parent  # go to the agnpy root
# where to read sampled files
data_dir = agnpy_dir / "agnpy/data"
# where to save figures, clean-up before making the new

E1, EdNdE1= np.genfromtxt(f"{data_dir}/reference_seds/Kelner_Aharonian_2008/Figure16/photon.txt",
                                     delimiter=',', dtype = 'float', comments = '#', usecols = (0,1), unpack = True)

E2, EdNdE2= np.genfromtxt(f"{data_dir}/reference_seds/Kelner_Aharonian_2008/Figure16/electron.txt",
                                     delimiter=',',dtype = 'float', comments = '#', usecols = (0,1), unpack = True)

E3, EdNdE3= np.genfromtxt(f"{data_dir}/reference_seds/Kelner_Aharonian_2008/Figure16/positron.txt",
                                     delimiter=',',dtype = 'float', comments = '#', usecols = (0,1), unpack = True)


# E1, EdNdE1= np.genfromtxt(f"{data_dir}/reference_seds/Kelner_Aharonian_2008/Figure15/nu_electron.txt",
#                                      delimiter=',', dtype = 'float', comments = '#', usecols = (0,1), unpack = True)
#
# E2, EdNdE2= np.genfromtxt(f"{data_dir}/reference_seds/Kelner_Aharonian_2008/Figure15/antinu_electron.txt",
#                                      delimiter=',',dtype = 'float', comments = '#', usecols = (0,1), unpack = True)
#
# E3, EdNdE3= np.genfromtxt(f"{data_dir}/reference_seds/Kelner_Aharonian_2008/Figure15/nu_muon.txt",
#                                      delimiter=',',dtype = 'float', comments = '#', usecols = (0,1), unpack = True)
#
# E4, EdNdE4= np.genfromtxt(f"{data_dir}/reference_seds/Kelner_Aharonian_2008/Figure15/antinu_muon.txt",
#                                      delimiter=',',dtype = 'float', comments = '#', usecols = (0,1), unpack = True)

# nu1 = (E1*u.eV  / h.to('eV s'))
# gammas1 = (E1*u.eV / mec2).value
# nu2 = (E2*u.eV / h.to('eV s'))
# gammas2 = (E2*u.eV/ mec2).value
# nu3 = (E3*u.eV / h.to('eV s'))
# gammas3 = (E3*u.eV / mec2).value
# nu4 = (E4*u.eV / h.to('eV s'))
# gammas4 = (E4*u.eV / mec2).value

nu = np.logspace(30,36,40)*u.Hz
gammas = (h.to('eV s') * nu / mec2).value
e = nu * h.to('eV s')

A1 = (0.26506*1e11)/(mpc2.value**2) * u.Unit('cm-3')
A2 = (0.24153*1e11)/(mpc2.value**2) * u.Unit('cm-3')
A3 = (0.22170*1e11)/(mpc2.value**2) * u.Unit('cm-3')
A4 = (0.19054*1e11)/(mpc2.value**2) * u.Unit('cm-3')


p = 2.
E_star = 3*1e20 * u.eV
E_cut = 10 * E_star # change to 0.1 etc.. for the other example figures of the reference paper
gamma_cut = E_cut / mpc2

p_dist = ECPL(A3, p, gamma_cut, 10, 1e20)
#
proton_gamma = PhotoHadronicInteraction_Reference3(p_dist, BlackBody)
#
# spec1 = proton_gamma.spectrum(nu, 'nu_electron')
# spec2 = proton_gamma.spectrum(nu, 'antinu_electron')
# spec3 = proton_gamma.spectrum(nu, 'nu_muon')
# spec4 = proton_gamma.spectrum(nu, 'antinu_muon')
spec1 = proton_gamma.spectrum(nu, 'photon')
spec2 = proton_gamma.spectrum(gammas, 'electron')
spec3 = proton_gamma.spectrum(gammas, 'positron')
#
EdNdE_agnpy1 = e * spec1
EdNdE_agnpy2 = e * spec2
EdNdE_agnpy3 = e * spec3
# EdNdE_agnpy4 = e * spec4
#
plt.loglog(E1, EdNdE1, 'o', color = 'black', label = 'Kelner and Aharonian 2008',)
plt.loglog(E2, EdNdE2, 'o', color = 'blue')
plt.loglog(E3, EdNdE3, 'o', color = 'red')
# plt.loglog(E4, EdNdE4, 'o', color = 'green')
#
plt.loglog(e, EdNdE_agnpy1, '-', linewidth = 3 , color = 'black', label = r'$\gamma$' )
plt.loglog(e, EdNdE_agnpy2, '--', linewidth = 3 , color = 'blue',label = r'$e^{-}$' )
plt.loglog(e, EdNdE_agnpy3, '-.', linewidth = 3 , color = 'red',label = r'$e^{+}$' )
# # #
# plt.loglog(e, EdNdE_agnpy1, '-', linewidth = 3 , color = 'black', label = r'$\nu_{e}$' )
# plt.loglog(e, EdNdE_agnpy2, '--', linewidth = 3 , color = 'blue',label = r'$\bar{\nu}_{e} $' )
# plt.loglog(e, EdNdE_agnpy3, '-.', linewidth = 3 , color = 'red',label = r'$\nu_{\mu}$' )
# plt.loglog(e, EdNdE_agnpy4, ':', linewidth = 3, color = 'green',label = r'$\bar{\nu}_{\mu}$' )
# #
# # plt.ylim(9e-29, 2e-25)
# # plt.xlim(9e16, 2e21)
# #
plt.xlabel(r"$E [eV]$")
plt.ylabel (r"$EdNdE [cm^{-3} s^{-1}]$")
# #
# import matplotlib.font_manager as font_manager
# font = font_manager.FontProperties(size=14)

plt.legend(loc = 'upper left')
plt.show()
#
#
# nu = np.logspace(31,37,20)*u.Hz
# gammas = (h.to('eV s') * nu / mec2).value
# e = nu * h.to('eV s')
#
# A1 = (0.26506*1e11)/(mpc2.value**2) * u.Unit('cm-3')
# A2 = (0.24153*1e11)/(mpc2.value**2) * u.Unit('cm-3')
# A3 = (0.22170*1e11)/(mpc2.value**2) * u.Unit('cm-3')
# A4 = (0.19054*1e11)/(mpc2.value**2) * u.Unit('cm-3')
#
#
# p = 2.
# E_star = 3*1e20 * u.eV
# E_cut = 0.1 * E_star # change to 0.1 etc.. for the other example figures of the reference paper
# gamma_cut = E_cut / mpc2
# gamma_cut = 1e7
# p_dist = ECPL(1e4* u.Unit('cm-3'), p, gamma_cut, 10, 1e12)
# # p_dist = ECPL(A2, p, gamma_cut, 10, 1e12)
#
#
# blob = Blob(R_b=R,
#         z=redshift,
#         delta_D=doppler_s,
#         Gamma=Gamma_bulk,
#         B=B,
#         n_p = p_dist,
# )
#
# def fph(epsilon):
#     return (1e-5 * epsilon**(-2))*u.Unit('cm-3')
#
# blazar = photomeson(blob, fph)
# nu = np.logspace(30,34,30)*u.Hz
#
# gammas = (h.to('eV s') * nu / mec2).value
# e = nu * h.to('eV s')
# spec1 = blazar.sed_flux(nu, 'photon')
# # spec2 = blazar.sed_flux(gammas, 'electron')
# # spec3 = blazar.sed_flux(gammas, 'positron')
# spec4 = blazar.sed_flux(nu, 'nu_electron')
# spec5 = blazar.sed_flux(nu, 'antinu_electron')
# # spec6 = blazar.sed_flux(nu, 'nu_muon')
# # spec7 = proton_gamma.spectrum(nu, 'antinu_muon')
#
# EdNdE_agnpy1 =  spec1
# EdNdE_agnpy2 =  spec4
# EdNdE_agnpy3 =  spec5
# # EdNdE_agnpy4 =  spec4
#
# # plt.loglog(gammas, EdNdE_agnpy1, '-', linewidth = 3 , color = 'black', label = r'$e^{-}$' )
# # plt.loglog(gammas, EdNdE_agnpy2, '--', linewidth = 3 , color = 'blue',label = r'$e^{+}$' )
#
# #
# plt.loglog(nu, EdNdE_agnpy1, '-', linewidth = 3 , color = 'black', label = r'$\gamma$' )
# plt.loglog(nu, EdNdE_agnpy2, '--', linewidth = 3 , color = 'blue',label = r'$\nu_{e}$' )
# plt.loglog(nu, EdNdE_agnpy3, '-.', linewidth = 3 , color = 'red',label = r'$\bar{\nu}_{e} $' )
# # # plt.loglog(e, EdNdE_agnpy1, '-', linewidth = 3 , color = 'black', label = r'$\nu_{e}$' )
# # # plt.loglog(e, EdNdE_agnpy2, '--', linewidth = 3 , color = 'blue',label = r'$\bar{\nu}_{e} $' )
# # # plt.loglog(e, EdNdE_agnpy3, '-.', linewidth = 3 , color = 'red',label = r'$\nu_{\mu}$' )
# # # plt.loglog(e, EdNdE_agnpy4, ':', linewidth = 3, color = 'green',label = r'$\bar{\nu}_{\mu}$' )
#
# # plt.xlabel(r"$ \gamma $")
# # plt.ylabel (r"$ {f}_{\gamma} \, [erg \, cm^{-2} \,  s^{-1}]$")
#
# plt.xlabel(r"$ \nu [Hz]$")
# plt.ylabel (r"$ {f}_{\epsilon} \, [erg \, cm^{-2} \,  s^{-1}]$")
# plt.legend(loc = 'lower left')
# plt.show()
