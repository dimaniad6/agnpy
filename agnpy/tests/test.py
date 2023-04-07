import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h, k_B
from astropy.coordinates import Distance
from scipy.integrate import quad, dblquad, nquad, simps, trapz
import numpy as np
from agnpy.spectra import ExpCutoffPowerLaw as ECPL
import pytest
import matplotlib.pyplot as plt
from agnpy.photomeson import PhotoHadronicInteraction_Reference
from pathlib import Path
mpc2 = (m_p * c ** 2).to('eV')
mec2 = (m_e * c ** 2).to('eV')
def extract_columns_sample_file(sample_file, x_unit, y_unit=None):
    """Return two arrays of quantities from a sample file."""
    sample_table = np.loadtxt(sample_file, delimiter=",", comments="#")
    x = sample_table[:, 0] * u.Unit(x_unit)
    y = sample_table[:, 1] if y_unit is None else sample_table[:, 1] * u.Unit(y_unit)
    return x, y

E, EdNdE= extract_columns_sample_file("/home/dimitris/Desktop/agnpy/agnpy/agnpy/data/reference_seds/Kelner_Aharonian_2008/Figure15/photon.txt",
                                    x_unit = 'eV', y_unit = 'cm-3 s-1')

def BlackBody(gamma):
    """ CMB radiation for target photons """
    T = 2.7 *u.K
    kT = (k_B * T).to('eV').value
    c1 = c.to('cm s-1').value
    h1 = h.to('eV s').value
    norm = 8*np.pi/(h1**3*c1**3)
    num = (mpc2.value * gamma) ** 2
    denom = np.exp(mpc2.value * gamma / kT) - 1
    return norm * (num / denom)*u.Unit("cm-3")

nu = (E / h.to('eV s'))
gammas = E * u.eV / mpc2
A = (0.265*1e11)/(mpc2.value**2) * u.Unit('cm-3')
p = 2.
E_star = 3*1e20 * u.eV
E_cut = 1 * E_star # change to 0.1 etc.. for the other example figures of the reference paper
gamma_cut = E_cut / mpc2
p_dist = ECPL(A, p, gamma_cut, 1, 1e12)
proton_gamma = PhotoHadronicInteraction_Reference(p_dist, BlackBody)
spec = proton_gamma.spectrum(nu, 'photon')
EdNdE_agnpy = E * spec
print (EdNdE_agnpy)
