import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h, k_B
from astropy.coordinates import Distance
from scipy.integrate import quad, dblquad, nquad, simps, trapz
import numpy as np
from agnpy.spectra import ExpCutoffPowerLaw as ECPL
import pytest
import matplotlib.pyplot as plt
from agnpy.photomeson import Kelner as PhotoHadronicInteraction
from pathlib import Path
from .utils import (
    make2_comparison_plot,
    extract_columns_sample_file,
    check_deviation,
    clean_and_make_dir,
)

#problems with paths


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

mpc2 = (m_p * c ** 2).to('eV')
mec2 = (m_e * c ** 2).to('eV')

agnpy_dir = Path(__file__).parent.parent.parent  # go to the agnpy root
# where to read sampled files
data_dir = agnpy_dir / "agnpy/data"
# where to save figures, clean-up before making the new
figures_dir = clean_and_make_dir(agnpy_dir, "crosschecks/figures/photomeson_reference")

class Test_PhotomesonReference:
        #EdNdE, E = np.genfromtxt(f"{data_dir}/reference_seds/kelner_Aharonian_2008/Figure15/{}.txt".format(particle),
    def test_aharonian_example(self):
        E, EdNdE= extract_columns_sample_file(f"{data_dir}/reference_seds/Kelner_Aharonian_2008/Figure17/photon.txt",
                                            x_unit = 'eV', y_unit = 'cm-3 s-1')
        #
        nu = (E / h.to('eV s'))
        gammas = (E / mec2).value
        # nu = np.logspace(27,37,50)*u.Hz
        A4 = (0.19054*1e11)/(mpc2.value**2) * u.Unit('cm-3')
        A = (0.24153*1e11)/(mpc2.value**2) * u.Unit('cm-3')
        p = 2.
        E_star = 3*1e20 * u.eV
        E_cut = 1000 * E_star # change to 0.1 etc.. for the other example figures of the reference paper
        gamma_cut = E_cut / mpc2

        p_dist = ECPL(A4, p, gamma_cut, 10, 1e13)

        proton_gamma = PhotoHadronicInteraction(p_dist, BlackBody)
        spec = proton_gamma.spectrum(nu, 'photon')
        EdNdE_agnpy = E * spec

        E_range = (1e17,1e21)

        make2_comparison_plot(
            E,
            EdNdE_agnpy,
            EdNdE,
            "agnpy",
            "Kelner Aharonian 2008",
            "Comparison with the literature: electron anti-neutrino spectrum",
            f"{figures_dir}/photomeson.eps",
            "spectrum E dN/dE",
            y_range=[1e-28, 1e-25],
            comparison_range=E_range
        )

        assert u.allclose(EdNdE_agnpy, EdNdE, atol= 0 * u.Unit("cm-3 s-1"), rtol=0.2)
