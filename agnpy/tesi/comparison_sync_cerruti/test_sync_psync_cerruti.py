# test the synchrotron module
import numpy as np
import astropy.units as u
from astropy.constants import m_e, m_p
from astropy.coordinates import Distance
import pytest
from pathlib import Path
from agnpy.emission_regions import Blob
from agnpy.spectra import ExpCutoffPowerLaw, ExpCutoffBrokenPowerLaw
from agnpy.synchrotron import Synchrotron, nu_synch_peak
from agnpy.synchrotron import ProtonSynchrotron
from agnpy.utils.math import trapz_loglog
from .utils import (
    make_comparison_plot,
    extract_columns_sample_file,
    check_deviation,
    clean_and_make_dir,
)

agnpy_dir = Path(__file__).parent.parent.parent  # go to the agnpy root
# where to read sampled files
data_dir = agnpy_dir / "agnpy/data"
# where to save figures, clean-up before making the new
figures_dir = clean_and_make_dir(agnpy_dir, "crosschecks/figures/proton_synchrotron")


# Define source parameters
B = 62.8936 * u.G
redshift = 0.044
distPKS = Distance(z=redshift) # Already inside blob definition
doppler_s = 30
Gamma_bulk = 16 #?? Come lo calcolo senza angolo?
R = 9.06234e+14 * u.cm #radius of the blob


class TestProtonSynchrotron2:
    """Class grouping all tests related to the Proton Synchrotron class."""

    def test_synch_reference_sed(self):
        """Test agnpy proton synchrotron SED against the SED produced by the same particle distribution from Matteo Cerruti's code."""
        # reference SED
        lognu, lognuFnu = np.genfromtxt(f"/Users/ilaria/Desktop/Dottorato_data/AGNpy/AGNpy_github/agnpy/agnpy/tesi/data/Cerruti/third_email/new175_30_1147_pss.dat",  dtype = 'float', comments = '#', usecols = (0,4), unpack = True)
        nu_ref = 10**lognu * u.Unit('Hz')
        sed_ref = 10**lognuFnu *u.Unit("erg cm-2 s-1")

        # agnpy
        n_p = ExpCutoffBrokenPowerLaw(k=12e4 / u.Unit('cm3'), #12e3 / u.Unit('cm3'),
            p1 = 1.5 ,
            p2 = 2.5,
            gamma_c= 3.64227e+09,
            gamma_min= 1,
            gamma_max=1e20,
            mass=m_p
        )

        blob = Blob(R_b=R,
                z=redshift,
                delta_D=doppler_s,
                Gamma=Gamma_bulk,
                B=B,
                n_p=n_p
        )

        psynch = ProtonSynchrotron(blob, ssa = True)
        sed_agnpy = psynch.sed_flux(nu_ref)

        # sed comparison plot
        nu_range = [1e10, 1e28] * u.Hz
        make_comparison_plot(
            nu_ref,
            sed_agnpy,
            sed_ref,
            "agnpy",
            "Cerruti model",
            "Proton Synchrotron",
            f"/Users/ilaria/Desktop/Dottorato_data/AGNpy/AGNpy_github/agnpy/agnpy/figures/proton_synch_comparison_Cerruti.png",
            "sed",
            # y_range=[1e-16, 1e-8],
            comparison_range=nu_range.to_value("Hz"),
        )
        
        # requires that the SED points deviate less than 25% from the figure
        assert check_deviation(nu_ref, sed_agnpy, sed_ref, 0.8, nu_range)


    # def test_sed_integration_methods(self):
    #     """Test different integration methods against each other:
    #     simple trapezoidal rule vs trapezoidal rule in log-log space.
    #     """
    #     n_p = ExpCutoffPowerLaw(k=12e4 / u.Unit('cm3'), #12e3 / u.Unit('cm3'),
    #         p = 2.2 ,
    #         gamma_c= 2.5e9,
    #         gamma_min= 1,
    #         gamma_max=1e20,
    #         mass=m_p
    #     )

    #     blob = Blob(R_b=R,
    #             z=redshift,
    #             delta_D=doppler_s,
    #             Gamma=Gamma_bulk,
    #             B=B,
    #             n_p=n_p
    #     )

    #     psynch = ProtonSynchrotron(blob)

    #     # sed comparison plot
    #     nu = [1e10, 1e30] * u.Hz
    #     synch_trapz = ProtonSynchrotron(blob, integrator=np.trapz)
    #     synch_trapz_loglog = ProtonSynchrotron(blob, integrator=trapz_loglog)

    #     nu = np.logspace(8, 23) * u.Hz
    #     sed_synch_trapz = synch_trapz.sed_flux(nu)
    #     sed_synch_trapz_loglog = synch_trapz_loglog.sed_flux(nu)

    #     # sed comparison plot
    #     nu_range = [1e8, 1e22] * u.Hz
    #     make_comparison_plot(
    #         nu,
    #         sed_synch_trapz_loglog,
    #         sed_synch_trapz,
    #         "trapezoidal log-log integration",
    #         "trapezoidal integration",
    #         "Proton Synchrotron",
    #         f"/Users/ilaria/Desktop/Dottorato_data/AGNpy/AGNpy_github/agnpy/agnpy/tesi/comparison_sync_cerruti/figures/proton_synch_comparison_integration_methods_model2.png/figures/proton_synch_comparison_integration_methods.png",
    #         "sed",
    #         y_range=[1e-16, 1e-10],
    #         comparison_range=nu_range.to_value("Hz"),
    #     )

    #     # requires that the SED points deviate less than 10%
    #     assert check_deviation(
    #         nu, sed_synch_trapz_loglog, sed_synch_trapz, 0.1, nu_range
    #     )

    # def test_nu_synch_peak(self):
    #     """Test peak synchrotron frequency for a given magnetic field and Lorentz factor."""
    #     gamma = 100
    #     nu_synch = nu_synch_peak(1 * u.G, gamma, mass = m_p).to_value("Hz")
    #     assert np.isclose(nu_synch, 15245186.451944835, atol=0)
