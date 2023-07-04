import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h
from astropy.coordinates import Distance
from scipy.integrate import quad, dblquad, nquad, simps, trapz
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import timeit
import re
from ..utils.math import axes_reshaper, gamma_p_to_integrate, eta_range
from ..utils.conversion import nu_to_epsilon_prime, B_to_cgs, lambda_c_e

from .kelners import *

# to be used in the future to make the code faster:
#import numba as nb

''' Photo-meson production process

    Reference for all expressions:
    Kelner, S.R., Aharonian, 2008, Phys.Rev.D 78, 034013
    (`arXiv:astro-ph/0803.0688 <https://arxiv.org/abs/0803.0688>`_).

    The code is using the results from the above reference. The emission region is
    modified to be the blob of the blazar. Instead of the energies of protons, soft photons
    and neutrinos, lorentz factors and the dimentionless energies are being used.
    Neutrinos are being treated as photons but as particles in the transformation of one
    reference frame to the other.

    The class is initiated by the blob and the soft photon distribution. It computes
    the SED for the final products of the interaction. In such approach, the secondaries
    are integrated out, and therefore their cooling is not described.

    The wanted output particle is given by the user from the variable $particle,
    which can take the following values:

    photon, positron, antinu_muon, nu_electron, nu_muon, electron, antinu_electron

'''

__all__ = ['PhotoMesonProduction']

mpc2 = (m_p * c ** 2).to('eV')
mec2 = (m_e * c ** 2).to('eV')


def H_log(y, eta, y_limit, particle_distribution, soft_photon_dist, particle):
# the integrand (not integral) of equation (70) but in a logarithmic space
# y = log10(gamma) (leptons) or log10(epsilon) (photons, neutrinos)

    u = 10**y
    u_limit = 10**y_limit

    return (1/ u**2 *
        particle_distribution(u).value *
        soft_photon_dist((eta /  (4*u))).value*
        phi_gamma(eta, u_limit/u , particle)
        * u * np.log(10))

class KelnerAharonian2008:
# takes as an input the blob and the soft photon distribution from the user
# and returns the SED of leptons or photons/neutrinos

    def __init__(self, blob, soft_photon_distribution, integrator = np.trapz):

        self.blob = blob
        self.soft_photon_distribution = soft_photon_distribution
        self.integrator = integrator

    @staticmethod
    # calculates the spectrum (eq. 69): the array of gammas/freqs initiated by the user is taken and for each one of
    # the array elements, the spectrum is calculated. For example, in the case of the calculation of the spectrum of photons,
    # we have the array (n_1,n_2,....n_N) which first is tranformed to epsilons and then is being taken as an input in this statismethod
    # function "spectrum". For each element array epsilon, the dNdE is calculated. So basically, it takes the x axis, and calculates the y axis.

    def spectrum(
        gammas,
        particle_distribution,
        soft_photon_distribution,
        particle
    ):
        output_spec = gammas #it is either gammas for electrons, positrons or epsilon for photons, neutrinos
        spectrum_array = np.zeros(len(output_spec))

    @staticmethod
    def evaluate_sed_flux_photons(
        nu, # freq of photons, for the other particles it is gamma
        soft_photon_distribution,
        z,
        d_L,
        delta_D,
        B,
        R_b,
        n_p,
        integrator=np.trapz,
        #gamma_p=gamma_p_to_integrate, # Non mi serve perché dipende dal parametro input
        eta = np.linspace(0.3443,31.3,100)
    ):
        
        epsilon = nu_to_epsilon_prime(nu, 0., delta_D, m = m_p) # dimensionless energy of the produced photons
        spectrum_array = np.zeros(len(epsilon))
        w = np.log10(epsilon)
        
        for i, g in enumerate(w):

            y_p_limit = g # y_p = log10(gamma_p)
            y_p_max = 15 # Poi proviamo valori più alti
            y = np.logspace(g,15,100)

            # Reshape y, eta
            _y, _eta = axes_reshaper(y, eta)

            H_integrand = 1/ (10**_y)**2 * n_p(10**_y).value * \
                soft_photon_dist((_eta /  (4*10**_y))).value * \
                phi_photon(_eta, 10**y_limit/10**_y) * 10**_y * np.log(10) # the term u * np.log(10) comes from the differential of 

            H = (1 / 4) * (mpc2.value) * integrator(H_integrand, 10**_y, axis=0)

            dNdE = integrator(H, _eta, axis=0)








        # volume of blob
        vol = ((4. / 3) * np.pi * R_b ** 3)
        # Area for the calculation of the flux: area of sphere with R = luminosity distance
        area = (4 * np.pi * d_L ** 2)

        # Compute dimensionless energy for photons
        epsilon = nu_to_epsilon_prime(input, z, delta_D, m = m_p)
        massa = mpc2.to('erg')

        sed_source_frame = (
                KelnerAharonian2008.spectrum(
                epsilon, n_p, soft_photon_distribution,particle
                ) * (vol / area) * (epsilon * massa) ** 2
        ).to("erg cm-2 s-1")

        sed = sed_source_frame * np.power(delta_D, 4)

        return sed

    # the user uses this function:
    def sed_flux(self, input, particle):
        r"""Evaluates the photomeson flux SED for a photomeson object built
        from a Blob."""
        return self.evaluate_sed_flux(
            input, # either frequencies or gammas
            particle,
            self.soft_photon_distribution,
            self.blob.z,
            self.blob.d_L,
            self.blob.delta_D,
            self.blob.B,
            self.blob.R_b,
            self.blob.n_p,
            integrator=self.integrator,
            gamma=self.blob.gamma_p,
        )

    def sed_luminosity(self, input, particle):
        r"""Evaluates the synchrotron luminosity SED
        :math:`\nu L_{\nu} \, [\mathrm{erg}\,\mathrm{s}^{-1}]`
        for a PhotoMesonProduction object built from a blob."""
        sphere = 4 * np.pi * np.power(self.blob.d_L, 2)
        return (sphere * self.sed_flux(nu, particle)).to("erg s-1")

    def sed_peak_flux(self, input, particle):
        """provided a grid of frequencies nu or Lorentz factors gamma, returns the peak flux of the SED
        """
        return self.sed_flux(input, particle).max()

    def sed_peak_nu(self, input):
        """provided a grid of frequencies nu or Lorentz factors gamma, returns the frequency or Lorentz factor
        at which the SED peaks
        """
        idx_max = self.sed_flux(input, particle).argmax()
        return nu[idx_max]
