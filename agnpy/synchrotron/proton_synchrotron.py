# module containing the synchrotron radiative process
import numpy as np
import astropy.units as u
<<<<<<< HEAD
from astropy.constants import e, h, c, m_e, m_p, sigma_T, G
from ..utils.math import axes_reshaper, gamma_e_to_integrate
from ..utils.conversion import nu_to_epsilon_prime, B_to_cgs, lambda_c_p
from .synchrotron import Synchrotron
=======
from astropy.constants import e, c, m_p
from ..utils.math import axes_reshaper, gamma_e_to_integrate
from ..utils.conversion import nu_to_epsilon_prime, B_to_cgs, lambda_c_p
>>>>>>> c3777f3170841b02c759e5f26ba3ba3574508de2
from .synchrotron import single_particle_synch_power, tau_to_attenuation

__all__ = ["ProtonSynchrotron"]

e = e.gauss
B_cr = 4.414e13 * u.G  # critical magnetic field

class ProtonSynchrotron:
    """Class for synchrotron radiation computation

    Parameters
    ----------
    blob : :class:`~agnpy.emission_region.Blob`
<<<<<<< HEAD
        emitting region and electron distribution
=======
        emitting region and proton distribution
>>>>>>> c3777f3170841b02c759e5f26ba3ba3574508de2
    ssa : bool
        whether or not to consider synchrotron self absorption (SSA).
        The absorption factor will be taken into account in
        :func:`~agnpy.synchrotron.Synchrotron.com_sed_emissivity`, in order to be
        propagated to :func:`~agnpy.synchrotron.Synchrotron.sed_luminosity` and
        :func:`~agnpy.synchrotron.Synchrotron.sed_flux`.
    integrator : func
        function to be used for integration (default = `np.trapz`)
	"""

<<<<<<< HEAD
    def __init__(self, blob, ssa=False, integrator=np.trapz):
        self.blob = blob
        self.ssa = ssa
        self.integrator = integrator

    @staticmethod
    def evaluate_tau_ssa(
        nu,
        z,
        d_L,
        delta_D,
        B,
        R_b,
        n_p,
        *args,
        integrator=np.trapz,
        gamma=gamma_e_to_integrate,
    ):
        """Computes the syncrotron self-absorption opacity for a general set
        of model parameters, see :func:`~agnpy:sycnhrotron.Synchrotron.evaluate_sed_flux`
        for parameters defintion. Eq. before 7.122 in [DermerMenon2009]_."""
        # conversions
        epsilon = nu_to_epsilon_prime(nu, z, delta_D, m = m_p)
        B_cgs = B_to_cgs(B)
        # multidimensional integration
        _gamma, _epsilon = axes_reshaper(gamma, epsilon)
        SSA_integrand = n_p.evaluate_SSA_integrand(_gamma, *args)
        integrand = SSA_integrand * single_particle_synch_power(B_cgs, _epsilon, _gamma, mass = m_p)
        integral = integrator(integrand, gamma, axis=0)
        prefactor_k_epsilon = (
            -1 / (8 * np.pi * m_p * np.power(epsilon, 2)) * np.power(lambda_c_p / c, 3)
        )
        k_epsilon = (prefactor_k_epsilon * integral).to("cm-1")
        return (2 * k_epsilon * R_b).to_value("")
=======
    #def __init__(self, blob, ssa=False, integrator=np.trapz):
    def __init__(self, blob, integrator=np.trapz):
        self.blob = blob
        #self.ssa = ssa
        self.integrator = integrator

    # @staticmethod
    # def evaluate_tau_ssa(
    #     nu,
    #     z,
    #     delta_D,
    #     B,
    #     R_b,
    #     n_p,
    #     *args,
    #     integrator=np.trapz,
    #     gamma=gamma_e_to_integrate,
    # ):
    #     """Computes the synchrotron self-absorption opacity for a general set
    #     of model parameters, see :func:`~agnpy:sycnhrotron.Synchrotron.evaluate_sed_flux`
    #     for parameters defintion. Eq. before 7.122 in [DermerMenon2009]_."""
    #     # conversions
    #     epsilon = nu_to_epsilon_prime(nu, z, delta_D, m = m_p)
    #     B_cgs = B_to_cgs(B)
    #     # multidimensional integration
    #     _gamma, _epsilon = axes_reshaper(gamma, epsilon)
    #     SSA_integrand = n_p.evaluate_SSA_integrand(_gamma, *args)
    #     integrand = SSA_integrand * single_particle_synch_power(B_cgs, _epsilon, _gamma, mass = m_p)
    #     integral = integrator(integrand, gamma, axis=0)
    #     prefactor_k_epsilon = (
    #         -1 / (8 * np.pi * m_p * np.power(epsilon, 2)) * np.power(lambda_c_p / c, 3)
    #     )
    #     k_epsilon = (prefactor_k_epsilon * integral).to("cm-1")
    
    #     return (2 * k_epsilon * R_b).to_value("")
>>>>>>> c3777f3170841b02c759e5f26ba3ba3574508de2

    @staticmethod
    def evaluate_sed_flux(
        nu,
        z,
        d_L,
        delta_D,
        B,
        R_b,
        n_p,
        *args,
<<<<<<< HEAD
        ssa=False,
=======
        # ssa=False,
>>>>>>> c3777f3170841b02c759e5f26ba3ba3574508de2
        integrator=np.trapz,
        gamma=gamma_e_to_integrate,
    ):
        r"""Evaluates the flux SED (:math:`\nu F_{\nu}`) due to synchrotron radiation,
<<<<<<< HEAD
        for a general set of model parameters. Eq. 21 in [Finke2008]_.
=======
        for a general set of model parameters. As for electrons, we implement Eq. 21 in
        [Finke2008]_ with just a change in the mass value (we are using the proton mass now).
        For a reference on proton synchrotron and other hadronic processes see [Cerruti2015]_.
>>>>>>> c3777f3170841b02c759e5f26ba3ba3574508de2

        **Note** parameters after \*args need to be passed with a keyword

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed
            **note** these are observed frequencies (observer frame)
        z : float
            redshift of the source
        d_L : :class:`~astropy.units.Quantity`
            luminosity distance of the source
        delta_D : float
            Doppler factor of the relativistic outflow
        B : :class:`~astropy.units.Quantity`
            magnetic field in the blob
        R_b : :class:`~astropy.units.Quantity`
            size of the emitting region (spherical blob assumed)
<<<<<<< HEAD
        n_e : :class:`~agnpy.spectra.ElectronDistribution`
            electron energy distribution
        \*args
            parameters of the electron energy distribution (k_e, p, ...)
        ssa : bool
            whether to consider or not the self-absorption, default false
        integrator : func
            which function to use for integration, default `numpy.trapz`
        gamma : :class:`~numpy.ndarray`
            array of Lorentz factor over which to integrate the electron
=======
        n_p : :class:`~agnpy.spectra.ProtonDistribution`
            proton energy distribution
        \*args
            parameters of the proton energy distribution (k_e, p, ...)
        # ssa : bool
        #     whether to consider or not the self-absorption, default false
        integrator : func
            which function to use for integration, default `numpy.trapz`
        gamma : :class:`~numpy.ndarray`
            array of Lorentz factor over which to integrate the proton
>>>>>>> c3777f3170841b02c759e5f26ba3ba3574508de2
            distribution

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the SED values corresponding to each frequency
        """
        # conversions
        epsilon = nu_to_epsilon_prime(nu, z, delta_D, m = m_p)
        B_cgs = B_to_cgs(B)
        # reshape for multidimensional integration
        _gamma, _epsilon = axes_reshaper(gamma, epsilon)
        V_b = 4 / 3 * np.pi * np.power(R_b, 3)
        N_p = V_b * n_p.evaluate(_gamma, *args)
<<<<<<< HEAD
        # fold the electron distribution with the synchrotron power
=======
        # fold the proton distribution with the synchrotron power
>>>>>>> c3777f3170841b02c759e5f26ba3ba3574508de2
        integrand = N_p * single_particle_synch_power(B_cgs, _epsilon, _gamma, mass=m_p)
        emissivity = integrator(integrand, gamma, axis=0)
        prefactor = np.power(delta_D, 4) / (4 * np.pi * np.power(d_L, 2))
        sed = (prefactor * epsilon * emissivity).to("erg cm-2 s-1")

<<<<<<< HEAD
        if ssa:
            tau = ProtonSynchrotron.evaluate_tau_ssa(
                nu,
                z,
                d_L,
                delta_D,
                B,
                R_b,
                n_p,
                *args,
                integrator=integrator,
                gamma=gamma,
            )
            attenuation = tau_to_attenuation(tau)
            sed *= attenuation
=======
        # if ssa:
        #     tau = ProtonSynchrotron.evaluate_tau_ssa(
        #         nu,
        #         z,
        #         d_L,
        #         delta_D,
        #         B,
        #         R_b,
        #         n_p,
        #         *args,
        #         integrator=integrator,
        #         gamma=gamma,
        #     )
        #     attenuation = tau_to_attenuation(tau)
        #     sed *= attenuation
>>>>>>> c3777f3170841b02c759e5f26ba3ba3574508de2

        return sed

    def sed_flux(self, nu):
        r"""Evaluates the synchrotron flux SED for a Synchrotron object built
        from a Blob."""
        return self.evaluate_sed_flux(
            nu,
            self.blob.z,
            self.blob.d_L,
            self.blob.delta_D,
            self.blob.B,
            self.blob.R_b,
            self.blob.n_p,
            *self.blob.n_p.parameters,
<<<<<<< HEAD
=======
            # ssa=self.ssa,
>>>>>>> c3777f3170841b02c759e5f26ba3ba3574508de2
            integrator=self.integrator,
            gamma=self.blob.gamma_p,
        )

    def sed_luminosity(self, nu):
        r"""Evaluates the synchrotron luminosity SED
        :math:`\nu L_{\nu} \, [\mathrm{erg}\,\mathrm{s}^{-1}]`
        for a a Synchrotron object built from a blob."""
        sphere = 4 * np.pi * np.power(self.blob.d_L, 2)
        return (sphere * self.sed_flux(nu)).to("erg s-1")

    def sed_peak_flux(self, nu):
        """provided a grid of frequencies nu, returns the peak flux of the SED
        """
        return self.sed_flux(nu).max()

    def sed_peak_nu(self, nu):
        """provided a grid of frequencies nu, returns the frequency at which the
        SED peaks
        """
        idx_max = self.sed_flux(nu).argmax()
        return nu[idx_max]
