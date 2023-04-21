import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h, hbar, k_B
from astropy.table import Table, Column
from astropy.coordinates import Distance
from scipy.integrate import quad, dblquad, nquad, simps, trapz
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import timeit
import re
# File with all available soft photon distributions:
# to be used in the future to make the code faster:
import numba as nb

def axes_reshaper(*args):
    """reshape 1-dimensional arrays of different lengths in order for them to be
    broadcastable in multi-dimensional operations

    the rearrangement scheme for a list of n arrays is the following:
    `args[0]` is reshaped as `(args[0].size, 1, 1, ..., 1)` -> axis 0
    `args[1]` is reshaped as `(1, args[1].size, 1, ..., 1)` -> axis 1
    `args[2]` is reshaped as `(1, 1, args[2].size ..., 1)` -> axis 2
        .
        .
        .
    `args[n-1]` is reshaped as `(1, 1, 1, ..., args[n-1].size)` -> axis n-1
    Parameters
    ----------
    args: 1-dimensional `~numpy.ndarray`s to be reshaped
    """
    n = len(args)
    dim = (1,) * n
    reshaped_arrays = []
    for i, arg in enumerate(args):
        reshaped_dim = list(dim)  # the tuple is copied in the list
        reshaped_dim[i] = arg.size
        reshaped_array = np.reshape(arg, reshaped_dim)
        reshaped_arrays.append(reshaped_array)
    return reshaped_arrays

def epsilon_equivalency(nu, m = m_e):
    if m == m_e:
        epsilon_equivalency = h.to('eV s') * nu / mec2

    elif m == m_p:
        epsilon_equivalency = h.to('eV s')* nu / mpc2

    return epsilon_equivalency


''' Photomeson process.

    Reference for all expressions:
    Kelner, S.R., Aharonian, 2008, Phys.Rev.D 78, 034013
    (`arXiv:astro-ph/0803.0688 <https://arxiv.org/abs/0803.0688>`_).

    This script is used to reproduce the examples from the reference paper.
    They results are being tested on the pytest.

'''

__all__ = ['PhotoHadronicInteraction_Reference']

mpc2 = (m_p * c ** 2).to('eV')
mec2 = (m_e * c ** 2).to('eV')

# Added a soft photon distribution function like this, necessary for the class to be
# able to take a soft photon distribution as an input.
particles = ('photon','electron','positron','nu_electron','nu_electron','nu_muon','antinu_electron','antinu_muon' )

# The file changes according to the type of particle

def lookup_tab1(eta, particle):

    for i in particles:
        if i == particle:
            interp_file = "../data/interpolation_tables/{}.txt".format(i)

    eta_eta0, s, delta, B = np.genfromtxt(interp_file, dtype = 'float',  comments = '#', usecols = (0,1,2,3), unpack = 'True')

    s_int = interp1d(eta_eta0, s, kind='linear', bounds_error=False, fill_value="extrapolate")
    delta_int = interp1d(eta_eta0, delta, kind='linear', bounds_error=False, fill_value="extrapolate")
    B_int = interp1d(eta_eta0, B, kind='linear', bounds_error=False, fill_value="extrapolate")

    return s_int(eta), delta_int(eta), B_int(eta)


def x_plus_minus(eta, particle):

    r = 0.146 # r = m_pi / M_p
    x_1 = eta + r ** 2
    x_2 = np.sqrt((eta - r ** 2 - 2 * r) * (eta - r ** 2 + 2 * r))
    x_3 = 1 / (2 * (1 + eta))

    x_plus = x_3 * (x_1 + x_2)
    x_minus = x_3 * (x_1 - x_2)

    if particle == 'photon':
        return x_plus, x_minus

    elif particle in ('positron', 'antinu_muon', 'nu_electron'):
        return x_plus, x_minus / 4

    elif particle in ('electron', 'antinu_electron'):
        r = 0.146
        x_1 = 2 * (1 + eta)
        x_2 = eta - (2 * r)
        x_3 = np.sqrt(eta * (eta - 4 * r * (1 + r)))

        x_plus = (x_2 + x_3) / x_1
        x_minus = (x_2 - x_3) / x_1

        return x_plus, x_minus / 2

    elif particle == 'nu_muon':
        rho = eta / 0.313
        if rho < 2.14:
            xp = 0.427 * x_plus
        elif rho > 2.14 and rho < 10:
            xp = (0.427 + 0.0729 * (rho - 2.14)) * x_plus
        elif rho > 10:
            xp = x_plus

        return xp, (x_minus * 0.427)


def phi_gamma(eta, x, particle):

    x_p, x_n = x_plus_minus(eta, particle)

    s, delta, B = lookup_tab1(eta / 0.313, particle) # eta_0 = 0.313

    if particle == 'photon':
        psi = 2.5 + 0.4 * np.log(eta / 0.313)
    elif particle in ('positron', 'antinu_muon', 'nu_electron', 'nu_muon'):
        psi = 2.5 + 1.4 * np.log(eta / 0.313)
    elif particle in ('electron', 'antinu_electron'):
        psi = 6 * (1 - np.exp(1.5 * (4 - eta/0.313))) * (np.sign(eta/0.313 - 4) + 1) / 2.
        # the np.sign part is the heavinside function of (rho - 4) where rho = eta/eta0

    if x > x_n and x < x_p:
        y = (x - x_n) / (x_p - x_n)
        ln1 = np.exp(- s * (np.log(x / x_n)) ** delta)
        ln2 = np.log(2. / (1 + y**2))
        return B * ln1 * ln2 ** psi

    elif x < x_n:
        return B * (np.log(2)) ** psi

    elif x > x_p:
        return 0


def H_integrand(gamma, eta, gamma_limit, particle_distribution, soft_photon_dist, particle):
    return (1 / gamma ** 2  *
        particle_distribution(gamma).value *
        soft_photon_dist((eta /  (4*gamma))).value*
        phi_gamma(eta, gamma_limit/gamma , particle)
    )


class PhotoHadronicInteraction_Reference:

    def __init__(self, particle_distribution, soft_photon_distribution):

        self.particle_distribution = particle_distribution
        self.soft_photon_distribution = soft_photon_distribution


    @staticmethod
    def spectrum_calculator(
        gammas,
        particle_distribution,
        soft_photon_distribution,
        particle
    ):
        output_spec = gammas #it is either gammas for electrons, positrons or epsilon for photons, neutrinos
        spectrum_array = np.zeros(len(output_spec))

        for i, g in enumerate(output_spec):

            if particle in ('electron','positron'):
                gamma_limit = g * (mec2/mpc2)
                gamma = np.logspace(np.log10(gamma_limit.value), 14, 100)
            else:
                gamma_limit = g
                gamma = np.logspace(np.log10(gamma_limit.value), 14, 100)

            if particle in ('electron', 'antinu_electron'):
                eta = np.linspace(0.945,31.3,100)
            else:
                eta = np.linspace(0.3443,31.3,100)

            #_gamma, _eta = axes_reshaper(gamma,eta)

            H_int = np.zeros(len(eta))

            print ('gamma_limit: ', gamma_limit)
            #print ('len eta: ', len(eta))
            print ('len_gamma: ', len(gamma))
            print ('gamma: ',gamma)
            #print ('eta: ',eta)
            H = np.zeros(len(gamma))

            for j in range(1,len(gamma)):

                for k in range(1,len(eta)):
                    H_int[k] = (H_integrand(float(gamma[j]),eta[k],gamma_limit # H[η1, γ], H[η2, γ], ...
                                    ,particle_distribution,soft_photon_distribution,particle))
                    # print ('k is: ', k)
                    # if H_int[k] > 0:
                    #     print ('integral is: ', H_int[k])

                H[j] = (1 / 4) * (mpc2.value) * np.trapz(H_int, gamma)

            dNdE = np.trapz(H, eta, axis=0) #dN/dE = dN/dE (γ), γ taken from the output particle array

            spectrum_array[i] = dNdE
            print ('SPECTRUM IS: ', dNdE)
            print ("Computing {} spectrum: {}% is completed..."
                .format(particle ,int(100*(i+1) / len(output_spec))))

        return (spectrum_array * u.Unit('eV-1 cm-3 s-1'))


    @staticmethod
    def evaluate_spectrum(
        input,
        particle_distribution,
        soft_photon_distribution,
        particle
    ):

        if particle not in ('electron', 'positron'):
            input = epsilon_equivalency(input, m = m_p)

        spectrum = PhotoHadronicInteraction_Reference.spectrum_calculator(
                input , particle_distribution, soft_photon_distribution, particle
                )

        return spectrum

    def spectrum(self, input, particle):
        return self.evaluate_spectrum(
            input,
            self.particle_distribution,
            self.soft_photon_distribution,
            particle
        )


if __name__ == '__main__':
    from agnpy.spectra import ExpCutoffPowerLaw as ECPL

    # SoftPhotonDistribution
    def BlackBody(gamma):
        T = 2.7 *u.K
        kT = (k_B * T).to('eV').value
        c1 = c.to('cm s-1').value
        h1 = h.to('eV s').value
        norm = 8*np.pi/(h1**3*c1**3)
        num = (mpc2.value * gamma) ** 2
        denom = np.exp(mpc2.value * gamma / kT) - 1
        return norm * (num / denom)*u.Unit('cm-3')

    start = timeit.default_timer()

    # EXAMPLE AHARONIAN:

    # Proton distribution: ExpCutoffPowerLaw with Ec = 0.1, 1 * E_star, Figures 14,15
    # Soft photon distribution: CMB
    mpc2 = (m_p * c ** 2).to('eV')
    nu = np.logspace(20,36,25)*u.Hz
    gammas = epsilon_equivalency(nu, m = m_p)
    ene = nu * h.to('eV s')
    print (gammas)
    A = (0.265*1e11)/(mpc2.value**2) * u.Unit('cm-3')
    p = 2.
    E_star = 3*1e20 * u.eV
    E_cut = 1 * E_star
    gamma_cut = E_cut / mpc2

    p_dist = ECPL(A, p, gamma_cut, 10, 1e12)

    proton_gamma = PhotoHadronicInteraction_Reference(p_dist, BlackBody)


    # spec_ele = proton_gamma.spectrum(gamma2, 'electron')
    spec = proton_gamma.spectrum(nu, 'photon')
    # spec_posi = proton_gamma.spectrum_electron(gammas, 'positron')
    # spec_nu_muon = proton_gamma.spectrum(nu3, 'nu_muon')
    # spec_antinu_muon = proton_gamma.spectrum(nu, 'antinu_muon')
    # spec_nu_electron = proton_gamma.spectrum(nu, 'nu_electron')

    #
    plt.loglog((ene), (spec * ene ), color='orange',label = 'photons')
    # plt.loglog((E2), (spec_ele * E2 ), '.', color='blue',label = 'electrons')
    # plt.loglog((ene), (spec_posi * ene ), lw=2.2, ls='-', color='red',label = 'positrons')
    # plt.loglog((E3), (spec_nu_muon * E3), lw=2.2, ls='-', color='red',label = 'spec_nu_muon')
    # plt.loglog((ene), (spec_antinu_muon * ene ), lw=2.2, ls='-', color='blue',label = 'spec_antinu_muon')
    # plt.loglog((ene), (spec_nu_electron * ene ), lw=2.2, ls='-', color='green',label = 'spec_nu_electron')

    plt.legend()
    plt.show()

    stop = timeit.default_timer()
    print("Elapsed time for computation = {} secs".format(stop - start))
