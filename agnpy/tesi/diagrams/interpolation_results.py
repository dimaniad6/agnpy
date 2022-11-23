import numpy as np
import astropy.units as u
from agnpy.spectra import PowerLaw, BrokenPowerLaw, LogParabola, ExpCutoffPowerLaw, InterpolatedDistribution
from agnpy.utils.math import trapz_loglog
from agnpy.utils.conversion import mec2
import matplotlib.pyplot as plt
from scipy.misc import derivative
from scipy.interpolate import CubicSpline


'''1. Initializing the four distributions'''

k_e_test = 1e-13 * u.Unit("cm-3")
p_test = 2.1
gamma_min_test = 10
gamma_max_test = 1e7
pwl_test = PowerLaw(
    k_e_test, p_test, gamma_min_test, gamma_max_test
)
# global BrokenPowerLaw
p1_test = 2.1
p2_test = 3.1
gamma_b_test = 1e3
bpwl_test = BrokenPowerLaw(
    k_e_test, p1_test, p2_test, gamma_b_test, gamma_min_test, gamma_max_test
)
# global LogParabola
q_test = 0.2
gamma_0_test = 1e4
lp_test = LogParabola(
    k_e_test, p_test, q_test, gamma_0_test, gamma_min_test, gamma_max_test
)
# global PowerLaw exp Cutoff
gamma_c_test = 1e3
epwl_test = ExpCutoffPowerLaw(
    k_e_test, p_test, gamma_c_test, gamma_min_test, gamma_max_test
)

""" Data and Interpolation """

gamma1 = np.logspace(np.log10(gamma_min_test),np.log10(gamma_max_test),10)
gamma2 = np.logspace(np.log10(gamma_min_test),np.log10(1e5),10) #just for exp

g1 = np.logspace(np.log10(gamma_min_test),np.log10(gamma_max_test),100)
g2 = np.logspace(np.log10(gamma_min_test),np.log10(1e5),100) #just for exp
#PowerLaw
pwl_data = pwl_test(gamma1)
pwl_inter = InterpolatedDistribution(
    gamma1, pwl_data
)
n_pwl = pwl_inter(g1)
# BrokenPowerLaw
bpwl_data = bpwl_test(gamma1).value
bpwl_inter = InterpolatedDistribution(
    gamma1, bpwl_data *u.Unit('cm-3')
)
n_bpwl = bpwl_inter(g1)
#LogParabola
lp_data = lp_test(gamma1).value
lp_inter = InterpolatedDistribution(
    gamma1, lp_data *u.Unit('cm-3')
)
n_lp = lp_inter(g1)
#ExpCutoffPowerLaw
epwl_data = epwl_test(gamma2).value
epwl_inter = InterpolatedDistribution(
    gamma2, epwl_data *u.Unit('cm-3')
)
n_epwl = epwl_inter(g2)

""" Diagrams """
#Style
plt.style.use('interp')

# Interpolation function vs Original
fig,ax=plt.subplots(2,2)
# Power law

ax[0][0].loglog(g1, n_pwl, label = 'IDF', c = 'orange')
ax[0][0].loglog(gamma1, pwl_data, '.', label='Power Law Data', c = 'black', markersize=8)
ax[0][0].set_xlabel(' γ ')
ax[0][0].set_ylabel('$ n $ [{0}]'.format(pwl_data.unit.to_string('latex_inline')) )
#plt.ylim(1e-12, 1e-7)

ax[0][0].legend(loc='lower left')


# Broken Power Law Parabola

ax[0][1].loglog(g1, n_bpwl, label = 'IDF', c = 'orange')
ax[0][1].loglog(gamma1, bpwl_data, '.', label='Broken Power Law Data' , c = 'black', markersize=8)
ax[0][1].set_xlabel('γ')
ax[0][1].set_ylabel('$ n $ [{0}]'.format(pwl_data.unit.to_string('latex_inline')) )
#plt.ylim(1e-12, 1e-7)
ax[0][1].legend(loc='lower left')

# Log Parabola

ax[1][0].loglog(g1, n_lp, label = 'IDF', c = 'orange')
ax[1][0].loglog(gamma1, lp_data, '.', label='Log Parabola Data' , c = 'black', markersize=8)
ax[1][0].set_xlabel('γ')
ax[1][0].set_ylabel('$ n $ [{0}]'.format(pwl_data.unit.to_string('latex_inline')))
#plt.ylim(1e-12, 1e-7)
ax[1][0].legend(loc='lower left')

# Exp cut off power law

ax[1][1].loglog(g2, n_epwl, label = 'IDF', c = 'orange')
ax[1][1].loglog(gamma2, epwl_data, '.', label='Exp Cut-off Power Law Data' , c = 'black', markersize=8)
ax[1][1].set_xlabel('γ')
ax[1][1].set_ylabel('$ n $ [{0}]'.format(pwl_data.unit.to_string('latex_inline')))
#plt.ylim(1e-12, 1e-7)
ax[1][1].legend(loc='lower left')
plt.tight_layout()
plt.show()


# SSA
fig,ax=plt.subplots(2,2)
# Power Law
SSA_inter = pwl_inter.SSA_integrand(g1).value
SSA_pwl = pwl_test.SSA_integrand(g1).value

ax[0][0].loglog(g1, abs(SSA_inter), label = 'IDF', c = 'orange' ,linewidth=3)
ax[0][0].loglog(g1, abs(SSA_pwl), '--', label='Power Law function' , c = 'black')
ax[0][0].set_xlabel('γ' )
ax[0][0].set_ylabel('$ n $ [{0}]'.format(pwl_data.unit.to_string('latex_inline')) )
ax[0][0].legend(loc='lower left')

#plt.ylim(1e-12, 1e-7)

# Broken Power Law
SSA_inter = bpwl_inter.SSA_integrand(g1).value
SSA_bpwl = bpwl_test.SSA_integrand(g1).value

ax[0][1].loglog(g1, abs(SSA_inter), label = 'IDF', c = 'orange',linewidth=3)
ax[0][1].loglog(g1, abs(SSA_bpwl), '--', label='Broken Power Law function' , c = 'black')
ax[0][1].set_xlabel(' γ ' )
ax[0][1].set_ylabel('$ n $ [{0}]'.format(pwl_data.unit.to_string('latex_inline')) )
ax[0][1].legend(loc='lower left')

#plt.ylim(1e-12, 1e-7)


# Log Parabola
SSA_inter = lp_inter.SSA_integrand(g1).value
SSA_lp = lp_test.SSA_integrand(g1).value

ax[1][0].loglog(g1, abs(SSA_inter), label = 'IDF', c = 'orange',linewidth=3)
ax[1][0].loglog(g1, abs(SSA_lp), '--', label='Log Parabola function' , c = 'black')
ax[1][0].set_xlabel('γ' )
ax[1][0].set_ylabel('$ n $ [{0}]'.format(pwl_data.unit.to_string('latex_inline')) )
ax[1][0].legend(loc='lower left')
#plt.ylim(1e-12, 1e-7)

# Exp cut off
SSA_inter = epwl_inter.SSA_integrand(g2).value
SSA_epwl = epwl_test.SSA_integrand(g2).value

ax[1][1].loglog(g2, abs(SSA_inter), label = 'IDF', c = 'orange',linewidth=3)
ax[1][1].loglog(g2, abs(SSA_epwl), '--' , label='Exp Cut-off Power Law function' , c = 'black')
ax[1][1].set_xlabel('γ' )
ax[1][1].set_ylabel('$ n $ [{0}]'.format(pwl_data.unit.to_string('latex_inline')) )
ax[1][1].legend(loc='lower left')
#plt.ylim(1e-12, 1e-7)


plt.tight_layout()
plt.show()