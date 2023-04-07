import numpy as np
import astropy.units as u
from agnpy.spectra import PowerLaw, BrokenPowerLaw, LogParabola, ExpCutoffPowerLaw
import matplotlib.pyplot as plt
plt.style.use('interp')
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

# 10 points used for the interpolation: they are fed first to the initial distributions,
# get the particle densities and then fed to the interpolation functions
gamma1 = np.logspace(np.log10(gamma_min_test),np.log10(gamma_max_test),100)
gamma2 = np.logspace(np.log10(gamma_min_test),np.log10(1e5),100) #just for exp



#PowerLaw
pwl_data = pwl_test(gamma1)

# BrokenPowerLaw
bpwl_data = bpwl_test(gamma1)

#LogParabola
lp_data = lp_test(gamma1)

#ExpCutoffPowerLaw
epwl_data = epwl_test(gamma2)


fig,ax=plt.subplots(2,2)

# Power law

ax[0][0].loglog(gamma1, pwl_data,  label='Power Law', c = 'black', markersize=8)
ax[0][0].set_xlabel(' γ ')
ax[0][0].set_ylabel('$ n $ [{0}]'.format(pwl_data.unit.to_string('latex_inline')) )
#plt.ylim(1e-12, 1e-7)

ax[0][0].legend(loc='lower left')

# Broken Power Law Parabola

ax[0][1].loglog(gamma1, bpwl_data, label='Broken Power Law' , c = 'black', markersize=8)
ax[0][1].set_xlabel('γ')
ax[0][1].set_ylabel('$ n $ [{0}]'.format(pwl_data.unit.to_string('latex_inline')) )
#plt.ylim(1e-12, 1e-7)
ax[0][1].legend(loc='lower left')

# Log Parabola

ax[1][0].loglog(gamma1, lp_data, label='Log Parabola' , c = 'black', markersize=8)
ax[1][0].set_xlabel('γ')
ax[1][0].set_ylabel('$ n $ [{0}]'.format(pwl_data.unit.to_string('latex_inline')))
#plt.ylim(1e-12, 1e-7)
ax[1][0].legend(loc='lower left')

# Exp cut off power law

ax[1][1].loglog(gamma2, epwl_data, label='Exp Cut-off Power Law' , c = 'black', markersize=8)
ax[1][1].set_xlabel('γ')
ax[1][1].set_ylabel('$ n $ [{0}]'.format(pwl_data.unit.to_string('latex_inline')))
#plt.ylim(1e-12, 1e-7)
ax[1][1].legend(loc='lower left')
plt.tight_layout()
plt.show()
