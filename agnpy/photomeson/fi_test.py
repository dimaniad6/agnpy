from agnpy.spectra import PowerLaw as PL
import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h, hbar, k_B
from scipy.integrate import quad, dblquad, nquad, simps, trapz
import numpy as np
import matplotlib.pyplot as plt

p_dist = PL(1 * u.Unit('cm-3'), 2., 1, 2e30)

def dist(x):
    return p_dist(x).value
    return x**(-2)

int = []
a = np.logspace(4,10,100)
k = -1
for i in a:
    k +=1
    x_range = [1,i]
    int.append(quad(dist, 1e3, i  )[0])


plt.loglog(a, int)
plt.show()
