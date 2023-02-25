from agnpy.spectra import PowerLaw as PL
import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h, hbar, k_B
from scipy.integrate import quad, dblquad, nquad, simps, trapz
import numpy as np
import matplotlib.pyplot as plt

def dist(x):
    return x**(-2)

min = 1
int = []
a = np.logspace(np.log10(min)+1,30,100)
k = -1

for i in a:
    k +=1
    int.append(quad(dist, min, i  )[0])

plt.loglog(a, int)
plt.show()
