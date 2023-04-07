from agnpy.spectra import PowerLaw as PL
import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h, hbar, k_B
from scipy.integrate import quad, dblquad, nquad, simps, trapz
import numpy as np
import matplotlib.pyplot as plt
import timeit

plt.style.use('integration')

def dist(x):
    return x**(-2)

def return_results(min, b, int):

    a = np.logspace(np.log10(min),15,40)
    k = -1
    param = 0
    correct = []

    for i in a:
        correct.append(quad(dist, min, min*1e2)[0])
        k += 1
        inte = quad(dist, min, i)[0]

        if inte > 0 and param == 0:
            int.append(inte)
            b.append(i)

    return a,b,int,correct


min = 1e3
b1,b2,b3,b4 = [],[],[],[]
int1,int2,int3,int4 = [],[],[],[]
a1,a2,a3,a4 = [],[],[],[]
c1,c2,c3,c4 = [],[],[],[]

a1,b1,int1,c1 = return_results(1, b1, int1)
a2,b2,int2,c2 = return_results(1e3, b2, int2)
a3,b3,int3,c3 = return_results(1e6, b3, int3)
a4,b4,int4,c4 = return_results(1e9, b4, int4)

fig,ax=plt.subplots(2,2)

ax[0][0].loglog(a1, c1, '--', color = 'green', label = 'Correct results')
ax[0][0].loglog(b1, int1, 'o' , label = 'Integration result')
ax[0][0].set_xlabel('Upper limit')
ax[0][0].set_ylabel('Value of the integral')
ax[0][0].legend(loc="lower left")
ax[0][0].set_xlim(1,1e15)

ax[0][1].loglog(a2, c2, '--', color = 'green', label = 'Correct result')
ax[0][1].loglog(b2, int2, 'o' , label = 'Integration result')
ax[0][1].set_xlabel('Upper limit')
ax[0][1].set_ylabel('Value of the integral')
ax[0][1].legend(loc="lower left")
ax[0][1].set_xlim(1e3,1.3e15)

ax[1][0].loglog(a3, c3, '--', color = 'green', label = 'Correct result')
ax[1][0].loglog(b3, int3, 'o' , label = 'Integration result')
ax[1][0].set_xlabel('Upper limit')
ax[1][0].set_ylabel('Value of the integral')
ax[1][0].legend(loc="lower left")
ax[1][0].set_xlim(1e6,1.3e15)

ax[1][1].loglog(a4, c4, '--', color = 'green', label = 'Correct result')
ax[1][1].loglog(b4, int4, 'o' , label = 'Integration result')
ax[1][1].set_xlabel('Upper limit')
ax[1][1].set_ylabel('Value of the integral')
ax[1][1].legend(loc="lower left")
ax[1][1].set_xlim(1e9,1e15)


ax[0][0].title.set_text('Lower limit = 1')
ax[0][1].title.set_text('Lower limit = $10^3$')
ax[1][0].title.set_text('Lower limit = $10^6$')
ax[1][1].title.set_text('Lower limit = $10^9$')

plt.tight_layout()
plt.show()
