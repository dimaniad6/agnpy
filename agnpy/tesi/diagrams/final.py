from agnpy.spectra import PowerLaw as PL
import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h, hbar, k_B
from scipy.integrate import quad, dblquad, nquad, simps, trapz
import numpy as np
import matplotlib.pyplot as plt
import timeit
plt.style.use('proton_synchrotron')

def dist(x):
    return x**(-2)

def return_results(min, b, int):

    a = np.logspace(np.log10(min),15,40)
    k = -1
    param = 0
    correct = []

    for i in a:
        correct.append(quad(dist, min, min*1e3)[0])
        k += 1
        inte = quad(dist, min, i)[0]

        if inte > 0 and param == 0:
            int.append(inte)
            b.append(i)

    return a,b,int,correct

def trick(max):

    inv = 1e3
    a = 1 #min
    inte = []

    while a * inv < max:

        b = inv * a
        minimum = a
        maximum = b
        inte.append(quad(dist, minimum, maximum)[0])
        b = inv * a
        a = b

    minimum = a
    maximum = max
    inte.append(quad(dist, minimum, maximum)[0])

    return sum(inte)


min = 1e3
b1,b2,b3,b4 = [],[],[],[]
int1,int2,int3,int4 = [],[],[],[]
a1,a2,a3,a4 = [],[],[],[]
c1,c2,c3,c4 = [],[],[],[]

a1,b1,int1,c1 = return_results(1, b1, int1)
a2,b2,int2,c2 = return_results(1e3, b2, int2)
a3,b3,int3,c3 = return_results(1e6, b3, int3)
a4,b4,int4,c4 = return_results(1e9, b4, int4)

b = []
int = []
a = np.logspace(1,15,40)
kappa = []
c = []
start = timeit.default_timer()
for i in a:
    kappa.append(trick(i))
    c.append(1)
stop = timeit.default_timer()
print("Elapsed time for computation = {} secs".format(stop - start))

fig, ax = plt.subplots()
plt.loglog(a1, c, '--', color = 'green', label = 'Correct result')
plt.loglog(a1,kappa, 'o' , label = 'Integration result')
plt.legend(loc = 'lower right')
plt.xlabel('Upper limit')
plt.ylabel('Value of the integral')
# plt.ylim(1e-10, 3)
plt.show()


plt.loglog(a1, c1, '--', color = 'green', label = 'Correct result')
plt.loglog(b1 ,int1, 'o' , label = 'Integration result')
plt.legend(loc = 'lower left')
plt.xlabel('Upper limit')
plt.ylabel('Value of the integral')
plt.ylim(bottom = 1e-12, top = 3)
plt.show()

plt.show()
