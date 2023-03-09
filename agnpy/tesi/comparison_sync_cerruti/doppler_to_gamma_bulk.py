import numpy as np
import astropy.units as u
from astropy.constants import m_e, m_p
from agnpy.emission_regions import Blob
from agnpy.spectra import ExpCutoffPowerLaw

def gamma_bulk(beta):
    return 1/np.sqrt(1-beta**2)

""" Takes as an input the doppler factor produced by the motion
    of the jet and the viewing angle on degrees. It returns the
    β (the u/c) and the bulk Lorentz factor Γ = 1 / sqrt(1 - β ** 2)
    doppler = 1 / ( Γ * (1 - cos(θ) )
"""

ang_d = float(input('Angle on degrees: '))
ang_r = (np.pi/180)*ang_d
mu = np.cos(ang_r)
doppler = float(input('doppler: '))

a = (doppler**2)*(mu**2)+1
b = -(2*mu*doppler**2)
c = doppler**2 - 1

delta = b**2 - 4*a*c
if delta < 0:
    raise ValueError(
        f"Delta is negative."
    )

beta1 = ( -b + np.sqrt(delta) ) / ( 2 * a)
beta2 = ( -b - np.sqrt(delta) ) / ( 2 * a)

print ('beta1 is: ',beta1, 'and the gamma bulk is: ', gamma_bulk(beta1))
print ('beta2 is: ',beta2, 'and the gamma bulk is: ', gamma_bulk(beta2))

# Check if computation is correct

# Define source parameters
B = 62.8936 * u.G
redshift = 0.044
#distPKS = Distance(z=redshift) # Already inside blob definition
doppler_s = 30
Gamma_bulk = 15e4 #15
R = 9.06234e+14 * u.cm #radius of the blob

# Distribution
n_e = ExpCutoffPowerLaw(k=2.5e2 / u.Unit('cm3'), #2.5e2 #7.726e-4 / u.Unit('cm3'), 
            p = 2.5 ,
            gamma_c= 26441.5,
            gamma_min= 200,
            gamma_max=1e15,
            mass=m_e
        )

# Emission region
blob = Blob(R_b=R,
        z=redshift,
        delta_D=doppler_s,
        Gamma=Gamma_bulk,
        B=B,
        n_e=n_e
)

doppler_test = blob.set_delta_D(Gamma=15.0269, theta_s=0.1 * u.deg)
print(f"{blob.delta_D:.2f}")