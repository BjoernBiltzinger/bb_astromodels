from numba import njit
import numpy as np

@njit
def powerlaw(E, C, gamma):
    return C*E**-gamma

def calc_ion_spec_numba(gamma, base_energy, deltaE):
    F = powerlaw(base_energy, 1, gamma)*base_energy
    return F*deltaE/np.sum(F*deltaE)/base_energy
