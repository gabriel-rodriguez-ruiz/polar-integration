#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 14:24:31 2025

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from get_pockets import plot_interpolated_contours, integrate_pocket
from diagonalization import get_Energies_in_polars

plt.rcParams.update({
  "text.usetex": True,
})

c = 3e18 # nm/s  #3e9 # m/s
m_e =  5.1e8 / c**2 # meV s²/m²
m = 0.403 * m_e # meV s²/m²
hbar = 6.58e-13 # meV s
gamma = hbar**2 / (2*m) # meV (nm)²
E_F = 50.6 # meV
k_F = np.sqrt(E_F / gamma ) # 1/nm

Delta = 0.08   #  meV
mu = 632 * Delta   #50.6  #  meV
B = 2*Delta  # 2 * Delta
phi_x = 4.870678455592439e-06   #0.002 * k_F  # 0.002 * k_F
gamma = 9479 # meV (nm)²
Lambda = 8 * Delta # 8 * Delta  #0.644 meV 

N = 500

#pockets_dictionary = get_pockets_contour(N, mu, B, Delta, phi_x, gamma, Lambda, k_F)
#plot_interpolated_contours(N, mu, B, Delta, phi_x, gamma, Lambda, k_F)

#integrals = integrate_pocket(N, mu, B, Delta, phi_x, gamma, Lambda, k_F)

#%%

phi_x_values = np.linspace(0, 0.0002 * k_F, 10)
energy_phi = np.zeros_like(phi_x_values)

for i, phi_x in enumerate(phi_x_values):
    print(phi_x)
    integrals = integrate_pocket(N, mu, B, Delta, phi_x, gamma, Lambda, k_F)
    energy_phi[i] = np.sum(integrals[:3])
    
fig, ax = plt.subplots()
ax.plot(phi_x_values, energy_phi, "o")