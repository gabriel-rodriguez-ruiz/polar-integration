#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:59:20 2025

@author: gabriel
"""

import numpy as np
import multiprocessing
from pathlib import Path
import scipy
from get_pockets import plot_interpolated_contours, integrate_pocket, get_pockets_contour, integrate_brute_force
from diagonalization import get_Energies_in_polars
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

c = 3e18 # nm/s  #3e9 # m/s
m_e =  5.1e8 / c**2 # meV s²/m²
m = 0.403 * m_e # meV s²/m²
hbar = 6.58e-13 # meV s
gamma = hbar**2 / (2*m) # meV (nm)²
E_F = 50.6 # meV
k_F = np.sqrt(E_F / gamma ) # 1/nm

Delta = 0.08   #  meV
mu = 50.6   # 623 Delta #50.6  #  meV
gamma = 9479 # meV (nm)²
Lambda = 8 * Delta # 8 * Delta  #0.644 meV 

h = 1e-6
phi_x_values = np.array([-h, 0, h])
cut_off = 1.1*k_F # 1.1 k_F

B_y = 0
B_x = 0

N = 200    #100
n_cores = 16
points = 1* n_cores


parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta, "phi_x_values": phi_x_values,
              "Lambda": Lambda, "N": N,
              "cut_off": cut_off
              }

def integrate_B_y(B_y):
    energy_phi = np.zeros_like(phi_x_values)

    for j, phi_x in enumerate(phi_x_values):
        integral, low_integral, high_integral = integrate_brute_force(N, mu, B_y, Delta, phi_x, gamma, Lambda, k_F, cut_off, B_x)
        energy_phi[j] = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
    fundamental_energy = energy_phi + np.pi/2 * cut_off**2 * (2*gamma*phi_x_values**2 - 2*mu + gamma*cut_off**2)
    zero_index = int( (len(phi_x_values)-1)/2 )
    superfluid_density_0 = (fundamental_energy[zero_index+1]-2*fundamental_energy[zero_index]+fundamental_energy[zero_index-1])/np.diff(phi_x_values)[0]**2
    
    return superfluid_density_0

def integrate_B_x(B_x):
    energy_phi = np.zeros_like(phi_x_values)

    for j, phi_x in enumerate(phi_x_values):
        integral, low_integral, high_integral = integrate_brute_force(N, mu, B_y, Delta, phi_x, gamma, Lambda, k_F, cut_off, B_x)
        energy_phi[j] = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
    fundamental_energy = energy_phi + np.pi/2 * cut_off**2 * (2*gamma*phi_x_values**2 - 2*mu + gamma*cut_off**2)
    zero_index = int( (len(phi_x_values)-1)/2 )
    superfluid_density_0 = (fundamental_energy[zero_index+1]-2*fundamental_energy[zero_index]+fundamental_energy[zero_index-1])/np.diff(phi_x_values)[0]**2


    return superfluid_density_0




if __name__ == "__main__":
    B_values = np.linspace(0.9*Delta, 1.1*Delta, points)
    integrate = integrate_B_x
    B_direction = "x"
    # integrate = integrate_B_y
    with multiprocessing.Pool(n_cores) as pool:
        superfluid_density_0 = pool.map(integrate, B_values)
    superfluid_density_0 = np.array(superfluid_density_0)
    data_folder = Path("Data/")
    name = f"superfluid_density_finite_differences_B_in_{B_direction}_({np.round(np.min(B_values/Delta),3)}-{np.round(np.max(B_values/Delta),3)})_phi_x_in_({np.round(np.min(phi_x_values/k_F), 3)}-{np.round(np.max(phi_x_values/k_F),3)})_Delta={Delta}_lambda={np.round(Lambda, 2)}_points={points}_h={h}_N={N}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open, superfluid_density_0=superfluid_density_0,
             B_values=B_values, **parameters)
    print("\007")
