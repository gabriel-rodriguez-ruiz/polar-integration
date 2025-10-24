#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 13:07:49 2025

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
Lambda = 0 * Delta # 8 * Delta  #0.644 meV 
B = 1 * Delta

N_phi = 31  # it should be odd to include zero
phi_x_values = np.linspace(-0.0002 * k_F, 0.0002 * k_F, N_phi)
cut_off = 1.1*k_F # 1.1 k_F

N = 100
n_cores = 16
points = 1 * n_cores
N_polifit = 6


parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta, "phi_x_values": phi_x_values,
              "N_phi": N_phi, "Lambda": Lambda, "N": N,
              "cut_off": cut_off, "B": B
              }

def integrate(phi_x):
    integral, low_integral, high_integral = integrate_brute_force(N, mu, B, Delta, phi_x, gamma, Lambda, k_F, cut_off)
    energy_phi = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
    fundamental_energy = energy_phi + np.pi/2 * cut_off**2 * (2*gamma*phi_x_values**2 - 2*mu + gamma*cut_off**2)
    return fundamental_energy

if __name__ == "__main__":
    phi_x_values = np.linspace(-0.0002, 0.0002, points) * k_F
    with multiprocessing.Pool(n_cores) as pool:
        results_pooled = pool.map(integrate)
    fundamental_energy = np.array(results_pooled)
    data_folder = Path("Data/")
    name = f"fundamental_energy_B={B}_phi_x_in_({np.round(np.min(phi_x_values/k_F), 3)}-{np.round(np.max(phi_x_values/k_F),3)})_Delta={Delta}_lambda={np.round(Lambda, 2)}_points={points}_N_phi={N_phi}_N={N}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open, fundamental_energy=fundamental_energy,
             phi_x_values=phi_x_values, **parameters)
    print("\007")
