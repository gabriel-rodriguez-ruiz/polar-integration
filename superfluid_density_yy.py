# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 22:57:45 2025

@author: Gabriel
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

N_phi = 101  # 101  # it should be odd to include zero
# phi_x_values = np.linspace(-0.002 * k_F, 0.002 * k_F, N_phi)   #np.linspace(-0.003 * k_F, 0.003 * k_F, N_phi)
cut_off = 1.1*k_F # 1.1 k_F

B_y = 0
B_x = 0
theta = 0.   # float

N = 300    #300
n_cores = 16
points = 3* n_cores
N_polifit = 2  # 4
h = 1e-4 * k_F

parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta,
              "N_phi": N_phi, "Lambda": Lambda, "N": N,
              "cut_off": cut_off
              }

data_folder = Path(r"./Data")

# file_to_open = data_folder / "superfluid_density_B_in_x_(0.8-2.0)_phi_x_in_(-0.003-0.003)_Delta=0.08_lambda=0.64_points=48_N_phi=101_N=300.npz"
file_to_open = data_folder / "superfluid_density_B_in_y_(0.8-2.0)_phi_x_in_(-0.003-0.003)_Delta=0.08_lambda=0.64_points=48_N_phi=101_N=300.npz"


Data = np.load(file_to_open)
phi_eq_B = Data["phi_eq"]
B_values = Data["B_values"]

def integrate_phi_eq_B(i):
    B = Data["B_values"][i]

    B_x = B * np.cos(theta)
    B_y = B * np.sin(theta)

    phi_x_values = np.array([-h, 0, h])
    energy_phi = np.zeros_like(phi_x_values)
    for j, phi_x in enumerate(phi_x_values):
        print(j)
        integral, low_integral, high_integral = integrate_brute_force(N, mu, B_y, Delta, phi_x, gamma, Lambda, k_F, cut_off, B_x, phi_eq_B[i])
        energy_phi[j] = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
    fundamental_energy = energy_phi + np.pi/2 * cut_off**2 * (2*gamma*(phi_eq_B[i]**2 + phi_x_values**2) - 2*mu + gamma*cut_off**2)
   
    superfluid_density_yy = (fundamental_energy[2]-2*fundamental_energy[1]+fundamental_energy[0])/h**2
    
    return superfluid_density_yy

if __name__ == "__main__":
    i_values = np.arange(48)   #48
    integrate = integrate_phi_eq_B
    B_direction = f"{theta:.2}"
    # integrate = integrate_B_y
    with multiprocessing.Pool(n_cores) as pool:
        superfluid_density_yy = pool.map(integrate, i_values)
    superfluid_density_yy = np.array(superfluid_density_yy)
    data_folder = Path("Data/")
    name = f"superfluid_density_xx_B_in_{B_direction}_({np.round(np.min(B_values/Delta),3)}-{np.round(np.max(B_values/Delta),3)})_phi_x_in_({np.round(np.min(phi_eq_B/k_F), 3)}-{np.round(np.max(phi_eq_B/k_F),3)})_Delta={Delta}_lambda={np.round(Lambda, 2)}_points={points}_N_phi={N_phi}_N={N}_h={h}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open, superfluid_density_yy=superfluid_density_yy, **parameters)
    print("\007")