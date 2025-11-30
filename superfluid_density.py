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
from get_pockets import plot_interpolated_contours, integrate_pocket, get_pockets_contour, integrate_brute_force, integrate_Romberg
from diagonalization import get_Energies_in_polars
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from skopt import gp_minimize
from skopt.space import Space

c = 3e18 # nm/s  #3e9 # m/s
m_e =  5.1e8 / c**2 # meV s²/m²
m = 0.403 * m_e # meV s²/m²
hbar = 6.58e-13 # meV s
gamma = hbar**2 / (2*m) # meV (nm)²
E_F = 50.6 # meV
k_F = np.sqrt(E_F / gamma ) # 1/nm

Delta = 0.08   #  meV
mu = 50.6   # 623 Delta #50.6  #  meV
# gamma = 9479 # meV (nm)²
Lambda = 8.76 #8*Delta # meV*nm    # 8 * Delta  #0.644 meV 

N_phi = 101  # 101  # it should be odd to include zero
phi_x_values = np.linspace(-0.002 * k_F, 0.002 * k_F, N_phi)   #np.linspace(-0.003 * k_F, 0.003 * k_F, N_phi)
cut_off = 1.1*k_F # 1.1 k_F

theta = -np.pi/4 #np.pi/2   # float

N = 514  #514   #300
n_cores = 16
points = 1* n_cores
N_polifit = 2  # 4
C = 0


parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta, "phi_x_values": phi_x_values,
              "N_phi": N_phi, "Lambda": Lambda, "N": N,
              "cut_off": cut_off, "C": C
              }

# Define the search space
search_space = [(-0.002 * k_F, 0.002 * k_F)]  # Search from -3 to 3

def function(phi, B_x, B_y):
    phi_x = phi * np.cos(theta - np.pi/2)
    phi_y = phi * np.sin(theta - np.pi/2)
    integral, low_integral, high_integral = integrate_Romberg(N, mu, B_y, Delta, phi_x, gamma, Lambda, k_F, cut_off, B_x, phi_y)
    energy_phi = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral) + np.pi/2 * cut_off**2 * (2*gamma*(phi_x**2 + phi_y**2) - 2*mu + gamma*cut_off**2)
    return energy_phi

def E_0(phi_x, phi_y, B_x, B_y):
    integral, low_integral, high_integral = integrate_Romberg(N, mu, B_y, Delta, phi_x, gamma, Lambda, k_F, cut_off, B_x, phi_y)
    energy_phi = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral) + np.pi/2 * cut_off**2 * (2*gamma*(phi_x**2 + phi_y**2) - 2*mu + gamma*cut_off**2)
    return energy_phi

def get_minima(search_space, B_x, B_y):
    def objective_function(x):
        """Objective function to minimize"""
        return function(x[0], B_x, B_y)
    result = gp_minimize(
        func=objective_function,
        dimensions=search_space,
        n_calls=15,                    # Only 25 expensive evaluations!
        n_initial_points=5,           # Start with 10 random points
        random_state=None,
        acq_func='LCB',  #Lower Confidence Bound (more exploratory) #"EI"  Expected Improvement
        noise=0.0,
        initial_point_generator="lhs",
        x0=[0.]
    )
    return result.x[0]

def integrate_B(B):
    B_x = B * np.cos(theta)
    B_y = B * np.sin(theta)
    energy_phi = np.zeros_like(phi_x_values)

    for j, phi_x in enumerate(phi_x_values):
        print(j)
        integral, low_integral, high_integral = integrate_brute_force(N, mu, B_y, Delta, phi_x, gamma, Lambda, k_F, cut_off, B_x, phi_y=-phi_x)
        energy_phi[j] = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
    fundamental_energy = energy_phi + np.pi/2 * cut_off**2 * (2*gamma*(phi_x_values**2 + (-phi_x_values)**2 ) - 2*mu + gamma*cut_off**2)
    # print(B)
    peaks_data, properties_data = find_peaks(-fundamental_energy)
    minima_index = peaks_data[np.argmax(-fundamental_energy[peaks_data])]
    a = minima_index - N_polifit
    b = minima_index + (N_polifit + 1)
    p = np.polyfit(phi_x_values[a:b], fundamental_energy[a:b], deg=2)
    superfluid_density = 2*p[0]
    
    
    zero_index = int( (len(phi_x_values)-1)/2 )
    a = zero_index - N_polifit
    b = zero_index + N_polifit + 1
    p_0 = np.polyfit(phi_x_values[a:b], fundamental_energy[a:b], deg=2)
    superfluid_density_0 = 2*p_0[0]
    
    superfluid_density_finite_differences = (fundamental_energy[minima_index+1]-2*fundamental_energy[minima_index]+fundamental_energy[minima_index-1])/np.diff(phi_x_values)[0]**2
    superfluid_density_finite_differences_0 = (fundamental_energy[zero_index+1]-2*fundamental_energy[zero_index]+fundamental_energy[zero_index-1])/np.diff(phi_x_values)[0]**2
        
    if len(peaks_data) >= 2:
        peaks_data_sorted = peaks_data[np.argsort(-fundamental_energy[peaks_data])]
        second_minima_index = peaks_data_sorted[1]
        superfluid_density_finite_differences_second_minima = (fundamental_energy[second_minima_index+1]-2*fundamental_energy[second_minima_index]+fundamental_energy[second_minima_index-1])/np.diff(phi_x_values)[0]**2
    else:
        second_minima_index = zero_index
        superfluid_density_finite_differences_second_minima = (fundamental_energy[second_minima_index+1]-2*fundamental_energy[second_minima_index]+fundamental_energy[second_minima_index-1])/np.diff(phi_x_values)[0]**2

    if len(peaks_data) >= 3:
        peaks_data_sorted = peaks_data[np.argsort(-fundamental_energy[peaks_data])]
        third_minima_index = peaks_data_sorted[2]
        superfluid_density_finite_differences_third_minima = (fundamental_energy[third_minima_index+1]-2*fundamental_energy[third_minima_index]+fundamental_energy[third_minima_index-1])/np.diff(phi_x_values)[0]**2
    else:
        third_minima_index = zero_index
        superfluid_density_finite_differences_third_minima = (fundamental_energy[third_minima_index+1]-2*fundamental_energy[third_minima_index]+fundamental_energy[third_minima_index-1])/np.diff(phi_x_values)[0]**2
    
    h = 1e-5 * k_F
    # phi_y_values = np.array([-h, 0, h])
    phi_y_values = np.array([-h, 0, h]) - phi_x_values[minima_index]
    energy_phi = np.zeros_like(phi_y_values)
    energy_phi_0 = np.zeros_like(phi_y_values)

    for j, phi_y in enumerate(phi_y_values):
        integral, low_integral, high_integral = integrate_brute_force(N, mu, B_y, Delta, phi_x_values[minima_index], gamma, Lambda, k_F, cut_off, B_x, phi_y)
        energy_phi[j] = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
        integral, low_integral, high_integral = integrate_brute_force(N, mu, B_y, Delta, 0, gamma, Lambda, k_F, cut_off, B_x, phi_y)
        energy_phi_0[j] = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
    fundamental_energy = energy_phi + np.pi/2 * cut_off**2 * (2*gamma*(phi_x_values[minima_index]**2 + phi_y_values**2) - 2*mu + gamma*cut_off**2)
    fundamental_energy_0 = energy_phi_0 + np.pi/2 * cut_off**2 * (2*gamma*(np.array([-h, 0, h])**2) - 2*mu + gamma*cut_off**2) + C/k_F * ( phi_x_values[minima_index]**2 + phi_y_values**2)

    superfluid_density_yy = (fundamental_energy[2]-2*fundamental_energy[1]+fundamental_energy[0])/h**2
    superfluid_density_yy_0 = (fundamental_energy_0[2]-2*fundamental_energy_0[1]+fundamental_energy_0[0])/h**2
    
    Phi_x_values = np.array([-h, 0, h]) + phi_x_values[minima_index]
    energy_phi = np.zeros_like(phi_y_values)
    energy_phi_0 = np.zeros_like(phi_y_values)

    for j, Phi_x in enumerate(Phi_x_values):
        integral, low_integral, high_integral = integrate_brute_force(N, mu, B_y, Delta, Phi_x, gamma, Lambda, k_F, cut_off, B_x, phi_y=-phi_x_values[minima_index])
        energy_phi[j] = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
        integral, low_integral, high_integral = integrate_brute_force(N, mu, B_y, Delta, 0, gamma, Lambda, k_F, cut_off, B_x, phi_y=0)
        energy_phi_0[j] = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
    fundamental_energy = energy_phi + np.pi/2 * cut_off**2 * (2*gamma*(Phi_x_values[minima_index]**2 + (-phi_x_values[minima_index])**2) - 2*mu + gamma*cut_off**2)
    fundamental_energy_0 = energy_phi_0 + np.pi/2 * cut_off**2 * (2*gamma*(np.array([-h, 0, h])**2) - 2*mu + gamma*cut_off**2) + C/k_F * ( phi_x_values[minima_index]**2 + phi_y_values**2)
    superfluid_density_xx = (fundamental_energy[2]-2*fundamental_energy[1]+fundamental_energy[0])/h**2
    superfluid_density_xx_0 = (fundamental_energy_0[2]-2*fundamental_energy_0[1]+fundamental_energy_0[0])/h**2

    """
    cs = CubicSpline(phi_x_values, fundamental_energy)
    x = np.linspace(min(phi_x_values), max(phi_x_values), 500)
    peaks, properties = find_peaks(-cs(x))
    minima = peaks[np.argmax(-cs(x[peaks]))]
    second_derivative = float(cs.derivative(nu=2)(x[minima]))
    """
    return superfluid_density, phi_x_values[minima_index], superfluid_density_0, superfluid_density_finite_differences, superfluid_density_finite_differences_0, superfluid_density_finite_differences_second_minima, phi_x_values[second_minima_index], superfluid_density_finite_differences_third_minima, phi_x_values[third_minima_index], superfluid_density_yy, superfluid_density_yy_0, superfluid_density_xx, superfluid_density_xx_0

def integrate_B_Romberg(B):
    B_x = B * np.cos(theta)
    B_y = B * np.sin(theta)
    q_B = get_minima(search_space, B_x, B_y)
    h = 1e-4 * k_F
    phi_x_values = np.array([-h, 0, h]) + q_B * np.cos(theta - np.pi/2)
    phi_y_values = np.array([-h, 0, h]) + q_B * np.sin(theta - np.pi/2)
    superfluid_density_xx = (E_0(phi_x_values[2], phi_y_values[1], B_x, B_y) - 2*E_0(phi_x_values[1], phi_y_values[1], B_x, B_y) + E_0(phi_x_values[0], phi_y_values[1], B_x, B_y))/h**2
    superfluid_density_xx_0 = (E_0(h, 0, B_x, B_y) - 2*E_0(0, 0, B_x, B_y) + E_0(-h, 0, B_x, B_y))/h**2
    superfluid_density_yy = (E_0(phi_x_values[1], phi_y_values[2], B_x, B_y) - 2*E_0(phi_x_values[1], phi_y_values[1], B_x, B_y) + E_0(phi_x_values[1], phi_y_values[0], B_x, B_y))/h**2
    superfluid_density_yy_0 = (E_0(0, h, B_x, B_y) - 2*E_0(0, 0, B_x, B_y) + E_0(0, -h, B_x, B_y))/h**2
    superfluid_density_xy = (E_0(phi_x_values[2], phi_y_values[2], B_x, B_y) - E_0(phi_x_values[2], phi_y_values[0], B_x, B_y) - E_0(phi_x_values[0], phi_y_values[2], B_x, B_y) + E_0(phi_x_values[0], phi_y_values[0], B_x, B_y))/(4*h**2)
    superfluid_density_xy_0 = (E_0(h, h, B_x, B_y) - E_0(h, -h, B_x, B_y) - E_0(-h, h, B_x, B_y) + E_0(-h, -h, B_x, B_y))/(4*h**2)
    return q_B, superfluid_density_xx, superfluid_density_xx_0, superfluid_density_yy, superfluid_density_yy_0, superfluid_density_xy, superfluid_density_xy_0

# if __name__ == "__main__":
#     B_values = np.linspace(0.8*Delta, 2*Delta, points)
#     integrate = integrate_B
#     B_direction = f"{theta:.2}"
#     # integrate = integrate_B_y
#     with multiprocessing.Pool(n_cores) as pool:
#         superfluid_density, phi_eq, superfluid_density_0, superfluid_density_finite_differences, superfluid_density_finite_differences_0, superfluid_density_finite_differences_second_minima, phi_eq_second_minima, superfluid_density_finite_differences_third_minima, phi_eq_third_minima, superfluid_density_yy, superfluid_density_yy_0, superfluid_density_xx, superfluid_density_xx_0 = zip(*pool.map(integrate, B_values))
#     superfluid_density = np.array(superfluid_density)
#     superfluid_density_0 = np.array(superfluid_density_0)
#     superfluid_density_finite_differences = np.array(superfluid_density_finite_differences)
#     superfluid_density_finite_differences_0 = np.array(superfluid_density_finite_differences_0)
#     superfluid_density_finite_differences_second_minima = np.array(superfluid_density_finite_differences_second_minima)
#     superfluid_density_finite_differences_third_minima = np.array(superfluid_density_finite_differences_third_minima)
#     superfluid_density_yy = np.array(superfluid_density_yy)
#     superfluid_density_yy_0 = np.array(superfluid_density_yy_0)
#     superfluid_density_xx = superfluid_density_xx
#     superfluid_density_xx_0 = superfluid_density_xx_0 
#     phi_eq = np.array(phi_eq)
#     phi_eq_second_minima = np.array(phi_eq_second_minima)
#     phi_eq_third_minima = np.array(phi_eq_third_minima)
#     data_folder = Path("Data/")
#     name = f"superfluid_density_B_in_{B_direction}_({np.round(np.min(B_values/Delta),3)}-{np.round(np.max(B_values/Delta),3)})_phi_x_in_({np.round(np.min(phi_x_values/k_F), 3)}-{np.round(np.max(phi_x_values/k_F),3)})_Delta={Delta}_lambda={np.round(Lambda, 2)}_points={points}_N_phi={N_phi}_N={N}_C={C}.npz"
#     file_to_open = data_folder / name
#     np.savez(file_to_open, superfluid_density=superfluid_density, superfluid_density_0=superfluid_density_0,
#              superfluid_density_finite_differences= superfluid_density_finite_differences,
#              superfluid_density_finite_differences_0=superfluid_density_finite_differences_0,
#              superfluid_density_finite_differences_second_minima=superfluid_density_finite_differences_second_minima,
#              superfluid_density_finite_differences_third_minima = superfluid_density_finite_differences_third_minima,
#              superfluid_density_yy=superfluid_density_yy,
#              superfluid_density_yy_0=superfluid_density_yy_0,
#              phi_eq_third_minima=phi_eq_third_minima,
#              phi_eq_second_minima = phi_eq_second_minima,
#              superfluid_density_xx = superfluid_density_xx,
#              superfluid_density_xx_0 = superfluid_density_xx_0,
#              B_values=B_values, phi_eq=phi_eq, **parameters)
#     print("\007")

if __name__ == "__main__":
    B_values = np.linspace(0.8*Delta, 2*Delta, points)
    B_direction = f"{theta:.2}"
    integrate = integrate_B_Romberg
    with multiprocessing.Pool(n_cores) as pool:
        q_B, superfluid_density_xx, superfluid_density_xx_0, superfluid_density_yy, superfluid_density_yy_0, superfluid_density_xy, superfluid_density_xy_0 = zip(*pool.map(integrate, B_values))
        q_B = np.array(q_B)
        superfluid_density_xx = np.array(superfluid_density_xx)
        superfluid_density_xx_0 = np.array(superfluid_density_xx_0)
        superfluid_density_yy = np.array(superfluid_density_yy)
        superfluid_density_yy_0 = np.array(superfluid_density_yy_0)
        superfluid_density_xy = np.array(superfluid_density_xy)
        superfluid_density_xy_0 = np.array(superfluid_density_xy_0)
        data_folder = Path("Data/")
        name = f"superfluid_density_B_in_{B_direction}_({np.round(np.min(B_values/Delta),3)}-{np.round(np.max(B_values/Delta),3)})_phi_x_in_({np.round(np.min(phi_x_values/k_F), 3)}-{np.round(np.max(phi_x_values/k_F),3)})_Delta={Delta}_lambda={np.round(Lambda, 2)}_points={points}_N_phi={N_phi}_N={N}.npz"
        file_to_open = data_folder / name
        np.savez(file_to_open,
                 superfluid_density_xx = superfluid_density_xx,
                 superfluid_density_xx_0 = superfluid_density_xx_0,
                 superfluid_density_yy = superfluid_density_yy,
                 superfluid_density_yy_0 = superfluid_density_yy_0,
                 superfluid_density_xy = superfluid_density_xy,
                 superfluid_density_xy_0 = superfluid_density_xy_0,
                 B_values = B_values, q_B = q_B, **parameters)
        print("\007")
