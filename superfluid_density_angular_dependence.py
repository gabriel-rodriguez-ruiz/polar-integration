#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 18:51:28 2025

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


cut_off = 1.1*k_F # 1.1 k_F

B = 1.5*Delta
N = 514  #514  #514   #300
n_cores = 16
points = 1* n_cores
N_polifit = 2  # 4
C = 0

# Define the search space
search_space = [(-0.003 * k_F, 0.003 * k_F)]  # Search from -3 to 3

parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta, "search_space": search_space,
               "Lambda": Lambda, "N": N,
              "cut_off": cut_off, "C": C, "B": B
              }


def function(phi, B_x, B_y):
    theta = np.arctan(B_y / B_x)
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
        n_calls=15,                    # Only 15 expensive evaluations!
        n_initial_points=5,           # Start with 5 random points
        random_state=None,
        acq_func='LCB',  #Lower Confidence Bound (more exploratory) #"EI"  Expected Improvement
        noise=0.0,
        initial_point_generator="lhs",
        x0=[0.]
    )
    return result.x[0]

def integrate_theta_Romberg(theta):
    B_x = B * np.cos(theta)
    B_y = B * np.sin(theta)
    q_theta = get_minima(search_space, B_x, B_y)
    h = 1e-4 * k_F
    phi_x_values = np.array([-h, 0, h]) + q_theta * np.cos(theta - np.pi/2)
    phi_y_values = np.array([-h, 0, h]) + q_theta * np.sin(theta - np.pi/2)
    superfluid_density_xx = (E_0(phi_x_values[2], phi_y_values[1], B_x, B_y) - 2*E_0(phi_x_values[1], phi_y_values[1], B_x, B_y) + E_0(phi_x_values[0], phi_y_values[1], B_x, B_y))/h**2
    superfluid_density_xx_0 = (E_0(h, 0, B_x, B_y) - 2*E_0(0, 0, B_x, B_y) + E_0(-h, 0, B_x, B_y))/h**2
    superfluid_density_yy = (E_0(phi_x_values[1], phi_y_values[2], B_x, B_y) - 2*E_0(phi_x_values[1], phi_y_values[1], B_x, B_y) + E_0(phi_x_values[1], phi_y_values[0], B_x, B_y))/h**2
    superfluid_density_yy_0 = (E_0(0, h, B_x, B_y) - 2*E_0(0, 0, B_x, B_y) + E_0(0, -h, B_x, B_y))/h**2
    superfluid_density_xy = (E_0(phi_x_values[2], phi_y_values[2], B_x, B_y) - E_0(phi_x_values[2], phi_y_values[0], B_x, B_y) - E_0(phi_x_values[0], phi_y_values[2], B_x, B_y) + E_0(phi_x_values[0], phi_y_values[0], B_x, B_y))/(4*h**2)
    superfluid_density_xy_0 = (E_0(h, h, B_x, B_y) - E_0(h, -h, B_x, B_y) - E_0(-h, h, B_x, B_y) + E_0(-h, -h, B_x, B_y))/(4*h**2)
    return q_theta, superfluid_density_xx, superfluid_density_xx_0, superfluid_density_yy, superfluid_density_yy_0, superfluid_density_xy, superfluid_density_xy_0


if __name__ == "__main__":
    theta_values = np.linspace(0, np.pi/2, points)
    integrate = integrate_theta_Romberg
    with multiprocessing.Pool(n_cores) as pool:
        q_theta, superfluid_density_xx, superfluid_density_xx_0, superfluid_density_yy, superfluid_density_yy_0, superfluid_density_xy, superfluid_density_xy_0 = zip(*pool.map(integrate, theta_values))
        q_theta = np.array(q_theta)
        superfluid_density_xx = np.array(superfluid_density_xx)
        superfluid_density_xx_0 = np.array(superfluid_density_xx_0)
        superfluid_density_yy = np.array(superfluid_density_yy)
        superfluid_density_yy_0 = np.array(superfluid_density_yy_0)
        superfluid_density_xy = np.array(superfluid_density_xy)
        superfluid_density_xy_0 = np.array(superfluid_density_xy_0)
        data_folder = Path("Data/")
        name = f"superfluid_density_B=_{B}_theta_in_({np.round(np.min(theta_values),3)}-{np.round(np.max(theta_values),3)})_Delta={Delta}_lambda={np.round(Lambda, 2)}_points={points}_N={N}.npz"
        file_to_open = data_folder / name
        np.savez(file_to_open,
                 superfluid_density_xx = superfluid_density_xx,
                 superfluid_density_xx_0 = superfluid_density_xx_0,
                 superfluid_density_yy = superfluid_density_yy,
                 superfluid_density_yy_0 = superfluid_density_yy_0,
                 superfluid_density_xy = superfluid_density_xy,
                 superfluid_density_xy_0 = superfluid_density_xy_0,
                 theta_values = theta_values, q_theta = q_theta, **parameters)
        print("\007")
