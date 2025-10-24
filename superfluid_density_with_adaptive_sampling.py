#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 17:05:41 2025

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
import scipy.optimize as opt
from scipy.optimize import minimize_scalar

def adaptive_global_minimization(func, bounds, max_evals=200, tol=1e-6):
    """
    Adaptive sampling method for very expensive functions.
    """
    def evaluate_and_store(x, eval_dict):
        if x not in eval_dict:
            eval_dict[x] = func(x)
        return eval_dict[x]
    
    evaluations = {}
    a, b = bounds
    
    # Initial sampling
    x_vals = np.linspace(a, b, 10)
    for x in x_vals:
        evaluate_and_store(x, evaluations)
    
    # Adaptive refinement
    for iteration in range(max_evals - 10):
        x_sorted = sorted(evaluations.keys())
        intervals = []
        
        # Find promising intervals
        for i in range(len(x_sorted) - 1):
            x1, x2 = x_sorted[i], x_sorted[i + 1]
            f1, f2 = evaluations[x1], evaluations[x2]
            # Score interval based on function values and width
            score = (f1 + f2) / 2 * (x2 - x1)
            intervals.append((score, x1, x2))
        
        # Sample in most promising interval
        intervals.sort(key=lambda x: x[0])
        _, x1, x2 = intervals[0]  # Most promising interval
        x_new = (x1 + x2) / 2
        evaluate_and_store(x_new, evaluations)
        
        # Check convergence
        if len(evaluations) >= 4:
            best_points = sorted(evaluations.items(), key=lambda x: x[1])[:3]
            positions = [p[0] for p in best_points]
            if max(positions) - min(positions) < tol:
                break
    
    # Find best point
    best_x, best_f = min(evaluations.items(), key=lambda x: x[1])
    return best_x, best_f

# Usage for expensive functions
# x_min_adapt, f_min_adapt = adaptive_global_minimization(hard_numerical_function, bounds)
# f_second_deriv_adapt = numerical_second_derivative(hard_numerical_function, x_min_adapt)

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
phi_x_values = np.linspace(-0.0003 * k_F, 0.0003 * k_F, N_phi)
cut_off = 1.1*k_F # 1.1 k_F

B_y = 0
B_x = 0

N = 100    #100
n_cores = 16
points = 1* n_cores
bounds = [-0.002 * k_F, 0.002 * k_F]
max_evals = 20
tol=1e-6

parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta, "phi_x_values": phi_x_values,
              "N_phi": N_phi, "Lambda": Lambda, "N": N,
              "cut_off": cut_off
              }

def get_fundamental_energy_B_x(phi_x):
    integral, low_integral, high_integral = integrate_brute_force(N, mu, B_y, Delta, phi_x, gamma, Lambda, k_F, cut_off, B_x)
    energy_phi = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
    fundamental_energy = energy_phi + np.pi/2 * cut_off**2 * (2*gamma*phi_x**2 - 2*mu + gamma*cut_off**2)
    return fundamental_energy


def find_global_minimum_scipy(func, bounds, n_samples=10, method='bounded'):
    """
    Find global minimum using scipy with multiple strategies.
    """
    # Strategy 1: Direct bounded optimization
    if method == 'bounded':
        result = minimize_scalar(func, bounds=bounds, method='bounded', options={"maxiter": n_samples})
        return result.x, result.fun
    
    # Strategy 2: Brute force with refinement
    elif method == 'brute':
        # Initial coarse search
        x_vals = np.linspace(bounds[0], bounds[1], n_samples)
        f_vals = np.array([func(x) for x in x_vals])
        
        # Find best candidate
        best_idx = np.argmin(f_vals)
        x0 = x_vals[best_idx]
        
        # Refine with local optimization
        result = minimize_scalar(func, bracket=(min(bounds[0], x0-0.1*x0), 
                                              x0, 
                                              max(bounds[1], x0+0.1*x0)), options={"maxiter": n_samples})
        return result.x, result.fun

def numerical_derivative(func, x, h=1e-8):
    """Calculate numerical derivative using central differences"""
    return (func(x + h) - func(x - h)) / (2 * h)

def numerical_second_derivative(func, x, h=1e-6):
    """Calculate numerical second derivative"""
    return (func(x + h) - 2 * func(x) + func(x - h)) / (h**2)

# Example usage
bounds = (-0.002*k_F, 0.002*k_F)
hard_numerical_function = get_fundamental_energy_B_x
n_samples = 2
method='brute'

# Find global minimum
x_min, f_min = find_global_minimum_scipy(hard_numerical_function, bounds, n_samples, method)

# Calculate second derivative at minimum
f_second_deriv = numerical_second_derivative(hard_numerical_function, x_min)

print(f"Global minimum found at x = {x_min:.6f}")
print(f"Function value at minimum: {f_min:.6f}")
print(f"Second derivative at minimum: {f_second_deriv:.6f}")

