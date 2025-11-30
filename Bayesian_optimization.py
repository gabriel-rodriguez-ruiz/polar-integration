#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 09:42:54 2025

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Space
from skopt.plots import plot_convergence, plot_objective
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import scipy
from get_pockets import plot_interpolated_contours, integrate_pocket, get_pockets_contour, integrate_brute_force, integrate_Romberg
from diagonalization import get_Energies_in_polars
from pathlib import Path

data_folder = Path(r"./Data")
file_to_open = data_folder / "superfluid_density_B_in_y_(0.8-2.0)_phi_x_in_(-0.003-0.003)_Delta=0.08_lambda=0.64_points=48_N_phi=101_N=300.npz"
Data = np.load(file_to_open)
B_values = Data["B_values"]
phi_x_values = Data["phi_eq"]

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

Delta = 0.08 #0.08   #  meV
mu = 50.6  #632 * Delta   #50.6  #  meV
B = 2*Delta  #2*Delta  #B_values[15]   #1.1 * Delta
theta = np.pi/2
B_y = B * np.sin(theta) #  2 * Delta  # 2 * Delta
B_x = B * np.cos(theta)
phi_x = 0.   #0.002 * k_F  # 0.002 * k_F
gamma = 9479 # meV (nm)²
Lambda = 8.76  #8*Delta  #8*Delta #meV*nm # 8 * Delta  #0.644 meV 

phi_B = phi_x_values[15]

#phi_x_values = np.linspace(0, 0.0002 * k_F, 10)
N = 66  #100
N_phi = 31 #31  # it should be odd to include zero

cut_off = 1.1*k_F
phi_x_values = np.linspace(-0.003 * k_F, 0.003 * k_F, N_phi)

# Define the skewed Mexican hat function in 1D
def skewed_mexican_hat(x):
    def function(phi_x):
        phi_y = 0
        integral, low_integral, high_integral = integrate_Romberg(N, mu, B_y, Delta, phi_x, gamma, Lambda, k_F, cut_off, B_x, phi_y)
        energy_phi = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral) + np.pi/2 * cut_off**2 * (2*gamma*(phi_x**2 + phi_y**2) - 2*mu + gamma*cut_off**2)
        return energy_phi
    return function(x)

# Define the search space
search_space = [(-0.003 * k_F, 0.003 * k_F)]  # Search from -3 to 3

#%%


energy_phi = np.zeros_like(phi_x_values)

for i, phi_x in enumerate(phi_x_values):
    # phi_x = phi_x + phi_B
    # phi_y=-phi_x
    phi_y = 0
    print(phi_x)
    #integrals = integrate_pocket(N, mu, B, Delta, phi_x, gamma, Lambda, k_F)
    integral, low_integral, high_integral = integrate_Romberg(N, mu, B_y, Delta, phi_x, gamma, Lambda, k_F, cut_off, B_x, phi_y)
    energy_phi[i] = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral) + np.pi/2 * cut_off**2 * (2*gamma*(phi_x**2 + phi_y**2) - 2*mu + gamma*cut_off**2)
    


# Vectorized version for plotting
x_plot = phi_x_values
y_plot = energy_phi

# Plot the function to see what we're dealing with
plt.figure(figsize=(12, 5))
plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Skewed Mexican Hat')
plt.xlabel('x')
plt.ylabel('E_0(x)')
# plt.title('1D Skewed Mexican Hat Function')
plt.grid(True, alpha=0.3)
# plt.legend()
plt.show()

#%%3

# Bayesian Optimization setup
def objective_function(x):
    """Objective function to minimize"""
    return skewed_mexican_hat(x[0])

print("Running Bayesian Optimization...")

# Run Bayesian Optimization with only 25 function evaluations!
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

print("\n=== RESULTS ===")
print(f"Global minimum found at: x = {result.x[0]:.6f}")
print(f"Function value at minimum: f(x) = {result.fun:.6f}")

#%%

# Find the true global minimum for comparison (we only know this because we can evaluate densely)
true_min_idx = np.argmin(y_plot)
true_min_x = x_plot[true_min_idx]
true_min_val = y_plot[true_min_idx]

print(f"True global minimum: x = {true_min_x:.6f}, f(x) = {true_min_val:.6f}")
print(f"Error in position: {abs(result.x[0] - true_min_x):.6f}")

# Plot the results
plt.figure(figsize=(15, 5))

# Plot 1: Function with optimization points
plt.subplot(1, 2, 1)
plt.plot(x_plot/k_F, y_plot, 'b-', linewidth=2, label='Skewed Mexican Hat', alpha=0.7)

# Plot all evaluation points
for i, (x_val, y_val) in enumerate(zip([xi[0] for xi in result.x_iters], result.func_vals)):
    color = 'red' if i < 5 else 'green'  # Initial points in red, BO points in green
    marker = 'o' if i < 5 else 's'
    alpha = 0.7 if i < 5 else 1.0
    plt.scatter(x_val/k_F, y_val, c=color, marker=marker, alpha=alpha, s=50)

# Mark the best point found
plt.scatter(result.x[0]/k_F, result.fun, c='gold', marker='*', s=200, 
           label=f'Best found: x={result.x[0]/k_F:.5f}', edgecolors='black')

plt.xlabel('phi/k_F')
plt.ylabel('f(x)')
plt.title('Bayesian Optimization Progress\n(Red: Initial, Green: BO, Star: Best)')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Convergence plot
plt.subplot(1, 2, 2)
plot_convergence(result)
plt.title('Convergence Plot')

plt.tight_layout()
plt.show()

# Print the evaluation history
print("\n=== EVALUATION HISTORY ===")
print("Iteration |      x      |    f(x)    |   Best So Far")
print("-" * 55)
best_so_far = float('inf')
for i, (x_val, f_val) in enumerate(zip(result.x_iters/k_F, result.func_vals)):
    if f_val < best_so_far:
        best_so_far = f_val
    print(f"{i+1:>8} | {x_val[0]:>10.5f} | {f_val:>9.9f} | {best_so_far:>12.9f}")
    
