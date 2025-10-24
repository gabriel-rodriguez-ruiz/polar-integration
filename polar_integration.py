#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 14:24:31 2025

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from get_pockets import plot_interpolated_contours, integrate_pocket, get_pockets_contour, integrate_brute_force
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

Delta = 0.08 #0.08   #  meV
mu = 50.6  #632 * Delta   #50.6  #  meV
B = 1.3 * Delta
theta = np.pi/4
B_y = B * np.sin(theta) #  2 * Delta  # 2 * Delta
B_x = B * np.cos(theta)
phi_x = 0   #0.002 * k_F  # 0.002 * k_F
gamma = 9479 # meV (nm)²
Lambda = 8 * Delta # 8 * Delta  #0.644 meV 

#phi_x_values = np.linspace(0, 0.0002 * k_F, 10)
N = 100
N_phi = 31  # it should be odd to include zero

#cut_off = 2*k_F 
#pockets_dictionary, Energies = get_pockets_contour(N, mu, B, Delta, phi_x, gamma, Lambda, k_F)
#plot_interpolated_contours(N, mu, B, Delta, phi_x, gamma, Lambda, k_F)

#integrals = integrate_pocket(N, mu, B, Delta, phi_x, gamma, Lambda, k_F)
#integral, low_integral, high_integral = integrate_brute_force(N, mu, B, Delta, phi_x, gamma, Lambda, k_F, cut_off)

#energy_phi = np.sum(integral + low_integral + high_integral)

#sumand = np.pi/2 * cut_off**2 * (2*gamma*phi_x**2 - 2*mu + gamma*cut_off**2)

def analytic_term(mu, phi_x, gamma, cut_off):
    return np.pi/2 * cut_off**2 * (2*gamma*phi_x**2 - 2*mu + gamma*cut_off**2)

# value of the integral in the pocket region without pockets
#absolute_value = -2*np.pi*gamma*((1.01**4+0.99**4)*k_F**4/4-k_F**4*(1.01**2+0.99**2)/2+k_F**4/2)

#%% Calculate fundamental energy

#phi_x_values = np.linspace(-0.005 * k_F, 0.005 * k_F, 20)
phi_x_values = np.linspace(-0.003 * k_F, 0.003 * k_F, N_phi)

energy_phi = np.zeros_like(phi_x_values)
cut_off = 1.1*k_F # 1.1 k_F

for i, phi_x in enumerate(phi_x_values):
    print(phi_x)
    #integrals = integrate_pocket(N, mu, B, Delta, phi_x, gamma, Lambda, k_F)
    integral, low_integral, high_integral = integrate_brute_force(N, mu, B_y, Delta, phi_x, gamma, Lambda, k_F, cut_off, B_x)
    energy_phi[i] = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
    
    
#%% Plot fundamental energy

from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

fig, ax = plt.subplots()
#ax.plot(phi_x_values/k_F, energy_phi, "o")
fundamental_energy = energy_phi + np.pi/2 * cut_off**2 * (2*gamma*phi_x_values**2 - 2*mu + gamma*cut_off**2)
ax.plot(phi_x_values/k_F, fundamental_energy, "o")
 

cs = CubicSpline(phi_x_values, fundamental_energy)
x = np.linspace(-0.00002 * k_F, 0.00002 * k_F, 300)
# x = np.linspace(-0.005 * k_F, 0.005 * k_F, 300)

ax.plot(x/k_F, cs(x))



ax.set_xlabel(r"$\phi_x/k_F$")
ax.set_ylabel(r"$E(\phi_x)$")
ax.set_title(r"$B_y/\Delta=$" + f"{B_y/Delta:.5}" +
             r"; $B_x/\Delta=$" + f"{B_x/Delta:.5}" +
             r"; $\Lambda/k_F=$" + f"{1/k_F*cut_off:.1f}" +
             r"; $N=$" + f"{N}"+
             r"; $\lambda/\Delta=$" + f"{Lambda/Delta}"
             )

#minima = find_peaks(-cs(x))[0]
# peaks, properties = find_peaks(-cs(x))
# minima = peaks[np.argmax(-cs(x[peaks]))]

# ax.scatter(x[minima]/k_F, cs(x[minima]), color="red")
# second_derivative = float(cs.derivative(nu=2)(x[minima]))

# axs[1].plot(x/k_F, cs.derivative(nu=2)(x))
N_polifit = 2

peaks_data, properties_data = find_peaks(-fundamental_energy)
minima_index = peaks_data[np.argmax(-fundamental_energy[peaks_data])]
a = minima_index - N_polifit
b = minima_index + N_polifit + 1
p = np.polyfit(phi_x_values[a:b], fundamental_energy[a:b], deg=2)

u = phi_x_values[a:b]
ax.plot(u/k_F, p[0]*u**2 + p[1] * u +  p[2])
ax.scatter(phi_x_values[minima_index]/k_F, fundamental_energy[minima_index],
               color="red", zorder=2)

superfluid_density = 2*p[0]


zero_index = int( (len(phi_x_values)-1)/2 )
a = zero_index - N_polifit
b = zero_index + N_polifit + 1
p_0 = np.polyfit(phi_x_values[a:b], fundamental_energy[a:b], deg=2)
u = phi_x_values[a:b]
ax.plot(u/k_F, p_0[0]*u**2 + p_0[1] * u +  p_0[2])
ax.scatter(phi_x_values[zero_index]/k_F, fundamental_energy[zero_index],
               color="black", zorder=2)

superfluid_density_0 = 2*p_0[0]

# peaks_data_sorted = peaks_data[np.argsort(-fundamental_energy[peaks_data])]
# second_minima_index = peaks_data_sorted[1]
# ax.scatter(phi_x_values[second_minima_index]/k_F, fundamental_energy[second_minima_index],
#                color="green", zorder=2)

def finite_differences(fundamental_energy):
    return (fundamental_energy[minima_index+1]-2*fundamental_energy[minima_index]+fundamental_energy[minima_index-1])/np.diff(phi_x_values)[0]**2


#%%

B_values = np.linspace(0, 2, 6) * Delta
superfluid_density = np.zeros_like(B_values)

for i, B in enumerate(B_values):
    print(B)
    phi_x_values = np.linspace(-0.00002 * k_F, 0.00002 * k_F, 10)
    energy_phi = np.zeros_like(phi_x_values)
    cut_off = 30*k_F # 1.1 k_F

    for j, phi_x in enumerate(phi_x_values):
        #integrals = integrate_pocket(N, mu, B, Delta, phi_x, gamma, Lambda, k_F)
        integral, low_integral, high_integral = integrate_brute_force(N, mu, B, Delta, phi_x, gamma, Lambda, k_F, cut_off)
        energy_phi[j] = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
    fundamental_energy = energy_phi + np.pi/2 * cut_off**2 * (2*gamma*phi_x_values**2 - 2*mu + gamma*cut_off**2)
    cs = CubicSpline(phi_x_values, fundamental_energy)
    x = np.linspace(-0.00002 * k_F, 0.00002 * k_F, 300)
    minima = find_peaks(-cs(x))[0]
    superfluid_density[i] = float(cs.derivative(nu=2)(x[minima[0]]))



#%%

fig, ax = plt.subplots()
ax.plot(B_values/Delta, superfluid_density)

ax.set_xlabel(r"$B/\Delta$")
ax.set_ylabel(r"$D_s$")