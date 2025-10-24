#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 07:46:37 2025

@author: gabriel
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
  "text.usetex": True,
})

data_folder = Path(r"./Data")

file_to_open = data_folder / "superfluid_density_B_in_x_(0.8-2.0)_phi_x_in_(-0.003-0.003)_Delta=0.08_lambda=0.64_points=48_N_phi=101_N=300.npz"

Data = np.load(file_to_open)
superfluid_density_45 = Data["superfluid_density"]
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
phi_eq_B = Data["phi_eq"]
k_F = Data["k_F"]
superfluid_density_0 = Data["superfluid_density_0"]
superfluid_density_finite_differences_0 = Data["superfluid_density_finite_differences_0"]
superfluid_density_finite_differences_45 = Data["superfluid_density_finite_differences"]

fig, axs = plt.subplots(2, 1)
axs[0].plot(B_values/Delta, superfluid_density_finite_differences_45, "s", label=r"$n_{\parallel,S}(\phi=\phi_{eq}, \lambda/\Delta=$" + f"{Lambda/Delta})",
            markersize=3, zorder=5)
axs[0].plot(B_values/Delta, superfluid_density_0, "s", label=r"$n_{\parallel,S}(\phi=0, \lambda/\Delta=$" + f"{Lambda/Delta})")


axs[0].set_xlabel(r"$B/\Delta$")
axs[0].set_ylabel(r"$D_s$")
axs[0].set_title(r"$\lambda/\Delta=$" + f"{Lambda/Delta}")

axs[1].plot(B_values/Delta, phi_eq_B/k_F, "s")

axs[1].set_xlabel(r"$B/\Delta$")
axs[1].set_ylabel(r"$\phi_{eq}/k_F$")

#%%

file_to_open = data_folder / "superfluid_density_B_in_y_(0.8-2.0)_phi_x_in_(-0.003-0.003)_Delta=0.08_lambda=0.64_points=48_N_phi=101_N=300.npz"

Data = np.load(file_to_open)
superfluid_density_135 = Data["superfluid_density"]
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
phi_eq_B = Data["phi_eq"]
k_F = Data["k_F"]
superfluid_density_0 = Data["superfluid_density_0"]
superfluid_density_finite_differences_0 = Data["superfluid_density_finite_differences_0"]
superfluid_density_finite_differences_135 = Data["superfluid_density_finite_differences"]


axs[0].plot(B_values/Delta, superfluid_density_finite_differences_135, "v", label=r"$n_{\perp,S}(\phi=\phi_{eq}, \lambda/\Delta=$" + f"{Lambda/Delta})")
axs[0].plot(B_values/Delta, superfluid_density_finite_differences_0, "v", label=r"$n_{\perp,S}(\phi=0, \lambda/\Delta=$" + f"{Lambda/Delta})",
            markersize=3, zorder=5)


axs[0].set_xlabel(r"$B/\Delta$")
axs[0].set_ylabel(r"$D_s$")
axs[0].set_title(r"$\lambda/\Delta=$" + f"{Lambda/Delta}")

axs[1].plot(B_values/Delta, phi_eq_B/k_F,"vg")

axs[1].set_xlabel(r"$B/\Delta$")
axs[1].set_ylabel(r"$\phi_{eq}/k_F$")
axs[1].set_ylim(-0.002, 0.0005)
axs[0].legend(prop={'size': 4}, loc="upper right")

axs[0].grid()
axs[1].grid()

#%% Other directions

file_to_open = data_folder / "superfluid_density_B_in_0.79_(0.8-2.0)_phi_x_in_(-0.003-0.003)_Delta=0.08_lambda=0.64_points=48_N_phi=101_N=300.npz"

Data = np.load(file_to_open)
superfluid_density_90 = Data["superfluid_density"]
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
phi_eq_B = Data["phi_eq"]
k_F = Data["k_F"]
superfluid_density_0 = Data["superfluid_density_0"]
superfluid_density_finite_differences_0 = Data["superfluid_density_finite_differences_0"]
superfluid_density_finite_differences_90 = Data["superfluid_density_finite_differences"]


axs[0].plot(B_values/Delta, superfluid_density_finite_differences_90, "^", label=r"$n_{\perp,S}(\phi=\phi_{eq}, \lambda/\Delta=$" + f"{Lambda/Delta})")
axs[0].plot(B_values/Delta, superfluid_density_finite_differences_0, "^", label=r"$n_{\perp,S}(\phi=0, \lambda/\Delta=$" + f"{Lambda/Delta})",
            markersize=3, zorder=5)
