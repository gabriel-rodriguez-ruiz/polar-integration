#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 14:44:11 2025

@author: gabriel
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
  "text.usetex": True,
})

data_folder = Path(r"./Data")

file_to_open = data_folder / "superfluid_density_B_in_(0.0-3.0)_phi_x_in_(-0.005-0.005)_Delta=0.08_lambda=0.0_points=16_N_phi=101_N=100.npz"

Data = np.load(file_to_open)
superfluid_density = Data["superfluid_density"]
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
phi_eq_B = Data["phi_eq"]
k_F = Data["k_F"]
superfluid_density_0 = Data["superfluid_density_0"]

fig, axs = plt.subplots(2, 1)
axs[0].plot(B_values/Delta, superfluid_density, "o", label=r"$n_S(\phi=\phi_{eq}, \lambda/\Delta=$" + f"{Lambda/Delta})")
axs[0].plot(B_values/Delta, superfluid_density_0, "o", label=r"$n_S(\phi=0, \lambda/\Delta=$" + f"{Lambda/Delta})")


axs[0].set_xlabel(r"$B/\Delta$")
axs[0].set_ylabel(r"$D_s$")
axs[0].set_title(r"$\lambda/\Delta=$" + f"{Lambda/Delta}")

axs[1].plot(B_values/Delta, phi_eq_B/k_F, "o")

axs[1].set_xlabel(r"$B/\Delta$")
axs[1].set_ylabel(r"$\phi_{eq}/k_F$")

#%%

file_to_open = data_folder / "superfluid_density_B_in_(0.0-3.0)_phi_x_in_(-0.005-0.005)_Delta=0.08_lambda=0.64_points=32_N_phi=101_N=100.npz"

Data = np.load(file_to_open)
superfluid_density = Data["superfluid_density"]
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
phi_eq_B = Data["phi_eq"]
k_F = Data["k_F"]
superfluid_density_0 = Data["superfluid_density_0"]

axs[0].plot(B_values/Delta, superfluid_density, "o", label=r"$n_S(\phi=\phi_{eq}, \lambda/\Delta=$" + f"{Lambda/Delta})")
axs[0].plot(B_values/Delta, superfluid_density_0, "o", label=r"$n_S(\phi=0, \lambda/\Delta=$" + f"{Lambda/Delta})")


axs[0].set_xlabel(r"$B/\Delta$")
axs[0].set_ylabel(r"$D_s$")
axs[0].set_title(r"$\lambda/\Delta=$" + f"{Lambda/Delta}")

axs[1].plot(B_values/Delta, phi_eq_B/k_F, "o")

axs[1].set_xlabel(r"$B/\Delta$")
axs[1].set_ylabel(r"$\phi_{eq}/k_F$")

axs[0].legend(prop={'size': 6})