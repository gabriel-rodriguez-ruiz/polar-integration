#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 19:55:22 2025

@author: gabriel
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
  "text.usetex": True,
})

data_folder = Path(r"./Data")

file_to_open = data_folder / "superfluid_density_B_in_-0.79_(0.8-2.0)_phi_x_in_(-0.002-0.002)_Delta=0.08_lambda=8.76_points=16_N_phi=101_N=514.npz"


Data = np.load(file_to_open)
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
q_B = Data["q_B"]
k_F = Data["k_F"]

superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_xx_0 = Data["superfluid_density_xx_0"]
superfluid_density_yy = Data["superfluid_density_yy"]
superfluid_density_yy_0 = Data["superfluid_density_yy_0"]
superfluid_density_yy = Data["superfluid_density_yy"]
superfluid_density_yy_0 = Data["superfluid_density_yy_0"]
superfluid_density_xy = Data["superfluid_density_xy"]
superfluid_density_xy_0 = Data["superfluid_density_xy_0"]
fig, axs = plt.subplots(2, 1)

# axs[0].plot(B_values/Delta, superfluid_density_xx, "o")
# axs[0].plot(B_values/Delta, superfluid_density_xx_0, "o")
# axs[0].plot(B_values/Delta, superfluid_density_yy, "o")
# axs[0].plot(B_values/Delta, superfluid_density_yy_0, "o")
# axs[0].plot(B_values/Delta, superfluid_density_xy, "o")
axs[0].plot(B_values/Delta, superfluid_density_yy + superfluid_density_xy, "o")
# axs[0].plot(B_values/Delta, superfluid_density_yy_0, "o")
axs[0].legend(prop={'size': 4})
axs[0].set_xlabel(r"$B/\Delta$")
axs[0].set_ylabel(r"$D_s$")
axs[0].set_title(r"$\lambda/\Delta=$" + f"{Lambda/Delta}")

axs[1].plot(B_values/Delta, q_B/k_F, "o")

axs[1].set_xlabel(r"$B/\Delta$")
axs[1].set_ylabel(r"$\phi_{eq}/k_F$")

#%%

data_folder = Path(r"./Data")

file_to_open = data_folder / "superfluid_density_B_in_-0.79_(0.8-2.0)_phi_x_in_(-0.002-0.002)_Delta=0.08_lambda=8.76_points=16_N_phi=101_N=258.npz"


Data = np.load(file_to_open)
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
q_B = Data["q_B"]
k_F = Data["k_F"]

superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_xx_0 = Data["superfluid_density_xx_0"]
superfluid_density_yy = Data["superfluid_density_yy"]
superfluid_density_yy_0 = Data["superfluid_density_yy_0"]
superfluid_density_yy = Data["superfluid_density_yy"]
superfluid_density_yy_0 = Data["superfluid_density_yy_0"]
superfluid_density_xy = Data["superfluid_density_xy"]
superfluid_density_xy_0 = Data["superfluid_density_xy_0"]

# fig, axs = plt.subplots(2, 1)

# axs[0].plot(B_values/Delta, superfluid_density_xx, "o")
# axs[0].plot(B_values/Delta, superfluid_density_xx_0, "o")
# axs[0].plot(B_values/Delta, superfluid_density_yy, "o")
# axs[0].plot(B_values/Delta, superfluid_density_yy_0, "o")
# axs[0].plot(B_values/Delta, superfluid_density_xy, "o")
axs[0].plot(B_values/Delta, superfluid_density_xx + superfluid_density_xy, "o")
# axs[0].plot(B_values/Delta, superfluid_density_yy_0, "o")
axs[0].legend(prop={'size': 4})
axs[0].set_xlabel(r"$B/\Delta$")
axs[0].set_ylabel(r"$D_s$")
axs[0].set_title(r"$\lambda/\Delta=$" + f"{Lambda/Delta}")

axs[1].plot(B_values/Delta, q_B/k_F, "o")

axs[1].set_xlabel(r"$B/\Delta$")
axs[1].set_ylabel(r"$\phi_{eq}/k_F$")

