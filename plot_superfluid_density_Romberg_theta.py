#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 08:17:51 2025

@author: gabriel
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
  "text.usetex": True,
})

data_folder = Path(r"./Data")

file_to_open = data_folder / "superfluid_density_B=_0.16_theta_in_(0.0-1.571)_Delta=0.08_lambda=8.76_points=16_N=514.npz"


Data = np.load(file_to_open)
theta_values = Data["theta_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
q_theta = Data["q_theta"]
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
axs[0].plot(theta_values, superfluid_density_xx, "o", label=r"$n_{xx}(q\neq0)$")
axs[0].plot(theta_values, superfluid_density_yy, "o", label=r"$n_{yy}(q\neq0)$")
axs[0].plot(theta_values, superfluid_density_xx_0, "o", label=r"$n_{xx}(q=0)$")
axs[0].plot(theta_values, superfluid_density_yy_0, "o", label=r"$n_{yy}(q=0)$")

# axs[0].plot(B_values/Delta, superfluid_density_yy_0, "o")
axs[0].legend(prop={'size': 12})
axs[0].set_xlabel(r"$\theta$")
axs[0].set_ylabel(r"$D_s$")
axs[0].set_title(r"$\lambda/\Delta=$" + f"{Lambda/Delta}")

axs[1].plot(theta_values, q_theta/k_F, "o")

axs[1].set_xlabel(r"$\theta$")
axs[1].set_ylabel(r"$\phi_{eq}/k_F$")

#%%

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

ax.plot(theta_values, superfluid_density_xx, "o")
ax.plot(theta_values, superfluid_density_yy, "o")
ax.plot(theta_values, superfluid_density_xx_0, "o")
ax.plot(theta_values, superfluid_density_yy_0, "o")
# ax.plot(theta_values, superfluid_density_xy, "o")
# ax.plot(theta_values, superfluid_density_xx_0, "o")
# ax.plot(theta_values, superfluid_density_yy_0, "o")
