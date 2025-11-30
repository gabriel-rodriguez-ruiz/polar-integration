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

# file_to_open = data_folder / "superfluid_density_B_in_y_(0.8-2.0)_phi_x_in_(-0.003-0.003)_Delta=0.08_lambda=0.64_points=48_N_phi=101_N=300.npz"
# file_to_open = data_folder / "superfluid_density_B_in_1.6_(0.8-2.0)_phi_x_in_(-0.002-0.002)_Delta=0.08_lambda=0.64_points=32_N_phi=101_N=300_C=0.npz"
file_to_open = data_folder / "superfluid_density_B_in_1.6_(0.8-2.0)_phi_x_in_(-0.002-0.002)_Delta=0.08_lambda=8.76_points=32_N_phi=101_N=300_C=0.npz"


Data = np.load(file_to_open)
superfluid_density = Data["superfluid_density"]
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
phi_eq_B = Data["phi_eq"]
k_F = Data["k_F"]
# C = Data["C"]
superfluid_density_0 = Data["superfluid_density_0"]
superfluid_density_finite_differences_perpendicular = Data["superfluid_density_finite_differences"]
superfluid_density_finite_differences_perpendicular_0 = Data["superfluid_density_finite_differences_0"]
# superfluid_density_finite_differences_second_minima = Data["superfluid_density_finite_differences_second_minima"]
# superfluid_density_finite_differences_third_minima = Data["superfluid_density_finite_differences_third_minima"]

# phi_eq_second_minima = Data["phi_eq_second_minima"]
# phi_eq_third_minima = Data["phi_eq_third_minima"]

fig, axs = plt.subplots(2, 1)
# axs[0].plot(B_values/Delta, superfluid_density, "o", label=r"$\phi_{eq}$")
axs[0].plot(B_values/Delta, superfluid_density_finite_differences_perpendicular, "v", label=r"$D_{xx}(\phi_x=\Phi_B(B_y),\phi_y=0)$",
            color="green")

# axs[0].plot(B_values/Delta, superfluid_density_0, "o", label=r"$\phi=0$")
axs[0].plot(B_values/Delta, superfluid_density_finite_differences_perpendicular_0, "v", label=r"$D_{xx}(\phi_x=0,\phi_y=0)$",
            color="C2")
# axs[0].plot(B_values/Delta, superfluid_density_finite_differences_second_minima, "o", label="second minima")
# axs[0].plot(B_values/Delta, superfluid_density_finite_differences_third_minima, "o", label="third minima")

axs[0].legend(prop={'size': 4})
axs[0].set_xlabel(r"$B/\Delta$")
axs[0].set_ylabel(r"$D_s$")
axs[0].set_title(r"$\lambda/\Delta=$" + f"{Lambda/Delta}")
                 # + f"; C={C}")

axs[1].plot(B_values/Delta, phi_eq_B/k_F, "o")
# axs[1].plot(B_values/Delta, phi_eq_second_minima/k_F, "o")
# axs[1].plot(B_values/Delta, phi_eq_third_minima/k_F, "o")


axs[1].set_xlabel(r"$B/\Delta$")
axs[1].set_ylabel(r"$\phi_{eq}/k_F$")

superfluid_density_yy = Data["superfluid_density_yy"]
axs[0].plot(B_values/Delta, superfluid_density_yy, "o", label=r"$D_{yy}(\phi_x=\Phi_B(B_y), \phi_y=0)$",
            color="C0")

superfluid_density_yy_0 = Data["superfluid_density_yy_0"]
axs[0].plot(B_values/Delta, superfluid_density_yy_0, "o", label=r"$D_{yy}(\phi_x=\Phi_B(B_y), \phi_y=0)$",
            color="C1")


# superfluid_density_xx = Data["superfluid_density_xx"]
# axs[0].plot(B_values/Delta, superfluid_density_xx, "o", label=r"$D_{yy}(\phi_x=\Phi_B(B_y), \phi_y=0)$")
# superfluid_density_xx_0 = Data["superfluid_density_xx_0"]
# axs[0].plot(B_values/Delta, superfluid_density_xx_0, "o", label=r"$D_{yy}(\phi_x=\Phi_B(B_y), \phi_y=0)$")

#%%

# file_to_open = data_folder / "superfluid_density_yy_B_in_0.0_(0.8-2.0)_phi_x_in_(-0.002-0.002)_Delta=0.08_lambda=0.64_points=48_N_phi=101_N=300_h=7.3060176833886576e-06.npz"
file_to_open = data_folder / "superfluid_density_xx_B_in_0.0_(0.8-2.0)_phi_x_in_(-0.002-0.002)_Delta=0.08_lambda=0.64_points=48_N_phi=101_N=300_h=7.3060176833886576e-06.npz"

Data = np.load(file_to_open)
superfluid_density_yy = Data["superfluid_density_yy"]
axs[0].plot(B_values/Delta, superfluid_density_yy, "o", label=r"$D_{yy}(\phi_x=\Phi_B(B_y), \phi_y=0)$",
            color="C0")

file_to_open = data_folder / "superfluid_density_B_in_x_(0.8-2.0)_phi_x_in_(-0.003-0.003)_Delta=0.08_lambda=0.64_points=48_N_phi=101_N=300.npz"

Data = np.load(file_to_open)
superfluid_density_finite_differences_yy_0 = Data["superfluid_density_finite_differences_0"]
axs[0].plot(B_values/Delta, superfluid_density_finite_differences_yy_0, "o", label=r"$D_{yy}(\phi_x=0, \phi_y=0)$",
            color="blue")


axs[0].legend(fontsize=5, loc="upper right")


#%%

axs[0].plot(B_values/Delta, 1/2*(superfluid_density_finite_differences_perpendicular + superfluid_density_yy), "s", label=r"$D_{yy}(\phi_x=0, \phi_y=0)$",
            color="red")

axs[0].plot(B_values/Delta, 1/2*(superfluid_density_finite_differences_perpendicular_0 + superfluid_density_finite_differences_yy_0), "s", label=r"$D_{yy}(\phi_x=0, \phi_y=0)$",
            color="violet")