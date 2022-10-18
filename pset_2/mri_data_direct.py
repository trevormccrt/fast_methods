import matplotlib.pyplot as plt
import numpy as np
from scipy import io as scio

import util, helmholtz_solver

data = scio.loadmat("MRI_DATA.mat")
eps = np.flipud(data["e_r"][1:-1, 1:-1])
fig, axs = plt.subplots(nrows=1, ncols=1)
axs.imshow(np.real(eps), origin="lower", extent=(0, 1, 0, 1))

grid_spacing = 1/256
grid = np.arange(start=0, stop=1 + grid_spacing, step=grid_spacing)
grid_free = grid[1:-1]
x_grid, y_grid = np.meshgrid(grid_free, grid_free)


center_x = 0.5
center_y = 0.5
sigma = 0.01
v = util.gaussian_2d(x_grid, y_grid, center_x, center_y, sigma)

f = 298.3e6
omega = 2 * np.pi * f
c = 3e8
k_sq = omega**2/c**2

xlims = [0.35, 0.635]
ylims = [0.186, 0.629]

soln = helmholtz_solver.solve_direct_fd_variable_k(grid_spacing, k_sq * eps, v)

soln_fig, soln_axs = plt.subplots(nrows=1, ncols=2)
im1 = soln_axs[0].imshow(np.real(soln), origin="lower", extent=(0, 1, 0, 1))
soln_axs[0].set_xlim(xlims)
soln_axs[0].set_ylim(ylims)
soln_axs[0].set_title("Real Part")
soln_fig.colorbar(im1, ax=soln_axs[0])
im2 = soln_axs[1].imshow(np.imag(soln), origin="lower", extent=(0, 1, 0, 1))
soln_axs[1].set_xlim(xlims)
soln_axs[1].set_ylim(ylims)
soln_axs[1].set_title("Imaginary Part")
soln_fig.colorbar(im2, ax=soln_axs[1])
soln_fig.tight_layout()
plt.show()
print("")

