import numpy as np
import matplotlib.pyplot as plt

import helmholtz_solver, util


grid_spacing = 1/256
grid = np.arange(start=0, stop=1 + grid_spacing, step=grid_spacing)
grid_free = grid[1:-1]
x_grid, y_grid = np.meshgrid(grid_free, grid_free)

center_x = 0.6
center_y = 0.7
sigma = 0.01
v = util.gaussian_2d(x_grid, y_grid, center_x, center_y, sigma)
v_fig, v_axs = plt.subplots(nrows=1, ncols=1)
v_axs.imshow(v, origin="lower", extent=(0, 1, 0, 1))


f = 21.3e6
omega = 2 * np.pi * f
c = 3e8
k_sq = omega**2/c**2

soln = helmholtz_solver.solve_direct_fd_const_coeff(grid_spacing, k_sq, v)
soln_fig, soln_axs = plt.subplots(nrows=1, ncols=1)
im = soln_axs.imshow(soln, origin="lower", extent=(0, 1, 0, 1))
soln_fig.colorbar(im, orientation='vertical')

plt.show()
