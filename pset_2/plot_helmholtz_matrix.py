import numpy as np
import matplotlib.pyplot as plt

import helmholtz_solver

f = 21.3e6
omega = 2 * np.pi * f
c = 3e8
k_sq = omega**2/c**2

grid_spacing = 1/8
grid = np.arange(start=0, stop=1 + grid_spacing, step=grid_spacing)
grid_nonzero = grid[1:-1]
helm_mat = helmholtz_solver.construct_direct_fd_const_coeff(len(grid_nonzero), grid_spacing, k_sq)

helm_mat_dense = np.array(helm_mat.todense())
plt.imshow(helm_mat_dense)
plt.show()

