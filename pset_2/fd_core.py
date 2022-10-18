import numpy as np
from scipy import sparse


def homogenous_dirichlet_fd_1d(grid_size, grid_spacing):
    return 1/grid_spacing**2 * sparse.diags([np.ones(grid_size - 1),
                                             -2 * np.ones(grid_size),
                                             np.ones(grid_size - 1)], [-1, 0, 1])


def homogenous_dirichlet_fd_2d(grid_size, grid_spacing):
    fd_mat = homogenous_dirichlet_fd_1d(grid_size, grid_spacing)
    return sparse.kron(fd_mat, sparse.identity(grid_size)) + sparse.kron(sparse.identity(grid_size), fd_mat)
