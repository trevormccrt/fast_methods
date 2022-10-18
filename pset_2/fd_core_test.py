import numpy as np

import fd_core


def test_1d_dirichlet():
    grid_spacing = 0.01
    quad_coeff = np.random.uniform(-1, 1)
    x = np.arange(start=-1, stop=1, step=grid_spacing)
    y = quad_coeff * np.power(x, 2)
    fd_mat = fd_core.homogenous_dirichlet_fd_1d(len(x), grid_spacing)
    deriv = fd_mat.dot(y)
    np.testing.assert_allclose(deriv[1:-1], 2 * quad_coeff)
    y_pad = np.pad(y, (1, 1))
    fd_mat_pad = fd_core.homogenous_dirichlet_fd_1d(len(y_pad), grid_spacing)
    deriv_pad = fd_mat_pad.dot(y_pad)[1:-1]
    np.testing.assert_allclose(deriv_pad, deriv)


def test_2d_dirichlet():
    grid_spacing = 0.01
    quad_coeff_x = np.random.uniform(-1, 1)
    quad_coeff_y = np.random.uniform(-1, 1)
    grid_side = np.arange(start=-1, stop=1, step=grid_spacing)
    x_grid, y_grid = np.meshgrid(grid_side, grid_side)
    f = quad_coeff_x * x_grid**2 + quad_coeff_y * y_grid**2
    fd_mat = fd_core.homogenous_dirichlet_fd_2d(len(grid_side), grid_spacing)
    deriv = np.reshape(fd_mat.dot(f.flatten()), np.shape(f))
    np.testing.assert_allclose(deriv[1:-1, 1:-1], 2 * (quad_coeff_x + quad_coeff_y))
    f_pad = np.pad(f, ((1, 1), ))
    fd_mat_pad = fd_core.homogenous_dirichlet_fd_2d(np.shape(f_pad)[0], grid_spacing)
    deriv_pad = np.reshape(fd_mat_pad.dot(f_pad.flatten()), np.shape(f_pad))[1:-1, 1:-1]
    np.testing.assert_allclose(deriv_pad, deriv)

