import numpy as np

from pset_3 import chebychev
import ultraspherical_core


def test_ultraspherical_differentiation():
    N = 2**6
    grid = chebychev.extrema_grid(N)
    quad_coeff = np.random.uniform(-1, 1)
    y = quad_coeff * grid**3 + np.sin(grid) + np.tan(grid)
    dy = 3 * quad_coeff * grid**2 + np.cos(grid) + 1/np.cos(grid)**2
    ddy = 6 * quad_coeff * grid - np.sin(grid) + 2 * np.tan(grid)/np.cos(grid)**2
    y_cheb = chebychev.cheb(np.expand_dims(y,0))[0, :]
    d_mat = ultraspherical_core.differentiation_matrix(1, N)
    dy_ultra = d_mat.dot(y_cheb)
    conv_mat = ultraspherical_core.conversion_matrix(0, N)
    dy_ultra_true = conv_mat.dot(chebychev.cheb(np.expand_dims(dy, 0))[0, :])
    np.testing.assert_allclose(dy_ultra, dy_ultra_true, atol=1e-7)
    dd_mat = ultraspherical_core.differentiation_matrix(2, N)
    ddy_ultra = dd_mat.dot(y_cheb)
    conv_mat_2 = ultraspherical_core.conversion_matrix(1, N)
    ddy_ultra_true = conv_mat_2.dot(conv_mat.dot(chebychev.cheb(np.expand_dims(ddy, 0))[0, :]))
    np.testing.assert_allclose(ddy_ultra, ddy_ultra_true, atol=1e-7)

