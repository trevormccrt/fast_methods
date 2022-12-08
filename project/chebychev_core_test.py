import numpy as np
from scipy import integrate

from project import chebychev_core
from pset_3 import chebychev


def _chebychev_poly(m, x):
    return np.cos(m * np.arccos(x))


def test_cheb_integral():
    def integrand(x, m, n):
        return _chebychev_poly(m, x) * _chebychev_poly(n, x)
    m_max = 10
    S = chebychev_core.generate_cheb_integral_matrix(m_max)
    S_slow = np.zeros_like(S)
    for k in range(m_max + 1):
        for j in range(m_max + 1):
            S_slow[k, j] = integrate.quad(lambda x: integrand(x, k, j), -1, 1)[0]
    np.testing.assert_allclose(S, S_slow, atol=1e-8)


def test_multicontract():
    dim = 10
    mat = np.random.uniform(-1, 1, (dim, dim))
    vec_1d = np.random.uniform(-1, 1, dim)
    multi_results_1d = chebychev_core._contract_mat_nd(mat, vec_1d)
    true_result_1d = mat.dot(vec_1d)
    np.testing.assert_allclose(multi_results_1d, true_result_1d)
    vec_2d = np.random.uniform(-1, 1, (dim, dim))
    results_2d = np.zeros((dim, dim))
    for k1 in range(dim):
        for k2 in range(dim):
            for j1 in range(dim):
                for j2 in range(dim):
                    results_2d[k1, k2] = mat[k1, j1] * mat[k2, j2] * vec_2d[j1, j2]
    multi_resuts_2d = chebychev_core._contract_mat_nd(mat, vec_2d)
    print("")


def test_cheb_kernel_1d():
    def f_x(x):
        return x ** 2

    def f_w(x1, x2):
        return x1**3 + (x1 - x2)**4 + np.tanh(x2 * x1)

    grid_size = 2**5
    extrema_grid = chebychev.extrema_grid(grid_size)
    grid_x, grid_y = np.meshgrid(extrema_grid, extrema_grid)
    x_grid = f_x(extrema_grid)
    w_grid = f_w(grid_x, grid_y)
    x_cheb = chebychev_core.ncheb(x_grid)
    w_cheb = chebychev_core.ncheb(w_grid)
    cheb_int_mat = chebychev_core.generate_cheb_integral_matrix(grid_size-1)
    int_cheb = chebychev_core.chebychev_kernel_integral(x_cheb, w_cheb, cheb_int_mat)
    print("")


test_multicontract()
