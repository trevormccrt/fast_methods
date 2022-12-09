import numpy as np
from scipy import integrate

from project import chebychev_core


def _chebychev_poly(m, x):
    return np.cos(m * np.arccos(x))


def test_ncheb_inv():
    dim = 5
    x = np.random.uniform(-1, 1, (dim, dim, dim, dim))
    y = chebychev_core.nicheb(chebychev_core.ncheb(x))
    np.testing.assert_allclose(x, y)


def test_delta():
    N = 2**5
    grid = chebychev_core.extrema_grid(N)
    for m in range(N):
        input = np.zeros((1, N))
        input[0, m] = 1
        out = chebychev_core.icheb(input)
        desired = np.cos(m * np.arccos(grid))
        np.testing.assert_allclose(out[0, :], desired)


def test_coeffecients():
    def f_test(x):
        return x**2 + np.tanh(x)**2 + np.sin(x)

    def integrand(x, m):
        integ = 2/np.pi * f_test(x) * np.cos(m * np.arccos(x)) * 1/np.sqrt(1 - x**2)
        if m==0:
            return integ/2
        return integ

    N = 2**6
    for m in range(N):
        grid = np.expand_dims(chebychev_core.extrema_grid(N), 0)
        f_sampled = f_test(grid)
        f_trns = chebychev_core.cheb(f_sampled)
        exact = integrate.quad(lambda x: integrand(x, m), -1, 1)[0]
        fast = f_trns[0, m]
        np.testing.assert_allclose(exact, fast, atol=1e-5)


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
                    results_2d[k1, k2] += mat[k1, j1] * mat[k2, j2] * vec_2d[j1, j2]
    multi_resuts_2d = chebychev_core._contract_mat_nd(mat, vec_2d)
    np.testing.assert_allclose(results_2d, multi_resuts_2d)


def test_contract_half():
    dim = 10
    mat_1d = np.random.uniform(-1, 1, (dim, dim))
    vec_1d = np.random.uniform(-1, 1, dim)
    contract_results_1d = chebychev_core._contract_half_nd(mat_1d, vec_1d)
    expected_1d = mat_1d.dot(vec_1d)
    np.testing.assert_allclose(contract_results_1d, expected_1d)
    mat_2d = np.random.uniform(-1, 1, (dim, dim, dim, dim))
    vec_2d = np.random.uniform(-1, 1, (dim, dim))
    results_2d = np.zeros((dim, dim))
    for k1 in range(dim):
        for k2 in range(dim):
            for j1 in range(dim):
                for j2 in range(dim):
                    results_2d[k1, k2] += mat_2d[k1, k2, j1, j2] * vec_2d[j1, j2]
    contract_results_2d = chebychev_core._contract_half_nd(mat_2d, vec_2d)
    np.testing.assert_allclose(results_2d, contract_results_2d)


def test_cheb_kernel_1d():
    def f_x(x):
        return x ** 2

    def f_w(x1, x2):
        return x1**3 + (x1 - x2)**4 + np.tanh(x2 * x1)

    def exact_integrand(r, x):
        return f_w(x, r) * f_x(x)

    grid_size = 2**5
    extrema_grid = chebychev_core.extrema_grid(grid_size)
    grid_x, grid_y = np.meshgrid(extrema_grid, extrema_grid)
    x_grid = f_x(extrema_grid)
    w_grid = f_w(grid_x, grid_y)
    x_cheb = chebychev_core.ncheb(x_grid)
    w_cheb = chebychev_core.ncheb(w_grid)
    cheb_int_mat = chebychev_core.generate_cheb_integral_matrix(grid_size-1)
    int_cheb = chebychev_core.chebychev_kernel_integral(x_cheb, w_cheb, cheb_int_mat)
    int_grid = chebychev_core.nicheb(int_cheb)
    exact_results = []
    for this_r in extrema_grid:
        exact_results.append(integrate.quad(lambda x: exact_integrand(this_r, x), -1, 1)[0])
    np.testing.assert_allclose(int_grid, exact_results, atol=1e-5)


def test_cheb_kernel_2d():
    def f_x(x1, x2):
        return np.sin(10 * x1) + x2**2 + np.tanh(x1 * x2)

    def f_w(y1, y2, x1, x2):
        return np.sin(y1 * y2 + x1 * x2) + np.exp(x1 - x2 + y1) + y2**2 + (y1 - x2) ** 2

    def exact_integrand(y1, y2, x1, x2):
        return f_w(y1, y2, x1, x2) * f_x(x1, x2)

    grid_size = 2 ** 5
    extrema_grid = chebychev_core.extrema_grid(grid_size)
    grid_y2, grid_y1, grid_x1, grid_x2 = np.meshgrid(extrema_grid, extrema_grid, extrema_grid, extrema_grid)
    grid_x2_small, grid_x1_small = np.meshgrid(extrema_grid, extrema_grid)
    cheb_int_mat = chebychev_core.generate_cheb_integral_matrix(grid_size - 1)
    a = grid_x2[0, 0, :, :]
    x_grid = f_x(grid_x1_small, grid_x2_small)
    w_grid = f_w(grid_y1, grid_y2, grid_x1, grid_x2)
    x_cheb = chebychev_core.ncheb(x_grid)
    w_cheb = chebychev_core.ncheb(w_grid)
    int_cheb = chebychev_core.chebychev_kernel_integral(x_cheb, w_cheb, cheb_int_mat)
    int_grid = chebychev_core.nicheb(int_cheb)
    exact_results = np.zeros_like(grid_x1_small)
    for i, y1 in enumerate(extrema_grid):
        for j, y2 in enumerate(extrema_grid):
            exact_results[i, j] = integrate.dblquad(lambda x1, x2: exact_integrand(y1, y2, x1, x2), -1, 1, -1, 1)[0]
    np.testing.assert_allclose(exact_results, int_grid, atol=1e-5)

