import numpy as np

from project import conv_solver


def test_exponential_decay():
    x_dim = 2 ** 5
    init_x_coeffs = np.random.uniform(0, 1, x_dim).astype(complex)
    weight_fn = np.zeros((x_dim))
    f_b = lambda t: np.zeros_like(init_x_coeffs)
    soln_t, soln_y = conv_solver.solve_coeff_space(weight_fn, init_x_coeffs, f_b, 1, lambda x: x, 5)
    log_y_0 = np.log(init_x_coeffs)
    log_y = np.log(soln_y)
    a = (log_y - log_y_0) * -1
    np.testing.assert_allclose(np.transpose(a), np.tile(np.expand_dims(soln_t, 0), (x_dim, 1)), rtol=1e-3)


def test_linear():
    x_dim = 2 ** 3
    init_x_coeffs = np.random.uniform(0, 1, (x_dim, x_dim)).astype(complex)
    w_coeffs = np.random.uniform(0, 1, (x_dim, x_dim)).astype(complex)
    f_b = lambda t: np.zeros_like(init_x_coeffs)
    soln_t, soln_y = conv_solver.solve_coeff_space(w_coeffs, init_x_coeffs, f_b, 1, lambda x: x, 0.3, rtol=1e-7)
    a = np.log(soln_y) - np.log(init_x_coeffs)
    b = np.einsum("t,... -> t...", soln_t, (2 * np.pi)**2 * w_coeffs - 1)
    np.testing.assert_allclose(a, b, rtol=1e-3)
