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
    w_coeffs = np.random.uniform(-1, 0, (x_dim, x_dim)) + 1j * np.random.uniform(0, 2 * np.pi, (x_dim, x_dim))
    f_b = lambda t: np.zeros_like(init_x_coeffs)
    soln_t, soln_y = conv_solver.solve_coeff_space(w_coeffs, init_x_coeffs, f_b, 1, lambda x: x, 3, rtol=1e-7)
    exact_exp = np.einsum("ij, t -> tij", (2 * np.pi) ** 2 * w_coeffs - 1, soln_t)
    exact_soln = init_x_coeffs * np.exp(exact_exp)
    np.testing.assert_allclose(soln_y, exact_soln, atol=5e-4)
