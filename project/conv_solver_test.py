import numpy as np

from project import conv_solver


def test_exponential_decay():
    x_dim = 2 ** 5
    init_x_coeffs = np.random.uniform(0, 1, x_dim).astype(complex)
    weight_fn = np.zeros((x_dim))
    f_b = lambda t: np.zeros_like(init_x_coeffs)
    soln = conv_solver.solve_coeff_space(weight_fn, init_x_coeffs, f_b, 1, lambda x: x, 5)
    log_y_0 = np.log(init_x_coeffs)
    log_y = np.log(np.transpose(soln.y))
    a = (log_y - log_y_0) * -1
    np.testing.assert_allclose(np.transpose(a), np.tile(np.expand_dims(soln.t, 0), (x_dim, 1)), rtol=1e-3)


test_exponential_decay()
