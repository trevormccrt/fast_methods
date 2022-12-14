import numpy as np

import chebychev_core, chebychev_solver, fourier_core, conv_solver


def cheb_to_fourier(x):
    return np.pi * (x + 1)


def test_exponential_decay():
    x_dim = 2 ** 5
    init_x_coeffs = np.random.uniform(0, 1, x_dim)
    weight_matrix = np.zeros((x_dim, x_dim))
    f_b = lambda t: np.zeros_like(init_x_coeffs)
    soln = chebychev_solver.solve_coeff_space(weight_matrix, init_x_coeffs, f_b, 1, lambda x: x, 5)
    log_y_0 = np.log(init_x_coeffs)
    log_y = np.log(np.transpose(soln.y))
    a = (log_y - log_y_0) * -1
    np.testing.assert_allclose(np.transpose(a), np.tile(np.expand_dims(soln.t, 0), (x_dim, 1)), rtol=1e-3)


def test_1d_conv():

    def conv_kernel(x):
        return 2 * (0.5 - np.cos(x - np.pi/4))

    def conv_kernel_2d(x, xp):
        return np.pi * conv_kernel(cheb_to_fourier(x) - cheb_to_fourier(xp))

    def init_x(x):
        return np.exp(np.sin(2 * x)) + 1

    def init_x_cheb(x):
        return init_x(cheb_to_fourier(x))

    g = 1
    f_nonlin = np.tanh
    total_time = 10
    t_eval = np.arange(start=0, stop=total_time, step=0.1)

    fourier_k = 7
    fourier_grid = fourier_core.fourier_grid(fourier_k)
    fourier_kernel = conv_kernel(fourier_grid)
    fourier_init_x = init_x(fourier_grid)
    fourier_b = lambda t: np.zeros_like(fourier_init_x)
    fourier_kernel_coeffs = fourier_core.fourier_series_coeffs(fourier_kernel)
    fourier_init_x_coeffs = fourier_core.fourier_series_coeffs(fourier_init_x)
    conv_soln = conv_solver.solve_coeff_space(fourier_kernel_coeffs, fourier_init_x_coeffs, fourier_b, g, f_nonlin, total_time, t_eval, rtol=1e-7)
    conv_soln_grid = np.real_if_close(fourier_core.nifft(np.transpose(conv_soln.y), axes_from=1))

    cheb_k = 7
    cheb_grid = chebychev_core.extrema_grid(2 ** cheb_k)
    grid_y, grid_x = np.meshgrid(cheb_grid, cheb_grid, indexing="ij")
    cheb_kernel = conv_kernel_2d(grid_y, grid_x)
    cheb_x = init_x_cheb(cheb_grid)
    cheb_kernel_coeffs = chebychev_core.ncheb(cheb_kernel)
    cheb_x_coeffs = chebychev_core.ncheb(cheb_x)
    cheb_b = lambda t: np.zeros_like(cheb_x_coeffs)
    cheb_soln = chebychev_solver.solve_coeff_space(cheb_kernel_coeffs, cheb_x_coeffs, cheb_b, g, f_nonlin, total_time, t_eval, rtol=1e-7)
    cheb_soln_grid = np.real_if_close(chebychev_core.icheb(np.transpose(cheb_soln.y), axis=-1))

    fourier_freqs = fourier_core.get_k_vals(len(fourier_grid))
    interpolated_fourier_solns = []
    for time in range(np.shape(conv_soln.y)[-1]):
        coeffs = conv_soln.y[:, time]
        interpolated_fourier_solns.append(fourier_core.evaluate_fourier_series(fourier_freqs, coeffs, cheb_to_fourier(cheb_grid)))
    interpolated_fourier_solns = np.real_if_close(np.array(interpolated_fourier_solns))

    error = np.max(np.abs(cheb_soln_grid - interpolated_fourier_solns))/np.max(np.abs(cheb_soln_grid))

    assert error < 1e-5
