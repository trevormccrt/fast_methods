import numpy as np
from scipy import integrate

from project import fourier_core


def _eval_rhs(w_coeffs, x_coeffs, b_coeffs, f_nonlin, g):
    conv_int = fourier_core.fft_convolution_integral(w_coeffs, x_coeffs, axes_from=1)
    y_grid = f_nonlin(conv_int + fourier_core.nifft(b_coeffs))
    y_coeffs = fourier_core.fourier_series_coeffs(y_grid, axes_from=1)
    return -x_coeffs + g * y_coeffs


def solve_coeff_space(w_coeffs, init_x_coeffs, b_coeffs_func, g, f_nonlin, t_max, t_vals=None, rtol=1e-7):
    def integrand(t, y):
        y = np.transpose(y)
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, 0)
        this_x = np.reshape(y, (np.shape(y)[0], *init_x_coeffs.shape))
        b_coeffs = b_coeffs_func(t)
        this_rhs = _eval_rhs(w_coeffs, this_x, b_coeffs, f_nonlin, g)
        return np.reshape(this_rhs, (np.shape(y)[0], -1))

    soln = integrate.solve_ivp(integrand, [0, t_max], init_x_coeffs.flatten(), t_eval=t_vals, vectorized=True, rtol=rtol)
    return soln.t, np.reshape(np.transpose(soln.y), (-1, *np.shape(init_x_coeffs)))
