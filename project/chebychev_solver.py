import numpy as np
from scipy import integrate

from project import chebychev_core


def _eval_rhs(w_coeffs, x_coeffs, b_coeffs, f_nonlin, contractor_s_x, contractor_w_sx, g):
    integral_coeffs = chebychev_core.chebychev_kernel_integral(x_coeffs, w_coeffs, contractor_s_x, contractor_w_sx)
    integral_grid = chebychev_core.nicheb(integral_coeffs, axis_from=1)
    b_grid = chebychev_core.nicheb(b_coeffs, axis_from=1)
    y_grid = f_nonlin(integral_grid + b_grid)
    y_coeffs = chebychev_core.ncheb(y_grid, axis_from=1)
    return -x_coeffs + g * y_coeffs


def solve_coeff_space(w_coeffs, init_x_coeffs, b_coeffs_func, g, f_nonlin, t_max, t_vals=None, rtol=1e-7):
    n = int(len(np.shape(w_coeffs))/2)
    cheb_int_mat = chebychev_core.generate_cheb_integral_matrix(
        np.max(np.concatenate([w_coeffs.shape, init_x_coeffs.shape])) - 1)
    j_dims = np.shape(init_x_coeffs)
    k_dims = np.shape(w_coeffs)[n:]
    contractor_sx = chebychev_core.generate_s_x_contractor(k_dims, j_dims, cheb_int_mat)
    contractor_w_sx = chebychev_core.generate_w_sx_contractor(n)

    def integrand(t, y):
        y = np.transpose(y)
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, 0)
        this_x = np.reshape(y, (np.shape(y)[0], *init_x_coeffs.shape))
        b_coeffs = b_coeffs_func(t)
        this_rhs = _eval_rhs(w_coeffs, this_x, b_coeffs, f_nonlin, contractor_sx, contractor_w_sx, g)
        return np.reshape(this_rhs, (np.shape(y)[0], -1))

    soln = integrate.solve_ivp(integrand, [0, t_max], init_x_coeffs.flatten(), t_eval=t_vals, vectorized=True, rtol=rtol)
    return soln.t, np.reshape(np.transpose(soln.y), (-1, *np.shape(init_x_coeffs)))
