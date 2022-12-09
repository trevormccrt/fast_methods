import chebychev_core


def _eval_rhs(w_coeffs, x_coeffs, b_coeffs, f_nonlin, cheb_integral_matrix, g):
    integral_coeffs = chebychev_core.chebychev_kernel_integral(x_coeffs, w_coeffs, cheb_integral_matrix)
    integral_grid = chebychev_core.nicheb(integral_coeffs, axis_from=1)
    b_grid = chebychev_core.nicheb(b_coeffs, axis_from=1)
    y_grid = f_nonlin(integral_grid + b_grid)
    y_coeffs = chebychev_core.ncheb(y_grid, axis_from=1)
    return -x_coeffs + g * y_coeffs


def solve_coeffs(w_coeffs, init_x_coeffs, b_coeffs_func, g):
