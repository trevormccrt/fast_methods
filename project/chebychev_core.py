import numpy as np
from scipy import fftpack


def extrema_grid(N):
    n = np.arange(start=0, stop=N, step=1)
    return np.cos(np.pi * n/(N-1))


def cheb(x, axis=-1):
    N = np.shape(x)[axis]
    cheb = 1/(N-1) * fftpack.dct(x, axis=axis, type=1)
    cheb = np.swapaxes(cheb, -1, axis)
    cheb[..., 0] = cheb[..., 0]/2
    cheb[..., -1] = cheb[..., -1] / 2
    cheb = np.swapaxes(cheb, -1, axis)
    return cheb


def icheb(x, axis=-1):
    x = np.swapaxes(x, -1, axis)
    x[..., 0] = x[..., 0] * 2
    x[..., -1] = x[..., -1] * 2
    x = np.swapaxes(x, -1, axis)
    return 1/2 * fftpack.idct(x, axis=axis, type=1)


def ncheb(x, axis_from=0):
    to_transform = np.arange(start=axis_from, stop=len(np.shape(x)), step=1)
    x_coeff = x
    for ax in to_transform:
        x_coeff = cheb(x_coeff, axis=ax)
    return x_coeff


def nicheb(x, axis_from=0):
    to_transform = np.arange(start=axis_from, stop=len(np.shape(x)), step=1)
    x_coeff = x
    for ax in to_transform:
        x_coeff = icheb(x_coeff, axis=ax)
    return x_coeff


def _num_to_chr(n):
    return chr(n + 97)


def generate_s_x_contractor(k_dims, j_dims, full_s):
    s_mats = []
    n = len(k_dims)
    for k_dim, x_dim in zip(k_dims, j_dims):
        s_mats.append(full_s[:k_dim, :x_dim])
    chrs_in = [_num_to_chr(i) for i in range(n)]
    chrs_out = [_num_to_chr(i) for i in range(n, 2 * n)]
    first_sub = "".join(["{}{},".format(chr_out, chr_in) for chr_out, chr_in in zip(chrs_out, chrs_in)])
    second_sub = "...".join(chrs_in)
    third_sub = "...".join(chrs_out)
    sub = "{}{} -> {}".format(first_sub, second_sub, third_sub)
    return lambda x: np.einsum(sub, *s_mats, x)


def generate_w_sx_contractor(n):
    chrs_in = [_num_to_chr(i) for i in range(n)]
    chrs_out = [_num_to_chr(i) for i in range(n, 2 * n)]
    first_sub = "".join(chrs_out + chrs_in)
    second_sub = "...".join(chrs_in)
    third_sub = "...".join(chrs_out)
    sub = "{},{} -> {}".format(first_sub, second_sub, third_sub)
    return lambda w, x: np.einsum(sub, w, x)


def chebychev_kernel_integral(x_coeffs, w_coeffs, cheb_integral_matrix):
    return _contract_half_nd(w_coeffs, _contract_mat_nd(cheb_integral_matrix, x_coeffs))


def generate_cheb_integral_matrix(m_max):
    grid = np.arange(start=0, stop=m_max + 1, step=1)
    grid_j, grid_k = np.meshgrid(grid, grid)
    first_part = grid_j + grid_k
    second_part = np.abs(grid_j - grid_k)
    return 1/2 * (np.nan_to_num((((-1) ** first_part) + 1)/(1 - first_part**2))+
                  np.nan_to_num((((-1) ** second_part) + 1)/(1 - second_part**2)))
