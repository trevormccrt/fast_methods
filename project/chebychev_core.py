import numpy as np

from pset_3 import chebychev

def ncheb(x, axis_from=0):
    to_transform = np.arange(start=axis_from, stop=len(np.shape(x)), step=1)
    x_coeff = x
    for ax in to_transform:
        x_coeff = chebychev.cheb(x_coeff, axis=ax)
    return x_coeff

def nicheb(x, axis_from=0):
    to_transform = np.arange(start=axis_from, stop=len(np.shape(x)), step=1)
    x_coeff = x
    for ax in to_transform:
        x_coeff = chebychev.icheb(x_coeff, axis=ax)
    return x_coeff

def _num_to_chr(n):
    return chr(n + 97)

def _contract_mat_nd(mat, input):
    n = len(np.shape(input))
    mat_copies = [mat] * n
    chrs_in = [_num_to_chr(i) for i in range(n)]
    chrs_out = [_num_to_chr(i) for i in range(n, 2 * n)]
    first_sub = "".join(["{}{},".format(chr_out, chr_in) for chr_out, chr_in in zip(chrs_out, chrs_in)])
    second_sub = "".join(chrs_in)
    third_sub = "".join(chrs_out)
    sub = "{}{} -> {}".format(first_sub, second_sub, third_sub)
    return np.einsum(sub, *mat_copies, input)

def chebychev_kernel_integral(x_coeffs, w_coeffs, cheb_integral_matrix):
    n = len(np.shape(x_coeffs))
    num_k = np.shape(w_coeffs)[0]
    num_j = np.shape(w_coeffs)[-1]
    for this_n in range(n):
        pass

def generate_cheb_integral_matrix(m_max):
    grid = np.arange(start=0, stop=m_max + 1, step=1)
    grid_j, grid_k = np.meshgrid(grid, grid)
    first_part = grid_j + grid_k
    second_part = np.abs(grid_j - grid_k)
    return 1/2 * (np.nan_to_num((((-1) ** first_part) + 1)/(1 - first_part**2))+
                  np.nan_to_num((((-1) ** second_part) + 1)/(1 - second_part**2)))
