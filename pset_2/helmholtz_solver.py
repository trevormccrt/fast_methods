import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

import fd_core, dst


def construct_direct_fd_const_coeff(grid_size, grid_spacing, k_sq):
    return fd_core.homogenous_dirichlet_fd_2d(grid_size, grid_spacing) \
           + sparse.diags([np.ones(grid_size * grid_size) * k_sq], [0])


def construct_direct_fd_variable_k(grid_spacing, k_sq):
    grid_size = np.shape(k_sq)[0]
    return fd_core.homogenous_dirichlet_fd_2d(grid_size, grid_spacing) \
           + sparse.diags([k_sq.flatten()], [0])


def solve_direct_fd_const_coeff(grid_spacing, k_sq, v):
    grid_size = np.shape(v)[0]
    return np.reshape(splinalg.spsolve(construct_direct_fd_const_coeff(grid_size, grid_spacing, k_sq), v.flatten()), v.shape)


def solve_direct_fd_variable_k(grid_spacing, k_sq, v):
    return np.reshape(splinalg.spsolve(construct_direct_fd_variable_k(grid_spacing, k_sq), v.flatten()), v.shape)


def run_iteration_variable_k(grid_spacing, k_sq, k_sq_0, v, u_prev, alpha=1):
    rhs = v - alpha * (k_sq - k_sq_0) * u_prev
    return solve_precond_const_coeff(grid_spacing, k_sq_0, rhs)


def solve_precond_const_coeff(grid_spacing, k_sq, v):
    N = np.shape(v)[0] + 1
    v_freq = dst.dst_1(dst.dst_1(v, axis=1), axis=0).flatten()
    k_vals = np.arange(start=1, stop=N, step=1)
    lambda_vals = -1 * (2 - 2 * np.cos(np.pi * k_vals/N))
    lhs = 1/grid_spacing**2 * (np.kron(lambda_vals, np.ones(np.shape(v)[0])) +
                               np.kron(np.ones(np.shape(v)[0]), lambda_vals)) + k_sq
    soln_freq = v_freq/lhs
    return dst.idst_1(dst.idst_1(np.reshape(soln_freq, np.shape(v)), axis=1), axis=0)


def spectral_solver(N, k_sq, v):
    k_vals = np.arange(start=0, stop=N, step=1)
    lhs = np.pi**2 * (np.kron(-(k_vals + 1)**2, np.ones(N)) +
                               np.kron(np.ones(N), -(k_vals + 1)**2)) + k_sq
    v_freq = dst.dst_1_physical(dst.dst_1_physical(v, axis=1), axis=0)
    red_v_freq = v_freq[:N, :N]
    soln_freq = np.reshape(red_v_freq.flatten() / lhs, red_v_freq.shape)
    return dst.idst_1_physical(dst.idst_1_physical(soln_freq, axis=1), axis=0)
