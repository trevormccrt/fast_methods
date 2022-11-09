import numpy as np
from scipy import linalg as sclinalg

from pset_3 import colocation_core, chebychev


def solve_update(grid, u_prev, n):
    grid_x = (grid + 1)/2
    diff_mat = colocation_core.extrema_diff_mat(grid)
    lhs_mat = 4 * np.diag(grid_x).dot(diff_mat).dot(diff_mat) + \
          4 * diff_mat + n * np.diag(grid_x * u_prev ** (n - 1))
    lhs_mat[0, :] = np.zeros_like(grid)
    lhs_mat[0, 0] = 1
    rhs = -1 * (2 * np.diag(grid + 1).dot(diff_mat).dot(diff_mat).dot(u_prev) + 4 * diff_mat.dot(u_prev) + (grid+1)/2 * u_prev**n)
    rhs[0] = 0
    return np.linalg.solve(lhs_mat, rhs)

