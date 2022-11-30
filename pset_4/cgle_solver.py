import numpy as np
from scipy.sparse import linalg as splinalg

from pset_3 import chebychev
import ultraspherical_core


def construct_lhs(h, b, L, N):
    raw_lhs = (1/h - 1) * ultraspherical_core.conversion_matrix(1,N).dot(ultraspherical_core.conversion_matrix(0, N))\
              - (1 + 1j * b) * (2/L)**2 * ultraspherical_core.differentiation_matrix(2, N)
    idx = np.arange(start=0, stop=N, step=1)
    raw_lhs[-2, :] = (-1) ** idx
    raw_lhs[-1, :] = 1
    return raw_lhs


def construct_rhs_cheb(a_cheb, h, c):
    a_grid = chebychev.icheb(a_cheb)
    rhs_grid = 1/h * a_grid - (1 + 1j * c) * np.abs(a_grid)**2 * a_grid
    rhs_cheb = chebychev.cheb(rhs_grid)
    return rhs_cheb


def ultraspherical_rhs_generator(N):
    conv_mat_0 = ultraspherical_core.conversion_matrix(0, N)
    conv_mat_1 = ultraspherical_core.conversion_matrix(1, N)

    def construct_rhs_ultra(a_cheb, h, c):
        rhs_cheb = construct_rhs_cheb(a_cheb, h, c)
        converted = conv_mat_1.dot(conv_mat_0.dot(rhs_cheb))
        converted[..., -2:] = 0
        return converted

    return construct_rhs_ultra


def solve_iteratively(n_iter, N, h, b, c, L, a_0_grid, save_every=10):
    lhs = construct_lhs(h, b, L, N)
    lhs_lu = splinalg.splu(lhs.T)
    rhs_generator = ultraspherical_rhs_generator(N)
    a_0_cheb = chebychev.cheb(a_0_grid)
    solns = []
    this_soln_cheb = a_0_cheb
    for i in range(n_iter):
        this_soln_cheb = lhs_lu.solve(rhs_generator(this_soln_cheb, h, c), trans="T")
        if not i % save_every:
            this_soln_grid = chebychev.icheb(this_soln_cheb)
            solns.append(this_soln_grid)
            print(i)
    return np.array(solns)
