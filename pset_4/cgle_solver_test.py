import numpy as np

from pset_3 import chebychev
import cgle_solver


def test_rhs():
    N = 32
    h = 0.02
    c = 0.5
    a_grid = np.random.uniform(-1, 1, (1, N))
    rhs_grid = 1/h * a_grid - (1 + 1j * c) * np.abs(a_grid)**2 * a_grid
    rhs_cheb = chebychev.cheb(rhs_grid)
    rhs_cheb_test = cgle_solver.construct_rhs_cheb(chebychev.cheb(a_grid), h, c)
    np.testing.assert_allclose(rhs_cheb, rhs_cheb_test)
