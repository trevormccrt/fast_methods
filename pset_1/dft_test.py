import numpy as np
import scipy

from pset_1 import dft, util


def test_dft_matrix():
    N = np.random.randint(10, 100)
    my_mat = dft.DFT_matrix(N)
    scipy_mat = scipy.linalg.dft(N, None)
    np.testing.assert_allclose(my_mat, scipy_mat)


def test_dft():
    N = 512
    M = 1
    x = util.random_complex(M, N)
    my_dft = dft.apply_DFT(x)
    numpy_dft = np.fft.fft(x, axis=1)
    np.testing.assert_allclose(my_dft, numpy_dft)
