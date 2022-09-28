import numpy as np

from pset_1 import fft, util


def test_fft():
    N = 2**10
    M = 10
    x = util.random_complex(M, N)
    my_fft = fft.apply_FFT(x)
    numpy_fft = np.fft.fft(x, axis=-1)
    np.testing.assert_allclose(my_fft, numpy_fft)


test_fft()