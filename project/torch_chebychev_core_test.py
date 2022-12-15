import numpy as np
from scipy import fftpack
import torch

from project import torch_chebychev_core


def test_dct_equivalence():
    dims = 4
    trns_dim = np.random.randint(0, dims)
    x = np.random.uniform(-1, 1, [2**6] * dims)
    dct_1_scipy = fftpack.dct(x, type=1, axis=trns_dim)
    with torch.no_grad():
        dct_1_torch = torch_chebychev_core.torch_dct1(torch.from_numpy(x), axis=trns_dim)
    np.testing.assert_allclose(dct_1_scipy, dct_1_torch)
