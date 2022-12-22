import numpy as np
from scipy import fftpack
import torch

from project import torch_chebychev_core, chebychev_core


def test_transform_equivalence():
    dims = 4
    trns_dim = np.random.randint(0, dims)
    x = np.random.uniform(-1, 1, [2 ** 6] * dims)
    dct_1_scipy = fftpack.dct(x, type=1, axis=trns_dim)
    idct_1_scipy = fftpack.idct(x, type=1, axis=trns_dim)
    cheb_scipy = chebychev_core.cheb(x, trns_dim)
    icheb_scipy = chebychev_core.icheb(x, trns_dim)
    ncheb_scipy = chebychev_core.ncheb(x, trns_dim)
    nicheb_scipy = chebychev_core.nicheb(x, trns_dim)
    with torch.no_grad():
        dct_1_torch = torch_chebychev_core.torch_dct1(torch.from_numpy(x), axis=trns_dim)
        idct_1_torch = torch_chebychev_core.torch_dct1(torch.from_numpy(x), axis=trns_dim)
        cheb_torch = torch_chebychev_core.cheb(torch.from_numpy(x), axis=trns_dim)
        icheb_torch = torch_chebychev_core.icheb(torch.from_numpy(x), trns_dim)
        ncheb_torch = torch_chebychev_core.ncheb(torch.from_numpy(x), trns_dim)
        nicheb_torch = torch_chebychev_core.nicheb(torch.from_numpy(x), trns_dim)
    np.testing.assert_allclose(dct_1_scipy, dct_1_torch, atol=1e-12)
    np.testing.assert_allclose(idct_1_scipy, idct_1_torch, atol=1e-12)
    np.testing.assert_allclose(cheb_scipy, cheb_torch, atol=1e-12)
    np.testing.assert_allclose(icheb_scipy, icheb_torch, atol=1e-12)
    np.testing.assert_allclose(ncheb_scipy, ncheb_torch, atol=1e-12)
    np.testing.assert_allclose(nicheb_scipy, nicheb_torch, atol=1e-12)


def test_s_x_contractor_equivalence():
    dim_k1 = 12
    dim_k2 = 18
    dim_j1 = 5
    dim_j2 = 20
    mat = np.random.uniform(-1, 1, (30, 30))
    vec_2d = np.random.uniform(-1, 1, (dim_j1, dim_j2))
    contractor_2d_numpy = chebychev_core.generate_s_x_contractor([dim_k1, dim_k2], [dim_j1, dim_j2], mat)
    contractor_2d_torch = torch_chebychev_core.generate_s_x_contractor([dim_k1, dim_k2], [dim_j1, dim_j2], torch.from_numpy(mat))
    result_numpy = contractor_2d_numpy(vec_2d)
    with torch.no_grad():
        result_torch = contractor_2d_torch(torch.from_numpy(vec_2d))
    np.testing.assert_allclose(result_numpy, result_torch)


def test_w_sx_contractor_equivalence():
    dim = 10
    mat_2d = np.random.uniform(-1, 1, (dim, dim, dim, dim))
    vec_2d = np.random.uniform(-1, 1, (dim, dim))
    contractor_2d_numpy = chebychev_core.generate_w_sx_contractor(2)
    contractor_2d_torch = torch_chebychev_core.generate_w_sx_contractor(2)
    result_numpy = contractor_2d_numpy(mat_2d, vec_2d)
    result_torch = contractor_2d_torch(torch.from_numpy(mat_2d), torch.from_numpy(vec_2d))
    np.testing.assert_allclose(result_numpy, result_torch)
