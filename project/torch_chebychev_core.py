import torch

from project import chebychev_core


def _dct1_rfft_impl(x):
    return torch.view_as_real(torch.fft.rfft(x, dim=1))


def _dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))


def _idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)


def _dct1(x):
    x_shape = x.shape
    x = torch.reshape(x, (-1, x_shape[-1]))
    x = torch.cat([x, x.flip([1])[:, 1:-1]], dim=1)
    return _dct1_rfft_impl(x)[:, :, 0].view(*x_shape)


def extrema_grid(N):
    n = torch.arange(start=0, end=N, step=1)
    return torch.cos(torch.pi * n/(N-1))


def torch_dct1(x, axis=-1):
    x = torch.swapaxes(x, axis, -1)
    dct_x = _dct1(x)
    return torch.swapaxes(dct_x, axis, -1)


def cheb(x, axis=-1):
    N = x.size(axis)
    x = 1/(N-1) * torch_dct1(x, axis=axis)
    x = torch.swapaxes(x, -1, axis)
    x[..., 0] = x[..., 0]/2
    x[..., -1] = x[..., -1] / 2
    x = torch.swapaxes(x, -1, axis)
    return x


def icheb(x, axis=-1):
    x = torch.swapaxes(torch.clone(x), -1, axis)
    x[..., 0] *= 2
    x[..., -1] *= 2
    x = torch.swapaxes(x, -1, axis)
    return 1/2 * torch_dct1(x, axis=axis)


def ncheb(x, axis_from=0):
    to_transform = torch.arange(start=axis_from, end=x.dim(), step=1)
    x_coeff = x
    for ax in to_transform:
        x_coeff = cheb(x_coeff, axis=ax)
    return x_coeff


def nicheb(x, axis_from=0):
    to_transform = torch.arange(start=axis_from, end=x.dim(), step=1)
    x_coeff = x
    for ax in to_transform:
        x_coeff = icheb(x_coeff, axis=ax)
    return x_coeff


def generate_s_x_contractor(k_dims, j_dims, full_s):
    sub, s_mats = chebychev_core._sx_subscripts_mats(k_dims, j_dims, full_s)

    def contractor(x):
        return torch.einsum(sub, *s_mats, x)
    return contractor


def generate_w_sx_contractor(n):
    sub = chebychev_core._w_sx_subs(n)

    def contractor(w, x):
        return torch.einsum(sub, w, x)
    return contractor
