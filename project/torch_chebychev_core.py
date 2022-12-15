import torch
import torch_dct


def extrema_grid(N):
    n = torch.arange(start=0, end=N, step=1)
    return torch.cos(torch.pi * n/(N-1))


def torch_dct1(x, axis=-1):
    x = torch.swapaxes(x, axis, -1)
    dct_x = torch_dct.dct1(x)
    return torch.swapaxes(dct_x, axis, -1)

# def cheb(x, axis=-1):
#     N = x.size(axis)
#     cheb = 1/(N-1) * fftpack.dct(x, axis=axis, type=1)
#     cheb = np.swapaxes(cheb, -1, axis)
#     cheb[..., 0] = cheb[..., 0]/2
#     cheb[..., -1] = cheb[..., -1] / 2
#     cheb = np.swapaxes(cheb, -1, axis)
#     return cheb
