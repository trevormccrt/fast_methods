import torch
from torchdiffeq import odeint

from project import torch_fourier_core


def _complex_to_stack(imag_tensor):
    return torch.stack([torch.real(imag_tensor), torch.imag(imag_tensor)], dim=-1)


def _stack_to_complex(stack_tensor):
    return stack_tensor[..., 0] + 1j * stack_tensor[..., 1]


def odeint_complex(func, y0, t, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None, integrator=odeint):
    stacked_y0 = _complex_to_stack(y0)
    result = integrator(func, stacked_y0, t, rtol=rtol, atol=atol, method=method, options=options, event_fn=event_fn)
    return _stack_to_complex(result)


class ConvNFModel(torch.nn.Module):
    def __init__(self, weight_coeffs, b_func, g, nonlin=torch.tanh):
        super().__init__()
        self.weight_coeffs = torch.nn.Parameter(weight_coeffs)
        self.b_func = b_func
        self.nonlin = nonlin
        self.g = torch.nn.Parameter(g)

    def forward(self, t, x_coeffs):
        x_coeffs = _stack_to_complex(x_coeffs)
        axis_from = x_coeffs.dim() - self.weight_coeffs.dim()
        b_coeffs = self.b_func(t)
        conv_int = torch_fourier_core.fft_convolution_integral(self.weight_coeffs, x_coeffs, axes_from=axis_from)
        b_grid = torch_fourier_core.nifft(b_coeffs)
        y_grid = self.nonlin(conv_int + b_grid)
        y_coeffs = torch_fourier_core.fourier_series_coeffs(y_grid, axes_from=axis_from)
        return _complex_to_stack(-x_coeffs + self.g * y_coeffs)
