import numpy as np
import torch

from project import chebychev_core, torch_chebychev_core


class ChebychevNFModel(torch.nn.Module):
    def __init__(self, weight_coeffs, b_func, g, nonlin=torch.tanh):
        super().__init__()
        self.weight_coeffs = torch.nn.Parameter(weight_coeffs)
        self.b_func = b_func
        self.nonlin = nonlin
        self.n = int(weight_coeffs.dim()/2)
        w_dims = weight_coeffs.size()
        x_dims = w_dims[:self.n]
        cheb_int_mat = torch.tensor(chebychev_core.generate_cheb_integral_matrix(np.max(w_dims) - 1), dtype=self.weight_coeffs.dtype)
        self.s_x_contractor = torch_chebychev_core.generate_s_x_contractor(w_dims[self.n:], x_dims, cheb_int_mat)
        self.w_sx_contractor = torch_chebychev_core.generate_w_sx_contractor(self.n)
        self.g = torch.nn.Parameter(g)

    def forward(self, t, x_coeffs):
        axis_from = x_coeffs.dim() - self.n
        b_coeffs = self.b_func(t)
        integral_coeffs = chebychev_core.chebychev_kernel_integral(x_coeffs, self.weight_coeffs, self.s_x_contractor, self.w_sx_contractor)
        integral_grid = torch_chebychev_core.nicheb(integral_coeffs, axis_from=axis_from)
        b_grid = torch_chebychev_core.nicheb(b_coeffs, axis_from=axis_from)
        y_grid = self.nonlin(integral_grid + b_grid)
        y_coeffs = torch_chebychev_core.ncheb(y_grid, axis_from=axis_from)
        return -x_coeffs + self.g * y_coeffs




