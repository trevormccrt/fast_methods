import numpy as np
import torch
from torchdiffeq import odeint

from project import torch_conv_solver, conv_solver


def test_solve_consistency():
    g = 1.0
    batch_size = 5
    x_dims = (2 ** 5, 2 ** 7)
    w_coeffs = np.random.uniform(-1, 1, x_dims) + 1j * np.random.uniform(-1, 1, x_dims)
    init_x_coeffs = np.random.uniform(-1, 1, (batch_size, *x_dims)) + 1j * np.random.uniform(-1, 1, x_dims)
    numpy_const_input = np.zeros(x_dims)
    torch_const_input = torch.from_numpy(numpy_const_input)
    b_func_numpy = lambda x: numpy_const_input
    b_func_torch = lambda x: torch_const_input
    torch_dynamics = torch_conv_solver.ConvNFModel(torch.from_numpy(w_coeffs), b_func_torch, torch.tensor(g), nonlin=torch.tanh)
    t_max = 1
    soln_times = np.arange(start=0, stop=t_max, step=0.1)
    with torch.no_grad():
        torch_soln = torch_conv_solver.odeint_complex(torch_dynamics, torch.from_numpy(init_x_coeffs), torch.from_numpy(soln_times), rtol=1e-8)
    scipy_solns = []
    for init_x in init_x_coeffs:
        scipy_soln_t, scipy_soln_y = conv_solver.solve_coeff_space(w_coeffs, init_x, b_func_numpy, g, np.tanh, t_max, soln_times, rtol=1e-8)
        scipy_solns.append(scipy_soln_y)
    scipy_solns = np.array(scipy_solns)
    torch_soln = torch_soln.numpy()
    torch_soln = np.swapaxes(torch_soln, 0, 1)
    error = np.max(np.abs(torch_soln - scipy_solns), (2, 3))/np.max(np.abs(scipy_solns))
    assert np.mean(error) < 1e-4
