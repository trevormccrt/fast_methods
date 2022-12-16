import numpy as np
import torch
from torchdiffeq import odeint

from project import torch_chebychev_solver, chebychev_solver


def test_solve_consistency():
    g = 1
    batch_size = 5
    x_dims = (2 ** 5, 2 ** 7)
    w_dims = (*x_dims, 2**8, 2**4)
    w_coeffs = np.random.uniform(-1, 1, w_dims)
    init_x_coeffs = np.random.uniform(-1, 1, (batch_size, *x_dims))
    numpy_const_input = np.zeros(w_dims[:2])
    torch_const_input = torch.from_numpy(numpy_const_input)
    b_func_numpy = lambda x: numpy_const_input
    b_func_torch = lambda x: torch_const_input
    torch_dynamics = torch_chebychev_solver.ChebychevNFModel(torch.from_numpy(w_coeffs), b_func_torch, x_dims, g, nonlin=torch.tanh)
    t_max = 5
    soln_times = np.arange(start=0, stop=t_max, step=0.5)
    with torch.no_grad():
        torch_soln = odeint(torch_dynamics, torch.from_numpy(init_x_coeffs), torch.from_numpy(soln_times))
    scipy_solns = []
    for init_x in init_x_coeffs:
        scipy_soln_t, scipy_soln_y = chebychev_solver.solve_coeff_space(w_coeffs, init_x, b_func_numpy, g, np.tanh, t_max, soln_times)
        scipy_solns.append(scipy_soln_y)
    scipy_solns = np.array(scipy_solns)
    torch_soln = torch_soln.numpy()
    torch_soln = np.swapaxes(torch_soln, 0, 1)
    error = np.max(np.abs(torch_soln - scipy_solns), (2, 3))/np.max(np.abs(scipy_solns))
    np.testing.assert_allclose(error, np.zeros_like(error), atol=1e-3)
