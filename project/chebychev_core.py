import numpy as np


def generate_cheb_integral_matrix(m_max):
    grid = np.arange(start=0, stop=m_max + 1, step=1)
    grid_j, grid_k = np.meshgrid(grid, grid)
    first_part = grid_j + grid_k
    second_part = np.abs(grid_j - grid_k)


generate_cheb_integral_matrix(10)
