import numpy as np


def gaussian_2d(x_grid, y_grid, center_x, center_y, sigma):
    return 1/np.sqrt((2 * np.pi)**2 * sigma**4) * np.exp(-1/2 * ((x_grid - center_x)/sigma)**2) * np.exp(-1/2 * ((y_grid - center_y)/sigma)**2)
