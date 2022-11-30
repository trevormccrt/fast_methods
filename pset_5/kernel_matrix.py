import numpy as np


def newtonian_kernel_matrix(target_points, source_points):
    target_tiled = np.tile(np.expand_dims(target_points, 1), (1, len(source_points), 1))
    source_tiled = np.tile(np.expand_dims(source_points, 0), (len(target_points), 1, 1))
    return 1/np.sqrt(np.sum(np.square(target_tiled - source_tiled), axis=-1))


def random_points_in_cube(xlims, ylims, zlims, n_points):
    x_vals = np.random.uniform(*xlims, n_points)
    y_vals = np.random.uniform(*ylims, n_points)
    z_vals = np.random.uniform(*zlims, n_points)
    return np.stack([x_vals, y_vals, z_vals], axis=-1)



target_points = random_points_in_cube((0, 1), (0, 1), (0, 1), 20)

zeta = 10
source_points = random_points_in_cube((zeta, zeta + 1), (0, 1), (0, 1), 20)

kernel_mat = newtonian_kernel_matrix(target_points, source_points)
print("")
