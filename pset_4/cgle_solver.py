import numpy as np
import ultraspherical_core


def construct_lhs(h, b, L, N):
    raw_lhs = (1/h - 1) * ultraspherical_core.conversion_matrix(1,N).dot(ultraspherical_core.conversion_matrix(0, N))\
              - (1 + 1j * b) * (2/L)**2 * ultraspherical_core.differentiation_matrix(2, N)
    idx = np.arange(start=0, stop=N, step=1)
    raw_lhs[-2, :] = -1 ** idx
    raw_lhs[-1, :] = 1
    return raw_lhs
