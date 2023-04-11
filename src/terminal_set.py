import numpy as np
import scipy.linalg as la
from itertools import product


def check_symmetric(P, rtol=1e-05, atol=1e-08):
    return np.allclose(P, P.T, rtol=rtol, atol=atol)


def calculate_vertices_bbox(c, P, x_target):
    # R = (1 / (2 * c)) * P
    # Larger c larger ellipse

    # Compute eigenvalues and eigenvectors of R
    eigenvalues, eigenvectors = la.eig(P)
    eigenvalues = np.real(eigenvalues) * c

    num_vecs, _ = eigenvectors.shape

    vertices = []
    for signs in product([-1, 1], repeat=num_vecs):
        vertex = np.sum(signs * eigenvalues * eigenvectors, axis=0)
        vertices.append(vertex)

    vertices = np.array(vertices)
    vertices += x_target
    return vertices


def _check_state_c(
    ctrl,
    vertices,
):
    state_within_range = True
    for vertex in vertices:
        # state constraints
        if np.any(ctrl.xmin > vertex) or np.any(ctrl.xmax < vertex):
            state_within_range = False  # set the boolean variable to False if a vector is outside the range
            break

        # input constraints
        if np.any(ctrl.umin > (ctrl.K @ vertex)) or np.any(
            ctrl.umax < (ctrl.K @ vertex)
        ):
            state_within_range = False  # set the boolean variable to False if a vector is outside the range
            break
        # print("Satisfied")
    return state_within_range


def calculate_c(ctrl, x_target, c_init=2.5):
    c = c_init
    vertices = calculate_vertices_bbox(c, ctrl.P, x_target)
    state_within_range = _check_state_c(ctrl, vertices)

    # continue to decrease c while checking if we satisfy the constraints
    while (not state_within_range) and (c > 1e-3):
        c = c / 1.01
        vertices = calculate_vertices_bbox(c, ctrl.P, x_target)
        state_within_range = _check_state_c(ctrl, vertices)
    return c
