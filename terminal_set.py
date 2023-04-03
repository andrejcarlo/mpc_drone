import numpy as np
import scipy.linalg as la


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
    for i in range(0, num_vecs - 1):
        for j in range(i + 1, num_vecs):
            # 4 points per combination of 2 eigvecs
            # (i+j), (i-j), (-i+j), (-i-j)

            # Direction * semi axis length
            eig1 = eigenvectors[:, i] * eigenvalues[i]  # np.sqrt(1 / eigenvalues[i])
            eig2 = eigenvectors[:, j] * eigenvalues[i]  # np.sqrt(1 / eigenvalues[j])

            vertices.append(eig1 + eig2)
            vertices.append(eig1 - eig2)
            vertices.append(-eig1 + eig2)
            vertices.append(-eig1 - eig2)

    vertices = np.array(vertices)
    vertices += x_target
    return vertices


####################################### check state constraints:

# constraints += [x[6:9, k] >= np.array([-math.pi, -math.pi / 2, -math.pi])]
# constraints += [x[6:9, k] <= np.array([math.pi, math.pi / 2, math.pi])]


def _check_state_c(
    ctrl,
    vertices,
    use_state_constraints=False,
):
    state_within_range = True
    for vertex in vertices:
        # state constraints
        # phi, thetha, psi
        if use_state_constraints and (
            vertex[6] < -np.pi
            or vertex[6] > np.pi
            or vertex[7] < -1 / 2 * np.pi
            or vertex[7] > 1 / 2 * np.pi
            or vertex[8] < -np.pi
            or vertex[8] > np.pi
        ):
            state_within_range = False  # set the boolean variable to False if a vector is outside the range
            break

        # print("Input constraint at vertex LHS ", ctrl.W_inv @ (K @ vertex))
        # print("Input constraint at vertex RHS ", ulb)
        # input constraints rpm
        if np.all((ctrl.W_inv @ (ctrl.K @ vertex)) < -(ctrl.W_inv @ ctrl.u_op)):
            state_within_range = False  # set the boolean variable to False if a vector is outside the range
            break
        # print("Satisfied")
    return state_within_range


def calculate_c(ctrl, P, x_target, c_init=2000):
    c = c_init
    vertices = calculate_vertices_bbox(c, ctrl.P, x_target)
    state_within_range = _check_state_c(ctrl, vertices)

    # continue to decrease c while checking if we satisfy the constraints
    while (not state_within_range) and (c > 1e-3):
        c = c / 1.01
        vertices = calculate_vertices_bbox(c, ctrl.P, x_target)
        state_within_range = _check_state_c(ctrl, vertices)

    return c
