import numpy as np
from mpc import MPCControl

np.set_printoptions(suppress=True)
import math

####################################### get P and K, get input constraints:

dt = 0.10  # Sampling period
N = 20  # MPC Horizon
T = 100  # Duration of simulation
x_init = np.zeros(12)  # Initial conditions
x_target = np.zeros(12)  # State to reach
x_target[0:3] = np.array([1.0, 1.0, 1.0])


ctrl = MPCControl(
    mpc_horizon=N,
    timestep_mpc_stages=dt,
    terminal_set_level_c=2000,
    use_terminal_set=True,
    use_terminal_cost=True,
)
P, K = ctrl._getPK()

W_inv, ulb = ctrl._get_ulb()

# print("K_inv",K_inv)
# print("ulb",ulb)


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


print("P is symmetric ", check_symmetric(P))

# Make P symmetrical:
# P1 = P
# for i in range(12):
#     for j in range(12):
#         P1[i][j] = 0.5 * (P[i][j] + P[j][i])
# P = P1
# for row in P:
#     print(row)


####################################### get vertex:
def calculate_vertex(c, P):
    R = (1 / (2 * c)) * P
    # Larger c larger ellipse

    # Compute eigenvalues and eigenvectors of R
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    vertex = []
    for i in range(eigenvectors.shape[1]):
        direction = eigenvectors[:, i]
        semi_axis_length = np.sqrt(1 / eigenvalues[i])

        vertex1 = x_target + direction * semi_axis_length
        vertex2 = x_target - direction * semi_axis_length

        vertex.append(vertex1)
        vertex.append(vertex2)

    vertex = np.array(vertex)
    return vertex


def calculate_vertices_bbox(c, P):
    # R = (1 / (2 * c)) * P
    # Larger c larger ellipse

    # Compute eigenvalues and eigenvectors of R
    eigenvalues, eigenvectors = np.linalg.eigh(P)
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


def _check_state_c(c, vertices):
    state_within_range = True
    for vertex in vertices:
        # state constraints
        # phi, thetha, psi
        # if (
        #     vertex[6] < -math.pi
        #     or vertex[6] > math.pi
        #     or vertex[7] < -1 / 2 * math.pi
        #     or vertex[7] > 1 / 2 * math.pi
        #     or vertex[8] < -math.pi
        #     or vertex[8] > math.pi
        # ):
        #     state_within_range = False  # set the boolean variable to False if a vector is outside the range
        #     break

        # print("Input constraint at vertex LHS ", ctrl.W_inv @ (K @ vertex))
        # print("Input constraint at vertex RHS ", ulb)
        # input constraints rpm
        if np.all((ctrl.W_inv @ (K @ vertex)) < ulb):
            state_within_range = False  # set the boolean variable to False if a vector is outside the range
            break
        # print("Satisfied")
    return state_within_range


def calculate_c():
    c = 2000
    vertices = calculate_vertices_bbox(c, P)
    state_within_range = _check_state_c(c, vertices)

    # continue to decrease c while checking if we satisfy the constraints
    while (not state_within_range) and (c > 1e-3):
        c = c / 1.01
        vertices = calculate_vertices_bbox(c, P)
        state_within_range = _check_state_c(c, vertices)

    return c


# check  = np.dot((vertex[0]-x_target).T, np.dot(P, vertex[0]-x_target))/2
# print(check)


# for vertex in vertices:
#     print(W_inv @ K @ vertex)


# print("ulb", ulb)
# input_within_range = True

# for i in range(24):
#     u = K_inv@(K@vertex[i]-x_target)
#     diff = u - ulb
#     if not all(d >= 0 for d in diff):
#         input_within_range = False
