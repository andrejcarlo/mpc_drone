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
x_target[0:3] = np.array([10.0, 10.0, 10.0])


ctrl = MPCControl(
    mpc_horizon=N,
    timestep_mpc_stages=dt,
    terminal_set_level_c=2000,
    use_terminal_set=True,
    use_terminal_cost=True,
)
P, K = ctrl._getPK()

K_inv, ulb = ctrl._get_ulb()

# print("K_inv",K_inv)
# print("ulb",ulb)

# Make P symmetrical:
P1 = P
for i in range(12):
    for j in range(12):
        P1[i][j] = 0.5 * (P[i][j] + P[j][i])
P = P1
# for row in P:
#     print(row)

####################################### get vertex:
def calculate_vertex(c,P):
    R = (1 / (2 * c)) * P

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

####################################### check state constraints:



def _check_state_c(c, vertex):
    state_within_range = True  
    for i in range(24):
        if vertex[i][6] < -math.pi or vertex[i][6] > math.pi or \
            vertex[i][7] < -1/2*math.pi or vertex[i][7] > 1/2*math.pi or \
            vertex[i][8] < -math.pi or vertex[i][8] > math.pi:
            state_within_range = False  # set the boolean variable to False if a vector is outside the range
            break
    return state_within_range


c = 2000
vertex = calculate_vertex(c,P)
state_within_range = _check_state_c(c, vertex)
while not state_within_range:
    c = c/1.01
    vertex = calculate_vertex(c,P)
    state_within_range = _check_state_c(c, vertex)

print(c)



# check  = np.dot((vertex[0]-x_target).T, np.dot(P, vertex[0]-x_target))/2
# print(check)



for i in range(24):
    print(K_inv@K@vertex[i])


print("ulb",ulb)
input_within_range = True

# for i in range(24):
#     u = K_inv@(K@vertex[i]-x_target)
#     diff = u - ulb
#     if not all(d >= 0 for d in diff):
#         input_within_range = False

