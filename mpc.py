import math
import cvxpy as cp
import numpy as np
from math import sin, cos, tan
from scipy import linalg as la
import control as ct

from drone import drone_cf2x, drone_m_islam
import control
from scipy.linalg import solve_sylvester


class MPCControl:
    def __init__(
        self,
        mpc_horizon=3,
        timestep_mpc_stages=0.25,
        terminal_set_level_c=10,
        use_terminal_set=True,
        use_terminal_cost=True,
    ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.
        N : optimization horizon

        """
        self.t_s = timestep_mpc_stages  # time step per stage
        self.N = mpc_horizon
        self.c = terminal_set_level_c
        self.use_terminal_cost = use_terminal_cost
        self.use_terminal_set = use_terminal_set
        self._buildModelMatrices()
        self._buildMPCProblem()

    def _buildModelMatrices(self):
        drone = drone_m_islam  # drone_cf2x

        I_x = drone.I_x
        I_y = drone.I_y
        I_z = drone.I_z
        l = drone.l
        I_r = drone.I_r
        k_f = drone.k_f
        k_m = drone.k_m
        m = drone.m
        g = drone.g
        k_tx = drone.k_tx
        k_ty = drone.k_ty
        k_tz = drone.k_tz
        k_rx = drone.k_rx
        k_ry = drone.k_rx
        k_rz = drone.k_rx
        w_r = drone.w_r

        # matrix to convert inputs (=forces) to rpm^2
        # rpm^2 = W * u
        self.W_inv = np.array(
            [
                [1 / (4 * k_f), 0, 1 / (2 * k_f), 1 / (4 * k_m)],
                [1 / (4 * k_f), -1 / (2 * k_f), 0, -1 / (4 * k_m)],
                [1 / (4 * k_f), 0, -1 / (2 * k_f), 1 / (4 * k_m)],
                [1 / (4 * k_f), 1 / (2 * k_f), 0, -1 / (4 * k_m)],
            ]
        )

        # operating point for linearization
        self.x_op = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0])
        self.hover_rpm = np.full(4, math.sqrt(m * g / (4 * k_f)))
        self.u_op = np.matmul(np.linalg.inv(self.W_inv), np.square(self.hover_rpm))

        #      0, 1, 2, 3,     4,     5,     6,   7,     8,   9, 10,11
        # x = [x, y, z, x_dot, y_dot, z_dot, phi, theta, psi, p, q, r]
        # u = [u1, u2, u3, u4]
        x = self.x_op
        u = self.u_op

        # fmt: off

        dx_ddot_dphi = -u[0]/m * (cos(x[6])*sin(x[8]) - sin(x[6])*cos(x[8])* sin(x[7]))
        dx_ddot_dtheta = -u[0]/m * (sin(x[6])*sin(x[8]) + cos(x[6])*cos(x[8])* cos(x[7]))
        dx_ddot_dpsi = -u[0]/m * (sin(x[6])*cos(x[8]) - cos(x[6])*sin(x[8])* cos(x[7]))
        
        dy_ddot_dphi = -u[0]/m * (cos(x[6])*cos(x[8]) + sin(x[6])*sin(x[8])* sin(x[7]))
        dy_ddot_dtheta = -u[0]/m * (sin(x[6])*cos(x[8]) - cos(x[6])*sin(x[8])* cos(x[7]))
        dy_ddot_dpsi = -u[0]/m * (-sin(x[6])*sin(x[8]) - cos(x[6])*cos(x[8])* sin(x[7]))

        dz_ddot_dphi = u[0]/m * sin(x[6]) * cos(x[7])
        dz_ddot_dtheta = u[0]/m * cos(x[6]) * sin(x[7])

        dphi_dot_dphi = -x[11]* sin(x[6]) * tan(x[7]) + x[10] * cos(x[6]) * tan(x[7])
        dphi_dot_dtheta = x[11]* cos(x[6]) * 1/(cos(x[7]))**2 + x[10] * sin(x[6]) * 1/(cos(x[7]))**2
        dphi_dot_dq = sin(x[6])*tan(x[7])
        dphi_dot_dr = cos(x[6])*tan(x[7])

        dtheta_dot_dphi = -x[10]*sin(x[6]) - x[11]*cos(x[6])
        dtheta_dot_dq = cos(x[7])
        dtheta_dot_dr = -sin(x[6])

        dpsi_dot_dphi = 1/cos(x[7]) * cos(x[6]) * x[10] - sin(x[6])/cos(x[7]) * x[11]
        dpsi_dot_dtheta = x[10] * sin(x[6]) * 1/(cos(x[7]))**2 * sin(x[7]) + x[11] * cos(x[6]) * sin(x[7])/cos(x[7])**2
        dpsi_dot_dq = sin(x[6])/cos(x[7])
        dpsi_dot_dr = cos(x[6])/cos(x[7])
        
        dp_dot_dp = - k_rx/I_x
        dp_dot_dq = - I_r/I_x * w_r + (I_y - I_z)/I_x * x[11]
        dp_dot_dr = (I_y - I_z)/I_x * x[10]

        dq_dot_dp = - I_r/I_y * w_r + (I_z - I_x)/I_y * x[11]
        dq_dot_dq = + k_ry/I_y
        dq_dot_dr = (I_z - I_x)/I_y * x[9]

        dr_dot_dp = + (I_y - I_x)/I_z * x[10]
        dr_dot_dq = + (I_y - I_x)/I_z * x[9]
        dr_dot_dr = + k_rz/I_z


        self.A = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, -k_tx/m, 0, 0, dx_ddot_dphi, dx_ddot_dtheta, dx_ddot_dpsi, 0, 0, 0],
                           [0, 0, 0, 0, -k_ty/m, 0, dy_ddot_dphi, dy_ddot_dtheta, dy_ddot_dpsi, 0, 0, 0],
                           [0, 0, 0, 0, 0, -k_tz/m, dz_ddot_dphi, dz_ddot_dtheta, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, dphi_dot_dphi, dphi_dot_dtheta, 0, 1, dphi_dot_dq, dphi_dot_dr],
                           [0, 0, 0, 0, 0, 0, dtheta_dot_dphi, 0, 0, 0, dtheta_dot_dq, dtheta_dot_dr],
                           [0, 0, 0, 0, 0, 0, dpsi_dot_dphi, dpsi_dot_dtheta, 0, 0, dpsi_dot_dq, dpsi_dot_dr],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, dp_dot_dp, dp_dot_dq, dp_dot_dr],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, dq_dot_dp, dq_dot_dq, dq_dot_dr],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, dr_dot_dp, dr_dot_dq, dr_dot_dr]])
        

        self.B = np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [-(math.sin(x[6])*math.sin(x[8]) + math.cos(x[6])
                             * math.cos(x[8])*math.sin(x[7]))/m, 0, 0, 0],
                           [-(math.sin(x[6])*math.cos(x[8]) - math.cos(x[6])
                              * math.sin(x[8])*math.sin(x[7]))/m, 0, 0, 0],
                           [-(math.cos(x[6])*math.cos(x[7]))/m, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, l/I_x, 0, 0],
                           [0, 0, -l/I_y, 0],
                           [0, 0, 0, -l/I_z]])
        
        self.A = np.array([[0,0,0,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,-k_tx/m,0,0,(u[0]*(math.cos(x[6])*math.sin(x[8]) - math.cos(x[8])*math.sin(x[6])*math.sin(x[7])))/m,(u[0]*math.cos(x[6])*math.cos(x[7])*math.cos(x[8]))/m,(u[0]*(math.cos(x[8])*math.sin(x[6]) - math.cos(x[6])*math.sin(x[7])*math.sin(x[8])))/m,0,0,0],
        [0,0,0,0,-k_ty/m,0,-(u[0]*(math.cos(x[6])*math.cos(x[8]) + math.sin(x[6])*math.sin(x[7])*math.sin(x[8])))/m,(u[0]*math.cos(x[6])*math.cos(x[7])*math.sin(x[8]))/m,(u[0]*(math.sin(x[6])*math.sin(x[8]) + math.cos(x[6])*math.cos(x[8])*math.sin(x[7])))/m,0,0,0],
        [0,0,0,0,0,-k_tz/m,-(u[0]*math.cos(x[7])*math.sin(x[6]))/m,-(u[0]*math.cos(x[6])*math.sin(x[7]))/m,0,0,0,0],
        [0,0,0,0,0,0,x[10]*math.cos(x[6])*math.tan(x[7]) - x[11]*math.sin(x[6])*math.tan(x[7]),x[11]*math.cos(x[6])*(math.tan(x[7])**2 + 1) + x[10]*math.sin(x[6])*(math.tan(x[7])**2 + 1),0,1,math.sin(x[6])*math.tan(x[7]),math.cos(x[6])*math.tan(x[7])],
        [0,0,0,0,0,0,- x[11]*math.cos(x[6]) - x[10]*math.sin(x[6]),0,0,0,math.cos(x[6]),-math.sin(x[6])],
        [0,0,0,0,0,0,(x[10]*math.cos(x[6]))/math.cos(x[7]) - (x[11]*math.sin(x[6]))/math.cos(x[7]),(x[11]*math.cos(x[6])*math.sin(x[7]))/math.cos(x[7])**2 + (x[10]*math.sin(x[6])*math.sin(x[7]))/math.cos(x[7])**2,0,0,math.sin(x[6])/math.cos(x[7]),math.cos(x[6])/math.cos(x[7])],
        [0,0,0,0,0,0,0,0,0,-k_rx/I_x,-(I_r*w_r - I_y*x[11] + I_z*x[11])/I_x,(I_y*x[10] - I_z*x[10])/I_x],
        [0,0,0,0,0,0,0,0,0,-(I_r*w_r + I_x*x[11] - I_z*x[11])/I_y,-k_ry/I_y,-(I_x*x[9] - I_z*x[9])/I_y],
        [0,0,0,0,0,0,0,0,0,(I_x*x[10] - I_y*x[10])/I_z,(I_x*x[9] - I_y*x[9])/I_z,-k_rz/I_z]])
        self.B = np.array([[0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [(math.sin(x[6])*math.sin(x[8]) + math.cos(x[6])*math.cos(x[8])*math.sin(x[7]))/m,0,0,0],
        [-(math.cos(x[8])*math.sin(x[6]) - math.cos(x[6])*math.sin(x[7])*math.sin(x[8]))/m,0,0,0],
        [(math.cos(x[6])*math.cos(x[7]))/m,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,-l/I_x,0,0],
        [0,0,-l/I_y,0],
        [0,0,0,-l/I_z]])

        # fmt: on

        # time discretize system
        self.A = self.t_s * self.A + np.identity(12)
        self.B = self.t_s * self.B

        # state cost
        self.Q = np.diag([1, 1, 1, 1, 1, 1, 0.001, 0.001, 0.001, 0.05, 0.05, 0.05])
        # self.Q = 0.1 * np.identity(12)
        # input cost
        self.R = 0.01 * np.identity(4)

        # Solve discrete algebraic ricatii eq
        # P = dare solution
        # L = closed loop eigenvalues L
        # K = state-feedback gain
        self.P, self.L, self.K = ct.dare(self.A, self.B, self.Q, self.R)

        # Solve Lyapunov P:
        # P,L,K = control.dare( self.A, self.B, self.Q, self.R, S=None, E=None)
        # Ak =  self.A + self.B@K
        # Qk =  self.Q + K.transpose()@ self.R@K
        # Ak_inv = np.linalg.inv(Ak)
        # Ak_T = np.transpose(Ak)
        # C = -2 * np.matmul(Qk, Ak_inv)
        # self.P_l = solve_sylvester(Ak_T, Ak_inv, C)

    def _getPK(self):
        P, L, K = control.dare(self.A, self.B, self.Q, self.R, S=None, E=None)
        return P, K

    def _get_ulb(self):
        return self.W_inv, -np.matmul(self.W_inv, self.u_op)

    def _buildMPCProblem(self):
        cost = 0.0
        constraints = []

        # Parameters
        x_ref = cp.Parameter((12), name="x_ref")
        x_init = cp.Parameter((12), name="x_init")

        # Create the optimization variables
        x = cp.Variable((12, self.N + 1), name="x")
        u = cp.Variable((4, self.N), name="u")

        # For each stage in k = 0, ..., N-1
        for k in range(self.N):
            x_e = x[:, k] - x_ref

            # Cost
            cost += cp.quad_form(x_e, self.Q)
            cost += cp.quad_form(u[:, k], self.R)

            # System dynamics
            constraints += [x[:, k + 1] == self.A @ x[:, k] + self.B @ u[:, k]]

            # Constraints
            # constraints += [x[6:9, k] >= np.array([-math.pi, -math.pi / 2, -math.pi])]
            # constraints += [x[6:9, k] <= np.array([math.pi, math.pi / 2, math.pi])]
            # constraints += [self.W_inv @ u[:, k] >= -np.matmul(self.W_inv, self.u_op)]

        # terminal cost addition (estimate cost N->inf)
        Vf = cp.quad_form(x[:, self.N] - x_ref, self.P)
        if self.use_terminal_cost:
            cost += Vf

        # terminal set constraint
        if self.use_terminal_set:
            constraints += [Vf <= self.c]

        # Inital condition
        constraints += [x[:, 0] == x_init]

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def _computeRPMfromInputs(self, u_delta):
        """
        Computes the rpm given the small-signal u_delta around the operating point.
        """
        return np.sqrt(np.matmul(self.W_inv, self.u_op + u_delta))

    def computeControl(self, x_init, x_target):
        self.problem.param_dict["x_init"].value = x_init
        self.problem.param_dict["x_ref"].value = x_target

        self.problem.solve(solver=cp.GUROBI, verbose=False)
        if not (
            self.problem.status == "optimal"
            or self.problem.status == "inaccurate optimal"
        ):
            raise RuntimeError(
                f"MPC solver did not find a solution, due to it being {self.problem.status}"
            )

        # We return the MPC input and the next state (and also the plan for visualization)
        return (
            self.problem.var_dict["u"][:, 0].value,
            self.problem.var_dict["x"][:, 1].value,
            self.problem.var_dict["x"][:, :].value,
        )
