import math
import cvxpy as cp
import numpy as np
from math import sin, cos, tan
from scipy import linalg as la
import control as ct

from drone import drone_cf2x, drone_m_islam
import control
from scipy.linalg import solve_sylvester


class Controller:
    def __init__(
        self,
        mpc_horizon=3,
        timestep_mpc_stages=0.25,
        solver=cp.GUROBI,
        control_type="mpc",
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
        self.dt = timestep_mpc_stages  # time step per stage
        self.N = mpc_horizon

        # terminal set parameters
        self._c_level = None
        self._beta = 0.0

        # type of controller
        self.control_type = control_type

        # type of solver CVXPY should use
        self.solver = solver

        self._buildModelMatrices()
        if self.control_type == "mpc":
            self._buildOTSProblem()
            self._buildMPCProblem()

    @property
    def c_level(self):
        return self._c_level

    @c_level.setter
    def c_level(self, value: float):
        self._c_level = value
        # rebuild mpc problem with new constraints
        self._buildMPCProblem()

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value: float):
        self._beta = value
        # rebuild mpc problem with new constraints
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
        self.d = np.zeros(3)

        #      0, 1, 2, 3,     4,     5,     6,   7,     8,   9, 10,11
        # x = [x, y, z, x_dot, y_dot, z_dot, phi, theta, psi, p, q, r]
        # u = [u1, u2, u3, u4]
        x = self.x_op
        u = self.u_op

        # Bounds on inputs
        # According to C. Kanellakis, S. S. Mansouri and G. Nikolakopoulos, "Dynamic visual sensing based on MPC controlled UAVs," 2017 25th Mediterranean Conference on Control and Automation (MED), Valletta, Malta, 2017, pp. 1201-1206, doi: 10.1109/MED.2017.7984281.
        self.umin = np.array([-45.0720, -11.2680, -11.2680, -0.54])
        self.umax = np.array([45.0720, 11.2680, 11.2680, 0.54])

        # bounds on states
        self.xmin = np.array(
            [
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                -math.pi / 9,
                -math.pi / 9,
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
            ]
        )
        self.xmax = np.array(
            [
                +np.inf,
                +np.inf,
                +np.inf,
                +np.inf,
                +np.inf,
                +np.inf,
                +math.pi / 9,
                +math.pi / 9,
                +np.inf,
                +np.inf,
                +np.inf,
                +np.inf,
            ]
        )

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

        # fmt: on

        # Time discretize system
        self.A = self.dt * self.A + np.identity(12)
        self.B = self.dt * self.B

        # Output matrix
        self.C = np.zeros((3, 12))
        self.C[:3, :3] = np.identity(3)

        # Disturbance matrix
        self.Cd = np.identity(3)

        # observer gain
        self.L = 0.1 * np.identity(3)

        # state & input cost matrices
        # self.Q = np.diag([1, 1, 1, 1, 1, 1, 0.001, 0.001, 0.001, 0.05, 0.05, 0.05])
        self.Q = 0.1 * np.identity(12)
        self.R = 0.01 * np.identity(4)

        # Solve discrete algebraic ricatii eq
        # P = dare solution
        # K = state-feedback gain
        self.P, _, self.K = ct.dare(self.A, self.B, self.Q, self.R)
        self.K = -self.K

    def _buildMPCProblem(self):
        cost = 0.0
        constraints = []

        # Parameters
        x_ref = cp.Parameter((12), name="x_ref")
        u_ref = cp.Parameter((4), name="u_ref")
        x_init = cp.Parameter((12), name="x_init")

        # Create the optimization variables
        x = cp.Variable((12, self.N + 1), name="x")
        u = cp.Variable((4, self.N), name="u")

        # For each stage in k = 0, ..., N-1
        for k in range(self.N):
            x_e = x[:, k] - x_ref

            # Cost
            cost += cp.quad_form(x_e, self.Q)
            cost += cp.quad_form(u[:, k] - u_ref, self.R)

            # System dynamics
            constraints += [x[:, k + 1] == self.A @ x[:, k] + self.B @ u[:, k]]

            constraints += [self.xmin <= x[:, k], x[:, k] <= self.xmax]
            constraints += [self.umin <= u[:, k], u[:, k] <= self.umax]

        # terminal cost addition (estimate cost N->inf)
        Vf = cp.quad_form(x[:, self.N] - x_ref, self.P)

        # 0 if self.beta or self.c_level is not set
        cost += self.beta * Vf

        if self.c_level:
            # terminal set constraint
            # optimal stability hard constraint
            # constraints += [x[:, self.N] == x_ref]
            constraints += [Vf <= self.c_level]

        # Inital condition
        constraints += [x[:, 0] == x_init]

        self.problem_mpc = cp.Problem(cp.Minimize(cost), constraints)

    def _buildOTSProblem(self):
        cost = 0.0
        constraints = []

        # TODO: Change hardcoded dimensions to x and u shapes
        # Parameters
        y_ref = cp.Parameter((3), name="y_ref")
        d_hat = cp.Parameter((3), name="d_hat")

        # Create the optimization variable (contains x_r and u_r)
        xu_r = cp.Variable((16), name="xu_r")

        # cost matrices for the quadratic problem
        H = np.identity(16)
        h = np.zeros((16, 1))
        
        cost += (1 / 2) * cp.quad_form(xu_r, H) + h.T @ xu_r

        mat1 = np.concatenate((np.identity(12) - self.A, -self.B), axis=1)
        mat2 = np.concatenate((self.C, np.zeros((3, 4))), axis=1)
        A_cstr = np.concatenate((mat1, mat2), axis=0)
        b_cstr = cp.hstack((np.zeros(12), (y_ref - self.Cd @ d_hat)))

        constraints += [A_cstr @ xu_r == b_cstr]
        # state and input constraints
        constraints += [self.xmin <= xu_r[:12], xu_r[:12] <= self.xmax]
        constraints += [self.umin <= xu_r[12:], xu_r[12:] <= self.umax]

        self.problem_ots = cp.Problem(cp.Minimize(cost), constraints)

    def _computeRPMfromInputs(self, u_delta):
        """
        Computes the rpm given the small-signal u_delta around the operating point.
        """
        return np.sqrt(np.matmul(self.W_inv, self.u_op + u_delta))

    def computeOTS(self, y_ref, d_hat):
        ## Run OTS here
        self.problem_ots.param_dict["y_ref"].value = y_ref
        self.problem_ots.param_dict["d_hat"].value = d_hat

        self.problem_ots.solve(solver=self.solver, verbose=False)
        if not (
            self.problem_ots.status == "optimal"
            or self.problem_ots.status == "inaccurate optimal"
        ):
            raise RuntimeError(
                f"OTS solver did not find a solution, due to it being {self.problem_ots.status}"
            )

        return (
            self.problem_ots.var_dict["xu_r"].value[:12],
            self.problem_ots.var_dict["xu_r"].value[12:],
        )

    def computeControl(self, x_init, x_target, u_target):
        self.problem_mpc.param_dict["x_init"].value = x_init
        self.problem_mpc.param_dict["x_ref"].value = x_target
        self.problem_mpc.param_dict["u_ref"].value = u_target

        self.problem_mpc.solve(solver=self.solver, verbose=False)
        if not (
            self.problem_mpc.status == "optimal"
            or self.problem_mpc.status == "inaccurate optimal"
        ):
            raise RuntimeError(
                f"MPC solver did not find a solution, due to it being {self.problem_mpc.status}"
            )

        # We return the MPC input and the next state (and also the plan for visualization)
        return (
            self.problem_mpc.var_dict["u"][:, 0].value,
            self.problem_mpc.var_dict["x"][:, 1].value,
            self.problem_mpc.var_dict["x"][:, :].value,
        )
