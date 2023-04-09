import numpy as np
from tqdm import tqdm
import cvxpy as cp
import matplotlib.pyplot as plt
import time

from src.mpc import Controller, Dimension
import src.terminal_set as terminal_set
from src.visualise import (
    plot_3d_control,
    plot_action_history,
    plot_state_history,
    plot_terminal_cost_lyupanov,
    plot_disturbance,
)


def simulate(
    controller,
    x_init,
    y_target,
    T=50,
    use_terminal_set=False,
    plot=False,
    plots_suffix="",
):
    # Initialise the states
    x_real = np.zeros((controller.dim.nx, T + 1))
    x_all = np.zeros((controller.dim.nx, controller.N + 1, T + 1))
    u_real = np.zeros((controller.dim.nu, T))
    x_real[:, 0] = x_init

    # initialise the outputs and disturbance estimate
    y_real = np.zeros((controller.dim.ny, T))
    d_hat = np.zeros((controller.dim.nd, T + 1))
    measurement_noise = np.zeros(controller.dim.ny)

    # terminal cost and stage cost for housekeeping
    Vf = np.zeros(T)
    stage_cost = np.zeros(T)

    # compute OTS with no disturbance
    (x_target, u_target) = controller.computeOTS(y_target, np.zeros(3))

    if use_terminal_set:
        ctrl.c_level = terminal_set.calculate_c(ctrl, x_target)

    start = time.time()
    for t in tqdm(range(0, T), "Simulating"):
        if controller.control_type == "lqr":
            # only apply lqr control law
            u_out = controller.K @ (x_real[:, t] - x_target)
            u_real[:, t] = u_out + u_target
            x_real[:, t + 1] = controller.A @ x_real[:, t] + controller.B @ u_real[:, t]
            y_real[:, t] = (
                controller.C @ x_real[:, t + 1]
                + controller.Cd @ controller.d
                + measurement_noise
            )

        elif controller.control_type == "mpc":
            # if using disturbance recompute OTS and terminal cost, every time with the estimate of d_hat
            if np.any(controller.d):
                (x_target, u_target) = controller.computeOTS(y_target, d_hat[:, t])
                if use_terminal_set:
                    ctrl.c_level = terminal_set.calculate_c(ctrl, x_target)
                measurement_noise = 0.01 * np.random.randn(3)

            # action @ k, next state at k+1, plan at k+N
            (u_out, x_out, x_all_out) = controller.computeControl(
                x_init=x_real[:, t], x_target=x_target, u_target=u_target
            )

            # Next x is the x in the second state
            x_real[:, t + 1] = x_out
            x_all[:, :, t] = x_all_out  # Save the plan (for visualization)

            # Used input is the first input
            u_real[:, t] = u_out + u_target

            # update system reponse
            y_real[:, t] = (
                controller.C @ x_real[:, t + 1]
                + controller.Cd @ controller.d
                + measurement_noise
            )

            # recompute disturbance estimate using Luenberger observer
            d_hat[:, t + 1] = d_hat[:, t] + controller.L @ (
                y_real[:, t] - controller.C @ x_real[:, t + 1] - d_hat[:, t]
            )

            # x[N] - x_target
            x_e = x_all_out[:, -1] - x_target

            # log tcost and stagecost for plotting
            Vf[t] = controller.beta * (x_e.T @ controller.P @ x_e)
            stage_cost[t] = (
                x_e.T
                @ (controller.Q + controller.K.T @ controller.R @ controller.K)
                @ x_e
            )
    end = time.time()
    time_cost = end - start

    # Function that plots the trajectories.
    # The plot is stored with the name of the first parameter
    if plot:
        plot_state_history(
            "figures/"
            + controller.control_type
            + "_state_history"
            + plots_suffix
            + ".png",
            x_real,
            T,
        )
        plot_action_history(
            "figures/"
            + controller.control_type
            + "_action_history"
            + plots_suffix
            + ".png",
            u_real,
            T,
        )
        plot_3d_control(x_init[:3], y_target, y_real.T)
        if controller.control_type == "mpc" and controller.beta:
            plot_terminal_cost_lyupanov(Vf, stage_cost, T, None)
        if np.any(controller.d):
            plot_disturbance(controller.d, d_hat, T)
        plt.show()

    return (x_real, u_real, x_all, Vf, stage_cost, y_real, d_hat, time_cost)


if __name__ == "__main__":
    dt = 0.10  # Sampling period
    N = 20  # MPC Horizon
    T = 100  # Duration of simulation
    dim = Dimension(nx=12, nu=4, ny=3, nd=3)

    x_init = np.zeros(dim.nx)  # Initial conditions
    y_target = np.zeros(dim.ny)  # State to reach
    y_target[0:3] = np.array([1.0, 0.0, 0.0])
    # y_target[3:6] = np.array([0.0, 0.0, 0.0])

    print("Initial state is ", x_init)
    print("Target state to reach is ", y_target)

    # Controller
    ctrl = Controller(
        dim=dim,
        mpc_horizon=N,
        timestep_mpc_stages=dt,
        solver=cp.GUROBI,
        control_type="mpc",  # 'lqr' or 'mpc'
    )

    # Set disturbance and terminal cost scaling
    ctrl.d = np.array([0.0, 0.5, 0.0])
    ctrl.beta = 2.0  # << (very small) beta means you get a behaviour as if tcost was not there at all

    states, inputs, plans, Vf, l, outputs, disturbance_est, time_cost = simulate(
        controller=ctrl,
        x_init=x_init,
        y_target=y_target,
        T=T,
        use_terminal_set=False,
        plot=True,
        plots_suffix="_disturbance",
    )

    print(
        "Simulation of {} with horizon {}, took {}s.".format(
            ctrl.control_type, N, time_cost
        )
    )
