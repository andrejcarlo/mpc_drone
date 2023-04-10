import numpy as np
from tqdm import tqdm
import cvxpy as cp
import matplotlib.pyplot as plt
import time
import argparse

from src.mpc import Controller, Dimension
import src.terminal_set as terminal_set
from src.visualise import (
    plot_action_history,
    plot_state_history,
)
from main import simulate


def simulate_path_following(ctrl, trajectory, T):
    x_init = np.zeros(ctrl.dim.nx)  # Initial conditions
    y_target = np.zeros(ctrl.dim.ny)  # State to reach

    _, path_len = trajectory.shape

    outputs_traj = np.zeros((path_len - 1, ctrl.dim.ny, T))
    states_traj = np.zeros((path_len, ctrl.dim.nx, T + 1))
    inputs_traj = np.zeros((path_len, ctrl.dim.nu, T))

    # first point in lissajous
    x_init[:3] = trajectory.T[0, :]

    for i, point in enumerate(trajectory.T[1:, :]):
        y_target[0:3] = point

        states, inputs, _, _, _, outputs, _, _ = simulate(
            controller=ctrl, x_init=x_init, y_target=y_target, T=T
        )
        x_init[:3] = outputs[:, -1]

        outputs_traj[i, :, :] = outputs
        states_traj[i, :, :] = states
        inputs_traj[i, :, :] = inputs

    # flatten trajectories for each simulation
    outputs_traj_flat = (
        np.transpose(outputs_traj, axes=(0, 2, 1)).reshape(-1, ctrl.dim.ny).T
    )
    states_traj_flat = (
        np.transpose(states_traj, axes=(0, 2, 1)).reshape(-1, ctrl.dim.nx).T
    )
    inputs_traj_flat = (
        np.transpose(inputs_traj, axes=(0, 2, 1)).reshape(-1, ctrl.dim.nu).T
    )

    return outputs_traj_flat, states_traj_flat, inputs_traj_flat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Path following simulation for a quadrotor using MPC"
    )
    parser.add_argument(
        "-t",
        "--trajectory",
        help="Path to trajectory file to simulate",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-s",
        "--subsample",
        help="Subsample trajectory every n step",
        required=False,
        type=int,
        default=1,
    )

    args = vars(parser.parse_args())

    dt = 0.10  # Sampling period
    N = 20  # MPC Horizon
    T = 50  # Duration of simulation
    dim = Dimension(nx=12, nu=4, ny=3, nd=3)

    # Controller
    ctrl = Controller(
        dim=dim,
        mpc_horizon=N,
        timestep_mpc_stages=dt,
        solver=cp.GUROBI,
        control_type="mpc",  # 'lqr' or 'mpc'
    )

    # Set disturbance and terminal cost scaling
    # ctrl.d = np.array([0.0, 0.5, 0.0])
    # ctrl.beta = 2.0  # << (very small) beta means you get a behaviour as if tcost was not there at all

    subsample_step = args["subsample"]
    lissajous = np.load(args["trajectory"])
    lissajous_subsampled = lissajous[:, ::subsample_step]
    print(
        "Subsampled trajectory from ",
        lissajous.shape,
        " to ",
        lissajous_subsampled.shape,
    )
    start = time.time()
    outputs, states, inputs = simulate_path_following(ctrl, lissajous_subsampled, T)
    end = time.time()
    print(
        "Simulation of {} with horizon {}, took {}s.".format(
            ctrl.control_type,
            N,
            end - start,
        )
    )

    plot_state_history("figures/state_hist_path_follow.png", states, states.shape[1])
    plot_action_history("figures/action_hist_path_follow.png", inputs, inputs.shape[1])

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.plot(
        lissajous_subsampled[0, :],
        lissajous_subsampled[1, :],
        lissajous_subsampled[2, :],
        label="reference",
        color="b",
        linestyle="--",
    )
    ax.plot(
        outputs[0, :],
        outputs[1, :],
        outputs[2, :],
        label="actual",
        color="red",
        linestyle="-",
    )
    ax.legend()
    plt.show()
