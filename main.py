import numpy as np
from tqdm import tqdm
from mpc import MPCControl
import matplotlib.pyplot as plt

from visualise import plot_3d_control, plot_action_history, plot_state_history


def simulate(controller, x_init, x_target, dt=0.05, T=50, N=3, plot=False):
    # Initialise the output arrays
    x_real = np.zeros((12, T + 1))
    x_all = np.zeros((12, N + 1, T + 1))
    u_real = np.zeros((4, T))
    x_real[:, 0] = x_init
    timesteps = np.linspace(0, dt, T)

    # while target_index != trajectory.number_of_points - 1 or trans_error > 0.05:
    for t in tqdm(range(0, T), "Simulating"):
        # action @ k, next state at k+1, plan at k+N
        (u_out, x_out, x_all_out) = controller.computeControl(
            x_init=x_real[:, t], x_target=x_target
        )

        # Next x is the x in the second state
        x_real[:, t + 1] = x_out
        x_all[:, :, t] = x_all_out  # Save the plan (for visualization)

        # Used input is the first input
        u_real[:, t] = u_out

        # print("Input: " + str(u_out))

    # Function that plots the trajectories.
    # The plot is stored with the name of the first parameter
    if plot:
        plot_state_history("figures/mpc_state_history.png", x_real, T)
        plot_action_history("figures/mpc_action_history.png", u_real, T)
        plot_3d_control(np.vstack((x_init, x_target)), x_real.T)
        plt.show()

    return (
        x_real,
        u_real,
        x_all,
        timesteps,
    )


if __name__ == "__main__":
    dt = 0.25  # Sampling period
    N = 20  # MPC Horizon
    T = 50  # Duration of simulation
    x_init = np.zeros(12)  # Initial conditions
    x_target = np.zeros(12)  # State to reach
    x_target[0:3] = np.array([1, 1, 1])

    print("Initial state is ", x_init)
    print("Target state to reach is ", x_target)

    # Controller
    ctrl = MPCControl(
        mpc_horizon=N,
        timestep_mpc_stages=dt,
    )

    states, inputs, plans, timesteps = simulate(
        controller=ctrl, x_init=x_init, x_target=x_target, dt=dt, T=T, N=N, plot=True
    )
