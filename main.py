import numpy as np
from tqdm import tqdm
from mpc import MPCControl
import matplotlib.pyplot as plt

from visualise import (
    plot_3d_control,
    plot_action_history,
    plot_state_history,
    plot_terminal_cost_lyupanov,
)

import terminal_set


def simulate(
    controller,
    x_init,
    x_target,
    T=50,
    compute_Vf_weird=False,
    plot=False,
    plots_suffix="",
):
    # Some checks
    # if (controller.use_terminal == 2) and not controller.beta:
    #     print("Using default mpc config perhaps you forgot to set beta!")
    # if (controller.use_terminal == 1) and not controller.c_level:
    #     print("Using default mpc config perhaps you forgot to set c_level!")

    # Initialise the output arrays
    x_real = np.zeros((12, T + 1))
    x_all = np.zeros((12, controller.N + 1, T + 1))
    u_real = np.zeros((4, T))
    x_real[:, 0] = x_init
    timesteps = np.linspace(0, controller.dt, T)

    # terminal cost and stage cost for housekeeping
    Vf = np.zeros(T)
    stage_cost = np.zeros(T)

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

        # -> log terminal cost for stability analysis

        # x[N] - x_target
        x_e = x_all_out[:, -1] - x_target

        if compute_Vf_weird:
            Ak = controller.A + controller.B @ controller.K
            Vf[t] = (Ak @ x_e).T @ controller.P @ (Ak @ x_e)
            stage_cost[t] = (
                x_e.T
                @ (controller.Q + controller.K.T @ controller.R @ controller.K)
                @ x_e
            )
        else:
            Vf[t] = (controller.beta if controller.beta else 1.0) * (
                x_e.T @ controller.P @ x_e
            )
            # stage_cost[t] = (
            #     x_e.T @ controller.Q @ x_e
            #     + u_real[:, t].T @ controller.R @ u_real[:, t]
            # )
            stage_cost[t] = (
                x_e.T
                @ (controller.Q + controller.K.T @ controller.R @ controller.K)
                @ x_e
            )

    # Function that plots the trajectories.
    # The plot is stored with the name of the first parameter
    if plot:
        plot_state_history(
            "figures/mpc_state_history" + plots_suffix + ".png",
            x_real,
            T,
        )
        plot_action_history(
            "figures/mpc_action_history" + plots_suffix + ".png",
            u_real,
            T,
        )
        plot_3d_control(np.vstack((x_init, x_target)), x_real.T)
        plot_terminal_cost_lyupanov(Vf, stage_cost, T, None)
        plt.show()

    return (x_real, u_real, x_all, timesteps, Vf, stage_cost)


if __name__ == "__main__":
    dt = 0.10  # Sampling period
    N = 20  # MPC Horizon
    T = 100  # Duration of simulation
    x_init = np.zeros(12)  # Initial conditions
    x_target = np.zeros(12)  # State to reach
    x_target[0:3] = np.array([5.0, 4.0, 8.0])
    x_target[3:6] = np.array([0.0, 0.0, 0.0])

    print("Initial state is ", x_init)
    print("Target state to reach is ", x_target)

    # Controller
    ctrl = MPCControl(
        mpc_horizon=N,
        timestep_mpc_stages=dt,
        use_terminal=0,
    )

    # this also rebuilds mpc problem to include the new constraint
    ctrl.c_level = terminal_set.calculate_c(ctrl, x_target)

    states, inputs, plans, timesteps = simulate(
        controller=ctrl,
        x_init=x_init,
        x_target=x_target,
        T=T,
        plot=True,
        plots_suffix="_terminal",
    )


# Plot for different mpc predictions horizons (with TCost and Tset)
## - plot stability (what they have in the example report)
## - plot computational time versus choice of N

# Investigate effect of c, motivate choice of c
# Discuss why it cannot find a solution with a lower N and a huge c
## - U is not upper bounded

# Andrei
# We could also do the terminal set with penalise thing? - Done
# Generate some statistics about these 3 methods, are they faster compared to one another?


# set upper boundary input constraint for rpm speeds for each motor

# tracking error statistics, how close you follow the target

# Initial condition in xf, vs init condition outside of xf


# Questions to ask prof:
# Why do we get the terminal cost bigger than the stage cost in the graph (the red bump)
# Do we need to verify the stability of the nonlinear system numerically?
# The choice of c and horizon is dependent on the problem to solve.
# #Ex: reference is at a different location
## is there any way to solve for c for different locations?
## in the example that you need to follow multiple targets, this c would need to change?
## What would be the drawback if we had a really large c
