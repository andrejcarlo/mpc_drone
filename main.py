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


def simulate(
    controller, x_init, x_target, dt=0.05, T=50, N=3, plot=False, use_terminal=False
):
    # Initialise the output arrays
    x_real = np.zeros((12, T + 1))
    x_all = np.zeros((12, N + 1, T + 1))
    u_real = np.zeros((4, T))
    x_real[:, 0] = x_init
    timesteps = np.linspace(0, dt, T)

    # terminal cost and stage cost for housekeeping
    Vf = np.zeros(T)
    l = np.zeros(T)

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

        x_e = x_real[:, t] - x_target

        Vf[t] = x_e.T @ controller.P @ x_e
        l[t] = x_e.T @ controller.Q @ x_e + u_real[:, t].T @ controller.R @ u_real[:, t]

    # Function that plots the trajectories.
    # The plot is stored with the name of the first parameter
    if plot:
        plot_state_history(
            "figures/mpc_state_history"
            + ("_terminal" if use_terminal else "")
            + ".png",
            x_real,
            T,
        )
        plot_action_history(
            "figures/mpc_action_history"
            + ("_terminal" if use_terminal else "")
            + ".png",
            u_real,
            T,
        )
        plot_3d_control(np.vstack((x_init, x_target)), x_real.T)
        plot_terminal_cost_lyupanov(Vf, l, T, controller.c)
        plt.show()

    return (
        x_real,
        u_real,
        x_all,
        timesteps,
    )


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
        terminal_set_level_c=100,
        use_terminal_set=True,
        use_terminal_cost=True,
    )

    print("K is ", ctrl.K)
    print("L is ", ctrl.L)

    print("Terminal cost ", ctrl.use_terminal_cost)
    print("Terminal set ", ctrl.use_terminal_set)

    states, inputs, plans, timesteps = simulate(
        controller=ctrl,
        x_init=x_init,
        x_target=x_target,
        dt=dt,
        T=T,
        N=N,
        plot=True,
        use_terminal=ctrl.use_terminal_cost,
    )

# Plots for stagecost and terminal cost over simulation time,
# #essentially checking if it's a lyuapunov function

# Plot for different mpc predictions horizons (with TCost and Tset)
## - plot stability (what they have in the example report)
## - plot computational time versus choice of N

# Investigate effect of c, motivate choice of c
# Discuss why it cannot find a solution with a lower N and a huge c
## - U is not upper bounded

# Andrei
# We could also do the box optimisation thing?
# We could also do the terminal set with penalise thing?
# Generate some statistics about these 3 methods, are they faster compared to one another?

# set upper boundary input constraint for rpm speeds for each motor

# tracking error statistics, how close you follow the target

# Initial condition in xf, vs init condition outside of xf

# Verify the stability of the nonlinear system numerically


# Questions to ask prof:
# Why do we get the terminal cost bigger than the stage cost in the graph (the red bump)
# Do we need to verify the stability of the nonlinear system numerically?
# The choice of c and horizon is dependent on the problem to solve.
# #Ex: reference is at a different location
## is there any way to solve for c for different locations?
## in the example that you need to follow multiple targets, this c would need to change?
## What would be the drawback if we had a really large c
