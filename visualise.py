import matplotlib.pyplot as plt
import numpy as np


def plot_3d_control(x_init, y_target, actual):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.plot(
        [x_init[0], y_target[0]],
        [x_init[1], y_target[1]],
        [x_init[2], y_target[2]],
        label="reference",
        color="b",
        linestyle="--",
        zorder=10,
    )

    ax.plot(
        actual[:, 0],
        actual[:, 1],
        actual[:, 2],
        label="actual",
        color="g",
        linestyle="-",
        zorder=10,
    )
    # ax.margins(0.01)
    ax.autoscale(False)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(x_init[0], y_target[0])
    ax.set_ylim(x_init[1], y_target[1])
    ax.set_zlim(x_init[2], y_target[2])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    fig.tight_layout()


def plot_single_state(state, T, label):
    plt.plot(state)
    plt.yticks([np.min(state), 0, np.max(state)])
    plt.ylim([np.min(state) - 0.01, np.max(state) + 0.01])
    plt.xlim([0, T])
    plt.ylabel(label, fontsize=14)
    plt.grid()


def plot_state_history(filename, x, T):
    f = plt.figure(figsize=(12, 8))

    # Plot x (position)
    ax1 = f.add_subplot(4, 3, 1)
    plot_single_state(x[0, :], T, r"$x$ $(m)$")

    ax1 = f.add_subplot(4, 3, 2)
    plot_single_state(x[1, :], T, r"$y$ $(m)$")

    ax1 = f.add_subplot(4, 3, 3)
    plot_single_state(x[2, :], T, r"$z$ $(m)$")

    ax1 = f.add_subplot(4, 3, 4)
    plot_single_state(x[3, :], T, r"$\dot{x}$ $(m/s)$")

    ax1 = f.add_subplot(4, 3, 5)
    plot_single_state(x[4, :], T, r"$\dot{y}$ $(m/s)$")

    ax1 = f.add_subplot(4, 3, 6)
    plot_single_state(x[5, :], T, r"$\dot{z}$ $(m/s)$")

    ax1 = f.add_subplot(4, 3, 7)
    plot_single_state(x[6, :], T, r"$\phi$ $(rad)$")

    ax1 = f.add_subplot(4, 3, 8)
    plot_single_state(x[7, :], T, r"$\theta$ $(rad)$")

    ax1 = f.add_subplot(4, 3, 9)
    plot_single_state(x[8, :], T, r"$\psi$ $(rad)$")

    ax1 = f.add_subplot(4, 3, 10)
    plot_single_state(x[9, :], T, r"$p$ $(rad/s)$")

    ax1 = f.add_subplot(4, 3, 11)
    plot_single_state(x[10, :], T, r"$q$ $(rad/s)$")

    ax1 = f.add_subplot(4, 3, 12)
    plot_single_state(x[11, :], T, r"$r$ $(rad/s)$")

    plt.tight_layout()
    plt.savefig(filename, format="png")


def plot_action_history(filename, u, T):
    f2 = plt.figure(figsize=(12, 8))

    # Plot x (position)
    ax1 = f2.add_subplot(4, 1, 1)
    plot_single_state(u[0, :], T, r"$u1$ $(-)$")

    ax1 = f2.add_subplot(4, 1, 2)
    plot_single_state(u[1, :], T, r"$u2$ $(-)$")

    ax1 = f2.add_subplot(4, 1, 3)
    plot_single_state(u[2, :], T, r"$u3$ $(-)$")

    ax1 = f2.add_subplot(4, 1, 4)
    plot_single_state(u[3, :], T, r"$u4$ $(-)$")

    plt.tight_layout()
    plt.savefig(filename, format="png")


def plot_terminal_cost_lyupanov(Vf, l, T, c_level):
    """
    Plots terminal set and stage cost as a function of time.
    Inputs:
        Vf = ndarray(1,T)
        l = ndarray(1,T)

    Inequality
    Vf(x_plus) - Vf(x) < -l(x,u)
    """
    Vf_diff = Vf[1:] - Vf[0:-1]
    l_diff = l[0:-1]

    f3 = plt.figure(figsize=(12, 8))
    plt.title("Stability - Lyapunov decrease assumption")

    plt.plot(range(0, T - 1), Vf_diff, color="r", label=r"$V_f(f(x,u)) - V_f(x)$")
    plt.plot(range(0, T - 1), -l_diff, color="g", label=r"$-l(x,u)$")
    plt.plot(range(0, T - 1), Vf[1:], color="b", label=r"$V_f(f(x,u))$")
    plt.plot(range(0, T - 1), Vf[0:-1], color="m", label=r"$V_f(x)$")
    if c_level:
        plt.axhline(y=c_level, color="c", label=r"$c$ level set curve")
    plt.legend()
    plt.grid()


def plot_disturbance(d_const, d_hat, T):
    # norm error in disturbance
    f3 = plt.figure(figsize=(12, 8))
    plt.title("Disturbance estimation")

    # ax1 = f3.add_subplot(3, 1, 1)
    # plt.plot(d_hat[0, :], label="estimate")
    # plt.axhline(d_const[0], color="r", label="actual")
    # plt.ylabel(r"$d_x$ $(-)$", fontsize=14)

    # ax1 = f3.add_subplot(3, 1, 2)
    # plt.plot(d_hat[1, :], label="estimate")
    # plt.axhline(d_const[1], color="r", label="actual")
    # plt.ylabel(r"$d_y$ $(-)$", fontsize=14)

    # ax1 = f3.add_subplot(3, 1, 3)
    # plt.plot(d_hat[2, :], label="estimate")
    # plt.axhline(d_const[2], color="r", label="actual")
    # plt.ylabel(r"$d_z$ $(-)$", fontsize=14)

    d_diff = np.linalg.norm(d_hat.T - d_const, axis=1)
    plt.ylabel(r"$|| \hat{d} - d ||$", fontsize=14)
    plt.plot(d_diff)

    plt.legend()
    plt.grid()
    plt.tight_layout()
