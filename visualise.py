import matplotlib.pyplot as plt
import numpy as np


def plot_3d_control(reference, actual):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    fig.tight_layout()

    ax.plot(
        reference[:, 0],
        reference[:, 1],
        reference[:, 2],
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
    ax.margins(0.01)
    ax.set_box_aspect((1, 1, 1))
    # ax.get_xaxis().set_ticklabels([])
    # ax.get_yaxis().set_ticklabels([])
    # ax.get_zaxis().set_ticklabels([])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()


def plot_single_state(state, T, label):
    plt.plot(state)
    plt.yticks([np.min(state), 0, np.max(state)])
    plt.ylim([np.min(state) - 0.1, np.max(state) + 0.1])
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


def plot_trajectories(filename, x, u, T):
    f = plt.figure(figsize=(12, 6))

    # Plot position
    ax2 = f.add_subplot(311)
    x1 = x[0, :]
    plt.plot(x1)
    plt.ylabel(r"$p$ $(m)$", fontsize=14)
    plt.yticks([np.min(x1), 0, np.max(x1)])
    plt.ylim([np.min(x1) - 0.1, np.max(x1) + 0.1])
    plt.xlim([0, T])
    plt.grid()

    # Plot velocity
    ax3 = plt.subplot(3, 1, 2)
    x2 = x[1, :]
    plt.plot(x2)
    plt.yticks([np.min(x2), 0, np.max(x2)])
    plt.ylim([np.min(x2) - 0.1, np.max(x2) + 0.1])
    plt.xlim([0, T])
    plt.ylabel(r"$v$ $(m/s)$", fontsize=14)
    plt.grid()

    # Plot acceleration (input)
    ax1 = plt.subplot(3, 1, 3)

    plt.plot(u[0, :])
    plt.ylabel(r"$a$ $(m/s^2)$", fontsize=14)
    plt.yticks([np.min(u), 0, np.max(u)])
    plt.ylim([np.min(u) - 0.1, np.max(u) + 0.1])
    plt.xlabel(r"$t$", fontsize=14)
    plt.tight_layout()
    plt.grid()
    plt.xlim([0, T])
    plt.show()
