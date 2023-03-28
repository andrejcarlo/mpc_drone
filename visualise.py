import matplotlib.pyplot as plt


def plot_3d_control(history, actual):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    fig.tight_layout()

    ax.plot(
        actual[:, 0],
        actual[:, 1],
        actual[:, 2],
        label="reference",
        color="b",
        linestyle="--",
        zorder=10,
    )

    ax.plot(
        history[:, 0],
        history[:, 1],
        history[:, 2],
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

    plt.show()
