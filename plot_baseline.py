import matplotlib.pyplot as plt
import numpy as np

x_init = np.zeros(12)
y_target = np.array([1.0, 1.0, 1.0])

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection="3d")

ax.plot(
    [x_init[0], y_target[0]],
    [x_init[1], y_target[1]],
    [x_init[2], y_target[2]],
    label="reference",
    color="b",
    linestyle="--",
)
ax.scatter(x_init[0], x_init[1], x_init[2], label=r"start")
ax.scatter(y_target[0], y_target[1], y_target[2], color="g", label=r"goal")
ax.autoscale(False)
ax.set_box_aspect((1, 1, 1.0))
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.legend()
fig.tight_layout()
plt.savefig("reference_baseline_3D.png")


fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot()

ax.plot(
    [x_init[0], y_target[0]],
    [x_init[1], y_target[1]],
    [x_init[2], y_target[2]],
    label="reference",
    color="b",
    linestyle="--",
)
ax.scatter(x_init[0], x_init[1], x_init[2], label=r"start")
ax.scatter(y_target[0], y_target[1], y_target[2], color="g", label=r"goal")
ax.autoscale(False)
ax.set_box_aspect((1, 1, 1.0))
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.legend()
fig.tight_layout()
plt.savefig("reference_baseline.png")
plt.show()
