import numpy as np
from mpc import MPCControl
from time_parametrize import (
    trajectory_from_path_const_vel,
    trajectory_from_path_bang_bang,
)


def run(
    target_path,
    max_speed=1,  # m/s
    min_speed=0.3,
    max_acceleration=1.5,
    sampling_time=0.5,
    time_parametrize_method="const",
):
    # Create time parametrized trajectory from given path
    controller_time_step = sampling_time
    trajectory_time_step = sampling_time
    if time_parametrize_method == "bangbang":
        trajectory = trajectory_from_path_bang_bang(
            target_path,
            max_velocity=max_speed,
            sampling_time=trajectory_time_step,
            min_speed=min_speed,
            max_acc=max_acceleration,
        )
    elif time_parametrize_method == "const":
        trajectory = trajectory_from_path_const_vel(
            target_path, max_velocity=max_speed, sampling_time=trajectory_time_step
        )
    else:
        raise ValueError('time_parametrize_method must be one of "bangbang" or "const"')

    # Controller
    ctrl = MPCControl(
        timestep_reference=trajectory_time_step,
        timestep_mpc_stages=controller_time_step,
    )

    # Run the simulation
    action = np.array([0, 0, 0, 0])
    TIMESTEP = 1
    trans_error = 0
    target_index = 0
    i = 0

    current_state = np.zeros(12)
    history = [current_state]

    while target_index != trajectory.number_of_points - 1 or trans_error > 0.05:
        # compute control action
        (
            action,
            trans_error,
            current_target,
            target_index,
        ) = ctrl.computeControl(
            cur_state=current_state,
            current_time=i * TIMESTEP,
            target_time=trajectory.time_points,
            target_pos=trajectory.positions,
            target_rpy=trajectory.orientation_rpy,
            target_vel=trajectory.velocities,
            target_rpy_rates=trajectory.orientation_rpy_rates,
        )

        # apply control
        current_state = ctrl.applyControlInput(current_state, action)
        history.append(current_state)

        i += 1

    return np.array(history)


def plot_3d_control(history, actual, completion_rate=1):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    fig.tight_layout()

    ax.plot(
        actual[j, 0, :],
        actual[j, 1, :],
        actual[j, 2, :],
        label="reference",
        color="b",
        linestyle="--",
        zorder=10,
    )

    ax.plot(
        history[j, 0, :i],
        history[j, 1, :i],
        history[j, 2, :i],
        label="actual",
        color="g",
        linestyle="-",
        zorder=10,
    )
    ax.margins(0.01)
    ax.set_box_aspect((1, 1, 1))
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.get_zaxis().set_ticklabels([])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()


if __name__ == "__main__":
    # tuples of x, y, z
    target_path = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0)]

    # Some checks for input
    if not (
        isinstance(target_path, list)
        and len(target_path) >= 2
        and all(isinstance(x, tuple) and len(x) == 3 for x in target_path)
    ):
        raise ValueError("Target path is incorrect type/dimension")
    target_path = np.array(target_path)

    history_actual = run(target_path)
    trajectory_actual = history_actual[:, 0:3]

    print("Reference trajectory size", target_path.shape)
    print("Actual trajectory size", trajectory_actual.shape)
