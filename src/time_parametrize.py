import numpy as np
from dataclasses import dataclass
import math


@dataclass
class Trajectory:
    number_of_points: int
    time_points: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    orientation_rpy: np.ndarray
    orientation_rpy_rates: np.ndarray


def resample(orig_path, max_distance=0.2):
    """
    Insert intermediate linearly interpolated waypoints such that the maximum distance between two consecutive waypoints is <max_distance>
    """
    if orig_path.shape[0] > 0:
        number_points = 1
        for i in range(1, orig_path.shape[0]):
            additonal_points = int(
                np.floor(
                    np.linalg.norm(orig_path[i, :] - orig_path[i - 1, :]) / max_distance
                )
            )
            number_points += additonal_points + 1

        resampled_path = np.zeros([number_points, 3])
        resampled_path[0, :] = orig_path[0, :]

        p = 1
        for i in range(1, orig_path.shape[0]):
            additonal_points = int(
                np.floor(
                    np.linalg.norm(orig_path[i, :] - orig_path[i - 1, :]) / max_distance
                )
            )
            for j in range(1, additonal_points + 1):
                resampled_path[p, :] = orig_path[i - 1, :] + (
                    orig_path[i, :] - orig_path[i - 1, :]
                ) * j / (additonal_points + 1)
                p += 1
            resampled_path[p, :] = orig_path[i, :]
            p += 1

        return resampled_path


def time_parametrize_const_velocity(path, velocity=0.1):
    """
    Calculate time point of each waypoint given a constant velocity.
    """
    t = np.zeros(path.shape[0])
    for i in range(1, path.shape[0]):
        t[i] = t[i - 1] + np.linalg.norm(path[i, :] - path[i - 1, :]) / velocity
    return t


def get_velocties_from_path(positions, time_points):
    """
    Calculate the instantanous velocity at each waypoint given by each pair of <positions> and <time_points>.

    Returns
    velocities: ndarray
            (3,positions.shape[1])-shaped array of floats containing the velocities at the given positions
    """
    if positions.shape[1] == 1:
        return np.array([0, 0, 0])

    velocities = np.zeros_like(positions)
    velocities[:, 0] = (positions[:, 1] - positions[:, 0]) / (
        time_points[1] - time_points[0]
    )
    for i in range(1, positions.shape[1] - 1):
        pre_vel = (positions[:, i] - positions[:, i - 1]) / (
            time_points[i] - time_points[i - 1]
        )
        post_vel = (positions[:, i + 1] - positions[:, i]) / (
            time_points[i + 1] - time_points[i]
        )
        velocities[:, i] = pre_vel  # + post_vel) / 2
    velocities[:, -1] = np.array([0, 0, 0])
    return velocities


def trajectory_from_path_const_vel(target_path, max_velocity, sampling_time):
    """
    Create trajectory from given path with a constant velocity <max_velocity> (can be less than that for edge cases) and return waypoint
    samples spaced <sampling_time> seconds apart.
    """

    # XYZ positions of the drones
    positions = resample(
        target_path, max_distance=max_velocity * sampling_time
    ).transpose()
    number_of_points = positions.shape[1]

    # time_points = np.arange(number_of_points) * sampling_time + 1
    time_points = time_parametrize_const_velocity(
        positions.transpose(), velocity=max_velocity
    )

    # velocities = np.zeros_like(positions)
    velocities = get_velocties_from_path(positions, time_points)
    number_of_points = time_points.size

    #  roll, pitch, and yaw of the drones in radians
    orientation_rpy = np.zeros_like(positions)
    #  angular velocities (for roll, pitch, and yaw) of the drones in radians/s
    orientation_rpy_rates = np.zeros_like(positions)

    return Trajectory(
        number_of_points,
        time_points,
        positions,
        velocities,
        orientation_rpy,
        orientation_rpy_rates,
    )


def trajectory_from_path_bang_bang(
    target_path, max_velocity, sampling_time, min_speed=0, max_acc=1
):
    pass
