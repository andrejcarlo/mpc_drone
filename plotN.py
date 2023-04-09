import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from main import simulate
from src.mpc import Controller, Dimension

if __name__ == "__main__":
    dt = 0.10  # Sampling period
    N = 20  # MPC Horizon
    T = 100  # Duration of simulation

    dim = Dimension(nx=12, nu=4, ny=3, nd=3)

    x_init = np.zeros(dim.nx)  # Initial conditions
    y_target = np.zeros(dim.ny)  # State to reach
    y_target[0:3] = np.array([1.0, 1.0, 1.0])

    # Controller
    ctrl1 = Controller(
        dim=dim,
        mpc_horizon=3,
        timestep_mpc_stages=dt,
    )
    ctrl2 = Controller(
        dim=dim,
        mpc_horizon=5,
        timestep_mpc_stages=dt,
    )
    ctrl3 = Controller(
        dim=dim,
        mpc_horizon=10,
        timestep_mpc_stages=dt,
    )
    ctrl4 = Controller(
        dim=dim,
        mpc_horizon=20,
        timestep_mpc_stages=dt,
    )
    ctrl5 = Controller(
        dim=dim,
        mpc_horizon=30,
        timestep_mpc_stages=dt,
    )
    ctrl1.beta = 10.0
    ctrl2.beta = 10.0
    ctrl3.beta = 10.0
    ctrl4.beta = 10.0
    ctrl5.beta = 10.0

    _, _, _, _, _, _, _, time_cost1 = simulate(
        controller=ctrl1, x_init=x_init, y_target=y_target, T=T
    )
    _, _, _, _, _, _, _, time_cost2 = simulate(
        controller=ctrl1, x_init=x_init, y_target=y_target, T=T
    )
    _, _, _, _, _, _, _, time_cost3 = simulate(
        controller=ctrl1, x_init=x_init, y_target=y_target, T=T
    )
    _, _, _, _, _, _, _, time_cost4 = simulate(
        controller=ctrl1, x_init=x_init, y_target=y_target, T=T
    )
    _, _, _, _, _, _, _, time_cost5 = simulate(
        controller=ctrl1, x_init=x_init, y_target=y_target, T=T
    )

    ###################################### Plot position error

    # fN = plt.figure(figsize=(12, 8))

    # def error(states):
    #     # Euclidean error on x, y, z
    #     error = np.sqrt(np.square(x_target[0]-states[0, :])
    #                     +np.square(x_target[1]-states[1, :])
    #                     +np.square(x_target[2]-states[2, :]))
    #     return error

    # t = np.arange(0, 10.1, 0.1)

    # plt.plot(t,error(states1), color="y", label=r"$N = 3$")
    # plt.plot(t,error(states2), color="r", label=r"$N = 5$")
    # plt.plot(t,error(states3), color="g", label=r"$N = 10$")
    # plt.plot(t,error(states4), color="b", label=r"$N = 20$")
    # plt.plot(t,error(states5), color="c", label=r"$N = 30$")

    # plt.legend()
    # plt.xlim([0, 10.1])
    # plt.xlabel('Time (s)',fontsize = 14)
    # plt.ylabel(r"error $(m)$", fontsize=14)
    # plt.suptitle("Effect of horizon N on tracking error of position (without Terminal cost)")

    # plt.show()

    ####################################################Plot time cost
    N = np.array([5, 10, 20, 30])
    t = np.array([time_cost2, time_cost3, time_cost4, time_cost5])
    plt.xlim([0, 38])
    plt.ylim([0, 10])
    plt.scatter(N, t)
    plt.xlabel("Prediction Horizon", fontsize=14)
    plt.ylabel("Time cost (s)", fontsize=14)
    plt.suptitle("Computation time for different horizons")
    plt.show()
