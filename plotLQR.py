import numpy as np
from mpc import Controller
import matplotlib.pyplot as plt
import terminal_set
from main import simulate


dt = 0.10  # Sampling period
T = 100
x_init_out = np.zeros(12)  # Initial conditions
x_init_in = np.zeros(12)  # Initial conditions
x_target = np.zeros(12)  # State to reach 
x_init_out[0:3] = np.array([2.0, 2.0, 2.0])      # outside the Terminal set x.T@P@x = 18.85
x_init_in[0:3] = np.array([0.01, 0.01, 0.1])  # inside the Terminal set x.T@P@x = 0.013


ctrl = Controller(
    mpc_horizon=10,
    timestep_mpc_stages=dt,
)
ctrl.beta = 10.0
# ctrl.c_level = terminal_set.calculate_c(ctrl, x_target)

ctrl_lqr = Controller(
    mpc_horizon=10,
    timestep_mpc_stages=dt,
    control_type="lqr"
)

states_lqr, inputs_lqr, plans_lqr, timesteps_lqr, Vf_lqr, l_lqr, timecost_lqr = simulate(
    controller=ctrl_lqr,
    x_init=x_init_in,
    x_target=x_target,
    T=T,
    plot=False,
    plots_suffix="",
)

states, inputs, plans, timesteps, Vf, l, timecost = simulate(
    controller=ctrl,
    x_init=x_init_in,
    x_target=x_target,
    T=T,
    plot=False,
    plots_suffix="",
)

fN = plt.figure()

t = np.arange(0, 10.1, 0.1)

def _plot_single_state(state,state_lqr,label):
    plt.plot(t,state, color="y", label=r"$mpc$")
    plt.plot(t,state_lqr, color="r", label=r"$lqr$")
    plt.legend()
    plt.xlim([0, 10.1])
    plt.xlabel('Time (s)',fontsize = 14)
    plt.ylabel(label, fontsize=14)

fN.add_subplot(2, 3, 1)
_plot_single_state(states[0, :], states_lqr[0, :],r"$x$ $(m)$")

fN.add_subplot(2, 3, 2)
_plot_single_state(states[1, :],states_lqr[1, :], r"$y$ $(m)$")

fN.add_subplot(2, 3, 3)
_plot_single_state(states[2, :], states_lqr[2, :],r"$z$ $(m)$")


plt.suptitle("Comparison for initial condition inside Terminal set")
plt.show()