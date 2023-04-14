# A Model Predictive Control Approach for Trajectory Tracking of Quadrotors

This repo contains the code to simulate an MPC controller for a quadrotor. Offset-free MPC alongside with disturbance is implemented. Terminal set and terminal cost scaling have been developed to ensure lyapunov stability and recursive feasibility.

## Building the environment
To build the environment run

`conda env create -f environment.yml`

## Run the simulation

**Important**: In order to run the simulation, you must have GUROBI licensed and available to use as a solver. The environment contains the gurobi package, and if one of the following scripts is ran, it will look for a license on your local machine.

In addition, to the following `main.py` notebooks have been provided to run simulations for various parameter changes of the MPC.

### Reference tracking 

To simulate a simple point reference tracking run thef following

`python main.py`

### Path following

There are three pre-made trajectories to try. These are located in the `/trajectories` folder. Most of the hyperparameters are hard-coded in the file due to time constraints. To see the parameters of the simulation run 

`python main_path_following.py -h`

A runnable example:

`python main_path_following.py -t trajectories/lissajous.npy -s 4`
