# A Linear Model Predictive Control Approach for Trajectory Tracking of Quadrotors

This repo contains the code to simulate an MPC controller for a quadrotor. Offset-free MPC alongside with disturbance is implemented. Terminal set and terminal cost scaling have been developed to ensure lyapunov stability and recursive feasibility.

## Building the environment
To build the environment run

`conda env create -f environment.yml`

## Run the simulation

### Reference tracking 

`python main.py`

### Path following

There are three pre-made trajectories to try. These are located in the `/trajectories` folder. Most of the hyperparameters are hard-coded in the file due to time constraints. To see the parameters of the simulation run 

`python main_path_following.py -h`

A runnable example:

`python main_path_following.py -t trajectories/lissajous.npy -s 4`
