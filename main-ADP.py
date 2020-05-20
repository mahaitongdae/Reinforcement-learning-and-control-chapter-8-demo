"""
    <Reinforcement Learning and Control>(Year 2020)
    by Shengbo Eben Li
        @ Intelligent Driving Lab, Tsinghua University

    ADP example for lane keeping problem in a curve road

    [Method]
    Approximate dynamic programming with structured policy

    """
import Dynamics
import numpy as np
import torch
import os
from Network import Actor, Critic
from Train import Train
from datetime import datetime
from plot_figure import adp_simulation_plot, plot_comparison


# Parameters
MAX_ITERATION = 10000       # max iterations
LR_P = 3e-5                 # learning rate of policy net
LR_V = 1e-4                 # learning rate of value net
S_DIM = 5                   # state dimension
A_DIM = 1                   # action dimension
TRAIN_FLAG = 1
LOAD_PARA_FLAG = 0

# Set random seed
np.random.seed(0)
torch.manual_seed(0)

# initialize policy and value net, model of vehicle dynamics
policy = Actor(S_DIM, A_DIM, lr=LR_P)
value = Critic(S_DIM, A_DIM, lr=LR_V)
vehicleDynamics = Dynamics.VehicleDynamics()
state_batch = vehicleDynamics.initialize_state()

# load pre-trained parameters or train
iteration_index = 0

if LOAD_PARA_FLAG == 1:
    # load pre-trained parameters
    load_dir = "./Results_dir/2020-05-18-02-21-10000"
    policy.load_parameters(load_dir)
    value.load_parameters(load_dir)

if TRAIN_FLAG == 1:
    # train the network by policy iteration
    train = Train()
    train.agent_batch = vehicleDynamics.initialize_state()
    while True:
        train.update_state(policy, vehicleDynamics)
        value_loss = train.policy_evaluation(policy, value, vehicleDynamics)
        policy_loss = train.policy_improvement(policy, value)
        iteration_index += 1

        # print train information
        if iteration_index % 1 == 0:
            log_trace = "iteration:{:3d} |"\
                        "policy_loss:{:3.3f} |" \
                        "value_loss:{:3.3f}".format(iteration_index, float(policy_loss), float(value_loss))
            print(log_trace)

        # save parameters and plot figures
        if iteration_index % 5000 == 0 or iteration_index == MAX_ITERATION:
            # ==================== Set log path ====================
            log_dir = "./Results_dir/" + datetime.now().strftime("%Y-%m-%d-%H-%M-" + str(iteration_index))
            os.makedirs(log_dir, exist_ok=True)
            value.save_parameters(log_dir)
            policy.save_parameters(log_dir)
            train.print_loss_figure(iteration_index, log_dir)
            train.save_data(log_dir)
            adp_simulation_plot(log_dir)
            # plot_comparison("./Figures")
            break








