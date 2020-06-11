import Dynamics
import numpy as np
import torch
import time
import os
from Network import Actor, Critic
from Config import DynamicsConfig
S_DIM = 4
A_DIM = 1
from Solver import Solver

def simulation(log_dir):
    comparison_dir = "./Results_dir/Comparison_Data"
    policy = Actor(S_DIM, A_DIM)
    value = Critic(S_DIM, A_DIM)
    config = DynamicsConfig()
    solver=Solver()
    load_dir = log_dir
    policy.load_parameters(load_dir)
    value.load_parameters(load_dir)
    statemodel_plt = Dynamics.VehicleDynamics()
    plot_length = config.NP_TOTAL

    # Open-loop reference
    x_init = [0.0, 0.0, config.psi_init, 0.0, 0.0]
    op_state, op_control = solver.openLoopMpcSolver(x_init, config.NP_TOTAL)
    np.savetxt(os.path.join(comparison_dir, 'Open_loop_control.txt'), op_control)

    methods = ['ADP','MPC','OP']
    for method in methods:
        cal_time = 0
        state = torch.tensor([[0.0, 0.0, config.psi_init, 0.0, 0.0]])
        state.requires_grad_(False)
        x_ref = statemodel_plt.reference_trajectory(state[:, -1])
        state_r = state.detach().clone()
        state_r[:, 0:4] = state_r[:, 0:4] - x_ref

        state_history = state.detach().numpy()
        control_history = []

        for i in range(plot_length):
            if method == 'ADP':
                time_start = time.time()
                u = policy.forward(state_r[:, 0:4])
                cal_time += time.time() - time_start
            elif method == 'MPC':
                x = state_r.tolist()[0]
                time_start = time.time()
                _, control = solver.mpcSolver(x, config.NP)
                cal_time += time.time() - time_start
                u = np.array(control[0], dtype='float32').reshape(-1, config.ACTION_DIM)
                u = torch.from_numpy(u)
            else:
                u = np.array(op_control[i], dtype='float32').reshape(-1, config.ACTION_DIM)
                u = torch.from_numpy(u)
            state_next, deri_state, utility, F_y1, F_y2, alpha_1, alpha_2 = statemodel_plt.step(state, u)
            state_r_old, _, _, _, _, _, _ = statemodel_plt.step(state_r, u)
            state_r = state_r_old.detach().clone()
            state_r[:, [0, 2]] = state_next[:, [0, 2]]
            x_ref = statemodel_plt.reference_trajectory(state_next[:, -1])
            state_r[:, 0:4] = state_r[:, 0:4] - x_ref
            state = state_next.clone().detach()
            s = state_next.detach().numpy()
            state_history = np.append(state_history, s, axis=0)
            control_history = np.append(control_history, u.detach().numpy())

        if method == 'ADP':
            print("ADP calculating time: {:.3f}".format(cal_time) + "s")
            np.savetxt(os.path.join(comparison_dir, 'ADP_state.txt'), state_history)
            np.savetxt(os.path.join(comparison_dir, 'ADP_control.txt'), control_history)
            np.savetxt(os.path.join(log_dir, 'ADP_state.txt'), state_history)
            np.savetxt(os.path.join(log_dir, 'ADP_control.txt'), control_history)

        elif method == 'MPC':
            print("MPC calculating time: {:.3f}".format(cal_time) + "s")
            np.savetxt(os.path.join(comparison_dir, 'structured_MPC_state.txt'), state_history)
            np.savetxt(os.path.join(comparison_dir, 'structured_MPC_control.txt'), control_history)

        else:
            np.savetxt(os.path.join(comparison_dir, 'Open_loop_state.txt'), state_history)



if __name__ == '__main__':
    log_dir = "./Results_dir/2020-06-11-13-24-10"
    simulation(log_dir)