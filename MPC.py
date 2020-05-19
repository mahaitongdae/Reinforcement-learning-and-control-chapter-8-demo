"""
    <Reinforcement Learning and Control>(Year 2020)
    by Shengbo Eben Li
        @ Intelligent Driving Lab, Tsinghua University

    ADP example for lane keeping problem in a circle road

    [Method]
    Model predictive control(MPC) as comparison

"""
from matplotlib import pyplot as plt
from Solver import Solver
from Config import Dynamics_Config
import numpy as np
import time
import os

def main():
    log_dir = "Results_dir/Comparison_Data"
    config = Dynamics_Config()
    solver=Solver()
    x = [0.0, 0.0, 0.033, 0.0, 0.0]
    state_history = np.array(x)
    control_history = np.empty([0, 1])
    time_init = time.time()
    for i in range(config.NP_TOTAL):
        state, control = solver.structured_mpc_solver(x, config.NP)
        x = state[1]
        u = control[0]
        state_history = np.append(state_history, x)
        control_history = np.append(control_history, u)
        x = x.tolist()
        print("steps:{:3d}".format(i) + " | state: " + str(x))
    print("MPC calculating time: {:.3f}".format(time.time() - time_init) + "s")
    state_history = state_history.reshape(-1,config.DYNAMICS_DIM)
    np.savetxt(os.path.join(log_dir, 'structured_MPC_state.txt'), state_history)
    np.savetxt(os.path.join(log_dir, 'structured_MPC_control.txt'), control_history)

if __name__ == '__main__':
    main()