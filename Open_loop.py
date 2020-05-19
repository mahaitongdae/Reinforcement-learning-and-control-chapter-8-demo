"""
    <Reinforcement Learning and Control>(Year 2020)
    by Shengbo Eben Li
        @ Intelligent Driving Lab, Tsinghua University

    ADP example for lane keeping problem in a circle road

    [Method]
    Open loop solution as comparison

"""
from matplotlib import pyplot as plt
from Solver import Solver
from Config import Dynamics_Config
import numpy as np
import os

def main():
    log_dir = "Results_dir/Comparison_Data"
    config = Dynamics_Config()
    x_init = [0.0, 0.0, config.psi_init, 0.0, 0.0]
    solver=Solver()
    state, control = solver.mpc_solver(x_init, config.NP_TOTAL)
    np.savetxt(os.path.join(log_dir, 'Open_loop_state.txt'), state)
    np.savetxt(os.path.join(log_dir, 'Open_loop_control.txt'), control)

if __name__ == '__main__':
    main()
    # test()
