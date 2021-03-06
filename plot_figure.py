import numpy as np
import os
from Config import DynamicsConfig
from utils import myplot
S_DIM = 4
A_DIM = 1


def plot_comparison(picture_dir):
    '''
    Plot comparison figure among ADP, MPC & open-loop solution.
    Trajectory, tracking error and control signal plot

    Parameters
    ----------
    picture_dir: string
        location of figure saved.

    '''
    config = DynamicsConfig()
    comparison_dir = "Results_dir/Comparison_Data"
    if os.path.exists(os.path.join(comparison_dir, 'MPC_state.txt')) == 0 or \
        os.path.exists(os.path.join(comparison_dir, 'Open_loop_state.txt')) == 0:
        print('No comparison state data!')
    else:
        mpc_state = np.loadtxt(os.path.join(comparison_dir, 'structured_MPC_state.txt'))
        open_loop_state = np.loadtxt(os.path.join(comparison_dir, 'Open_loop_state.txt'))
        adp_state = np.loadtxt(os.path.join(comparison_dir, 'ADP_state.txt'))

        mpc_trajectory = (mpc_state[:, 4], mpc_state[:, 0])
        open_loop_trajectory = (open_loop_state[:, 4], open_loop_state[:, 0])
        adp_trajectory = (adp_state[:, 4], adp_state[:, 0])
        trajectory_data = [mpc_trajectory, adp_trajectory, open_loop_trajectory]
        myplot(trajectory_data, 3, "xy",
               fname=os.path.join(picture_dir, 'trajectory.png'),
               xlabel="longitudinal position [m]",
               ylabel="Lateral position [m]",
               legend=["MPC", "ADP", "Open-loop"],
               legend_loc="upper left"
               )

        mpc_error = (mpc_state[:, 4], mpc_state[:, 0] - config.a_curve * np.sin(config.k_curve * mpc_state[:, 4]))
        open_loop_error =  (open_loop_state[:, 4], open_loop_state[:, 0] - config.a_curve * np.sin(config.k_curve * open_loop_state[:, 4]))
        adp_error = (adp_state[:, 4], 1 * (adp_state[:, 0] - config.a_curve * np.sin(config.k_curve * adp_state[:, 4])))
        error_data = [mpc_error, adp_error, open_loop_error]
        myplot(error_data, 3, "xy",
               fname=os.path.join(picture_dir,'trajectory_error.png'),
               xlabel="longitudinal position [m]",
               ylabel="Lateral position error [m]",
               legend=["MPC", "ADP", "Open-loop"],
               legend_loc="lower left"
               )
        y_avs_error = []
        for [i, d] in enumerate(error_data):
            y_avs_error.append(np.mean(np.abs(d[1])))
        print("Tracking error of lateral position:")
        print("MPC:{:.3e} | ".format(y_avs_error[0]) +
              "ADP:{:.3e} | ".format(y_avs_error[1]) +
              "Open-loop:{:.3e} | ".format(y_avs_error[2]))

        mpc_psi_error = (mpc_state[:, 4], 180 / np.pi * (mpc_state[:, 2] -
                         np.arctan(config.a_curve * config.k_curve * np.cos(config.k_curve * mpc_state[:, 4]))))
        open_loop_psi_error = (open_loop_state[:, 4], 180 / np.pi * (open_loop_state[:, 2] -
                               np.arctan(config.a_curve * config.k_curve * np.cos(config.k_curve * open_loop_state[:, 4]))))
        adp_psi_error = (adp_state[:, 4], 180 / np.pi * (adp_state[:, 2] -
                         np.arctan(config.a_curve * config.k_curve * np.cos(config.k_curve * adp_state[:, 4]))))
        psi_error_data = [mpc_psi_error, adp_psi_error, open_loop_psi_error]
        myplot(psi_error_data, 3, "xy",
               fname=os.path.join(picture_dir, 'head_angle_error.png'),
               xlabel="longitudinal position [m]",
               ylabel="head angle error [degree]",
               legend=["MPC", "ADP", "Open-loop"],
               legend_loc="lower left"
               )

        psi_avs_error = []
        for [i, d] in enumerate(psi_error_data):
            psi_avs_error.append(np.mean(np.abs(d[1])))
        print("Tracking error of heading angle:")
        print("MPC:{:.3e} | ".format(psi_avs_error[0]) +
              "ADP:{:.3e} | ".format(psi_avs_error[1]) +
              "Open-loop:{:.3e} | ".format(psi_avs_error[2]))

        mpc_control = np.loadtxt(os.path.join(comparison_dir, 'structured_MPC_control.txt'))
        open_loop_control = np.loadtxt(os.path.join(comparison_dir, 'Open_loop_control.txt'))
        adp_control = np.loadtxt(os.path.join(comparison_dir, 'ADP_control.txt'))
        mpc_control_tuple = (mpc_state[1:, 4] ,180 / np.pi * mpc_control)
        open_loop_control_tuple = (open_loop_state[1:, 4], 180 / np.pi * open_loop_control)
        adp_control_tuple = (adp_state[1:, 4], 180 / np.pi * adp_control)
        control_plot_data = [mpc_control_tuple, adp_control_tuple, open_loop_control_tuple]
        myplot(control_plot_data, 3, "xy",
               fname=os.path.join(picture_dir, 'control.png'),
               xlabel="longitudinal position [m]",
               ylabel="steering angle [degree]",
               legend=["MPC", "ADP", "Open-loop"],
               legend_loc="upper left"
               )

        mpc_control_error = mpc_control - open_loop_control
        adp_control_error = adp_control - open_loop_control
        print("Control error:")
        print("MPC:{:.3e} | ".format(np.mean(np.abs(mpc_control_error))) +
              "ADP:{:.3e} | ".format(np.mean(np.abs(adp_control_error))))

def plot_loss_decent_compare(comparison_dir):
    fs_step = ["10","20","30"]
    value_loss = []
    policy_loss = []
    p_scatter_data = []
    v_scatter_data = []
    for [i,fs] in enumerate(fs_step):
        value_np = "value_loss_" + fs + ".txt"
        policy_np = "policy_loss_" + fs + ".txt"
        value_loss.append(np.loadtxt(os.path.join(comparison_dir, value_np)))
        policy_loss.append(np.loadtxt(os.path.join(comparison_dir, policy_np)))
        p_scatter_data.append((range(len(policy_loss[i])), policy_loss[i]))
        v_scatter_data.append((range(len(value_loss[i])), np.log10(value_loss[i])))
    myplot(v_scatter_data,3,"scatter",
           fname=(os.path.join(comparison_dir, "p_loss.png")),
           xlabel="iteration",
           ylabel="log value loss",
           legend=["10 steps","20 steps","30 steps"],
           xlim=[0, 5000],
           ylim=[0, 3]
           )

def plot_loss_decent(log_dir):
    value_loss = np.loadtxt(os.path.join(log_dir, "value_loss.txt"))
    policy_loss = np.loadtxt(os.path.join(log_dir, "policy_loss.txt"))
    value_loss_tuple = (range(len(value_loss)), np.log10(value_loss))
    policy_loss_tuple = (range(len(policy_loss)), np.log10(policy_loss))
    loss = [value_loss_tuple, policy_loss_tuple]
    myplot(loss, 2, "scatter",
           fname=(os.path.join(log_dir, "loss.png")),
           xlabel="iteration",
           ylabel="log value loss",
           legend=["PEV loss", "PIM loss"],
           xlim=[0, 5000],
           ylim=[-1, 3]
           )


def adp_simulation_plot(log_dir):
    '''
    Simulate and plot trajectory and control after ADP training algorithm.

    Parameters
    ----------
    log_dir: string
        location of data and figures saved.

    '''
    state_history = np.loadtxt(os.path.join(log_dir, 'ADP_state.txt'))
    control_history = np.loadtxt(os.path.join(log_dir, 'ADP_control.txt'))
    trajectory = (state_history[:, -1], state_history[:, 0])
    myplot(trajectory, 1, "xy",
           fname=os.path.join(log_dir, 'trajectory.png'),
           xlabel="longitudinal position [m]",
           ylabel="Lateral position [m]",
           legend=["trajectory"],
           legend_loc="upper left"
    )
    u_lat = (state_history[:, -1], state_history[:, 1])
    psi =(state_history[:, -1], state_history[:, 2])
    omega = (state_history[:, -1], state_history[:, 3])
    data = [u_lat, psi, omega]
    legend=["$u_{lat}$", "$\psi$", "$\omega$"]
    myplot(data, 3, "xy",
           fname=os.path.join(log_dir, 'other_state.png'),
           xlabel="longitudinal position [m]",
           legend=legend
           )
    control_history_plot = (state_history[1:, -1], 180 / np.pi * control_history)
    myplot(control_history_plot, 1, "xy",
           fname=os.path.join(log_dir, 'control.png'),
           xlabel="longitudinal position [m]",
           ylabel="steering angle [degree]"
           )


if __name__ == '__main__':
    Figures_dir = './Figures/'
    # adp_simulation_plot("./Results_dir/2020-06-10-19-15-5000")
    plot_comparison(Figures_dir)
    # plot_loss_decent("./Results_dir/2020-05-18-02-21-10000")