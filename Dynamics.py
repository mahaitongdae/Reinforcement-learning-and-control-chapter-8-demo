from __future__ import print_function
import torch
import numpy as np
from Config import DynamicsConfig
import math

PI = 3.1415926


class VehicleDynamics(DynamicsConfig):

    def __init__(self):
        self._state = torch.zeros([self.BATCH_SIZE, self.DYNAMICS_DIM])
        self.init_state = torch.zeros([self.BATCH_SIZE, self.DYNAMICS_DIM])
        self._reset_index = np.zeros([self.BATCH_SIZE, 1])
        self.initialize_state()
        super(VehicleDynamics, self).__init__()

    def initialize_state(self):
        """
        random initialization of state.

        Returns
        -------

        """
        self.init_state[:, 0] = torch.normal(0.0, 0.6, [self.BATCH_SIZE,])
        self.init_state[:, 1] = torch.normal(0.0, 0.4, [self.BATCH_SIZE,])
        self.init_state[:, 2] = torch.normal(0.0, 0.15, [self.BATCH_SIZE,])
        self.init_state[:, 3] = torch.normal(0.0, 0.1, [self.BATCH_SIZE,])
        self.init_state[:, 4] = torch.linspace(0.0, 1.5 * np.pi / self.k_curve, self.BATCH_SIZE)
        init_ref = self.reference_trajectory(self.init_state[:, 4])
        init_ref_all = torch.cat((init_ref, torch.zeros([self.BATCH_SIZE,1])),1)
        self._state = self.init_state
        init_state = self.init_state + init_ref_all
        return init_state

    def check_done(self, state):
        """
        Check if the states reach unreasonable zone and reset them
        Parameters
        ----------
        state: tensor   shape: [BATCH_SIZE, STATE_DIMENSION]
            state used for checking.
        Returns
        -------

        """
        threshold = np.kron(np.ones([self.BATCH_SIZE, 1]), np.array([self.y_range, self.psi_range, self.x_range]))
        threshold = np.array(threshold, dtype='float32')
        threshold = torch.from_numpy(threshold)
        check_state = state[:, [0, 2, 4]].clone()
        check_state.detach_()
        sign_error = torch.sign(torch.abs(check_state) - threshold) # if abs state is over threshold, sign_error = 1
        self._reset_index, _ = torch.max(sign_error, 1) # if one state is over threshold, _reset_index = 1
        reset_state = self._reset_state(state)
        return reset_state

    def _reset_state(self, state):
        """
        reset state to initial state.
        Parameters
        ----------
        state: tensor   shape: [BATCH_SIZE, STATE_DIMENSION]
            state used for checking.

        Returns
        -------
        state: state after reset.

        """
        for i in range(self.BATCH_SIZE):
            if self._reset_index[i] == 1:
                state[i, :] = self.init_state[i, :]
        return state

    def _state_function(self, state, control):
        """
        State function of vehicle with Pacejka tire model, i.e. \dot(x)=f(x,u)
        Parameters
        ----------
        state: tensor   shape: [BATCH_SIZE, STATE_DIMENSION]
            current state
        control: tensor   shape: [BATCH_SIZE, ACTION_DIMENSION]
            input
        Returns
        -------
        deri_state.T:   tensor shape: [BATCH_SIZE, ]
            f(x,u)
        F_y1:           tensor shape: [BATCH_SIZE, ]
            front axle lateral force
        F_y2:           tensor shape: [BATCH_SIZE, ]
            rear axle lateral force
        alpha_1:        tensor shape: [BATCH_SIZE, ]
            front wheel slip angle
        alpha_2:        tensor shape: [BATCH_SIZE, ]
            rear wheel slip angle

        """

        # state variable
        y = state[:, 0]                 # lateral position
        u_lateral = state[:, 1]         # lateral speed
        beta = u_lateral / self.u       # yaw angle
        psi = state[:, 2]               # heading angle
        omega_r = state[:, 3]           # yaw rate
        x = state[:, 4]                 # longitudinal position

        # inputs
        delta = control[:, 0]           # front wheel steering angle
        delta.requires_grad_(True)

        # slip angle of front and rear wheels
        alpha_1 = -delta + torch.atan(beta + self.a * omega_r / self.u)
        alpha_2 = torch.atan(beta - self.b * omega_r / self.u)

        # cornering force of front and rear angle, Pacejka tire model
        F_y1 = -self.D * torch.sin(self.C * torch.atan(self.B * alpha_1)) * self.F_z1
        F_y2 = -self.D * torch.sin(self.C * torch.atan(self.B * alpha_2)) * self.F_z2

        # derivative of state
        deri_y = self.u * torch.sin(psi) + u_lateral * torch.cos(psi)
        deri_u_lat = (torch.mul(F_y1, torch.cos(delta)) + F_y2) / (self.m) - self.u * omega_r
        deri_psi = omega_r
        deri_omega_r = (torch.mul(self.a * F_y1, torch.cos(delta)) - self.b * F_y2) / self.I_zz
        deri_x = self.u * torch.cos(psi) - u_lateral * torch.sin(psi)

        deri_state = torch.cat((deri_y[np.newaxis, :],
                                deri_u_lat[np.newaxis, :],
                                deri_psi[np.newaxis, :],
                                deri_omega_r[np.newaxis, :],
                                deri_x[np.newaxis, :]), 0)

        return deri_state.T, F_y1, F_y2, alpha_1, alpha_2

    def _state_function_linear(self, state, control):
        """
        State function of vehicle with linear tire model and linear approximation, i.e. \dot(x) = Ax + Bu
        Parameters
        ----------
        state: tensor   shape: [BATCH_SIZE, STATE_DIMENSION]
            current state
        control: tensor   shape: [BATCH_SIZE, ACTION_DIMENSION]
            input
        Returns
        -------
        deri_state.T:   tensor shape: [BATCH_SIZE, ]
            f(x,u)
        F_y1:           tensor shape: [BATCH_SIZE, ]
            front axle lateral force
        F_y2:           tensor shape: [BATCH_SIZE, ]
            rear axle lateral force
        alpha_1:        tensor shape: [BATCH_SIZE, ]
            front wheel slip angle
        alpha_2:        tensor shape: [BATCH_SIZE, ]
            rear wheel slip angle

        """

        # state variable
        y = state[:, 0]                 # lateral position
        u_lateral = state[:, 1]         # lateral speed
        beta = u_lateral / self.u       # yaw angle
        psi = state[:, 2]               # heading angle
        omega_r = state[:, 3]           # yaw rate
        x = state[:, 4]                 # longitudinal position

        # inputs
        delta = control[:, 0]           # front wheel steering angle
        delta.requires_grad_(True)

        # slip angle of front and rear wheels, with small angle approximation
        alpha_1 = -delta + beta + self.a * omega_r / self.u
        alpha_2 = beta - self.b * omega_r / self.u

        # cornering force of front and rear angle, linear tire model
        F_y1 = self.k1 * alpha_1
        F_y2 = self.k2 * alpha_2

        # derivative of state
        deri_y = self.u * psi + u_lateral
        deri_u_lat = (torch.mul(F_y1, torch.cos(delta)) + F_y2) / (self.m) - self.u * omega_r
        deri_psi = omega_r
        deri_omega_r = (torch.mul(self.a * F_y1, torch.cos(delta)) - self.b * F_y2) / self.I_zz
        deri_x = self.u * torch.cos(psi) - u_lateral * torch.sin(psi)

        deri_state = torch.cat((deri_y[np.newaxis, :],
                                deri_u_lat[np.newaxis, :],
                                deri_psi[np.newaxis, :],
                                deri_omega_r[np.newaxis, :],
                                deri_x[np.newaxis, :]), 0)

        return deri_state.T, F_y1, F_y2, alpha_1, alpha_2

    def reference_trajectory(self, state):
        k = self.k_curve
        a = self.a_curve
        y_ref = a * torch.sin(k * state)
        psi_ref = torch.atan(a * k * torch.cos(k * state))
        zeros = torch.zeros([len(state), ])
        state_ref = torch.cat((y_ref[np.newaxis, :],
                                zeros[np.newaxis, :],
                                psi_ref[np.newaxis, :],
                                zeros[np.newaxis, :]), 0)
        return state_ref.T

    @staticmethod
    def _utility(state, control):
        """

        Parameters
        ----------
        state: tensor   shape: [BATCH_SIZE, STATE_DIMENSION]
            current state
        control: tensor   shape: [BATCH_SIZE, ACTION_DIMENSION]
            current control signal

        Returns
        -------
        utility: tensor   shape: [BATCH_SIZE, ]
            utility, i.e. l(x,u)
        """
        utility = 0.01 * (10 * torch.pow(state[:, 0], 2) + 10 * torch.pow(state[:, 2], 2) + 0.1 * torch.pow(control[:, 0], 2))
        return utility

    def _utility_full(self, state, control):
        utility = 0.01 * (
                    10 * torch.pow(state[:, 0] - self.a_curve * torch.sin(self.k_curve * state[:,4]), 2)
                    + 10 * torch.pow(state[:, 2] - torch.atan(self.a_curve* self.k_curve * torch.cos(self.k_curve* state[:,4])), 2)
                    + 0.1 * torch.pow(control[:, 0], 2))
        return utility

    def step(self, state, control):
        """
        step ahead with discrete state function, i.e. x'=f(x,u)
        Parameters
        ----------
        state: tensor   shape: [BATCH_SIZE, STATE_DIMENSION]
            current state
        control: tensor   shape: [BATCH_SIZE, ACTION_DIMENSION]
            current control signal

        Returns
        -------
        state_next:     tensor shape: [BATCH_SIZE, ]
            x'
        f_xu:           tensor shape: [BATCH_SIZE, ]
            f(x,u)
        utility:        tensor shape: [BATCH_SIZE, ]
        utility, i.e. l(x,u)
        F_y1:           tensor shape: [BATCH_SIZE, ]
            front axle lateral force
        F_y2:           tensor shape: [BATCH_SIZE, ]
            rear axle lateral force
        alpha_1:        tensor shape: [BATCH_SIZE, ]
            front wheel slip angle
        alpha_2:        tensor shape: [BATCH_SIZE, ]
            rear wheel slip angle

        """
        deri_state, F_y1, F_y2, alpha_1, alpha_2 = self._state_function_linear(state, control)
        state_next = state + self.Ts * deri_state
        utility = self._utility_full(state, control) # todo:fully
        f_xu = deri_state[:, 0:4]
        return state_next, f_xu, utility, F_y1, F_y2, alpha_1, alpha_2

