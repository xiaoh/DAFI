# Copyright 2020 Virginia Polytechnic Institute and State University.
""" Dynamic model for solving the scalar inversion problem used for
uncertainty quantification. """

# standard library imports
import os

# third party imports
import numpy as np
import yaml

# local imports
from dafi import PhysicsModel


class Model(PhysicsModel):

    def __init__(self, inputs_dafi, inputs_model):
        # save the required inputs
        self.nsamples = inputs_dafi['nsamples']

        # read inputs
        self.init_state = np.array(inputs_model['x_init_mean'])
        self.state_std = np.array(inputs_model['x_init_std'])
        self.obs = np.array(inputs_model['obs'])
        self.obs_err = np.diag(np.array(inputs_model['obs_std'])**2)

        # required attributes.
        self.name = 'Scalar Inversion Case for UQ'

        # other attributes
        self.nstate = len(self.init_state)

    def __str__(self):
        return self.name

    def generate_ensemble(self):
        state_vec = np.empty([self.nstate, self.nsamples])
        for i in range(self.nstate):
            state_vec[i, :] = np.random.normal(
                self.init_state[i], self.state_std[i], self.nsamples)
        return state_vec

    def state_to_observation(self, state_vec):
        obs_1 = state_vec[0, :]
        obs_2 = state_vec[0, :] + state_vec[1, :]**3
        model_obs = np.array([obs_1, obs_2])
        return model_obs

    def get_obs(self, time):
        return self.obs, self.obs_err
