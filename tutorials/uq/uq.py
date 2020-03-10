# Copyright 2020 Virginia Polytechnic Institute and State University.
""" Dynamic model for solving the model used for uncertainty
quantification. """

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

        # read input file
        input_file = inputs_model['input_file']
        with open(input_file, 'r') as f:
            inputs_model = yaml.load(f, yaml.SafeLoader)

        x_mean = np.array(inputs_model['x_init_mean'])
        self.state_std =  np.array(inputs_model['x_init_std'])
        self.obs =  np.array(inputs_model['obs'])
        self.obs_err = np.diag(np.array(inputs_model['obs_std'])**2)

        # required attributes.
        self.name = 'uq'
        self.nstate = len(x_mean)
        self.nobs = len(self.obs)
        self.init_state = np.array(x_mean)


    def __str__(self):
        str_info = 'UQ model.'
        return str_info


    def generate_ensemble(self):
        state_vec = np.empty([self.nstate, self.nsamples])
        for i in range(self.nstate):
            state_vec[i,:] = np.random.normal(
                self.init_state[i], self.state_std[i], self.nsamples)
        return state_vec


    def forecast_to_time(self, state_vec_current, end_time):
        """ Not needed for UQ model. """
        return state_vec_current


    def state_to_observation(self, state_vec):
        obs_1 = state_vec[0, :]
        obs_2 = state_vec[0, :] + state_vec[1, :]**3
        model_obs = np.array([obs_1, obs_2])
        return model_obs


    def get_obs(self, time):
        return self.obs, self.obs_err


    def clean(self):
        """ Cleanup before exiting. """
        pass
