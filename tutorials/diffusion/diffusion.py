# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Physics model for the one dimension diffusion equation."""

# standard library imports
import ast
import sys
import math
import os

# third party imports
import numpy as np
import scipy.sparse as sp
import yaml

# local imports
from dafi import PhysicsModel
from dafi import random_field as rf


class Model(PhysicsModel):
    """ Dynamic model for solving one dimensional diffusion equation.
    """

    def __init__(self, inputs_dafi, inputs_model):
        # save the main input
        self.nsamples = inputs_dafi['nsamples']
        self.max_iteration = inputs_dafi['max_iterations']

        # required attributes
        self.name = '1D diffusion Equation'

        # case properties
        self.space_interval = 0.01
        self.max_length = 1

        # counter for number of state_to_observation calls
        self.counter = 0

        # read input file
        input_file = inputs_model['input_file']
        with open(input_file, 'r') as f:
            inputs_model = yaml.load(f, yaml.SafeLoader)
        mu_init = inputs_model['prior_mean']
        self.stddev = inputs_model['stddev']
        self.length_scale = inputs_model['length_scale']
        self.obs_loc = inputs_model['obs_locations']
        obs_rel_std = inputs_model['obs_rel_stddev']
        obs_abs_std = inputs_model['obs_abs_stddev']
        self.nmodes = inputs_model['nmodes']
        self.calculate_kl_flag = inputs_model['calculate_kl_flag']

        # create save directory
        self.savedir = './results_diffusion'
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

        # create spatial coordinate and save
        self.x_coor = np.arange(
            0, self.max_length+self.space_interval, self.space_interval)
        np.savetxt(os.path.join(self.savedir, 'x_coor.dat'), self.x_coor)

        # dimension of state space
        self.ncells = self.x_coor.shape[0]

        # create source term fx
        source = np.zeros(self.ncells)
        for i in range(self.ncells - 1):
            source[i] = math.sin(0.2*np.pi*self.x_coor[i])
        source = np.mat(source).T
        self.fx = source.A

        # create or read modes for K-L expansion
        if self.calculate_kl_flag:
            # covariance
            cov = rf.covariance.generate_cov(
                'sqrexp', self.stddev, coords=self.x_coor,
                length_scales=[self.length_scale])
            # KL decomposition
            eig_vals, kl_modes = rf.calc_kl_modes(
                cov, self.nmodes, self.space_interval, normalize=False)
            # save
            np.savetxt(os.path.join(self.savedir, 'KLmodes.dat'), kl_modes)
            np.savetxt(os.path.join(self.savedir, 'eigVals.dat'), eig_vals)
            np.savetxt(os.path.join(self.savedir, 'eigValsNorm.dat'),
                       eig_vals/eig_vals[0])
            self.kl_modes = kl_modes
        else:
            self.kl_modes = np.loadtxt('KLmodes.dat')

        # create observations
        true_obs = self._truth()
        std_obs = obs_rel_std * true_obs + obs_abs_std
        self.obs_error = np.diag(std_obs**2)
        self.obs = np.random.multivariate_normal(true_obs, self.obs_error)
        self.nobs = len(self.obs)

    def __str__(self):
        return '1-D heat diffusion model.'

    def generate_ensemble(self):
        """ Return states at the first data assimilation time-step. """
        return np.random.normal(0, 1, [self.nmodes, self.nsamples])

    def state_to_observation(self, state_vec):
        """ Map the states to observation space (X to HX).

        Parameters
        ----------
        state_vec: ndarray
            Ensemble of state variables

        Returns
        -------
        model_obs: ndarray
            Ensemble of states in observation space (HX).
            *dtype=float*, *ndim=2*, *shape=(nstate_obs, nsamples)*
        """
        self.counter += 1
        u_mat = np.zeros([self.ncells, self.nsamples])
        model_obs = np.zeros([self.nobs, self.nsamples])
        # solve model in observation space for each sample
        for isample in range(self.nsamples):
            mu, mu_dot = self._get_mu_mudot(state_vec[:, isample])
            u_vec = self._solve_diffusion_equation(mu, mu_dot)
            model_obs[:, isample] = np.interp(self.obs_loc, self.x_coor, u_vec)
            u_mat[:, isample] = u_vec
        np.savetxt(os.path.join(self.savedir, f'U.{self.counter}'), u_mat)
        return model_obs

    def get_obs(self, time):
        """ Return the observation and observation covariance. """
        return self.obs, self.obs_error

    def _truth(self):
        """Return synthetic truth in obervation space.

        Returns
        -------
        model_obs: ndarray
            Synthetic truth in observation space.
            *dtype=float*, *ndim=1*, *shape=(nstate_obs)*
        """
        # Synthetic truth, omega=1,1,1,0,0,0,...
        synthetic_mu = np.zeros((self.ncells))
        omega = np.zeros((self.nmodes))
        omega[0:3] = np.array([1, 1, 1])
        np.savetxt(os.path.join(self.savedir, 'omega_truth.dat'), omega)
        for i in range(self.nmodes):
            synthetic_mu += omega[i] * self.kl_modes[:, i]
        mu = np.exp(synthetic_mu[:-1])
        mu_dot = np.gradient(mu, self.space_interval)
        np.savetxt(os.path.join(self.savedir, 'mu_truth.dat'), mu)

        # solve
        u_vec = self._solve_diffusion_equation(mu, mu_dot)
        np.savetxt(os.path.join(self.savedir, 'u_truth.dat'), u_vec)

        # interpolate
        return np.interp(self.obs_loc, self.x_coor, u_vec)

    def _solve_diffusion_equation(self, mu, mu_dot):
        """ Solve the one-dimensional diffusion equation.

        Parameters
        ----------
        mu : ndarray
            Diffusivity field. *dtype=float*, *ndim=1*, *shape=(ncells)*
        mu_dot : ndarray
            Spatial derivative (d/dx) of the diffusivity field.
            *dtype=float*, *ndim=1*, *shape=(ncells)*

        Returns
        -------
        u : ndarray
            Output field (e.g. temperature).
            *dtype=float*, *ndim=1*, *shape=(ncells)*
        """
        # calculate coeffient of Du = E
        dx = self.space_interval
        A = 0.5 * mu_dot / dx + mu / dx / dx
        B = -2 * mu / dx / dx
        C = -0.5 * mu_dot / dx + mu / dx / dx
        B1 = np.append(B, 1)
        D1 = np.diag(B1)
        D1[0][0] = 1
        D2 = np.diag(C, 1)   # C above the main diagonal
        D2[0][1] = 0
        A1 = np.append(A, 0)
        D3 = np.diag(A1[1:self.ncells], -1)   # A below the main diagonal
        D = D1+D2+D3
        D = np.mat(D)
        u = - (D.I)*(self.fx)
        return u.getA1()

    def _get_mu_mudot(self, state):
        """ Convert state (KL coefficients) to diffusivity field and
        calculate spatial derivative.

        Parameters
        ----------
        state : ndarray
            KL coefficients. *dtype=float*, *ndim=1*, *shape=(nstate)*

        Returns
        -------
        mu : ndarray
            Diffusivity field. *dtype=float*, *ndim=1*, *shape=(ncells)*
        mu_dot : ndarray
            Spatial derivative (d/dx) of the diffusivity field.
            *dtype=float*, *ndim=1*, *shape=(ncells)*
        """
        mu = np.zeros(self.ncells-1)
        mu_dot = np.zeros(self.ncells-1)
        # obtain diffusivity based on KL coefficient and KL mode
        for imode in range(self.nmodes):
            mode_dot = np.gradient(self.kl_modes[:-1, imode], self.space_interval)
            mu_dot += state[imode] * mode_dot
        # for imode in range(self.nmodes):
            mu += state[imode] * self.kl_modes[:-1, imode]
        mu = np.exp(mu)
        mu_dot *= mu
        return mu, mu_dot
