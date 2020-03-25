# Copyright 2020 Virginia Polytechnic Institute and State University.
""" Physics model for the one dimension diffusion equation."""

# standard library imports
import os

# third party imports
import numpy as np
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

        # required attributes
        self.name = '1D diffusion Equation'

        # case properties
        self.space_interval = 0.01
        max_length = 1.0

        # counter for number of state_to_observation calls
        self.counter = 0

        # read input file
        input_file = inputs_model['input_file']
        with open(input_file, 'r') as f:
            inputs_model = yaml.load(f, yaml.SafeLoader)
        mu_init = inputs_model['prior_mean']
        stddev = inputs_model['stddev']
        length_scale = inputs_model['length_scale']
        self.obs_loc = inputs_model['obs_locations']
        obs_rel_std = inputs_model['obs_rel_stddev']
        obs_abs_std = inputs_model['obs_abs_stddev']
        self.nmodes = inputs_model['nmodes']
        calculate_kl_flag = inputs_model.get('calculate_kl_flag', True)

        # create save directory
        self.savedir = './results_diffusion'
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

        # create spatial coordinate and save
        self.x_coor = np.expand_dims(np.arange(
            0, max_length+self.space_interval, self.space_interval), 1)
        np.savetxt(os.path.join(self.savedir, 'x_coor.dat'), self.x_coor)

        # dimension of state space
        self.ncells = self.x_coor.shape[0]

        # create source term
        period = 10 * max_length
        mag = 1.0
        self.fx = mag * np.sin(2.0 * np.pi / (period) * self.x_coor)
        self.fx[-1] = 0.0  # Q

        # create or read modes for K-L expansion
        if calculate_kl_flag:
            # covariance
            cov = rf.covariance.generate_cov(
                'sqrexp', stddev, coords=self.x_coor,
                length_scales=[length_scale])
            # KL decomposition
            eig_vals, kl_modes = rf.calc_kl_modes(
                cov, self.nmodes, self.space_interval, normalize=False)
            # save
            np.savetxt(os.path.join(self.savedir, 'KLmodes.dat'), kl_modes)
            np.savetxt(os.path.join(self.savedir, 'eigVals.dat'), eig_vals)
            np.savetxt(os.path.join(self.savedir, 'eigValsNorm.dat'),
                       eig_vals/eig_vals[0])
        else:
            kl_modes = np.loadtxt('KLmodes.dat')
        kl_modes = kl_modes[:-1, :]  # Q

        # create random field
        self.rf = rf.LogNormal(kl_modes, mu_init, self.space_interval)

        # create observations
        true_obs = self._truth()
        std_obs = obs_rel_std * true_obs + obs_abs_std
        self.obs_error = np.diag(std_obs**2)
        self.obs = np.random.multivariate_normal(true_obs, self.obs_error)
        self.nobs = len(self.obs)
        np.savetxt(os.path.join(self.savedir, 'obs.dat'), self.obs)
        np.savetxt(os.path.join(self.savedir, 'std_obs.dat'), std_obs)

        # solve baseline
        mu = np.ones(self.ncells-1) * mu_init  # Q
        mu_dot = np.zeros(self.ncells-1)  # Q
        u_vec = self._solve_diffusion_equation(mu, mu_dot)
        np.savetxt(os.path.join(self.savedir, 'u_baseline.dat'), u_vec)

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
            mu = np.squeeze(self.rf.reconstruct_func(state_vec[:, isample]))
            mu_dot = np.gradient(mu, self.space_interval)
            u_vec = self._solve_diffusion_equation(mu, mu_dot)
            model_obs[:, isample] = np.interp(
                self.obs_loc, np.squeeze(self.x_coor), u_vec)
            u_mat[:, isample] = u_vec
        np.savetxt(os.path.join(self.savedir, f'U.{self.counter}'), u_mat)
        return model_obs

    def get_obs(self, time):
        """ Return the observation and observation covariance. """
        return self.obs, self.obs_error

    def _truth(self):
        """Return synthetic truth in observation space.

        Returns
        -------
        model_obs: ndarray
            Synthetic truth in observation space.
            *dtype=float*, *ndim=1*, *shape=(nstate_obs)*
        """
        # Synthetic truth, omega=1,1,1,0,0,0,...
        omega = np.zeros((self.nmodes))
        omega[0:3] = np.array([1, 1, 1])
        np.savetxt(os.path.join(self.savedir, 'omega_truth.dat'), omega)
        mu = np.squeeze(self.rf.reconstruct_func(omega))
        mu_dot = np.gradient(mu, self.space_interval)
        np.savetxt(os.path.join(self.savedir, 'mu_truth.dat'), mu)

        # solve
        u_vec = self._solve_diffusion_equation(mu, mu_dot)
        np.savetxt(os.path.join(self.savedir, 'u_truth.dat'), u_vec)

        # interpolate
        return np.interp(self.obs_loc, np.squeeze(self.x_coor), u_vec)

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
        D2 = np.diag(C, 1)  # C above the main diagonal
        D2[0][1] = 0
        A1 = np.append(A, 0)
        D3 = np.diag(A1[1:self.ncells], -1)  # A below the main diagonal
        D = D1+D2+D3
        D = np.mat(D)
        u = - (D.I)*(self.fx)
        return u.getA1()
