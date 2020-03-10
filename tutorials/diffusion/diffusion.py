# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Dynamic model for solving one dimension heat diffusion system."""

# standard library imports
import ast
import sys
import math
import os

# third party imports
import numpy as np
from numpy import linalg as la
import scipy.sparse as sp
from scipy.integrate import ode
import yaml

# local imports
from dafi import PhysicsModel

# from dafi.dyn_model import DynModel
# from dafi.utilities import read_input_data, str2bool


counter = 0

# create save directory
savedir = './results_diffusion'
if not os.path.exists(savedir):
    os.makedirs(savedir)

class Model(PhysicsModel):
    """ Dynamic model for solving one dimension heat diffusion equation. """

    def __init__(self, inputs_dafi, inputs_model):
        # save the main input
        self.nsamples = inputs_dafi['nsamples']
        self.max_iteration = inputs_dafi['max_iterations']
        
        # read input file
        input_file = inputs_model['input_file']
        with open(input_file, 'r') as f:
            inputs_model = yaml.load(f, yaml.SafeLoader)


        self.obs_rel_std = float(inputs_model['obs_rel_std'])
        self.obs_abs_std = float(inputs_model['obs_abs_std'])
        self.length_scale = float(inputs_model['length_scale'])
        self.nmodes = int(inputs_model['nmodes'])
        self.sigma = float(inputs_model['sigma'])
        # self.projected_truth_flag = str2bool(inputs_model['projected_truth_flag'])
        self.calculate_kl_flag = int(inputs_model['calculate_kl_flag'])

        mu_init = float(inputs_model['mu_init'])

        # required attributes
        self.name = '1D heat diffusion Equation'
        self.space_interval = 0.01
        self.max_length = 1
        self.nstate_obs = 3



        # create spatial coordinate and save
        self.x_coor = np.arange(
            0, self.max_length+self.space_interval, self.space_interval)
        np.savetxt('x_coor.dat', self.x_coor)

        # initialize state vector
        self.omega_init = np.zeros((self.nmodes))
        self.init_state = np.zeros(self.x_coor.shape)
        # dimension of state space
        self.nstate = len(self.init_state)

        # create source term fx
        S = np.zeros(self.nstate)
        for i in range(self.nstate - 1):
            S[i] = math.sin(0.2*np.pi*self.x_coor[i])
        S = np.mat(S).T
        self.fx = S.A

        if self.calculate_kl_flag:
        # create modes for K-L expansion
            cov = np.zeros((self.nstate, self.nstate))
            for i in range(self.nstate):
                for j in range(self.nstate):
                    cov[i][j] = self.sigma * self.sigma * math.exp(
                        -abs(self.x_coor[i]-self.x_coor[j])**2/self.length_scale**2/2)
            cov_weigted = cov * self.space_interval
            eigVals, eigVecs = sp.linalg.eigsh(cov, k=self.nmodes)
            ascendingOrder = eigVals.argsort()
            descendingOrder = ascendingOrder[::-1]
            eigVals = eigVals[descendingOrder]
            eigVecs = eigVecs[:, descendingOrder]
            eigVecs_weighed = np.dot(np.diag(self.space_interval * np.ones(len(self.x_coor))), eigVecs)
            # calculate KL modes: eigVec * sqrt(eigVal)
            self.KL_mode = np.zeros([self.nstate, self.nmodes])
            for i in np.arange(self.nmodes):
                self.KL_mode[:, i] = eigVecs_weighed[::-1, i] * np.sqrt(eigVals[i])
            np.savetxt('KLmodes.dat', self.KL_mode)
            np.savetxt('norm_eigVals.dat', eigVals/eigVals[0])
        else:
            self.KL_mode = np.loadtxt('KLmodes.dat')
        # project the baseline to KL basis
        log_mu_init = [np.log(mu_init)] * self.nstate
        for i in range(self.nmodes):
            self.omega_init[i] = np.trapz(
                log_mu_init * self.KL_mode[:, i], x=self.x_coor)
            self.omega_init[i] /= np.trapz(self.KL_mode[:, i]
                                           * self.KL_mode[:, i], x=self.x_coor)
        np.savetxt('omega_init.dat', self.omega_init)

    def __str__(self):
        str_info = '1-D heat diffusion model.'
        return str_info

    # required methods
    def generate_ensemble(self):
        """ Return states at the first data assimilation time-step.

        Args:
        -----
        nstate : size of state space
        nsamples : number of samples
        nstate_obs : size of observation space
        nmodes : number of modes
        omega_init : initial KL-expansion coefficient


        Returns
        -------
        state_init : ndarray
            State variables at current time.
            ``dtype=float``, ``ndim=2``, ``shape=(nstate_aug, nsamples)``
        model_obs : ndarray
            Forcast ensemble in observation space.
            ``dtype=float``, ``ndim=2``, ``shape=(nstate_aug, nsamples)``
        """

        state_init = np.zeros([self.nstate, self.nsamples])
        model_obs = np.zeros([self.nstate_obs, self.nsamples])
        para_init = np.zeros([self.nmodes, self.nsamples])
        # generate initial KL expansion coefficient
        for i in range(self.nmodes):
            para_init[i, :] = self.omega_init[i] + \
                np.random.normal(0, 1, self.nsamples)
        # augment the state with KL expansion coefficient
        state_init = para_init # np.concatenate((state_init, para_init)) TODO
        model_obs = self.state_to_observation(state_init)
        return state_init

    def forecast_to_time(self, state_vec_current, next_end_time):
        """ Returns states at the next end time."""

        return state_vec_current

    def state_to_observation(self, state_vec):
        """ Forward the states to observation space (from X to HX).

        Parameters
        ----------
        state_vec: ndarray
            current state variables

        Args:
        -----
        nstate : size of state space
        nsamples : number of samples
        nstate_obs : size of observation space
        nmodes : number of modes
        KL_mode : basis set of KL expansion
        space_interval : space interval
        fx : force term

        Returns
        -------
        model_obs: ndarray
            ensemble realization in observation space (HX).
            ``dtype=float``, ``ndim=2``,
            ``shape=(nstate_obs, nsamples)``
        """
        u_mat = np.zeros((self.nstate, self.nsamples))
        model_obs = np.zeros((self.nstate_obs, self.nsamples))
        omega = state_vec  # [self.nstate:, :]
        # solve model in observation space for each sample
        for i_nsample in range(self.nsamples):
            mu_dot = np.zeros(self.nstate-1)
            mu = np.zeros(self.nstate-1)
            # obtain diffusivity based on KL coefficient and KL mode
            for i in range(self.nstate-1):
                for imode in range(self.nmodes):
                    mu_dot[i] += omega[imode, i_nsample] * (
                        self.KL_mode[i+1, imode] - self.KL_mode[i, imode]) / \
                        self.space_interval
            for imode in range(self.nmodes):
                mu += omega[imode, i_nsample] * self.KL_mode[:-1, imode]
            mu = np.exp(mu)
            mu_dot = mu*mu_dot

            # calculate coeffient of Du = E
            A = 0.5 * mu_dot / self.space_interval + mu / \
                self.space_interval/self.space_interval
            B = -2 * mu / self.space_interval / self.space_interval
            C = -0.5 * mu_dot / self.space_interval + \
                mu / self.space_interval / self.space_interval
            B1 = np.append(B, 1)
            D1 = np.diag(B1)
            D1[0][0] = 1
            D2 = np.diag(C, 1)   # C above the main diagonal
            D2[0][1] = 0
            A1 = np.append(A, 0)
            D3 = np.diag(A1[1:101], -1)   # A below the main diagonal
            D = D1+D2+D3
            D = np.mat(D)
            u = - (D.I)*(self.fx)
            u = u.getA1()
            u_mat[:, i_nsample] = u
        global counter
        np.savetxt(os.path.join(savedir, f'U.{counter}'), u_mat)
        counter += 1
        model_obs[0,:] = u_mat[25, :]
        model_obs[1,:] = u_mat[50, :]
        model_obs[2,:] = u_mat[75, :]
        return model_obs

    def get_obs(self, next_end_time):
        """ Return the observation and observation covariance.

        Parameters
        ----------
        next_end_time : float
            Next end time at which observation is requested.

        Returns
        -------
        obs : ndarray
            Ensemble observations.
            ``dtype=float``, ``ndim=2``,
            ``shape=(nstate_obs, nsamples)``
        obs_error : ndarray
            Observation error covariance (R).
            ``dtype=float``, ``ndim=2``,
            ``shape=(nstate_obs, nstate_obs)``
        """
        truth_mat = np.zeros((self.nstate_obs))
        # obtain the truth via forward model
        truth_mat = self._obs_forward()
        obs = truth_mat.reshape(-1)
        std_obs = self.obs_rel_std * obs + self.obs_abs_std
        # import pdb; pdb.set_trace()
        self.obs_error = np.diag(std_obs**2)
        obs_error = self.obs_error
        return obs, obs_error

    def clean(self):
        """ Perform any necessary cleanup before exiting. """
        pass

    # private method
    def _obs_forward(self):
        """Return synthetic truth.

        Returns
        -------
        model_obs: ndarray
            ensemble realizations in observation space.
            ``dtype=float``, ``ndim=2``,
            ``shape=(nstate_obs, nsamples)``
        """
        u_mat = np.zeros((self.nstate))
        model_obs = np.zeros((self.nstate_obs))

        # synthectic diffusivity field
        # mu = np.exp(self.KL_mode[:,0] + self.KL_mode[:,1] + self.KL_mode[:,2])
        # mu_dot = np.gradient(mu, self.x_coor)
        # np.savetxt('mu_truth.dat', mu[:-1])
        # log_mu = np.log(mu)
        # obtain the synthetic truth of KL expansion coefficient
        omega = np.zeros((self.nmodes))
        synthetic_mu = np.zeros((self.nstate))
        # for i in range(self.nmodes):
        #     omega[i] = np.trapz(log_mu * self.KL_mode[:, i], x=self.x_coor)
        #     norm_mode = np.trapz(
        #         self.KL_mode[:, i] * self.KL_mode[:, i], x=self.x_coor)
        #     omega[i] = omega[i] / norm_mode

        omega = np.zeros(15)
        omega[0:3] = np.array([1, 1, 1])

        np.savetxt('omega_truth.dat', omega)
        # save the misfit of the synthetic truth and the projected truth
        for i in range(self.nmodes):
            synthetic_mu += omega[i] * self.KL_mode[:, i]
	# # import pdb; pdb.set_trace()
 #        misfit = project_mu - log_mu
 #        norm_misfit = np.sqrt(np.trapz(misfit * misfit, x=self.x_coor))
 #        np.savetxt('project_misfit.dat', [norm_misfit, 0])
        # solve forward model
        mu = np.exp(synthetic_mu[:-1])
        mu_dot = np.gradient(mu, self.space_interval)
        np.savetxt('mu_truth.dat', mu[:-1])

        # calculate coeffient of Du = E
        A = 0.5 * mu_dot / self.space_interval + mu / \
            self.space_interval/self.space_interval
        B = -2 * mu/self.space_interval/self.space_interval
        C = -0.5 * mu_dot / self.space_interval + \
            mu / self.space_interval / self.space_interval
        B1 = np.append(B, 1)
        D1 = np.diag(B1)
        D1[0][0] = 1
        D2 = np.diag(C, 1)   # C above the main diagonal
        D2[0][1] = 0
        A1 = np.append(A, 0)
        D3 = np.diag(A1[1:101], -1)   # A below the main diagonal
        D = D1+D2+D3
        D = np.mat(D)
        u = - (D.I)*(self.fx)
        u = u.getA1()
        u_mat = u.copy()
        model_obs = u_mat[[25, 50, 75]]
        np.savetxt('u_truth.dat', u_mat)
        return model_obs
