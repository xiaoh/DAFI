# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Dynamic model for solving one dimension heat diffusion system.
"""

# standard library imports
import ast
import sys
import math

# third party imports
import numpy as np
from numpy import linalg as la
import scipy.sparse as sp
from scipy.integrate import ode

# local import
from data_assimilation.dyn_model import DynModel
from data_assimilation.utilities import read_input_data


class Solver(DynModel):
    """ Dynamic model for solving one dimension heat diffusion equation. """

    def __init__(self, nsamples, da_interval, t_end, max_pseudo_time,
                 model_input):
        """ Parse input file and assign values to class attributes.

        Parameters
        ----------
        nsamples : int
            Ensemble size.
        da_interval : float
            Time interval between data assimilation steps.
        t_end : float
            Final time.
        max_da_iteration : int
            Maximum number of DA iterations at a given time-step.
        input_file : str
            Input file name.

        Note
        ----
        Inputs in ``model_input``:
            * **n_mode** (``int``) -
              Number of modes for K-L expansion.
            * **mu_init** (``float``) -
              Prior diffusivity.
            * **sigma** (``float``) -
              Constant variance for kernel function.
            * **x_rel_std** (``float``) -
              Relative standard deviation of prior state vector.
            * **x_abs_std** (``float``) -
              Absolute standard deviation of prior state vector.
            * **std_coef** (``float``) -
              coefficient of standard deviation of prior state vector.
            * **obs_rel_std** (``float``) -
              Relative standard deviation of observation.
        """

        # save the main input
        self.nsamples = nsamples
        self.da_interval = da_interval
        self.t_end = t_end
        self.max_da_iteration = max_da_iteration

        # read input file
        param_dict = read_input_data(model_input)
        self.x_rel_std = float(param_dict['x_rel_std'])
        self.x_abs_std = float(param_dict['x_abs_std'])
        self.std_coef = float(param_dict['std_coef'])
        self.obs_rel_std = float(param_dict['obs_rel_std'])
        self.obs_abs_std = float(param_dict['obs_abs_std'])
        self.nmodes = int(param_dict['nmodes'])
        self.sigma = float(param_dict['sigma'])
        mu_init = float(param_dict['mu_init'])

        # required attributes
        self.name = '1D heat diffusion Equation'
        self.space_interval = 0.1
        self.max_length = 5
        self.nstate_obs = 10

        # create spatial coordinate and save
        self.x_coor = np.arange(
            0, self.max_length+self.space_interval, self.space_interval)
        np.savetxt('x_coor.dat', self.x_coor)

        # initialize state vector
        self.omega_init = np.zeros((self.nmodes))
        self.init_state = np.zeros(self.x_coor.shape)
        self.augstate_init = np.concatenate((
            self.init_state, np.zeros(self.nmodes)))
        # dimension of state space
        self.nstate = len(self.init_state)
        # dimension of augmented state space
        self.nstate_aug = len(self.augstate_init)

        # create source term fx
        S = np.zeros(self.nstate)
        for i in range(self.nstate - 1):
            S[i] = math.sin(2*np.pi*self.x_coor[i]/5)
        S = np.mat(S).T
        self.fx = S.A

        # create modes for K-L expansion
        cov = np.zeros((self.nstate, self.nstate))
        for i in range(self.nstate):
            for j in range(self.nstate):
                cov[i][j] = self.sigma * self.sigma * math.exp(
                    -abs(self.x_coor[i]-self.x_coor[j])**2/self.max_length**2)
        eigVals, eigVecs = sp.linalg.eigsh(cov, k=self.nmodes)
        ascendingOrder = eigVals.argsort()
        descendingOrder = ascendingOrder[::-1]
        eigVals = eigVals[descendingOrder]
        eigVecs = eigVecs[:, descendingOrder]

        # calculate KL modes: eigVec * sqrt(eigVal)
        KL_mode_raw = np.zeros([self.nstate, self.nmodes])
        for i in np.arange(self.nmodes):
            KL_mode_raw[:, i] = eigVecs[:, i] * np.sqrt(eigVals[i])
        # normalize the KL modes
        self.KL_mode = np.zeros([self.nstate, self.nmodes])
        for i in range(self.nmodes):
            self.KL_mode[:, i] = KL_mode_raw[:, i] / \
                np.linalg.norm(KL_mode_raw[:, i])
        np.savetxt('KLmodes.dat', self.KL_mode)

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
        x_rel_std : relative standard deviation of state vector
        x_abs_std : absolute standard deviation of state vector
        std_coef : standard deviation coefficient
        omega_init : initial KL-expansion coefficient


        Returns
        -------
        augstate_init : ndarray
            Augmented state variables at current time.
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
            dx_std = self.std_coef * abs(
                self.x_rel_std * self.omega_init[i] + self.x_abs_std)
            para_init[i, :] = self.omega_init[i] + \
                np.random.normal(0, dx_std, self.nsamples)
        # augment the state with KL expansion coefficient
        augstate_init = np.concatenate((state_init, para_init))
        model_obs = self.forward(augstate_init)
        return augstate_init, model_obs

    def forecast_to_time(self, state_vec_current, next_end_time):
        """ Returns states at the next end time."""

        return state_vec_current

    def forward(self, state_vec):
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
        omega = state_vec[self.nstate:, :]
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
            D3 = np.diag(A1[1:51], -1)   # A below the main diagonal
            D = D1+D2+D3
            D = np.mat(D)
            u = - (D.I)*(self.fx)
            u = u.getA1()
            u_mat[:, i_nsample] = u
        for j in range(1, 11):
            model_obs[j-1, :] = u_mat[5*j, :]
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
        mu = 0.5 + 0.02 * self.x_coor**2
        mu_dot = 0.04 * self.x_coor
        np.savetxt('mu_truth.dat', mu[:-1])
        log_mu = np.log(mu)
        # obtain the synthetic truth of KL expansion coefficient
        omega = np.zeros((self.nmodes))
        project_mu = np.zeros((self.nstate))
        for i in range(self.nmodes):
            omega[i] = np.trapz(log_mu * self.KL_mode[:, i], x=self.x_coor)
            norm_mode = np.trapz(
                self.KL_mode[:, i] * self.KL_mode[:, i], x=self.x_coor)
            omega[i] = omega[i] / norm_mode
        np.savetxt('omega_truth.dat', omega)
        # save the misfit of the synthetic truth and the projected truth
        for i in range(self.nmodes):
            project_mu += omega[i] * self.KL_mode[:, i]
        misfit = project_mu - log_mu
        norm_misfit = np.sqrt(np.trapz(misfit * misfit, x=self.x_coor))
        np.savetxt('project_misfit.dat', [norm_misfit, 0])
        # solve forward model
        mu = mu[:-1]
        mu_dot = mu_dot[:-1]
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
        D3 = np.diag(A1[1:51], -1)   # A below the main diagonal
        D = D1+D2+D3
        D = np.mat(D)
        u = - (D.I)*(self.fx)
        u = u.getA1()
        u_mat = u.copy()
        for j in range(1, 11):
            model_obs[j-1] = u_mat[5*j]
        np.savetxt('u_truth.dat', u_mat)
        return model_obs
