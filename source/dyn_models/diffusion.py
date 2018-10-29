# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Dynamic model for solving the Lorenz system.
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
from dainv.dyn_model import DynModel
from dainv.utilities import read_input_data


class Solver(DynModel):
    """ Dynamic model for solving the one dimension diffusion equation.

    The state vector includes the time-dependent positions (x, y, z) and the
    three constant coefficients (rho, beta, sigma). The observations consist
    of the position at the given time (x, y, z).
    """

    def __init__(self, nsamples, da_interval, t_end, forward_interval,
                 max_pseudo_time, model_input):
        """ Initialize the dynamic model and parse input file.

        Parameters
        ----------
        nsamples : int
            Ensemble size.
        da_interval : float
            Time interval between data assimilation steps.
        t_end : float
            Final time.
        forward_interval : float
            forward interval at current time.
        max_pseudo_time : float
            Max forward step.
        input_file : str
            Input file name.

        Note
        ----
        Inputs in ``model_input``:
            * **n_mode** (``int``) -
              number of modes for K-L expansion.
            * **true_omega1** (``float``) -
              True value of parameter omega1.
            * **true_omega2** (``float``) -
              True value of parameter omega2.
            * **true_omega3** (``float``) -
              True value of parameter omega3.
            * **obs_rel_std** (``float``) -
              Relative standard deviation of observation.
        """

        self.name = '1D diffusion Equation'
        # number of samples in the ensemble
        self.nsamples = nsamples
        # Data assimilation inverval
        self.da_interval = da_interval
        # End time
        self.t_end = t_end
        # Extract forward Model Input parameters
        param_dict = read_input_data(model_input)
        # forward time inverval
        self.forward_interval = forward_interval
        # forward maximum pseudo step
        self.max_pseudo_time = max_pseudo_time
        # initial state varibles: x, y, z
        self.space_interval = 0.1  # diffential space step Todo move to input file
        self.max_length = 5    # maximum max_length Todo move to input file

        # synthetic mode parameters
        self.true_omega1 = float(param_dict['true_omega1'])
        self.true_omega2 = float(param_dict['true_omega2'])
        self.true_omega3 = float(param_dict['true_omega3'])
        self.true_omega = (
            self.true_omega1, self.true_omega2, self.true_omega3)
        # relative standard deviation of observation
        self.obs_rel_std = float(param_dict['obs_rel_std'])
        # relative standard deviation of state variables
        #self.state_rel_std = float(param_dict['state_rel_std'])
        self.nmodes = int(param_dict['nmodes'])
        # set 1-D space coordinate
        self.x_coor = np.arange(
            0, self.max_length+self.space_interval, self.space_interval)
        self.state_vec_init = np.zeros(self.x_coor.shape)
        self.omega_init = np.array([])
        # state augmentation
        self.augstate_init = np.concatenate((
            self.state_vec_init, np.zeros(self.nmodes)))
        self.omega_min = [0, 0, 0]
        self.omega_max = [50, 15, 25]
        # dimension of state space
        self.nstate = len(self.state_vec_init)
        # dimension of augmented state space
        self.nstate_aug = len(self.augstate_init)
        # dimension of observation space
        self.nstate_obs = 10

        # source term fx
        S = np.zeros(self.nstate)
        for i in range(self.nstate - 1):
            S[i] = math.sin(self.x_coor[i])
        S = np.mat(S).T
        self.fx = 1000 * S.A

        # calculate sigma field
        sigma = np.zeros(self.nstate)
        for i in range(self.nstate):
            if (self.x_coor[i] >= 0) & (self.x_coor[i] <= self.max_length/2.0):
                sigma[i] = 8.0/self.max_length*self.x_coor[i]+1
            else:
                sigma[i] = -(8.0/self.max_length)*self.x_coor[i]+9

        # calculate the modes
        cov = np.zeros((self.nstate, self.nstate))
        for i in range(self.nstate):
            for j in range(self.nstate):
                cov[i][j] = sigma[i]*sigma[j] * math.exp(
                    -abs(self.x_coor[i]-self.x_coor[j])**2/self.max_length**2)

        eigVals, eigVecs = sp.linalg.eigsh(cov, k=self.nmodes)
        # sort the eig-value and eig-vectors in a descending order
        ascendingOrder = eigVals.argsort()
        descendingOrder = ascendingOrder[::-1]
        eigVals = eigVals[descendingOrder]
        eigVecs = eigVecs[:, descendingOrder]

        # calculate KL basis-set: eigVec * sqrt(eigVal)
        KL_mode_raw = np.zeros([self.nstate, self.nmodes])
        for i in np.arange(self.nmodes):
            KL_mode_raw[:, i] = eigVecs[:, i] * np.sqrt(eigVals[i])

        # normalize the KL basis set
        self.KL_mode = np.zeros([self.nstate, self.nmodes])
        for i in range(self.nmodes):
            self.KL_mode[:, i] = KL_mode_raw[:, i] / \
                np.linalg.norm(KL_mode_raw[:, i])

        np.savetxt('x_coor.dat', self.x_coor)
        np.savetxt('KLmodes.dat', self.KL_mode)

    def __str__(self):
        s = '1-D diffusion model.'
        return s

    def generate_ensemble(self):
        """ Generate initial ensemble state X and observation HX

        Args:
        -----
        nstate : size of state space
        nsamples : number of samples
        nstate_obs : size of observation space
        nmodes : number of modes
        omega_min : min range value for randomization
        omega_max : max range value for randomization

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

        # generate initial state
        for i in range(self.nmodes):
            self.omega_init = np.concatenate(
                (self.omega_init,
                    np.random.uniform(self.omega_min[i],
                                      self.omega_max[i], self.nsamples)))

        self.omega_init = self.omega_init.reshape(self.nmodes, self.nsamples)
        augstate_init = np.concatenate(
            (state_init, self.omega_init))
        augstate_init, model_obs = self.forward(augstate_init, 0)
        return augstate_init, model_obs

    def forecast_to_time(self, state_vec_current, next_end_time):
        """ Returns states at the next end time."""

        return state_vec_current

    def forward(self, X, next_pseudo_time):
        """
        Returns states at the next end time.

        Parameters
        ----------
        X: ndarray
            current state variables
        next_pseudo_time: float
            next pseudo time

        Returns
        -------
        X: ndarray
            forwarded ensemble state variables by forward model
        HX: ndarray
            ensemble in observation space
        """

        u_mat = np.zeros((self.nstate, self.nsamples))
        model_obs = np.zeros((self.nstate_obs, self.nsamples))
        for i_nsample in range(self.nsamples):
            omega1 = X[-3, i_nsample]
            omega2 = X[-2, i_nsample]
            omega3 = X[-1, i_nsample]
            vx_dot = np.zeros(self.nstate-1)
            vx = np.zeros(self.nstate-1)
            for i in range(self.nstate-1):
                vx_dot[i] = omega1 * (self.KL_mode[i+1, 0] -
                                      self.KL_mode[i, 0])/self.space_interval \
                    + omega2*(self.KL_mode[i+1, 1] -
                              self.KL_mode[i, 1])/self.space_interval \
                    + omega3*(self.KL_mode[i+1, 2] -
                              self.KL_mode[i, 2])/self.space_interval
                vx[i] = omega1 * self.KL_mode[i, 0] + \
                    omega2*self.KL_mode[i, 1] + omega3*self.KL_mode[i, 2]
            # calculate coeffient of Du = E
            A = 0.5 * vx_dot / self.space_interval + vx / \
                self.space_interval/self.space_interval
            B = -2 * vx/self.space_interval/self.space_interval
            C = -0.5 * vx_dot / self.space_interval + \
                vx/self.space_interval/self.space_interval
            B1 = np.append(B, 1)  # bounadary condition
            D1 = np.diag(B1)
            D1[0][0] = 1  # bounadary condition

            D2 = np.diag(C, 1)   # C above the main diagonal
            D2[0][1] = 0

            A1 = np.append(A, 0)
            D3 = np.diag(A1[1:51], -1)   # A below the main diagonal

            D = D1+D2+D3
            D = np.mat(D)
            u = (D.I)*(self.fx)
            u = u.getA1()
            u_mat[:, i_nsample] = u.copy()
        for j in range(1, 11):
            model_obs[j-1, :] = u_mat[5*j, :]
        return X, model_obs

    def get_obs(self, next_end_time):
        """ Return the observation and observation covariance.

        Parameters
        ----------
        next_end_time : float
            Next end time.

        Returns
        -------
        obs : ndarray
            Ensemble observations. ``dtype=float``, ``ndim=2``,
            ``shape=(nstate_obs, nsamples)``
        obs_error : ndarray
            Observation error covariance. ``dtype=float``, ``ndim=2``,
            ``shape=(nstate_obs, nstate_obs)``
        """
        obs, obs_perturb = self.observe(next_end_time)
        obs_error = self.get_obs_error()
        return obs, obs_perturb, obs_error

    def get_obs_error(self):
        """ Return the observation error covariance. """
        return self.obs_error.todense()

    def report(self):
        """ Report summary information. """
        pass

    def plot(self):
        """ Plot relevant model results. """
        pass

    def clean(self):
        """ Perform any necessary cleanup before exiting. """
        pass

    def observe(self, next_end_time):
        """ Return the observation data at a given time.

        Parameters
        ----------
        next_end_time: float
            Next end time.

        Returns
        -------
        obs_mat: ndarray
            Ensemble observations. ``dtype=float``, ``ndim=2``,
            ``shape=(nstate_obs, nsamples)``
        obs_perturb: ndarray
            Ensemble observation perturbation. ``dtype=float``, ``ndim=2``,
            ``shape=(nstate_obs, nsamples)``  
        """
        # get synthetic omega
        self.synthetic_omega = np.array(
            [self.true_omega1, self.true_omega2, self.true_omega3])
        self.synthetic_omega_mat = np.tile(
            self.synthetic_omega, self.nsamples).reshape(
                self.nsamples, self.nmodes).T
        # run the forward model to obtain synthetic truth
        truth_mat = np.zeros((self.nstate_obs, self.nsamples))
        self.synthetic_omega_mat, truth_mat = self.forward(
            self.synthetic_omega_mat, 0)
        observe_vec = truth_mat[:, 0].reshape(-1)
        # change 0 to 0.0001, otherwise there is no inverse of the matrix
        observe_vec[9] = 0.0001

        self.obs_error = sp.diags((self.obs_rel_std*observe_vec)**2, 0)
        R = self.obs_error.todense()
        obs_error_mean = np.zeros(self.nstate_obs)

        obs_perturb = np.random.multivariate_normal(
            obs_error_mean, R, self.nsamples)
        obs_mat = np.tile(observe_vec, (self.nsamples, 1)) + obs_perturb
        return obs_mat.T, obs_perturb.T
