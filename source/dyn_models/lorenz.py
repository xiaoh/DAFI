# Copyright 2018 Virginia Polytechnic Institute and State University.
""" 

This module is dynamic model for solving the Lorenz attractor

It  consist of 3 functions:
    1 generateEnsemble: generate ensemble
    2 forcastToTime: evolve ensemble to next time using forward model
    3 observe: Get observations and observation error covariance

"""

# standard library imports
import ast
import sys

# third party imports
import numpy as np
from numpy import linalg as la
import scipy.sparse as sp
from scipy.integrate import ode

# local import
from dainv.dyn_model import DynModel
from dainv.utilities import read_input_data

class Solver(DynModel):
    """
        Dynamic forward model: Lorenz 63
        The state variable include: x, y, z
        The parameters need to be augmented: coefficients for rho, beta, sigma
        The Observation include: x, y, z
    """

    def __init__(self, nsamples, da_interval, t_end,  model_input):
        """
            Initialization
        """
        self.name = 'Lorenz63'
        # number of samples in the ensemble
        self.nsamples = nsamples
        # Data assimilation inverval
        self.da_interval = da_interval
        # End time
        self.t_end = t_end
        # Extract forward Model Input parameters
        paramDict = read_input_data(model_input)
        # case folder name
        self.caseName = paramDict['caseName']
        # forward time inverval
        self.dt_interval = float(paramDict['dt_interval'])
        # initial state varibles: x, y, z
        self.x = float(paramDict['x'])
        self.y = float(paramDict['y'])
        self.z = float(paramDict['z'])
        # initial parameters: rho, beta, sigma
        self.rho = float(paramDict['rho'])
        self.beta = float(paramDict['beta'])
        self.sigma = float(paramDict['sigma'])
        # relative standard deviation of observation
        self.obs_rel_std = float(paramDict['obs_rel_std'])
        # relative standard deviation of x, y, z, rho, beta, sigma
        self.x_rel_std = float(paramDict['x_rel_std'])
        # switch control which parameters are perturbed
        self.perturb_rho = ast.literal_eval(paramDict['perturb_rho'])
        self.perturb_beta = ast.literal_eval(paramDict['perturb_beta'])
        self.perturb_sigma = ast.literal_eval(paramDict['perturb_sigma'])
        # initial  state condition
        self.x_init = [self.x, self.y, self.z]
        
        # specify state augmented by which parameter and count
        self.num_params = 0
        if self.perturb_rho:
            self.x_init.append(self.rho)
            self.num_params += 1
        if self.perturb_beta:
            self.x_init.append(self.beta)
            self.num_params += 1
        if self.perturb_sigma:
            self.x_init.append(self.sigma)
            self.num_params += 1
        # dimension of state space
        self.nstate = len(self.x_init)
        # dimension of observation space
        self.nstate_obs = 3

    def __str__(self):
        s = 'Lorenz 63 model.'
        return s

    def generate_ensemble(self):
        """ Generate initial ensemble state X and observation HX

        Args:
        -----
        DAInterval: DA step interval

        Returns
        -------
        X: ndarray
            ensemble of whole states
        HX: ndarray
            ensemble of whole states in observation space
        """

        X = np.zeros([self.nstate, self.nsamples])
        HX = np.zeros([self.nstate_obs, self.nsamples])

        # generate initial X
        for iDim in np.arange(self.nstate):
            dxStd = self.x_rel_std * self.x_init[iDim]
            X[iDim, :] = self.x_init[iDim] + np.random.normal(0, dxStd, self.nsamples)
        # operation operator
        H = self._constructHMatrix()
        #H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).T
        HX = H.dot(X)
        return X, HX

    def lorenz63(self, t, x):
        """ 
        Define Lorenz 63 system 
        
        Parameters
        ----------
        t: float
            current time
        x: adrray
            state variables at current time

        Returns
        -------
        dx: differential value for each state varible
        """
        
        # intial system parameters
        rho = self.rho
        beta = self.beta
        sigma = self.sigma
        # initial differential vector for each state variable
        dx = np.zeros([self.nstate, 1])
        # switch perturbation of each parameter
        if self.nstate == 4:
            dx[3] = 0
            if self.perturb_rho: rho = x[3]
            if self.perturb_beta: beta = x[3]
            if self.perturb_sigma: sigma = x[3]
        if self.nstate == 5:
            dx[3] = 0
            dx[4] = 0
            if not self.perturb_rho: beta = x[3]; sigma = x[4]
            if self.perturb_beta: rho = x[3]; sigma = x[4]
            if self.perturb_sigma: rho = x[3]; beta = x[4]
        if self.nstate == 6:
            dx[3] = 0
            dx[4] = 0
            dx[5] = 0
            rho = x[3]
            beta = x[4]
            sigma = x[5]
        # ODEs
        dx[0] = sigma * (-x[0] + x[1])
        dx[1] = rho * x[0] - x[1] - x[0] * x[2]
        dx[2] = x[0] * x[1] - beta * x[2]
        
        return dx

    def forecast_to_time(self, X, next_end_time):
        """
        Returns states at the next end time.
        
        Parameters
        ----------
        X: ndarray
            current state variables
        next_end_time: float
            next end time

        Returns
        -------
        X: ndarray
            forwarded ensemble state variables by forward model
        HX: ndarray
            ensemble in observation space
        """
        # Set start time
        new_start_time = next_end_time - self.da_interval
        # Make time series from start time to end time
        time_series = np.arange(new_start_time + self.dt_interval, next_end_time + self.dt_interval, self.dt_interval)
        # Ode solver setup
        self.solver = ode(self.lorenz63)
        self.solver.set_integrator('dopri5')
        # Solve ode for each sample
        for i in range(self.nsamples):
            # Set initial value for ode solver
            self.solver.set_initial_value(X[:,i], new_start_time)
            x = np.empty([len(time_series), self.nstate])
            for t in np.arange(len(time_series)):
                if not self.solver.successful():
                    print "solver failed"
                    sys.exit(1)
                self.solver.integrate(time_series[t])
                x[t] = self.solver.y
            # Save current state in state matrix X
            X[:,i] = x[-1]
        # Construct observation operator
        H = self._constructHMatrix()
        HX = H.dot(X)

        return X, HX

    def get_obs(self, next_end_time):
        """ 
        Return the observation and observation error covariance
        
        Parameters
        ----------
        next_end_time: float
            next end time

        Returns
        -------
        obs: ndarray
            ensemble observations
        R_obs: ndarray
            observation error covariance
        """
        da_step = (next_end_time - self.da_interval) / self.da_interval
        obs = self.observe(next_end_time)
        R_obs = self.get_Robs()

        return obs, R_obs

    def get_Robs(self):
        """ Return the observation error covariance. """

        return self.Robs.todense()

    def clean(self):
        """ Perform any necessary cleanup before exiting. """
        pass

    def observe(self, next_end_time):
        """ 
        Return the observation data
        
        Parameters
        ----------
        next_end_time: float
            next end time

        Returns
        -------
        obsM: ndarray
            ensemble observations
        """
        # initial observation and observation standard deviation 
        obs = np.empty(self.nstate_obs)
        obs_std_vec = np.empty(self.nstate_obs)
        # calculate current da_step
        da_step = (next_end_time - self.da_interval) / self.da_interval
        # read observation at next end time
        obs_vec = np.loadtxt('obs.txt')[int(da_step)*10,1:-1]
        # add noise in observation
        for iDim in np.arange(self.nstate_obs):
            obs_std = self.obs_rel_std * np.abs(obs_vec[iDim])
            obs_std_vec[iDim] = obs_std
            obs[iDim] = obs_vec[iDim] + np.random.normal(0, obs_std, 1)
        # construct ensemble observation
        obsM = np.empty([self.nstate_obs, self.nsamples])
        for i in np.arange(self.nsamples):
            obsM[:, i] = obs
        # construct the observation error covariance
        self.Robs = sp.diags(obs_std_vec**2,0)

        return obsM

    def _constructHMatrix(self):
        """construct the observation operator"""

        H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for i in range(self.num_params): H.append([0,0,0]) 
        H = np.array(H).T

        return H
