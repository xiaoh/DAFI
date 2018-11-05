# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Dynamic model for solving the Lorenz system.
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
    """ Dynamic model for solving the Lorenz system.

    The state vector includes the time-dependent positions (x, y, z) and the
    three constant coefficients (rho, beta, sigma). The observations consist
    of the position at the given time (x, y, z).
    """

    def __init__(self, nsamples, da_interval, t_end, max_da_iteration,
                 model_input):
        """ Initialize the dynamic model and parse input file.

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
        model_input : str
            Input file name.

        Note
        ----
        Inputs in ``model_input``:
            * **dt_interval** (``str``) -
              Time step for the dynamic model.
            * **x** (``float``) -
              True initial x-position.
            * **y** (``float``) -
              True initial y-position.
            * **z** (``float``) -
              True initial z-position.
            * **rho** (``float``) -
              True value of parameter rho.
            * **beta** (``float``) -
              True value of parameter beta.
            * **sigma** (``float``) -
              True value of parameter sigma.
            * **x_rel_std** (``float``) -
              Relative standard deviation of x, y, z, rho, beta, sigma.
              E.g. std(rho) = rho * x_rel_std
            * **obs_rel_std** (``float``) -
              Relative standard deviation of observation. See x_rel_std for
              details.
            * **perturb_rho** (``bool``) -
              Whether to infer the value of rho.
            * **perturb_beta** (``bool``) -
              Whether to infer the value of beta.
            * **perturb_sigma** (``bool``) -
              Whether to infer the value of sigma.
        """
        # TODO: Simplify
        self.name = 'Lorenz63'
        # number of samples in the ensemble
        self.nsamples = nsamples
        # Data assimilation inverval
        self.da_interval = da_interval
        # End time
        self.t_end = t_end
        # Extract forward Model Input parameters
        param_dict = read_input_data(model_input)
        # forward time inverval
        self.dt_interval = float(param_dict['dt_interval'])
        # initial state varibles: x, y, z
        self.x = float(param_dict['x'])
        self.y = float(param_dict['y'])
        self.z = float(param_dict['z'])
        # initial parameters: rho, beta, sigma
        self.rho = float(param_dict['rho'])
        self.beta = float(param_dict['beta'])
        self.sigma = float(param_dict['sigma'])
        # relative standard deviation of observation
        self.obs_rel_std = float(param_dict['obs_rel_std'])
        # relative standard deviation of x, y, z, rho, beta, sigma
        self.x_rel_std = float(param_dict['x_rel_std'])
        # switch control which parameters are perturbed
        self.perturb_rho = ast.literal_eval(param_dict['perturb_rho'])
        self.perturb_beta = ast.literal_eval(param_dict['perturb_beta'])
        self.perturb_sigma = ast.literal_eval(param_dict['perturb_sigma'])
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
        str_info = 'Lorenz 63 model.'
        # TODO: expand with truth
        return str_info

    def generate_ensemble(self):
        """ Generate initial ensemble state X and observation HX

        Args:
        -----
        DAInterval : DA step interval

        Returns
        -------
        state_vec : ndarray
            State variables at current time.
            ``dtype=float``, ``ndim=2``, ``shape=(nstate, nsamples)``
        model_obs : ndarray
            Forecast ensemble in observation space.
            ``dtype=float``, ``ndim=2``, ``shape=(nstate, nsamples)``
        """

        state_vec = np.zeros([self.nstate, self.nsamples])
        model_obs = np.zeros([self.nstate_obs, self.nsamples])

        # generate initial state_vec
        for idim in np.arange(self.nstate):
            dx_std = self.x_rel_std * self.x_init[idim]
            state_vec[idim, :] = self.x_init[idim] + \
                np.random.normal(0, dx_std, self.nsamples)
        # operation operator
        h_matrix = self._construct_h_matrix()
        # h_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).T
        model_obs = h_matrix.dot(state_vec)
        return state_vec, model_obs

    def lorenz63(self, time, state_vec):
        """ Define Lorenz 63 system.

        Parameters
        ----------
        time : float
           Current time.
        state_vec : ndarray
            State variables at current time.
            ``dtype=float``, ``ndim=2``, ``shape=(nstate, nsamples)``

        Returns
        -------
        delta_x : differential value for each state varible
        """
        # TODO: time not used?
        # intial system parameters
        rho = self.rho
        beta = self.beta
        sigma = self.sigma
        # initial differential vector for each state variable
        delta_x = np.zeros([self.nstate, 1])
        # switch perturbation of each parameter
        if self.nstate == 4:
            delta_x[3] = 0
            if self.perturb_rho:
                rho = state_vec[3]
            if self.perturb_beta:
                beta = state_vec[3]
            if self.perturb_sigma:
                sigma = state_vec[3]
        if self.nstate == 5:
            delta_x[3] = 0
            delta_x[4] = 0
            if not self.perturb_rho:
                beta = state_vec[3]
                sigma = state_vec[4]
            if self.perturb_beta:
                rho = state_vec[3]
                sigma = state_vec[4]
            if self.perturb_sigma:
                rho = state_vec[3]
                beta = state_vec[4]
        if self.nstate == 6:
            delta_x[3] = 0
            delta_x[4] = 0
            delta_x[5] = 0
            rho = state_vec[3]
            beta = state_vec[4]
            sigma = state_vec[5]
        # ODEs
        delta_x[0] = sigma * (-state_vec[0] + state_vec[1])
        delta_x[1] = rho * state_vec[0] - state_vec[1] - \
            state_vec[0] * state_vec[2]
        delta_x[2] = state_vec[0] * state_vec[1] - beta * state_vec[2]
        return delta_x

    def forecast_to_time(self, state_vec_current, next_end_time):
        """ Returns states at the next end time.

        Parameters
        ----------
        state_vec_current : ndarray
            State variables at current time.
            ``dtype=float``, ``ndim=2``, ``shape=(nstate, nsamples)``
        next_end_time : float
            Next end time.

        Returns
        -------
        state_vec_forecast: ndarray
            Forecast ensemble of state variables by forward model.
            ``dtype=float``, ``ndim=2``, ``shape=(nstate, nsamples)``
        model_obs: ndarray
            Forecast ensemble in observation space.
            ``dtype=float``, ``ndim=2``, ``shape=(nstate, nsamples)``
        """
        # Set start time
        new_start_time = next_end_time - self.da_interval
        # Make time series from start time to end time
        time_series = np.arange(new_start_time + self.dt_interval,
                                next_end_time + self.dt_interval,
                                self.dt_interval)  # TODO
        # time_series = np.arange(  # TODO
        #     new_start_time, next_end_time, self.dt_interval) # TODO
        # Ode solver setup
        self.solver = ode(self.lorenz63)
        self.solver.set_integrator('dopri5')
        state_vec_forecast = np.zeros(state_vec_current.shape)
        # Solve ode for each sample
        for isamp in range(self.nsamples):
            # Set initial value for ode solver
            self.solver.set_initial_value(
                state_vec_current[:, isamp], new_start_time)
            forecast_state = np.empty([len(time_series), self.nstate])
            for t in np.arange(len(time_series)):
                if not self.solver.successful():
                    print("solver failed")
                    sys.exit(1)
                self.solver.integrate(time_series[t])
                forecast_state[t] = self.solver.y
            # Save current state in state matrix state_vec_current
            state_vec_forecast[:, isamp] = forecast_state[-1]
        # Construct observation operator
        return state_vec_forecast

    def forward(self, X):
        """
        Forwards the states from X to HX.

        Parameters
        ----------
        X: ndarray
            Current state variables.

        Returns
        -------
        HX: ndarray
            Ensemble in observation space.
        """
        # Construct observation operator
        h_matrix = self._construct_h_matrix()
        model_obs = h_matrix.dot(X)
        return model_obs

    def get_obs(self, time):
        """ Return the observation and observation covariance.

        Parameters
        ----------
        time : float
            Time at which observation is requested.

        Returns
        -------
        obs_vec : ndarray
            Vector of observation data. ``dtype=float``, ``ndim=1``,
            ``shape=(nstate_obs)``
        obs_error : ndarray
            Observation error covariance. ``dtype=float``, ``ndim=2``,
            ``shape=(nstate_obs, nstate_obs)``
        """
        # get truth
        da_step = time / self.da_interval
        truth = np.loadtxt('truth.dat')[int(da_step)*10, 1:]
        # create synthetic observation
        obs_std_vec = self.obs_rel_std * np.abs(truth) + 0.1
        obs_vec = truth + np.random.normal(0, obs_std_vec)
        # calculate observation error matrix
        # obs_std_vec = self.obs_rel_std * np.abs(obs_vec) + 0.1
        # TODO: change truth to obs_vec, add 0.1 as some abs_err option, modify the test.
        obs_error = sp.diags(obs_std_vec**2, 0)
        obs_error = obs_error.todense()
        return obs_vec, obs_error

    def report(self):
        """ Report summary information. """
        pass

    def plot(self):
        """ Plot relevant model results. """
        pass

    def clean(self):
        """ Perform any necessary cleanup before exiting. """
        pass

    def _construct_h_matrix(self):
        """ Construct the observation operator. """
        h_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for i in range(self.num_params):
            h_matrix.append([0, 0, 0])
        h_matrix = np.array(h_matrix).T
        return h_matrix
