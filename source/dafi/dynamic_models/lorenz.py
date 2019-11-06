# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Dynamic model for solving the Lorenz system. """

# standard library imports
import os

# third party imports
import numpy as np
from scipy.integrate import ode

# local imports
from dafi.dyn_model import DynModel
import dafi.utilities as utils


def solve_lorenz(time_series, init_state, parameters):
    """ Solve the Lorenz 63 equations for given initial condition.

    Parameters
    ----------
    time_series : list
        Time-series for the integration of the Lorenz system.
        The first entry is the initial time.
        ``len=3``, ``type=float``
    init_state : list
        The state [x, y, z] at the initial time.
        ``len=3``, ``type=float``
    parameters : list
        The values of the three parameters [rho, beta, sigma].
        ``len=3``, ``type=float``

    Returns
    -------
    state : ndarray
        The state [x, y, z] at all times specified in the time_series.
        ``dtype=float``, ``ndim=2``, ``shape=(len(time_series), 3)``


    """
    # define Lorenz system
    def lorenz63(time, state, parameters):
        """ Calculate the velocity at a given state.

        This function is integrated using scipy.integrate.ode.

        Parameters
        ---------
        time : float
            The time.
            Not used. Required input for the integrator.
        state : list
            The state [x, y, z] state at the current time.
            ``len=3``, ``type=float``
        parameters : list
            The values of the three parameters [rho, beta, sigma].
            ``len=3``, ``type=float``

        Returns
        -------
        velocities : list
            The derivative with respect to time of the 3 state components.
            ``len=3``, ``type=float``
        """
        xstate, ystate, zstate = state
        rho, beta, sigma = parameters
        ddt_x = sigma * (ystate - xstate)
        ddt_y = rho*xstate - ystate - xstate*zstate
        ddt_z = xstate*ystate - beta*zstate
        return [ddt_x, ddt_y, ddt_z]

    # solve lorenz system
    solver = ode(lorenz63)
    solver.set_integrator('dopri5')
    solver.set_initial_value(init_state, time_series[0])
    solver.set_f_params(parameters)
    state = np.expand_dims(np.array(init_state), axis=0)
    for time in time_series[1:]:
        if not solver.successful():
            raise RuntimeError('Solver failed at time: {} s'.format(time))
        else:
            solver.integrate(time)
        state = np.vstack((state, solver.y))
    return state


class Solver(DynModel):
    """ Dynamic model for solving the Lorenz system. """

    def __init__(self, nsamples, da_interval, t_end, max_da_iteration,
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
            Not used. Maximum number of DA iterations at a given
            time-step.
        input_file : str
            Input file name.
        """
        # save the main inputs
        self.nsamples = nsamples
        self.da_interval = da_interval
        self.t_end = t_end
        self.max_da_iteration = max_da_iteration

        # read input file
        param_dict = utils.read_input_data(model_input)
        dt_interval = float(param_dict['dt_interval'])
        x_init_mean = float(param_dict['x_init_mean'])
        y_init_mean = float(param_dict['y_init_mean'])
        z_init_mean = float(param_dict['z_init_mean'])
        rho_init_mean = float(param_dict['rho_init_mean'])
        x_init_std = float(param_dict['x_init_std'])
        y_init_std = float(param_dict['y_init_std'])
        z_init_std = float(param_dict['z_init_std'])
        rho_init_std = float(param_dict['rho_init_std'])
        beta = float(param_dict['beta'])
        sigma = float(param_dict['sigma'])
        x_obs_rel_std = float(param_dict['x_obs_rel_std'])
        z_obs_rel_std = float(param_dict['z_obs_rel_std'])
        x_obs_abs_std = float(param_dict['x_obs_abs_std'])
        z_obs_abs_std = float(param_dict['z_obs_abs_std'])
        x_true = float(param_dict['x_true'])
        y_true = float(param_dict['y_true'])
        z_true = float(param_dict['z_true'])
        rho_true = float(param_dict['rho_true'])
        beta_true = float(param_dict['beta_true'])
        sigma_true = float(param_dict['sigma_true'])

        # required attributes.
        self.name = 'Lorenz63'
        self.nstate = 4
        self.nstate_obs = 2
        self.init_state = [x_init_mean, y_init_mean, z_init_mean,
                           rho_init_mean]

        # save other inputs for future use
        self.dt = dt_interval
        self.beta = beta
        self.sigma = sigma
        self.init_std = [x_init_std, y_init_std, z_init_std, rho_init_std]
        self.obs_rel_std = [x_obs_rel_std, z_obs_rel_std]
        self.obs_abs_std = [x_obs_abs_std, z_obs_abs_std]

        # create save directory
        self.dir = 'results_lorenz'
        utils.create_dir(self.dir)

        # create synthetic observations
        true_init_orgstate = [x_true, y_true, z_true]
        true_params = [rho_true, beta_true, sigma_true]
        self.obs = self._create_synthetic_observations(
            true_init_orgstate, true_params)

    def __str__(self):
        str_info = 'Lorenz 63 model.'
        return str_info

    # required methods
    def generate_ensemble(self):
        """ Return states at the first data assimilation time-step.

        Returns
        -------
        state_vec : ndarray
            Ensemble matrix of states (Xi).
            ``dtype=float``, ``ndim=2``, ``shape=(nstate, nsamples)``
        model_obs : ndarray
            Ensemble matrix of states in observation space (HX).
            ``dtype=float``, ``ndim=2``,
            ``shape=(nstate_obs, nsamples)``
        """
        state_vec = np.empty([self.nstate, self.nsamples])
        for isamp in range(self.nsamples):
            state_vec[:, isamp] = self.init_state \
                + np.random.normal(0, self.init_std)
        model_obs = self.state_to_observation(state_vec)
        return state_vec, model_obs

    def forecast_to_time(self, state_vec_current, end_time):
        """ Return states at the next end time.

        Parameters
        ----------
        state_vec : ndarray
            Current ensemble matrix of states (Xa). [nstate x nsamples]
        next_end_time : float
            Next end time.

        Returns
        -------
        state_vec : ndarray
            Updated ensemble matrix of states (Xf).
            ``dtype=float``, ``ndim=2``, ``shape=(nstate, nsamples)``
        """
        # create time series
        start_time = end_time - self.da_interval
        time_series = np.arange(start_time, end_time + self.dt/2.0, self.dt)
        # initialize variables
        state_vec_forecast = np.empty([self.nstate, self.nsamples])
        savedir = self.dir + os.sep + 'states'
        utils.create_dir(savedir)
        da_step = int((start_time + self.dt/2.0) / self.da_interval) + 1
        for isamp in range(self.nsamples):
            # solve
            parameters = [state_vec_current[3, isamp], self.beta, self.sigma]
            init_orgstate = state_vec_current[:3, isamp]
            orgstate = solve_lorenz(time_series, init_orgstate, parameters)
            # create forecasted vector
            state_vec_forecast[:3, isamp] = orgstate[-1, :]
            state_vec_forecast[3, isamp] = state_vec_current[3, isamp]
            # save
            fname = 'dastep_{}_samp_{}'.format(da_step, isamp)
            np.savetxt(savedir + os.sep + fname, orgstate)
        np.savetxt(savedir + os.sep + 'time_{}'.format(da_step), time_series)
        return state_vec_forecast

    def state_to_observation(self, state_vec):
        """ Map the states to observation space (from X to HX).

        Parameters
        ----------
        state_vec: ndarray
            Current state variables (X).
            ``dtype=float``, ``ndim=2``, ``shape=(nstate, nsamples)``

        Returns
        -------
        model_obs: ndarray
            Ensemble in observation space (HX).
            ``dtype=float``, ``ndim=2``,
            ``shape=(nstate_obs, nsamples)``
        """
        return state_vec[[0, 2]]

    def get_obs(self, time):
        """ Return the observation and error matrix.

        Parameters
        ----------
        time : float
            Time at which observation is requested.

        Returns
        -------
        obs : ndarray
            Observations.
            ``dtype=float``, ``ndim=2``,
            ``shape=(nstate_obs, nsamples)``
        obs_error : ndarray
            Observation error covariance matrix (R).
            ``dtype=float``, ``ndim=2``,
            ``shape=(nstate_obs, nstate_obs)``
        """
        da_step = int(time / self.da_interval)
        obs_vec = self.obs[da_step-1, :]
        obs_stddev = np.abs(obs_vec) * self.obs_rel_std + self.obs_abs_std
        obs_error = np.diag(obs_stddev**2)
        return obs_vec, obs_error

    def clean(self):
        """ Cleanup before exiting. """
        pass

    # internal methods
    def _create_synthetic_observations(self, state, parameters):
        """ Create synthetic truth and observations.

        Parameters
        ----------
        state : list
            True value of the initial state [x, y, z].
        parameters : list
            True value of the paramters [rho, beta, sigma].

        Returns
        -------
        obs : ndarray
            Values of observations at different times.
            ``dtype=float``, ``ndim=2``,
            ``shape=(number of observation times, nstate_obs)``
        """
        # create truth
        time_series = np.arange(0.0, self.t_end + self.dt/2.0, self.dt)
        truth = solve_lorenz(time_series, state, parameters)
        # create observations
        ndt = int(self.da_interval/self.dt)
        observe_orgstate = [True, False, True]
        obs_time = time_series[ndt::ndt]
        obs = truth[ndt::ndt, observe_orgstate]
        obs_std = np.abs(obs) * np.tile(self.obs_rel_std, (obs.shape[0], 1)) \
            + np.tile(self.obs_abs_std, (obs.shape[0], 1))
        obs = obs + np.random.normal(0.0, obs_std)
        # save
        time_series = np.expand_dims(time_series, axis=1)
        obs_time = np.expand_dims(obs_time, axis=1)
        np.savetxt(self.dir + os.sep + 'truth.dat',
                   np.append(time_series, truth, axis=1))
        np.savetxt(self.dir + os.sep + 'obs.dat',
                   np.append(obs_time, obs, axis=1))
        np.savetxt(self.dir + os.sep + 'rho.dat',
                   np.expand_dims(np.array(parameters[0]), axis=1))
        np.savetxt(self.dir + os.sep + 'params.dat', [self.beta, self.sigma])
        return obs
