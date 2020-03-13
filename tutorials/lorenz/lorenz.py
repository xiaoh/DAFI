# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Dynamic model for solving the Lorenz system. """

# standard library imports
import os

# third party imports
import numpy as np
from scipy.integrate import ode
import yaml

# local imports
from dafi import PhysicsModel

NSTATE = 3
NSTATEAUG = 4
OBSERVE_STATE = [True, False, True]


def lorenz(time_series, init_state, parameters):
    """ Solve the Lorenz 63 equations for given initial condition.

    Parameters
    ----------
    time_series : list
        Time-series for the integration of the Lorenz system.
        The first entry is the initial time. *len=3*, *type=float*
    init_state : list
        The state [x, y, z] at the initial time.
        *len=3*, *type=float*
    parameters : list
        The values of the three parameters [rho, beta, sigma].
        *len=3*, *type=float*

    Returns
    -------
    state : ndarray
        The state [x, y, z] at all times specified in the time_series.
        *dtype=float*, *ndim=2*, *shape=(len(time_series), 3)*


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
            *len=3*, *type=float*
        parameters : list
            The values of the three parameters [rho, beta, sigma].
            *len=3*, *type=float*

        Returns
        -------
        velocities : list
            The derivative with respect to time of the 3 state components.
            *len=3*, *type=float*
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
            raise RuntimeError(f'Solver failed at time: {time} s')
        else:
            solver.integrate(time)
        state = np.vstack((state, solver.y))
    return state


class Model(PhysicsModel):
    """ Dynamic model for solving the Lorenz system. """

    def __init__(self, inputs_dafi, inputs_model):
        # save the required dafi inputs
        self.nsamples = inputs_dafi['nsamples']
        tend = inputs_dafi['ntime']

        # modify DAFI inputs
        if inputs_dafi['convergence_option'] != 'max':
            inputs_dafi['convergence_option'] = 'max'
            warning.warn("User-supplied 'convergence_option' modified.")
        if inputs_dafi['max_iterations'] != 1:
            inputs_dafi['max_iterations'] = 1
            warning.warn("User-supplied 'max_iterations' modified.")

        # read input file
        input_file = inputs_model['input_file']
        with open(input_file, 'r') as f:
            inputs_model = yaml.load(f, yaml.SafeLoader)

        dt_interval = inputs_model['dt_interval']
        da_interval = inputs_model['da_interval']
        x_init_mean = inputs_model['x_init_mean']
        y_init_mean = inputs_model['y_init_mean']
        z_init_mean = inputs_model['z_init_mean']
        rho_init_mean = inputs_model['rho_init_mean']
        x_init_std = inputs_model['x_init_std']
        y_init_std = inputs_model['y_init_std']
        z_init_std = inputs_model['z_init_std']
        rho_init_std = inputs_model['rho_init_std']
        beta = inputs_model['beta']
        sigma = inputs_model['sigma']
        x_obs_rel_std = inputs_model['x_obs_rel_std']
        z_obs_rel_std = inputs_model['z_obs_rel_std']
        x_obs_abs_std = inputs_model['x_obs_abs_std']
        z_obs_abs_std = inputs_model['z_obs_abs_std']
        x_true = inputs_model['x_true']
        y_true = inputs_model['y_true']
        z_true = inputs_model['z_true']
        rho_true = inputs_model['rho_true']
        beta_true = inputs_model['beta_true']
        sigma_true = inputs_model['sigma_true']

        # required attributes.
        self.name = 'Lorenz63'

        # save other inputs for future use
        self.init_state = [x_init_mean, y_init_mean, z_init_mean,
                           rho_init_mean]
        self.dt = dt_interval
        self.da = da_interval
        self.beta = beta
        self.sigma = sigma
        self.init_std = [x_init_std, y_init_std, z_init_std, rho_init_std]
        self.obs_rel_std = [x_obs_rel_std, z_obs_rel_std]
        self.obs_abs_std = [x_obs_abs_std, z_obs_abs_std]

        # create save directory
        self.dir = 'results_lorenz'
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        # time series
        tend = tend * self.da * self.dt
        self.time = np.arange(0.0, tend - (self.da-0.5)*self.dt, self.dt)

        # create synthetic observations
        true_init_orgstate = [x_true, y_true, z_true]
        true_params = [rho_true, beta_true, sigma_true]
        self.obs = self._create_synthetic_observations(
            true_init_orgstate, true_params)

    def __str__(self):
        str_info = 'Lorenz 63 model.'
        return str_info

    def generate_ensemble(self):
        """ Return states at the first data assimilation time-step.

        Returns
        -------
        state_vec : ndarray
            Ensemble matrix of states (Xi).
            *dtype=float*, *ndim=2*, *shape=(nstate, nsamples)*
        model_obs : ndarray
            Ensemble matrix of states in observation space (HX).
            *dtype=float*, *ndim=2*, *shape=(nstate_obs, nsamples)*
        """
        state_vec = np.empty([NSTATEAUG, self.nsamples])
        for isamp in range(self.nsamples):
            state_vec[:, isamp] = self.init_state \
                + np.random.normal(0, self.init_std)
        model_obs = self.state_to_observation(state_vec)
        return state_vec

    def forecast_to_time(self, state_vec_current, end_time):
        """ Return states at the next end time.

        Parameters
        ----------
        state_vec_current : ndarray
            Current ensemble matrix of states (Xa). [nstate x nsamples]
        end_time : int
            Next DA time index.

        Returns
        -------
        state_vec_forecast : ndarray
            Updated ensemble matrix of states (Xf).
            *dtype=float*, *ndim=2*, *shape=(nstate, nsamples)*
        """
        # create time series
        end_ptime = end_time * self.da * self.dt
        start_time = end_ptime - self.da * self.dt
        time_series = np.arange(start_time, end_ptime + self.dt/2.0, self.dt)
        # initialize variables
        state_vec_forecast = np.empty([NSTATEAUG, self.nsamples])
        savedir = os.path.join(self.dir, 'states')
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        for isamp in range(self.nsamples):
            # solve
            parameters = [state_vec_current[3, isamp], self.beta, self.sigma]
            init_orgstate = state_vec_current[:3, isamp]
            orgstate = lorenz(time_series, init_orgstate, parameters)
            # create forecasted vector
            state_vec_forecast[:3, isamp] = orgstate[-1, :]
            state_vec_forecast[3, isamp] = state_vec_current[3, isamp]
            # save
            fname = f'dastep_{end_time}_samp_{isamp}'
            np.savetxt(os.path.join(savedir, fname), orgstate)
        np.savetxt(os.path.join(savedir, f'time_{end_time}'), time_series)
        return state_vec_forecast

    def state_to_observation(self, state_vec):
        """ Map the states to observation space (from X to HX).

        Parameters
        ----------
        state_vec: ndarray
            Ensemble of state variables (X).
            *dtype=float*, *ndim=2*, *shape=(nstate, nsamples)*

        Returns
        -------
        model_obs: ndarray
            Ensemble in observation space (HX).
            *dtype=float*, *ndim=2*, *shape=(nstate_obs, nsamples)*
        """
        return state_vec[[0, 2]]

    def get_obs(self, time):
        """ Return the observation and error matrix.

        Parameters
        ----------
        time : float
            DA time index at which observation is requested.

        Returns
        -------
        obs : ndarray
            Observations.
            *dtype=float*, *ndim=2*, *shape=(nstate_obs, nsamples)*
        obs_error : ndarray
            Observation error covariance matrix (R).
            *dtype=float*, *ndim=2*, *shape=(nstate_obs, nstate_obs)*
        """
        obs_vec = self.obs[time, :]
        obs_stddev = np.abs(obs_vec) * self.obs_rel_std + self.obs_abs_std
        obs_error = np.diag(obs_stddev**2)
        return obs_vec, obs_error

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
            *dtype=float*, *ndim=2*, *shape=(n_da_times, nstate_obs)*
        """
        # create truth
        truth = lorenz(self.time, state, parameters)
        # create observations
        obs_time = self.time[::self.da]
        obs = truth[::self.da, OBSERVE_STATE]
        obs_std = np.abs(obs) * np.tile(self.obs_rel_std, (obs.shape[0], 1)) \
            + np.tile(self.obs_abs_std, (obs.shape[0], 1))
        obs = obs + np.random.normal(0.0, obs_std)
        # save
        self.time = np.expand_dims(self.time, axis=1)
        obs_time = np.expand_dims(obs_time, axis=1)
        np.savetxt(os.path.join(self.dir, 'truth.dat'),
                   np.append(self.time, truth, axis=1))
        np.savetxt(os.path.join(self.dir, 'obs.dat'),
                   np.append(obs_time, obs, axis=1))
        np.savetxt(os.path.join(self.dir, 'rho.dat'),
                   np.expand_dims(np.array(parameters[0]), axis=0))
        np.savetxt(os.path.join(self.dir, 'params.dat'),
                   [self.beta, self.sigma])
        return obs
