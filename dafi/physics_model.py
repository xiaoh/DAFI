# Copyright 2020 Virginia Polytechnic Institute and State University.
""" Template for physics models. """

# third party imports
import numpy as np


class PhysicsModel(object):
    """ Parent class for physics models.

    Use this as a template to write new dynamic models.
    The required attributes and methods are summarized below.

    Attributes
    ----------
        name: Name of the forward model for reporting. ``str``
        nstate: Number of states in the state vector. ``int``
        nobs: Number of observations in the observation vector. ``int``
        init_state: Initial mean value of the state vector. ``ndarray``
    """

    def __init__(self, nsamples, t_interval, t_end, max_iterations,
                 input_dict):
        """ Parse input file and assign values to class attributes.

        Parameters
        ----------
        nsamples : int
            Ensemble size.
        t_interval : float
            Time interval between data assimilation steps.
        t_end : float
            Final time.
        max_iterations : int
            Maximum number of iterations at a given time-step.
        input_dict : dict
            Dictionary containing required model inputs.
        """
        self.name = 'Physics Model'
        self.nstate = 1
        self.nobs = 1
        self.init_state = np.zeros(self.nstate)
        self._nsamples = nsamples

    def __str__(self):
        str_info = 'An empty physics model.'
        return str_info

    def generate_ensemble(self):
        """ Return states at the first data assimilation time-step.

        Returns
        -------
        state : ndarray
            Ensemble matrix of states (Xi).
            ``dtype=float``, ``ndim=2``, ``shape=(nstate, nsamples)``
        state_obs : ndarray
            Ensemble matrix of states mapped to observation space (HXi).
            ``dtype=float``, ``ndim=2``,
            ``shape=(nobs, nsamples)``
        """
        state = np.zeros([self.nstate, self._nsamples])
        state_obs = np.zeros([self.nobs, self._nsamples])
        return state, state_obs

    def forecast_to_time(self, state, next_end_time):
        """ Return states at the next end time.

        Parameters
        ----------
        state : ndarray
            Current ensemble matrix of states (Xa). [nstate x nsamples]
        next_end_time : float
            Next end time.

        Returns
        -------
        state : ndarray
            Updated ensemble matrix of states (Xf).
            ``dtype=float``, ``ndim=2``, ``shape=(nstate, nsamples)``
        """
        state = np.zeros([self.nstate, self._nsamples])
        return state

    def state_to_observation(self, state):
        """ Map the states to observation space (X to HX).

        Parameters
        ----------
        state: ndarray
            Current state variables.
            ``dtype=float``, ``ndim=2``, ``shape=(nstate, nsamples)``

        Returns
        -------
        state_obs: ndarray
            Ensemble in observation space. ``dtype=float``, ``ndim=2``,
            ``shape=(nobs, nsamples)``
        """
        state_obs = np.zeros([self.nobs, self._nsamples])
        return state_obs

    def get_obs(time):
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
            ``shape=(nobs, nsamples)``
        obs_error : ndarray
            Observation error.
            ``dtype=float``, ``ndim=2``,
            ``shape=(nobs, nobs)``
        """
        obs = np.zeros([self.nobs, self._nsamples])
        obs_error = np.zeros([nobs, nobs])
        return obs, obs_error

    def clean(self):
        """ Cleanup before exiting. """
        pass
