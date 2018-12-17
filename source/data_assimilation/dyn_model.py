# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Template for dynamic models. """

# third party imports
import numpy as np


class DynModel(object):
    """ Parent class for dynamic models.

    Use this as a template to write new dynamic models.
    The required attributes and methods are summarized below.

    Attributes
    ----------
        name: Name of the forward model for reporting. ``str``
        nstate: Number of states in the state vector. ``int``
        nstate_obs: Number of states in the observation vector. ``int``
        init_state: Initial mean value of the state vector. ``ndarray``
    """

    def __init__(self, nsamples, da_interval, t_end, max_da_iteration,
                 input_file):
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
        """
        self.name = 'DynModel'
        self.nstate = 1
        self.nstate_obs = 1
        init_state = np.zeros(self.nstate)
        self._nsamples = nsamples
        pass

    def __str__(self):
        str_info = 'An empty dynamic model.'
        return str_info

    def generate_ensemble(self):
        """ Return states at the first data assimilation time-step.

        Returns
        -------
        state_vec : ndarray
            Ensemble matrix of states (Xi).
            ``dtype=float``, ``ndim=2``, ``shape=(nstate, nsamples)``
        model_obs : ndarray
            Ensemble matrix of states in observation space.
            ``dtype=float``, ``ndim=2``,
            ``shape=(nstate_obs, nsamples)``
        """
        state_vec = np.zeros([self.nstate, self.nsamples])
        model_obs = np.zeros([self.nstate_obs, self._nsamples])
        return state_vec, model_obs

    def forecast_to_time(self, state_vec, next_end_time):
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
        state_vec = np.zeros([self.nstate, self._nsamples])
        return state_vec

    def forward(self, state_vec):
        """ Forward the states to observation space (from X to HX).

        Parameters
        ----------
        state_vec: ndarray
            Current state variables.
            ``dtype=float``, ``ndim=2``, ``shape=(nstate, nsamples)``

        Returns
        -------
        model_obs: ndarray
            Ensemble in observation space. ``dtype=float``, ``ndim=2``,
            ``shape=(nstate_obs, nsamples)``
        """
        model_obs = np.zeros([self.nstate_obs, self._nsamples])
        return model_obs

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
            ``shape=(nstate_obs, nsamples)``
        obs_error : ndarray
            Observation error.
            ``dtype=float``, ``ndim=2``,
            ``shape=(nstate_obs, nstate_obs)``
        """
        obs = np.zeros([self.nstate_obs, self._nsamples])
        obs_error = np.zeros([nstate_obs, nstate_obs])
        return obs, obs_error

    def clean(self):
        """ Cleanup before exiting. """
        pass
