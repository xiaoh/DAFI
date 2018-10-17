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
        name: Name of the forward model for reporting.
        nstate: Number of states in the state vector.
        nstate_obs: Number of states in the observation vector.
    """

    def __init__(self, nsamples, da_interval, t_end,  input_file):
        """ Parse input file and assign values to class attributes.

        Parameters
        ----------
        nsamples : int
            Ensemble size.
        da_interval : float
            Iteration interval between data assimilation steps.
        t_end : float
            Final time.
        input_file : str
            Input file name.
        """
        self.name = 'DynModel'
        self.nstate = 0
        self.nstate_obs = 0
        self._nsamples = nsamples
        pass

    def __str__(self):
        s = 'An empty dynamic model.'
        return s

    def generate_ensemble(self):
        """ Return states at the first data assimilation time-step.

        Returns
        -------
        state_vec : ndarray
            Ensemble matrix of states.
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
            Current ensemble matrix of states. [nstate x nsamples]
        next_end_time : float
            Next end time.

        Returns
        -------
        state_vec : ndarray
            Updated ensemble matrix of states.
            ``dtype=float``, ``ndim=2``, ``shape=(nstate, nsamples)``
        model_obs : ndarray
            Updated ensemble matrix of states in observation space.
            ``dtype=float``, ``ndim=2``,
            ``shape=(nstate_obs, nsamples)``
        """
        state_vec = np.zeros([self.nstate, self._nsamples])
        model_obs = np.zeros([self.nstate_obs, self._nsamples])
        return state_vec, model_obs

    def get_obs(next_end_time):
        """ Return the observation and error matrix.

        Parameters
        ----------
        next_end_time : float
            Next end time.

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

    def report(self):
        """ Report summary information. """
        pass

    def plot(self):
        """ Plot relevant model results. """
        pass

    def clean(self):
        """ Cleanup before exiting. """
        pass
