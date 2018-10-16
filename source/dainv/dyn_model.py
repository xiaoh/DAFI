# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Template for dynamic models. """

# third party imports
import numpy as np

class DynModel(object):
    """ Parent class for dynamic models.

    Use this as a template to write new dynamic models.
    The required attributes and methods are summarized below.

    attributes:
        name: Name of the forward model for reporting.
        nstate: Number of states in the state vector.
        nstate_obs: Number of states in the observation vector.

    methods:
        X, HX  = generate_ensemble()
        X, HX  = forecast_to_time(X, next_end_time)
        obs, R_obs = get_obs(next_end_time)
        report()
        plot()
        clean()
    """

    def __init__(self, nsamples, da_interval, t_end,  input_file):
        """ Parse input file and assign values to class attributes.

        Args:
            nsamples: Ensemble size. [int]
            da_interval: Iteration interval between data assimilation
                        steps. [float]
            t_end: Final time. [float]
            input_file: Input file name. [string]
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

        Returns:
            X: Ensemble matrix of states. [nstate x nsamples]
            HX: Ensemble matrix of states in observation space.
                [nstate_obs x nsamples]
        """
        X = np.zeros([self.nstate, self.nsamples])
        HX = np.zeros([self.nstate_obs, self._nsamples])
        return X, HX

    def forecast_to_time(self, X, next_end_time):
        """ Return states at the next end time.

        Args:
            X: Current ensemble matrix of states. [nstate x nsamples]
            next_end_time: Next end time. [float]

        Returns:
            X: Updated ensemble matrix of states. [nstate x nsamples]
            HX: Updated ensemble matrix of states in observation
                space. [nstate_obs x nsamples]
        """
        X = np.zeros([self.nstate, self._nsamples])
        HX = np.zeros([self.nstate_obs, self._nsamples])
        return X, HX

    def get_obs(next_end_time):
        """ Return the observation and error matrix.

        Args:
            next_end_time: Next end time. [float]

        Returns:
            obs: Observations. [nstate_obs x nsamples]
            R_obs: Observation error. [nstate_obs x nstate_obs]
        """
        obs = np.zeros([self.nstate_obs, self._nsamples])
        R_obs = np.zeros([nstate_obs, nstate_obs])
        return obs, R_obs

    def report(self):
        """ Report summary information. """
        pass

    def plot(self):
        """ Plot relevant model results. """
        pass

    def clean(self):
        """ Cleanup before exiting. """
        pass
