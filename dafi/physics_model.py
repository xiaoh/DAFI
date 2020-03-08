# Copyright 2020 Virginia Polytechnic Institute and State University.
""" Template for physics models. """

# third party imports
import numpy as np


class PhysicsModel(object):
    """ Parent class for physics models.

    Accessible through **dafi.PhysicsModel**.
    Use this as a template to write new dynamic models.
    The required attributes and methods are summarized below.

    Attributes
    ----------
        * **name** - Name of the forward model for reporting. *str*
        * **nstate** - Number of states in the state vector. *int*
        * **nobs** - Number of observations in the observation vector.
            *int*
        * **init_state** - Initial mean value of the state vector.
            *ndarray*, *dtype=float*, *ndim=1*, *shape=(nstate)*

    Methods
    -------
    See the method's docstring for information on each.
        * **generate_ensemble**
        * **forecast_to_time**
        * **state_to_observation**
        * **get_obs**
    """

    def __init__(self, inputs_dafi, inputs):
        """ Parse input file and assign values to class attributes.

        Parameters
        ----------
        inputs_dafi : dict
            Dictionary containing all the dafi inputs in case the model
            requires access to this information.
        inputs : dict
            Dictionary containing required model inputs.
        """
        self.name = 'Physics Model'
        self.nstate = 1
        self.nobs = 1
        self.init_state = np.zeros(self.nstate)
        self._nsamples = inputs_dafi['nsamples']

    def __str__(self):
        str_info = 'An empty physics model.'
        return str_info

    def generate_ensemble(self):
        """ Return states at the first data assimilation time-step.

        Returns
        -------
        state : ndarray
            Ensemble matrix of states (X0).
            *dtype=float*, *ndim=2*, *shape=(nstate, nsamples)*
        """
        state = np.zeros([self.nstate, self._nsamples])
        return state

    def forecast_to_time(self, state, time):
        """ Return states at the next end time.

        Parameters
        ----------
        state : ndarray
            Current ensemble of states (Xa).
            *dtype=float*, *ndim=2*, *shape=(nstate, nsamples)*
        time : int
            Next end time index. Any concept of real time is implemented
            the physics model (e.g. this file).

        Returns
        -------
        state : ndarray
            Updated ensemble matrix of states (Xf).
            *dtype=float*, *ndim=2*, *shape=(nstate, nsamples)*
        """
        state = np.zeros([self.nstate, self._nsamples])
        return state

    def state_to_observation(self, state):
        """ Map the states to observation space (X to HX).

        Parameters
        ----------
        state : ndarray
            Ensemble of states.
            *dtype=float*, *ndim=2*, *shape=(nstate, nsamples)*

        Returns
        -------
        state_obs : ndarray
            Ensemble in observation space.
            *dtype=float*, *ndim=2*, *shape=(nobs, nsamples)*
        """
        state_obs = np.zeros([self.nobs, self._nsamples])
        return state_obs

    def get_obs(time):
        """ Return the observation and error matrix.

        Parameters
        ----------
        time : int
            Time index at which observation is requested. Any concept of
            real time is implemented the physics model (e.g. this file).

        Returns
        -------
        obs : ndarray
            Observations.
            *dtype=float*, *ndim=2*, *shape=(nobs, nsamples)*
        obs_error : ndarray
            Observation error (covariance) matrix.
            *dtype=float*, *ndim=2*, *shape=(nobs, nobs)*
        """
        obs = np.zeros([self.nobs, self._nsamples])
        obs_error = np.zeros([nobs, nobs])
        return obs, obs_error
