# Copyright 2020 Virginia Polytechnic Institute and State University.
""" Collection of ensemble-based Bayesian inversion techniques. """

# standard library imports
import warnings
import os
import sys
import importlib
import logging
logger = logging.getLogger(__name__)

# third party imports
import numpy as np


# template parent class
class InverseMethod(object):
    """ Parent class for ensemble-based Bayesian inversion techniques.

    Use this as a template to write new inversion classes.
    To implement a new inverse technique create a child class and
    override the ``analysis`` method.
    """

    def __init__(self, inputs_dafi, inputs):
        """ Parse input file and assign values to class attributes.

        Parameters
        ----------
        inputs_dafi : dict
            Dictionary containing all the dafi inputs in case the model
            requires access to this information.
        inputs : dict
            Dictionary containing required inverse method inputs.
        """
        self.nsamples = inputs_dafi['nsamples']
        self._debug = inputs_dafi['save_level'] == 'debug'
        self._debug_dir = os.path.join(inputs_dafi['save_dir'], 'debug')
        self.inputs_dafi = inputs_dafi
        self.time = 0


    def __str__(self):
        str_info = f'DAFI inverse method: \n    {self.name}'
        return str_info


    def analysis(self, iteration, state_forecast, state_in_obsspace, obs,
            obs_error, obs_vec):
        """ Correct the forecast ensemble states to analysis state.

        This is the data assimilation step.

        Parameters
        ----------
        iteration : int
            Iteration number at current DA time step.
        state_forecast : ndarray
            Ensemble of forecast states (Xf).
            ``dtype=float``, ``ndim=2``, ``shape=(nstate, nsamples)``
        state_in_obsspace : ndarray
            Ensemble forecast states mapped to observation space (Hx).
            ``dtype=float``, ``ndim=2``, ``shape=(nobs, nsamples)``.
        obs : ndarray
            Ensemble of (possibly perturbed) observations.
            ``dtype=float``, ``ndim=2``, ``shape=(nobs, nsamples)``
        obs_error : ndarray
            Observation error (covariance) matrix.
            ``dtype=float``, ``ndim=2``, ``shape=(nobs, nobs)``
        obs_vec : ndarray
            Unperturbed observation vector. This is the actual
            observation and is the mean of ``obs``.
            ``dtype=float``, ``ndim=1``, ``shape=(nobs)``

        Returns
        -------
        state_analysis : ndarray
            Ensemble matrix of updated states (xa).
            ``dtype=float``, ``ndim=2``, ``shape=(nstate, nsamples)``
        """
        dx = np.zeros(state_forecast.shape)
        state_analysis = state_forecast + dx
        return state_analysis

    def _save_debug(self, debug_dict, iteration):
        """ Save specified ndarrays to the debug directory.
        """
        if iteration == 0:
            self.time += 1
        for key, value in debug_dict.items():
            file = key + f'_{self.time-1}_{iteration}'
            np.savetxt(os.path.join(self._debug_dir, file), value)


# child classes (specific filtering techniques)
class EnKF(InverseMethod):
    """ Implementation of the ensemble Kalman Filter (EnKF).

    The EnKF is updated by: ``xa = xf + K*(obs - Hx)`` where *xf* is
    the forecasted state vector (by the forward model), *xa* is the
    updated vector after data-assimilation, *K* is the Kalman gain
    matrix, *obs* is the observation vector, and *Hx* is the forecasted
    state vector in observation space.
    """

    def __init__(self, inputs_dafi, inputs):
        """ See InverseMethod.__init__ for details. """
        super(self.__class__, self).__init__(inputs_dafi, inputs)
        self.name = 'Ensemble Kalman Filter (EnKF)'

    def analysis(self, iteration, state_forecast, state_in_obsspace, obs,
            obs_error, obs_vec):
        """ Correct the forecast ensemble states using EnKF.

        Note
        ----
        See InverseMethod.analysis for I/O details.
        """
        # calculate the Kalman gain matrix
        xp = _mean_subtracted_matrix(state_forecast)
        hxp = _mean_subtracted_matrix(state_in_obsspace)
        coeff = (1.0 / (self.nsamples - 1.0))
        pht = coeff * np.dot(xp, hxp.T)
        hpht = coeff * hxp.dot(hxp.T)
        conn = _check_condition_number(hpht + obs_error, '(HPHT + R)')
        inv = np.linalg.inv(hpht + obs_error)
        kalman_gain_matrix = pht.dot(inv)

        # analysis step
        dx = np.dot(kalman_gain_matrix, obs - state_in_obsspace)
        state_analysis = state_forecast + dx

        # debug
        if self._debug:
            debug_dict = {
                'K': kalman_gain_matrix, 'inv': inv, 'HPHT': hpht, 'PHT': pht,
                'Hxp': hxp, 'xp': xp}
            self._save_debug(debug_dict, iteration)
        return state_analysis


class EnRML(InverseMethod):
    """ Implementation of the ensemble Randomized Maximal Likelihood
    (EnRML).

    The EnRML is updated by: ``xa = xf + GN*(obs - Hx)+P``
    where *xf* is the forecasted state vector (by the forward model),
    *xa* is the updated vector after data-assimilation, *GN* is the
    Gauss-Newton matrix, *obs* is the observation vector, and *Hx* is
    the forecasted state vector in observation space, *P* is Penalty
    matrix.

    Note
    ----
    Required inputs in ``inputs`` dictionary:
        * **step_length** - ``float``
            EnRML step length parameter. has value between 0 and 1.
    """

    def __init__(self, inputs_dafi, inputs):
        """ See InverseMethod.__init__ for details. """
        super(self.__class__, self).__init__(inputs_dafi, inputs)
        self.name = 'Ensemble Randomized Maximal Likelihood (EnRML)'
        self.beta = inputs['step_length']

        # Override user-specified observation perturbation obtion
        # check and give warning
        check = inputs_dafi['perturb_obs_option'] == 'time'
        if not check:
            message = "warning: EnRML: 'perturb_obs_option' set to 'time'."
            logger.warning(message)
            # override
        inputs_dafi['perturb_obs_option'] = 'time'

    def analysis(self, iteration, state_forecast, state_in_obsspace, obs,
            obs_error, obs_vec):
        """ Correct the forecast ensemble states using EnRML.

        Note
        ----
        See InverseMethod.analysis for I/O details.
        """
        # save the prior state
        if iteration == 0:
            self.state_prior = state_forecast

        # calculate the Gauss-Newton matrix
        xp0 = _mean_subtracted_matrix(self.state_prior)
        p0 = (1.0 / (self.nsamples - 1.0)) * xp0.dot(xp0.T)
        x = state_forecast
        xp = _mean_subtracted_matrix(x)
        hxp = _mean_subtracted_matrix(state_in_obsspace)
        gen = np.dot(hxp, np.linalg.pinv(xp))
        sen_mat = p0.dot(gen.T)
        cyyi = np.dot(np.dot(gen, p0), gen.T)
        cyyi_R = cyyi + obs_error
        conn = _check_condition_number(cyyi_R, '(cyyi + R)')
        inv = np.linalg.inv(cyyi_R)
        gauss_newton_matrix = sen_mat.dot(inv)

        # calculate the penalty
        penalty = np.dot(gauss_newton_matrix, gen.dot(x - self.state_prior))

        # analysis step
        diff = obs - state_in_obsspace
        dx = np.dot(gauss_newton_matrix, diff) + penalty
        state_analysis = self.beta * self.state_prior + \
                         (1.0 - self.beta) * x + self.beta * dx

        # debug
        if self._debug:
            debug_dict = {
                'GN': gauss_newton_matrix, 'pen': penalty, 'inv': inv,
                'cyyi': cyyi, 'Hxp': hxp, 'xp': xp}
            self._save_debug(debug_dict, iteration)
        return state_analysis


class EnKF_MDA(InverseMethod):
    """ Implementation of the ensemble Kalman Filter-Multi data
    assimilaton (EnKF-MDA).

    The EnKF-MDA is updated by:
    ``xa = xf + K_mda*(obs - Hx - err_mda)`` where *xf* is the
    forecasted state vector (by the dynamic model),
    *xa* is the updated vector after data-assimilation, *K_mda* is the
    modified Kalman gain matrix, *obs* is the observation vector, and
    *Hx* is the forwarded state vector in observation space, 'err_mda'
    is inflated error.


    Note
    ----
    Required inputs in ``inputs`` dictionary:
        * **nsteps** - ``int``
            Number of steps used in the multiple data assimilation.
    """

    def __init__(self, inputs_dafi, inputs):
        """ See InverseMethod.__init__ for details. """
        super(self.__class__, self).__init__(inputs_dafi, inputs)
        self.name = 'Ensemble Kalman Filter-Multi Data Assimilation (EnKF-MDA)'
        self.alpha = inputs['nsteps']

        # Override user-specified convergence
        # check and give warning
        check1 = inputs_dafi['convergence_option'] == 'max'
        check2 = inputs_dafi['max_iterations'] == 1
        if not (check1 and check2):
            message = 'User-supplied convergence options ignored.'
            warnings.warn(message)
        # override
        inputs_dafi['convergence_option'] = 'max'
        inputs_dafi['max_iterations'] = self.alpha


    def analysis(self, iteration, state_forecast, state_in_obsspace, obs,
            obs_error, obs_vec):
        """ Correct the forecast ensemble states using EnKF-MDA.

        Note
        ----
        See InverseMethod.analysis for I/O details.
        """
        # calculate the Kalman gain matrix
        x = state_forecast
        xp = _mean_subtracted_matrix(state_forecast)
        hxp = _mean_subtracted_matrix(state_in_obsspace)
        coeff = (1.0 / (self.nsamples - 1.0))
        pht = coeff * np.dot(xp, hxp.T)
        hpht = coeff * hxp.dot(hxp.T)
        hpht_ar = hpht + self.alpha * obs_error
        conn =  _check_condition_number(hpht_ar, '(HPHT + aR)')
        inv = np.linalg.inv(hpht_ar)
        kalman_gain_matrix = pht.dot(inv)

        # analysis step
        nsamps = obs.shape[1]
        obs_mean = np.tile(obs_vec, (nsamps, 1)).T
        obs_perturbation = obs - obs_mean
        obs = obs_mean
        obs_mda = obs + np.sqrt(self.alpha) * obs_perturbation
        dx = np.dot(kalman_gain_matrix, obs_mda - state_in_obsspace)
        state_analysis = x + dx

        # debug
        if self._debug:
            debug_dict = {
                'K': kalman_gain_matrix, 'inv': inv, 'HPHT': hpht, 'PHT': pht,
                'Hxp': hxp, 'xp': xp}
            self._save_debug(debug_dict, iteration)
        return state_analysis


class REnKF(InverseMethod):
    """ Implementation of the regularized ensemble Kalman Filter
    (REnKF).

    Note
    ----
    Required inputs in ``inputs`` dictionary:
        * **penalties_python_file** (``string``) -
          Path to python file that contains function called
          ``penalties`` that returns a list of dictionaries.
          Each dictionary represents one penalty and includes:
          ``lambda`` (float), ``weight_matrix`` (ndarray),
          ``penalty`` (function), and ``gradient`` (function).
    """

    def __init__(self, inputs_dafi, inputs):
        """ See InverseMethod.__init__ for details. """
        super(self.__class__, self).__init__(inputs_dafi, inputs)
        self.name = 'Regularized Ensemble Kalman Filter (REnKF)'
        # load penalties
        pfile = inputs['penalties_python_file']
        sys.path.append(os.path.dirname(pfile))
        penalties = getattr(importlib.import_module(
            os.path.splitext(os.path.basename(pfile))[0]), 'penalties')
        self.penalties = penalties(self)

    def analysis(self, iteration, state_forecast, state_in_obsspace, obs,
            obs_error, obs_vec):
        """ Correct the forecast ensemble states using REnKF.

        Note
        ----
        See InverseMethod.analysis for I/O details.
        """
        # calculate the Kalman gain matrix
        xp = _mean_subtracted_matrix(state_forecast)
        hxp = _mean_subtracted_matrix(state_in_obsspace)
        coeff = (1.0 / (self.nsamples - 1.0))
        pht = coeff * np.dot(xp, hxp.T)
        hpht = coeff * hxp.dot(hxp.T)
        hpht_R = hpht + obs_error
        conn =  _check_condition_number(hpht_R, '(HPHT + R)')
        inv = np.linalg.inv(hpht_R)
        kalman_gain_matrix = pht.dot(inv)

        # calculate the "K2" matrix
        hxx = np.dot(hxp, xp.T)
        k2_gain_matrix = \
            coeff * np.dot(kalman_gain_matrix, hxx) - coeff*np.dot(xp, xp.T)

        # calculate penalty matrix
        penalty_mat = np.zeros([len(state_forecast), self.nsamples])
        for ipenalty in self.penalties:
            w_mat = ipenalty['weight_matrix']
            lamb = ipenalty['lambda']
            lamda = lamb(iteration)
            func_penalty = ipenalty['penalty']
            func_gradient = ipenalty['gradient']
            for isamp in range(self.nsamples):
                istate = state_forecast[:, isamp]
                gpw = np.dot(func_gradient(istate).T, w_mat)
                gpwg = np.dot(gpw, func_penalty(istate))
                penalty_mat[:, isamp] += lamda * gpwg

        # analysis step
        dx1 = np.dot(kalman_gain_matrix, obs - state_in_obsspace)
        dx2 = np.dot(k2_gain_matrix, penalty_mat)
        state_analysis = state_forecast + dx1 + dx2

        # debug
        if self._debug:
            # TODO: Save each penalty separately if debug
            # debug
            debug_dict = {
                'K': kalman_gain_matrix, 'inv': inv, 'HPHT': hpht, 'PHT': pht,
                'Hxp': hxp, 'xp': xp, 'lamda': lamda,
                'dx1': dx1, 'dx2': dx2, 'k2': k2_gain_matrix,
                'penalty': penalty_mat,
                }
            self._save_debug(debug_dict, iteration)
        return state_analysis


# functions
def _check_condition_number(mat, name='matrix', eps=1e16,):
    """ Calculate the condition number of a matrix and check it is below
    the specified threshold.
    """
    conn = np.linalg.cond(mat)
    if (conn > eps):
        message = "The matrix is singular, inverse will fail."
        warnings.warn(message, RuntimeWarning)
    debug_message = f"    Condition number of {name} is {conn}"
    logger.log(logging.DEBUG, debug_message)
    return conn


def _mean_subtracted_matrix(mat, samp_axis=1,):
    """ Return the mean subtracted matrix. """
    nsamps = mat.shape[samp_axis]
    mean_vec = np.array([np.mean(mat, axis=samp_axis)])
    mean_vec = mean_vec.T
    mean_vec = np.tile(mean_vec, (1, nsamps))
    mean_sub_mat = mat - mean_vec
    return mean_sub_mat
