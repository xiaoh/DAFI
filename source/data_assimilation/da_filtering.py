# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Collection of different filtering techniques. """

# standard library imports
import ast
import warnings
import os
import sys
import importlib

# third party imports
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

# local imports
import data_assimilation.utilities as utils


# parent classes (templates)
class DAFilter(object):
    """ Parent class for data assimilation filtering techniques.

    Use this as a template to write new filtering classes.
    The required methods are summarized below.
    """

    def __init__(self, nsamples, da_interval, t_end, max_da_iteration,
                 dyn_model, input_dict):
        """ Parse input file and assign values to class attributes.

        Parameters
        ----------
        nsamples : int
            Ensemble size.
        da_interval : float
            Iteration interval between data assimilation steps.
        t_end : float
            Final time.
        max_da_iteration : int
            Maximum number of DA iterations at a given time-step.
        dyn_model : DynModel
            Dynamic model.
        input_dict : dict[str]
            All filter-specific inputs.
        """
        self.dyn_model = dyn_model

    def __str__(self):
        str_info = 'An empty data assimilation filtering technique.'
        return str_info

    def solve(self):
        """ Implement the filtering technique. """
        pass

    def plot(self):
        """ Create any relevant plots. """
        pass

    def clean(self):
        """ Perform any necessary cleanup at completion. """
        self.dyn_model.clean()

    def save(self):
        """ Save any important results to text files. """
        pass


class DAFilter2(DAFilter):
    """ Parent class for DA filtering techniques.

    This class includes more implemented methods than the DAFilter
    class. The DAFilter class is a barebones template. DAFilter2
    contains several methods (e.g error checking, plotting) as well
    as a framework for the main self.solve() method. This can be used
    to quickly create new filter classes if some of the same methods
    are desired.
    """

    def __init__(self, nsamples, da_interval, t_end, max_da_iteration,
                 dyn_model, input_dict):
        """ Parse input file and assign values to class attributes.

        Parameters
        ----------
        nsamples : int
            Ensemble size.
        da_interval : float
            Iteration interval between data assimilation steps.
        t_end : float
            Final time.
        max_da_iteration : int
            Maximum number of DA iterations at a given time-step.
        dyn_model : DynModel
            Dynamic model.
        input_dict : dict[str]
            All filter-specific inputs.

        Note
        ----
        Inputs in ``input_dict``:
            * **save_dir** (``string``, ``./results_da``) -
              Folder where to save results.
            * **debug_flag** (``bool``, ``False``) -
              Save extra information for debugging.
            * **debug_dir** (``string``, ``./debug``) -
              Folder where to save debugging information
            * **verbosity** (``int``, ``1``) -
              Amount of information to print to screen.
            * **sensitivity_only** (``bool``, ``False``) -
              Perform initial perturbation but no data assimilation).
            * **reach_max_flag** (``bool``, ``True``) -
              Do not terminate simulation when converged.
            * **convergence_option** (``str``, ``variance``) -
              Convergence criteria to use if ``reach_max_flag`` is
              ``False``: ``variance`` to use the variance convergence
              criteria,  ``residual``. See documentation for more
              information.
            * **convergence_residual** (``float``, None) -
              Residual value for convergence if ``reach_max_flag`` is
              ``False`` and ``convergence_option`` is ``residual``.
            * **convergence_norm** (``int, inf``,``2``) -
              Order of norm to use for convergence criteria. can be an
              integer or ``np.inf``. Default is L2 norm.
            * **perturb_obs** (``bool``, ``True``) -
              Perturb the observations for each sample.
            * **obs_err_multiplier** (``float``, ``1.0``) -
              Factor by which to multiply the observation error (R).
        """
        self.name = 'Generic DA filtering technique'
        self.short_name = None
        self.dyn_model = dyn_model
        self.nsamples = nsamples  # number of samples in ensemble
        self.nstate = self.dyn_model.nstate  # number of states
        self.nstate_obs = self.dyn_model.nstate_obs  # number of observations
        self.da_interval = da_interval  # DA time step interval
        self.t_end = t_end  # total run time
        self.max_da_iteration = max_da_iteration  # max iterations per  DA step
        if t_end == da_interval:
            self._stationary_flag = True
        else:
            self._stationary_flag = False

        # filter-specific inputs
        try:
            self.reach_max_flag = ast.literal_eval(
                input_dict['reach_max_flag'])
        except:
            self.reach_max_flag = True
        try:
            self.convergence_option = input_dict['convergence_option']
        except:
            self.convergence_option = 'variance'
        if self.convergence_option not in ['variance', 'residual']:
            raise NotImplementedError('Invalid convergence_option.')
        if not self.reach_max_flag and self.convergence_option is 'variance':
            self.convergence_residual = float(
                input_dict['convergence_residual'])
        else:
            self.convergence_residual = None
        try:
            self.convergence_norm = input_dict['convergence_norm']
        except:
            self.convergence_norm = int(2)
        try:
            self.convergence_norm = int(self.convergence_norm)
        except:
            pass
        # private attributes
        try:
            self._sensitivity_only = ast.literal_eval(
                input_dict['sensitivity_only'])
        except:
            self._sensitivity_only = False
        try:
            self._debug_flag = ast.literal_eval(input_dict['debug_flag'])
        except:
            self._debug_flag = False
        try:
            self._debug_dir = input_dict['debug_dir']
        except:
            self._debug_dir = os.curdir + os.sep + 'debug'
        if self._debug_flag:
            utils.create_dir(self._debug_dir)
        try:
            self._save_dir = input_dict['save_dir']
        except:
            self._save_dir = os.curdir + os.sep + 'results_dafi'
        try:
            self._verb = int(input_dict['verbosity'])
        except:
            self._verb = int(1)
        try:
            self._perturb_obs = utils.str2bool(input_dict['perturb_obs'])
        except:
            self._perturb_obs = True
        if 'obs_err_multiplier' in input_dict:
            self.obs_err_factor = float(input_dict['obs_err_multiplier'])
        else:
            self.obs_err_factor = 1.0

        # initialize iteration array
        self.time_array = np.arange(0.0, self.t_end, self.da_interval)
        self.forwad_iteration_array = np.arange(0, self.max_da_iteration)
        # initialize states: these change at every iteration
        self.time = 0.0  # current time
        self.da_step = int(0)  # current DA step
        self.forward_step = int(0)  # current forward step
        # ensemble matrix (nsamples, nstate)
        self.state_vec_prior = np.zeros(
            [self.dyn_model.nstate, self.nsamples])
        self.state_vec_forecast = np.zeros(
            [self.dyn_model.nstate, self.nsamples])
        self.state_vec_analysis = np.zeros(
            [self.dyn_model.nstate, self.nsamples])
        # ensemble matrix projected to observation space
        # (nsamples, nstateSample)
        self.model_obs = np.zeros([self.dyn_model.nstate_obs, self.nsamples])
        # observation matrix (nstate_obs, nsamples)
        self.obs = np.zeros([self.dyn_model.nstate_obs, self.nsamples])
        # observation perturbation matrix (nstate_obs, nsamples)
        self.obs_perturbation = np.zeros(
            [self.dyn_model.nstate_obs, self.nsamples])
        # observation covariance matrix (nstate_obs, nstate_obs)
        self.obs_error = np.zeros(
            [self.dyn_model.nstate_obs, self.dyn_model.nstate_obs])

        # initialize misfit: these grow each iteration, but all values stored.
        self._misfit_norm = []
        self._sigma_hx_norm = []
        self._sigma_obs_norm = []
        self._misfit_norm_store = []
        self._sigma_hx_norm_store = []
        self._sigma_obs_norm_store = []

        # for storage and saving: these grow each iteration
        self.state_vec_analysis_all = []
        self.state_vec_forecast_all = []
        self.obs_all = []
        self.model_obs_all = []
        self.obs_error_all = []

    def __str__(self):
        str_info = self.name + \
            '\n   Number of samples: {}'.format(self.nsamples) + \
            '\n   Run time:          {}'.format(self.t_end) +  \
            '\n   DA interval:       {}'.format(self.da_interval) + \
            '\n   Forward model:     {}'.format(self.dyn_model.name)
        return str_info

    def solve(self):
        """ Solve the parameter estimation problem.

        This is the main method for the filter. It solves the inverse
        problem. It has options for sensitivity analysis, early
        stopping, and output debugging information. At each iteration
        it calculates misfits using various norms, and checks for
        convergence.

        **Updates:**
            * self.time
            * self.iter
            * self.da_step
            * self.state_vec_analysis
            * self.state_vec_forecast
            * self.model_obs
            * self.obs
            * self.obs_error
            * self.init_state
            * self.state_vec_prior
            * more through methods called.
        """
        # Generate initial state Ensemble
        self.state_vec_analysis, self.model_obs = \
            self.dyn_model.generate_ensemble()
        self.init_state = self.state_vec_analysis.copy()

        # sensitivity only
        if self._sensitivity_only:
            self.model_obs = self.dyn_model.state_to_observation(
                self.state_vec_analysis)
            if self.ver >= 1:
                print("\nSensitivity study completed.")
            sys.exit(0)
        # main DA loop - through time
        early_stop = False
        for time in self.time_array:
            self.time = time + self.da_interval
            self.da_step += 1
            if self._verb >= 1:
                print("\nData-assimilation step: {}".format(self.da_step) +
                      "\nTime: {}".format(self.time))
            # dyn_model: propagate the state ensemble to next DA time.
            self.state_vec_forecast = \
                self.dyn_model.forecast_to_time(
                    self.state_vec_analysis, self.time)
            self.state_vec_prior = self.state_vec_forecast
            # DA iterations at fixed time-step.
            self._misfit_norm = []
            self._sigma_hx_norm = []
            self._sigma_obs_norm = []
            for forward_step in self.forwad_iteration_array:
                self.forward_step = forward_step + 1
                if self._verb >= 1:
                    print("\n  Forward step: {}".format(self.forward_step))
                # forward the state vector to observation space
                if self.forward_step != 1:
                    self.state_vec_forecast = self.state_vec_analysis.copy()
                self.model_obs = self.dyn_model.state_to_observation(
                    self.state_vec_forecast)
                # get observation data at current step
                obs_vec, self.obs_error = self.dyn_model.get_obs(self.time)
                self.obs_error *= self.obs_err_factor
                if self._perturb_obs:
                    self.obs = self._perturb_vec(
                        obs_vec, self.obs_error, self.nsamples)
                else:
                    self.obs = self._vec_to_mat(obs_vec, self.nsamples)
                # data assimilation
                self._correct_forecasts()
                self._calc_misfits()
                # iteration: store results, report, check convergence
                if self._stationary_flag:
                    self._store_vars()
                    if self._verb >= 2:
                        print(self._report())
                conv_var, conv_res = self._check_convergence()
                if self.convergence_option is 'variance':
                    conv = conv_var
                else:
                    conv = conv_res
                if conv and not self.reach_max_flag:
                    early_stop = True
                    self._store_misfits()
                    break
                if self.forward_step == self.max_da_iteration:
                    self._store_misfits()
            # time-step: store results, report
            if not self._stationary_flag:
                self._store_vars()
                if self._verb >= 2:
                    print(self._report())
            if self._verb >= 1:
                if early_stop:
                    print("\n  DA Filtering completed: convergence early stop.")
                else:
                    print("\n  DA Filtering completed: max iteration reached.")

    def plot(self):
        """ Plot iteration convergence. """
        utils.create_dir(self._save_dir)
        if self._stationary_flag:
            iter = np.arange(self.forward_step) + 1
        else:
            iter = np.arange(self.da_step) + 1
        fig, ax = plt.subplots()
        ax.plot(iter, self._misfit_norm_store, '.-', label='norm(obs-HX)')
        ax.plot(iter, self._sigma_hx_norm_store, '.-', label='norm(std(HX))')
        ax.plot(iter, self._sigma_obs_norm_store, '.-', label='norm(std(obs))')
        ax.set_xlabel('Data-Assimilation Step')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.savefig(self._save_dir + os.sep + 'iteration_errors_plot.pdf')
        plt.close(fig)

    def clean(self):
        """ Cleanup before exiting. """
        self.dyn_model.clean()

    def save(self):
        """ Saves results to text files. """
        utils.create_dir(self._save_dir)
        sdir = self._save_dir + os.sep
        xadir = sdir + 'Xa'
        utils.create_dir(xadir)
        xfdir = sdir + 'Xf'
        utils.create_dir(xfdir)
        hxfdir = sdir + 'HXf'
        utils.create_dir(hxfdir)
        odir = sdir + 'obs'
        utils.create_dir(odir)
        oedir = sdir + 'R'
        utils.create_dir(oedir)
        np.savetxt(sdir + 'X_0_mean', self.dyn_model.init_state)
        np.savetxt(sdir + 'X_0', self.init_state)
        np.savetxt(sdir + 'misfit_norm', np.array(self._misfit_norm_store))
        np.savetxt(sdir + 'sigma_HX', np.array(self._sigma_hx_norm_store))
        np.savetxt(sdir + 'sigma_obs', np.array(self._sigma_obs_norm_store))
        for da_step, value in enumerate(self.state_vec_analysis_all):
            np.savetxt(xadir + os.sep + 'Xa_{}'.format(da_step+1), value)
        for da_step, value in enumerate(self.state_vec_forecast_all):
            np.savetxt(xfdir + os.sep + 'Xf_{}'.format(da_step+1), value)
        for da_step, value in enumerate(self.model_obs_all):
            np.savetxt(hxfdir + os.sep + 'HXf_{}'.format(da_step+1), value)
        for da_step, value in enumerate(self.obs_all):
            np.savetxt(odir + os.sep + 'obs_{}'.format(da_step+1), value)
        for da_step, value in enumerate(self.obs_error_all):
            np.savetxt(oedir + os.sep + 'R_{}'.format(da_step+1), value)

    # private methods
    def _correct_forecasts(self):
        """ Correct the propagated ensemble (filtering step).

        **Updates:**
            * self.state_vec_analysis
        """
        raise NotImplementedError(
            "Needs to be implemented in the child class!")

    def _store_misfits(self):
        """ Store the values of misfits at end of DA loop.

        **Updates:**
            * self._misfit_norm_store
            * self._sigma_hx_norm_store
            * self._sigma_obs_norm_store
        """
        self._misfit_norm_store.append(self._misfit_norm[-1])
        self._sigma_hx_norm_store.append(self._sigma_hx_norm[-1])
        self._sigma_obs_norm_store.append(self._sigma_obs_norm[-1])

    def _vec_to_mat(self, vec, ncol):
        "Tile a vector ncol times to form a matrix"
        return np.tile(vec, (ncol, 1)).T

    def _perturb_vec(self, mean, cov, nsamps):
        """ Create samples of random vector.

        Parameters
        ----------
        mean : ndarray
            Mean vector.
            ``dtype=float``, ``ndim=1``, ``shape=(ndim,)``
        cov : ndarray
            Covariance matrix.
            ``dtype=float``, ``ndim=2``, ``shape=(ndim, ndim)``
        snamps : int
            Number of samples to create.

        Returns
        -------
        samp : ndarray
            Array of sampled vectors.
            ``dtype=float``, ``ndim=2``, ``shape=(ndim, nsamps)``
        """
        # check symmetric
        if not np.allclose(cov, cov.T):
            raise ValueError('Covariance matrix is not symmetric.')
        # Cholesky decomposition
        l_mat = np.linalg.cholesky(cov)
        # create correlated perturbations
        ndim = len(mean)
        x_mat = np.random.normal(loc=0.0, scale=1.0, size=(ndim, nsamps))
        perturb = np.matmul(l_mat, x_mat)
        return np.tile(mean, (nsamps, 1)).T + perturb

    def _store_vars(self):
        """ Store the important variables at each iteration.

        **Updates:**
            * self.obs_all
            * self.model_obs_all
            * self.obs_error_all
            * self.state_vec_analysis_all
            * self.state_vec_forecast_all
        """
        # save results
        self.obs_all.append(self.obs.copy())
        self.model_obs_all.append(self.model_obs.copy())
        self.obs_error_all.append(self.obs_error.copy())
        self.state_vec_analysis_all.append(self.state_vec_analysis.copy())
        self.state_vec_forecast_all.append(self.state_vec_forecast.copy())

    def _save_debug(self, debug_dict):
        """ Save specified ndarrays to the debug directory. """
        if self._debug_flag:
            for key, value in debug_dict.items():
                fname = self._debug_dir + os.sep + key + \
                    '_{}'.format(self.da_step)
                np.savetxt(fname, value)

    def _calc_misfits(self):
        """ Calculate the misfit.

        **Updates:**
            * self._misfit_norm
            * self._sigma_hx_norm
            * self._sigma_obs_norm
        """
        if self.convergence_norm == np.inf:
            nnorm_vec = 1.0
            nnorm_mat = 1.0
        else:
            nnorm_vec = self.nstate_obs
            nnorm_mat = self.nstate_obs * self.nsamples
        diff = abs(self.obs - self.model_obs)
        misfit_norm = la.norm(diff, self.convergence_norm) / nnorm_mat
        sigma_hx = np.std(self.model_obs, axis=1)
        sigma_hx_norm = la.norm(sigma_hx, self.convergence_norm) / nnorm_vec
        sigma_obs = np.sqrt(np.diag(self.obs_error))
        sigma_obs_norm = la.norm(sigma_obs, self.convergence_norm) / nnorm_vec
        self._misfit_norm.append(misfit_norm)
        self._sigma_hx_norm.append(sigma_hx_norm)
        self._sigma_obs_norm.append(sigma_obs_norm)

    def _iteration_residual(self, list, iter):
        """ Calculate the residual at a given iteration. """
        if iter > 0:
            iterative_residual = abs(list[iter] - list[iter-1]) / abs(list[0])
        else:
            iterative_residual = None
        return iterative_residual

    def _check_convergence(self):
        """ Check iteration convergence. """
        conv_variance = self._misfit_norm[self.forward_step - 1] < \
            self._sigma_obs_norm[self.forward_step - 1]
        residual = self._iteration_residual(
            self._misfit_norm, self.forward_step-1)
        if self.convergence_residual is None:
            conv_residual = False
        else:
            conv_residual = residual < self.convergence_residual
        return conv_variance, conv_residual

    def _report(self):
        """ Create report for current iteration. """
        residual = self._iteration_residual(
            self._misfit_norm, self.forward_step-1)
        conv_var, conv_res = self._check_convergence()
        str_report = ''
        str_report += "    Norm of standard deviation of HX: {}".format(
            self._sigma_hx_norm[self.forward_step - 1]) + \
            "\n    Norm of standard deviation of observation: {}".format(
            self._sigma_obs_norm[self.forward_step - 1])
        str_report += "\n  Norm of misfit: {}".format(
            self._misfit_norm[self.forward_step - 1])
        str_report += "\n    Convergence (variance): {}".format(conv_var)
        str_report += "\n    Convergence (residual): {}".format(conv_res) \
            + "\n      Relative iterative residual: {}".format(residual) \
            + "\n      Relative convergence criterion: {}".format(
            self.convergence_residual)
        return str_report


# child classes (specific filtering techniques)
class EnKF(DAFilter2):
    """ Implementation of the ensemble Kalman Filter (EnKF).

    It inherits most methods from parent class (``DAFIlter2``), but
    replaces the ``correct_forecasts`` method to use EnKF for the
    data-assimilation step.

    The EnKF is updated by: ``Xa = Xf + K*(obs - HX)`` where *Xf* is
    the forecasted state vector (by the forward model), *Xa* is the
    updated vector after data-assimilation, *K* is the Kalman gain
    matrix, *obs* is the observation vector, and *HX* is the forecasted
    state vector in observation space. See the documentation for more
    information.
    """

    def __init__(self, nsamples, da_interval, t_end, max_da_iteration,
                 dyn_model, input_dict):
        """ Parse input file and assign values to class attributes.

        Note
        ----
        See DAFilter2.__init__ for details.
        """
        super(self.__class__, self).__init__(
            nsamples, da_interval, t_end, max_da_iteration, dyn_model,
            input_dict)
        self.name = 'Ensemble Kalman Filter'
        self.short_name = 'EnKF'

    def _correct_forecasts(self):
        """ Correct the propagated ensemble (filtering step) using EnKF

        **Updates:**
            * self.state_vec_analysis
        """

        # TODO: Consider moving these two functions to DAFilter2. Remove from all the child classes. Add one-line docstring.
        def _check_condition_number(hpht):
            con_inv = la.cond(hpht + self.obs_error)
            if self._verb >= 2:
                print("  Condition number of (HPHT + R) is {}".format(
                    con_inv))
            if (con_inv > 1e16):
                message = "The matrix (HPHT + R) is singular, inverse will" + \
                    "fail."
                warnings.warn(message, RuntimeWarning)

        def _mean_subtracted_matrix(mat, samps=self.nsamples):
            mean_vec = np.array([np.mean(mat, axis=1)])
            mean_vec = mean_vec.T
            mean_vec = np.tile(mean_vec, (1, samps))
            mean_sub_mat = mat - mean_vec
            return mean_sub_mat

        # calculate the Kalman gain matrix
        xp = _mean_subtracted_matrix(self.state_vec_forecast)
        hxp = _mean_subtracted_matrix(self.model_obs)
        coeff = (1.0 / (self.nsamples - 1.0))
        pht = coeff * np.dot(xp, hxp.T)
        hpht = coeff * hxp.dot(hxp.T)
        _check_condition_number(hpht)
        inv = la.inv(hpht + self.obs_error)
        kalman_gain_matrix = pht.dot(inv)
        # analysis step
        dx = np.dot(kalman_gain_matrix, self.obs - self.model_obs)
        self.state_vec_analysis = self.state_vec_forecast + dx
        # debug
        debug_dict = {
            'K': kalman_gain_matrix, 'inv': inv, 'HPHT': hpht, 'PHT': pht,
            'HXP': hxp, 'XP': xp}
        self._save_debug(debug_dict)


class EnRML(DAFilter2):
    """ Implementation of the ensemble Randomized Maximal Likelihood
    (EnRML).

    It inherits most methods from parent class (``DAFIlter2``), but
    replaces the ``correct_forecasts`` method to use EnRML for the
    data-assimilation step.

    The EnRML is updated by: ``Xa = Xf + GN*(obs - HX)+P``
    where *Xf* is the forecasted state vector (by the forward model),
    *Xa* is the updated vector after data-assimilation, *GN* is the
    Gauss-Newton matrix, *obs* is the observation vector, and *HX* is
    the forecasted state vector in observation space, *P* is Penalty
    matrix. See the documentation for more information.
    """

    def __init__(self, nsamples, da_interval, t_end, max_da_iteration,
                 dyn_model, input_dict):
        """ Parse input file and assign values to class attributes.

        Note
        ----
        See DAFilter2.__init__ for details.
        """
        # TODO: add the additional inputs to the docstring as in DAFilter2.__init__
        super(self.__class__, self).__init__(
            nsamples, da_interval, t_end, max_da_iteration, dyn_model,
            input_dict)
        self.name = 'Ensemble Randomized Maximal Likelihood'
        self.short_name = 'EnRML'
        self.beta = float(input_dict['beta'])
        self.criteria = ast.literal_eval(
            input_dict['criteria'])

    def _correct_forecasts(self):
        """ Correct the propagated ensemble (filtering step) using EnRML

        **Updates:**
            * self.state_vec_analysis
        """

        def _check_condition_number(hpht):
            con_inv = la.cond(hpht + self.obs_error)
            if self._verb >= 2:
                print("  Condition number of (HPHT + R) is {}".format(
                    con_inv))
            if (con_inv > 1e16):
                message = "The matrix (HPHT + R) is singular, inverse will" + \
                    "fail."
                warnings.warn(message, RuntimeWarning)

        def _mean_subtracted_matrix(mat, samps=self.nsamples):
            mean_vec = np.array([np.mean(mat, axis=1)])
            mean_vec = mean_vec.T
            mean_vec = np.tile(mean_vec, (1, samps))
            mean_sub_mat = mat - mean_vec
            return mean_sub_mat

        # calculate the Gauss-Newton matrix
        xp0 = _mean_subtracted_matrix(self.state_vec_prior)
        p0 = (1.0 / (self.nsamples - 1.0)) * xp0.dot(xp0.T)
        x = self.state_vec_forecast.copy()
        # TODO: I'm pretty sure this copy is unnecesary. Check.
        xp = _mean_subtracted_matrix(x)
        hxp = _mean_subtracted_matrix(self.model_obs)
        gen = np.dot(hxp, la.pinv(xp))
        sen_mat = p0.dot(gen.T)

        cyyi = np.dot(np.dot(gen, p0), gen.T)
        _check_condition_number(cyyi)
        inv = la.inv(cyyi + self.obs_error)
        inv = inv.A  # convert np.matrix to np.ndarray
        gauss_newton_matrix = sen_mat.dot(inv)

        # calculate the penalty
        penalty = np.dot(gauss_newton_matrix, gen.dot(
            x-self.state_vec_prior))

        # analysis step
        dx = np.dot(gauss_newton_matrix, self.obs - self.model_obs) + penalty
        x = self.beta * self.state_vec_prior + (
            1.0 - self.beta) * x + self.beta*dx

        self.state_vec_analysis = x.copy()
        # TODO: I'm pretty sure this copy is unnecesary. Check.
        # debug
        debug_dict = {
            'GN': gauss_newton_matrix, 'pen': penalty, 'inv': inv,
            'cyyi': cyyi, 'HXP': hxp, 'XP': xp}
        self._save_debug(debug_dict)


class EnKF_MDA(DAFilter2):
    """ Implementation of the ensemble Kalman Filter-Multi data
    assimilaton (EnKF-MDA).

    It inherits most methods from parent class (``DAFIlter2``), but
    replaces the ``correct_forecasts`` method to use EnKF-MDA for the
    data-assimilation step.

    The EnKF-MDA is updated by:
    ``Xa = Xf + K_mda*(obs - HX - err_mda)`` where *Xf* is the
    forecasted state vector (by the dynamic model),
    *Xa* is the updated vector after data-assimilation, *K_mda* is the
    modified Kalman gain matrix, *obs* is the observation vector, and
    *HX* is the forwarded state vector in observation space, 'err_mda'
    is inflated error. See the documentation for more information.
    """

    def __init__(self, nsamples, da_interval, t_end, max_da_iteration,
                 dyn_model, input_dict):
        """ Parse input file and assign values to class attributes.

        Note
        ----
        See DAFilter2.__init__ for details.
        """
        super(self.__class__, self).__init__(
            nsamples, da_interval, t_end, max_da_iteration, dyn_model,
            input_dict)
        self.name = 'Ensemble Kalman Filter-Multi Data Assimilation'
        self.short_name = 'EnKF-MDA'

    def _correct_forecasts(self):
        """ Correct the propagated ensemble (filtering step) using
        EnKF_MDA.

        **Updates:**
            * self.state_vec_analysis
        """

        def _check_condition_number(hpht):
            con_inv = la.cond(hpht + self.obs_error)
            if self._verb >= 2:
                print("  Condition number of (HPHT + R) is {}".format(
                    con_inv))
            if (con_inv > 1e16):
                message = "The matrix (HPHT + R) is singular, inverse will" + \
                    "fail."
                warnings.warn(message, RuntimeWarning)

        def _mean_subtracted_matrix(mat, samps=self.nsamples):
            mean_vec = np.array([np.mean(mat, axis=1)])
            mean_vec = mean_vec.T
            mean_vec = np.tile(mean_vec, (1, samps))
            mean_sub_mat = mat - mean_vec
            return mean_sub_mat

        alpha = self.max_da_iteration
        # calculate the Kalman gain matrix
        xp = _mean_subtracted_matrix(self.state_vec_forecast)
        hxp = _mean_subtracted_matrix(self.model_obs)
        coeff = (1.0 / (self.nsamples - 1.0))
        pht = coeff * np.dot(xp, hxp.T)
        hpht = coeff * hxp.dot(hxp.T)
        _check_condition_number(hpht)
        inv = la.inv(hpht + alpha * self.obs_error)
        inv = inv.A  # convert np.matrix to np.ndarray
        kalman_gain_matrix = pht.dot(inv)
        # analysis step
        d = self.obs - self.obs_perturb
        dx = np.dot(kalman_gain_matrix, d - self.model_obs +
                    np.sqrt(alpha) * self.obs_perturb)
        self.state_vec_analysis = self.state_vec_forecast + dx
        # debug
        debug_dict = {
            'K': kalman_gain_matrix, 'inv': inv, 'HPHT': hpht, 'PHT': pht,
            'HXP': hxp, 'XP': xp}
        self._save_debug(debug_dict)


class REnKF(DAFilter2):
    """ Implementation of the regularized ensemble Kalman Filter (REnKF).

    It inherits most methods from parent class (``DAFIlter2``), but
    replaces the ``correct_forecasts`` method to use penalized EnKF for
    the data-assimilation step.

    For detail on the implementation see
    [cite paper once it is published].

    Note
    ----
    Additional inputs in ``input_dict``:
        * **penalties_python_file** (``string``) -
          Path to python file that contains function called
          ``penalties`` that returns a list of dictionaries. Each
          dictionary represents one penalty and includes:
          ``lambda`` (float), ``weight_matrix`` (ndarray),
          ``penalty`` (function), and ``gradient`` (function).
    """

    def __init__(self, nsamples, da_interval, t_end, max_da_iteration,
                 dyn_model, input_dict):
        """ Parse input file and assign values to class attributes.

        Note
        ----
        See DAFilter2.__init__ for details.
        """
        super(self.__class__, self).__init__(
            nsamples, da_interval, t_end, max_da_iteration, dyn_model,
            input_dict)
        self.name = 'Regularized Ensemble Kalman Filter'
        self.short_name = 'REnKF'
        # load penalties
        pfile = input_dict['penalties_python_file']
        sys.path.append(os.path.dirname(pfile))
        penalties = getattr(importlib.import_module(
            os.path.splitext(os.path.basename(pfile))[0]), 'penalties')
        self.penalties = penalties(self)
        self.cost1_all = []
        self.cost2_all = []
        self.cost3_all = []
        self.lamda_all = []
        self.dx1_all = []
        self.dx2_all = []
        self.k2_all = []
        self.penalty_all = []

    def _correct_forecasts(self):
        """ Correct the propagated ensemble (filtering step) using EnKF

        **Updates:**
            * self.state_vec_analysis
        """

        # TODO: Consider moving these two functions to DAFilter2. Remove from all the child classes. Add one-line docstring.
        def _check_condition_number(hpht):
            con_inv = la.cond(hpht + self.obs_error)
            if self._verb >= 2:
                print("  Condition number of (HPHT + R) is {}".format(
                    con_inv))
            if (con_inv > 1e16):
                message = "The matrix (HPHT + R) is singular, inverse will" + \
                    "fail."
                warnings.warn(message, RuntimeWarning)

        def _mean_subtracted_matrix(mat, samps=self.nsamples):
            mean_vec = np.array([np.mean(mat, axis=1)])
            mean_vec = mean_vec.T
            mean_vec = np.tile(mean_vec, (1, samps))
            mean_sub_mat = mat - mean_vec
            return mean_sub_mat

        # calculate the Kalman gain matrix
        xp = _mean_subtracted_matrix(self.state_vec_forecast)
        hxp = _mean_subtracted_matrix(self.model_obs)
        coeff = (1.0 / (self.nsamples - 1.0))
        pht = coeff * np.dot(xp, hxp.T)
        hpht = coeff * hxp.dot(hxp.T)
        _check_condition_number(hpht)
        inv = la.inv(hpht + self.obs_error)
        kalman_gain_matrix = pht.dot(inv)
        # calculate the "K2" matrix
        hxx = np.dot(hxp, xp.T)
        k2_gain_matrix = \
            coeff* np.dot(kalman_gain_matrix, hxx) - coeff*np.dot(xp, xp.T)
        # calculate penalty matrix
        penalty_mat = np.zeros([len(self.state_vec_forecast), self.nsamples])
        for ipenalty in self.penalties:
            w_mat = ipenalty['weight_matrix']
            lamb = ipenalty['lambda']
            lamda = lamb(self.forward_step)
            if self.forward_step > 1 and self.cost3_all[-1] > 0.1*self.cost2_all[-1]:
                lamda=self.lamda_all[-1]

            func_penalty = ipenalty['penalty']
            func_gradient = ipenalty['gradient']

            for isamp in range(self.nsamples):
                istate = self.state_vec_forecast[:,isamp]
                gpw = np.dot(func_gradient(istate).T, w_mat)
                gpwg = np.dot(gpw, func_penalty(istate))
                penalty_mat[:, isamp] += lamda * gpwg
        penalty = func_penalty(istate)
        # analysis step
        dx1 = np.dot(kalman_gain_matrix, self.obs - self.model_obs)
        dx2 = np.dot(k2_gain_matrix, penalty_mat)
        self.state_vec_analysis = self.state_vec_forecast + dx1 + dx2
        dx = dx1 + dx2
        p = coeff*np.dot(xp, xp.T)
        cost1 = 0.5 * np.dot(dx.T.dot(la.inv(p)), dx)
        delta_y = self.obs - self.model_obs
        cost2 = 0.5 * np.dot(delta_y.T.dot(la.inv(self.obs_error)), delta_y)
        cost3 = 0.5 * lamda * np.dot(func_penalty(istate).T.dot(w_mat), func_penalty(istate))
        self.cost1_all.append(np.linalg.norm(cost1))
        self.cost2_all.append(np.linalg.norm(cost2))
        self.cost3_all.append(np.linalg.norm(cost3))
        self.lamda_all.append(lamda)
        self.dx1_all.append(np.linalg.norm(dx1))
        self.dx2_all.append(np.linalg.norm(dx2))
        self.k2_all.append(np.linalg.norm(k2_gain_matrix))
        self.penalty_all.append(np.linalg.norm(penalty))
        # debug
        debug_dict = {
            'K': kalman_gain_matrix, 'inv': inv, 'HPHT': hpht, 'PHT': pht,
            'HXP': hxp, 'XP': xp, 'cost1': self.cost1_all, 'cost2': self.cost2_all, 
            'cost3': self.cost3_all, 'lamda': self.lamda_all, 'dx1': self.dx1_all,
            'dx2': self.dx2_all, 'k2': self.k2_all, 'penalty': self.penalty_all}
        self._save_debug(debug_dict)
