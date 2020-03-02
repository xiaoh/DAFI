# Copyright 2020 Virginia Polytechnic Institute and State University.
""" Collection of different ensemble-based Bayesian inversion
techniques.
"""

# standard library imports
import warnings
import os
import sys
import importlib

# third party imports
import numpy as np
from numpy import linalg as la


# parent classes (templates)
class InverseMethodBase(object):
    """ Parent class for ensemble-based Bayesian inversion techniques.

    Use this as a template to write new inversion classes.
    The required methods are summarized below.
    """

    def __init__(self, nsamples, t_interval, t_end, max_iterations,
                 model, input_dict):
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
        model : dafi.PhysicsModel
            Physics model.
        input_dict : dict[str]
            Inverse method specific inputs.
        """
        self.name = 'Generic inverse method'
        self.model = model
        self.nsamples = nsamples

    def __str__(self):
        str_info = self.name + \
            '\n   Number of samples:    {}'.format(self.nsamples) + \
            '\n   Forward model:        {}'.format(self.model.name)
        return str_info

    def solve(self):
        """ Implement the inverse method. """
        pass

    def clean(self):
        """ Perform any necessary cleanup at completion. """
        try:
            self.model.clean()
        except AttributeError:
            pass

    def save(self):
        """ Save any important results to files. """
        pass


class InverseMethod(InverseMethodBase):
    """ Parent class for ensemble-based Bayesian inversion techniques.

    This class includes more implemented methods than the InverseMethod
    class. The Inverse class is a barebones template. InverseMethodDAFI
    contains several methods  as well as a framework for the main
    self.solve() method. This can be used to quickly create new filter
    classes if some of the same methods are desired. This correspond
    to the "general method" in the paper [1]. Child classes implement
    specific filtering techniques.

    [1]
    """

    def __init__(self, nsamples, t_interval, t_end, max_iterations,
                 model, input_dict,):
        """ Parse input file and assign values to class attributes.

        Parameters
        ----------
        nsamples : int
            Ensemble size.
        t_interval : float
            Time interval between data assimilation steps
        t_end : float
            Final time.
        max_iterations : int
            Maximum number of iterations at a given time-step.
        model : dafi.PhysicsModel
            Physics model.
        input_dict : dict
            Inverse method specific inputs.

        Note
        ----
        Inputs in ``input_dict``:
            * **save_dir** (``string``, ``./results_dafi``) -
              Folder where to save results.
            * **save_iterations** (``bool``, ``False``) -
              Save results at every iteration.
            * **debug** (``bool``, ``False``) -
              Save extra information for debugging.
            * **debug_dir** (``string``, ``./debug``) -
              Folder where to save debugging information
            * **verbosity** (``int``, ``1``) -
              Amount of information to print to screen.
            * **convergence_option** (``str``, ``max``) -
              Convergence criteria to use: ``variance`` to use the
              variance convergence criteria,  ``residual`` to use the
              residual convergence criteria. ``max`` to reach the max
              iterations. See documentation for more information.
            * **convergence_residual** (``float``, None) -
              Residual value for convergence if ``reach_max`` is
              ``False`` and ``convergence_option`` is ``residual``.
            * **perturb_obs_option** (``str``, ``iter``) -
              Option on when to perturb observations:
              ``iter`` to perturb at each iteration (inner loop),
              ``time`` to perturb only once each data assimilaton time
              (outer loop),  ``never`` to not perturb observations
            * **obs_err_multiplier** (``float``, ``1.0``) -
              Factor by which to multiply the observation error matrix.
              This is done by some authors in the literature.
            * **analysis_to_obs** (``bool``, ``False``) -
              Map analysis state to observation space
        """
        self.name = 'Generic inverse method'
        self.model = model
        self.nsamples = nsamples
        self.nstate = self.model.nstate
        self.nobs = self.model.nobs
        self.t_interval = t_interval
        self.t_end = t_end
        self.max_iterations = max_iterations

        # InverseMethodDAFI-specific inputs
        self.convergence_option = input_dict.get('convergence_option',
                                                 'max')
        if self.convergence_option not in ['discrepancy', 'residual', 'max']:
            raise NotImplementedError('Invalid convergence_option.')
        if self.convergence_option == 'residual':
            self.convergence_residual = input_dict['convergence_residual']
        else:
            self.convergence_residual = None
        if self.convergence_option == 'discrepancy':
            self.convergence_factor = input_dict['convergence_factor']
        else:
            self.convergence_factor = None
        self.obs_err_factor = input_dict.get('obs_err_multiplier', 1.)
        # private attributes
        self._debug = input_dict.get('debug', False)
        default_debug_dir = os.curdir + os.sep + 'debug'
        self._debug_dir = input_dict.get('debug_dir', default_debug_dir)
        if self._debug:
            _create_dir(self._debug_dir)
        default_save_dir = os.curdir + os.sep + 'results_dafi'
        self._save_dir = input_dict.get('save_dir', default_save_dir)
        self._verb = input_dict.get('verbosity', 1)
        self._perturb_obs = input_dict.get('perturb_obs_option', 'iter')
        self._save_iterations = input_dict.get('save_iterations', False)
        self._analysis_to_obs = input_dict.get('analysis_to_obs', False)

        # initialize iteration array
        stationary = t_end == t_interval
        if stationary:
            self.time_array = [0.]
        else:
            self.time_array = np.arange(0.0, self.t_end+self.t_interval, self.t_interval)
        self.iteration_array = np.arange(0, self.max_iterations)

        # initialize states: these change at every iteration
        self.time = 0.0  # current time
        self.i_time = int(0)  # current time step
        self.i_iteration = int(0)  # current iteration step

        # ensemble matrix (nsamples, nstate)
        self.state_prior = np.zeros(
            [self.model.nstate, self.nsamples])
        self.state_forecast = np.zeros(
            [self.model.nstate, self.nsamples])
        self.state_analysis = np.zeros(
            [self.model.nstate, self.nsamples])
        # ensemble matrix mapped to observation space (nsamples, nstateSample)
        self.state_in_obsspace = np.zeros([self.model.nobs, self.nsamples])
        # observation matrix (nobs, nsamples)
        self.obs = np.zeros([self.model.nobs, self.nsamples])
        # observation perturbation matrix (nobs, nsamples)
        self.obs_perturbation = np.zeros([self.model.nobs, self.nsamples])

        # observation covariance matrix (nobs, nobs)
        self.obs_error = np.zeros([self.model.nobs, self.model.nobs])

        # for storage and saving: these grow each iteration
        self._store_results = {'xa': [], 'xf': [], 'Hx': [], 'y': [], 'R': []}
        if self._analysis_to_obs:
            self._store_results['Hxa'] = []


    def solve(self):
        """ Solve the inverse problem.
        """
        # Generate initial state Ensemble
        self.state_forecast = self.model.generate_ensemble()
        # main loop - through time
        early_stop = False
        for time in self.time_array:
            self.time = time
            self.i_time += 1
            if self._verb >= 1:
                print(f"\nData-assimilation step: {self.i_time}" +
                      f"\nTime: {self.time}")
            # model: propagate the state ensemble to next DA time.
            if self.i_time != 1:
                self.state_forecast = self.model.forecast_to_time(
                                            self.state_analysis, self.time)

            # get observations at current step
            obs_vec, self.obs_error = self.model.get_obs(self.time)
            self.obs_error *= self.obs_err_factor
            self.obs = _vec_to_mat(obs_vec, self.nsamples)
            if self._perturb_obs == 'time':
                self.obs, self.obs_perturbation = _perturb_vec(
                    obs_vec, self.obs_error, self.nsamples)

            # prior state for iterative methods that require it
            self.state_prior = self.state_forecast

            # DA iterations at fixed time-step
            save_iter = self._save_iterations and len(self.iteration_array) > 1
            self._store_iter = {'xa': [], 'xf': [], 'Hx': [], 'y': []}
            std_y = _cov_to_std(self.obs_error)
            self._store_convergence = {
                'misfit': [], 'noise_level': [], 'residual': []}
            for iteration in self.iteration_array:
                self.i_iteration = iteration + 1
                if self._verb >= 1:
                    print("\n  Iteration: {}".format(self.i_iteration))
                # map the state vector to observation space
                if self.i_iteration != 1:
                    self.state_forecast = self.state_analysis.copy()
                self.state_in_obsspace = self.model.state_to_observation(
                    self.state_forecast)
                if self.i_iteration == 1:
                    self.state_in_obsspace_prior = self.state_in_obsspace
                # Perturb observations
                if self._perturb_obs == 'iter':
                    self.obs, self.obs_perturbation = _perturb_vec(
                        obs_vec, self.obs_error, self.nsamples)
                # data assimilation
                self.state_analysis, report_analysis = self.analysis()
                # iteration: store results, report, check convergence
                conv, report_conv = self._check_store_convergence()
                if save_iter:
                    self._store_inner_iter()
                if self._verb >= 2 and len(self.iteration_array) > 1:
                    print(report_analysis)
                    print(report_conv)
                if conv and (self.i_iteration < self.max_iterations):
                    early_stop = True
                    break
            # time-step
            # map xa to obs space
            if self._analysis_to_obs:
                print("\n  Mapping final analysis states " + \
                      "to observation space.")
                self.state_in_obsspace = self.model.state_to_observation(
                    self.state_forecast)
                if save_iter:
                    self._store_iter['Hx'].append(
                        self.state_in_obsspace.copy())
            # store results, report
            self._store()
            if save_iter:
                self._save_inner_iter()
            if self._verb >= 1:
                if early_stop:
                    message = "convergence early stop."
                else:
                    message = "max iteration reached."
                print("\n  DA Filtering completed: " + message)
        return self.state_analysis

    def save(self,):
        """ Saves results to text files. """
        var_list = ['xa', 'xf', 'Hx', 'y', 'R']
        if self._analysis_to_obs:
            var_list.append('Hxa')
        _create_dir(self._save_dir)
        np.savetxt(self._save_dir + os.sep + 'time',
            self.time_array)
        for var in var_list:
            dir = self._save_dir + os.sep + var
            _create_dir(dir)
            for iter, value in enumerate(self._store_results[var]):
                np.savetxt(dir + os.sep + var + f'_{iter+1}', value)

    def analysis(self,):
        """ Correct the forecasted ensemble states to analysis state.

        This is the data assimilation step.

        Returns
        -------
        state_analysis : ndarray
            Ensemble matrix of updated states (xa).
            ``dtype=float``, ``ndim=2``, ``shape=(nstate, nsamples)``
        """
        error_message = "Needs to be implemented in the child class."
        raise NotImplementedError(error_message)

    # private methods used to simplify main code
    def _store_inner_iter(self,):
        """ Store iteration results for current time step. """
        self._store_iter['y'].append(self.obs.copy())
        self._store_iter['Hx'].append(self.state_in_obsspace.copy())
        self._store_iter['xa'].append(self.state_analysis.copy())
        self._store_iter['xf'].append(self.state_forecast.copy())

    def _save_inner_iter(self,):
        """ Save iteration results for current time step. """
        tdir = self._save_dir + os.sep + f't_{self.i_time}'
        _create_dir(tdir)
        var_list = ['xa', 'xf', 'Hx', 'y']
        _create_dir(self._save_dir)
        for var in var_list:
            dir = tdir + os.sep + var
            _create_dir(dir)
            for iter, value in enumerate(self._store_iter[var]):
                np.savetxt(dir + os.sep + var + f'_{iter+1}', value)
        # save misfit
        var_list = ['misfit', 'noise_level', 'residual']
        for var in var_list:
            np.savetxt(tdir + os.sep + var,
                np.atleast_1d(np.array(self._store_convergence[var])))

    def _store(self,):
        """ Store results at each time. """
        # save results
        self._store_results['y'].append(self.obs.copy())
        self._store_results['Hx'].append(self.state_in_obsspace_prior.copy())
        if self._analysis_to_obs:
            self._store_results['Hxa'].append(self.state_in_obsspace.copy())
        self._store_results['R'].append(self.obs_error.copy())
        self._store_results['xa'].append(self.state_analysis.copy())
        self._store_results['xf'].append(self.state_prior.copy())

    def _save_debug(self, debug_dict, post_name=None):
        """ Save specified ndarrays to the debug directory. """
        for key, value in debug_dict.items():
            fname = self._debug_dir + os.sep + key + \
                f'_{self.i_time}_{self.i_iteration}'
            if post_name is not None:
                fname += post_name
            np.savetxt(fname, value)

    def _check_store_convergence(self,):
        """ Calculate and store misfits, and return iteration convergence. """
        compute_all = self._save_iterations or (self._verb >= 2)
        comput2_all = compute_all and (len(self.iteration_array) > 1)
        if self.convergence_option != 'max' or compute_all:
            diff = self.obs - self.state_in_obsspace
            misfit_norm = la.norm(np.mean(diff, axis=1))
            self._store_convergence['misfit'].append(misfit_norm)
        if self.convergence_option == 'discrepancy' or compute_all:
            noise_level = np.sqrt(np.trace(self.obs_error))
            if self.convergence_factor is None:
                conv_variance = False
            else:
                noise_criteria = self.convergence_factor * noise_level
                conv_variance = misfit_norm < noise_criteria
            self._store_convergence['noise_level'].append(noise_level)
        if self.convergence_option == 'residual' or compute_all:
            residual = _iteration_residual(
                self._store_convergence['misfit'], self.i_iteration-1)
            if self.convergence_residual is None:
                conv_residual = False
            else:
                conv_residual = residual < self.convergence_residual
            self._store_convergence['residual'].append(residual)
        # return iteration convergence (bool)
        if self.convergence_option == 'discrepancy':
            conv = conv_variance
        elif self.convergence_option == 'residual':
            conv = conv_residual
        elif self.convergence_option == 'max':
            conv = False
        # report
        if compute_all:
            report = f"\n    Convergence (variance): {conv_variance}"
            report += f"\n      Norm of misfit: {misfit_norm}"
            report += f"\n      Noise level: {noise_level}"
            report += f"\n    Convergence (residual): {conv_residual}"
            report += f"\n      Relative iterative residual: {residual}"
            report += "\n       Relative convergence criterion: " + \
                f"{self.convergence_residual}"
        else:
            report = ""
        return conv, report


# child classes (specific filtering techniques)
class EnKF(InverseMethod):
    """ Implementation of the ensemble Kalman Filter (EnKF).

    It inherits most methods from parent class (``InverseMethod``),
    but replaces the ``analysis`` method to use EnKF for the
    data-assimilation step.

    The EnKF is updated by: ``xa = xf + K*(obs - Hx)`` where *xf* is
    the forecasted state vector (by the forward model), *xa* is the
    updated vector after data-assimilation, *K* is the Kalman gain
    matrix, *obs* is the observation vector, and *Hx* is the forecasted
    state vector in observation space. See the documentation for more
    information.
    """

    def __init__(self, nsamples, t_interval, t_end, max_iterations,
                 model, input_dict,):
        """ Parse input file and assign values to class attributes.

        Note
        ----
        See InverseMethodDAFI.__init__ for details.
        """
        super(self.__class__, self).__init__(
            nsamples, t_interval, t_end, max_iterations, model,
            input_dict)
        self.name = 'Ensemble Kalman Filter (EnKF)'

    def analysis(self,):
        """ Correct the propagated ensemble (filtering step) using EnKF
        """
        # calculate the Kalman gain matrix
        xp = _mean_subtracted_matrix(self.state_forecast)
        hxp = _mean_subtracted_matrix(self.state_in_obsspace)
        coeff = (1.0 / (self.nsamples - 1.0))
        pht = coeff * np.dot(xp, hxp.T)
        hpht = coeff * hxp.dot(hxp.T)
        conn = _check_condition_number(hpht + self.obs_error)
        report = f"    Condition number of (HPHT + R) is {conn}"
        inv = la.inv(hpht + self.obs_error)
        kalman_gain_matrix = pht.dot(inv)
        # analysis step
        dx = np.dot(kalman_gain_matrix, self.obs - self.state_in_obsspace)
        state_analysis = self.state_forecast + dx
        # debug
        if self._debug:
            debug_dict = {
                'K': kalman_gain_matrix, 'inv': inv, 'HPHT': hpht, 'PHT': pht,
                'Hxp': hxp, 'xp': xp}
            self._save_debug(debug_dict)
        return state_analysis, report


class EnRML(InverseMethod):
    """ Implementation of the ensemble Randomized Maximal Likelihood
    (EnRML).

    It inherits most methods from parent class (``InverseMethodDAFI``),
    but replaces the ``analysis`` method to use EnRML for the
    data-assimilation step.

    The EnRML is updated by: ``xa = xf + GN*(obs - Hx)+P``
    where *xf* is the forecasted state vector (by the forward model),
    *xa* is the updated vector after data-assimilation, *GN* is the
    Gauss-Newton matrix, *obs* is the observation vector, and *Hx* is
    the forecasted state vector in observation space, *P* is Penalty
    matrix. See the documentation for more information.
    """

    def __init__(self, nsamples, t_interval, t_end, max_iterations,
                 model, input_dict,):
        """ Parse input file and assign values to class attributes.

        Note
        ----
        See InverseMethodDAFI.__init__ for details.
        """
        super(self.__class__, self).__init__(
            nsamples, t_interval, t_end, max_iterations, model,
            input_dict)
        self.name = 'Ensemble Randomized Maximal Likelihood (EnRML)'
        self.beta = float(input_dict['step_length'])
        self._perturb_obs = 'time'

    def analysis(self,):
        """ Correct the propagated ensemble (filtering step) using EnRML
        """
        # calculate the Gauss-Newton matrix
        xp0 = _mean_subtracted_matrix(self.state_prior)
        p0 = (1.0 / (self.nsamples - 1.0)) * xp0.dot(xp0.T)
        x = self.state_forecast

        xp = _mean_subtracted_matrix(x)
        hxp = _mean_subtracted_matrix(self.state_in_obsspace)
        gen = np.dot(hxp, la.pinv(xp))
        sen_mat = p0.dot(gen.T)
        cyyi = np.dot(np.dot(gen, p0), gen.T)
        conn = _check_condition_number(cyyi + self.obs_error)
        inv = la.inv(cyyi + self.obs_error)
        gauss_newton_matrix = sen_mat.dot(inv)
        # calculate the penalty
        penalty = np.dot(gauss_newton_matrix, gen.dot(x-self.state_prior))

        # analysis step
        dx = np.dot(gauss_newton_matrix,
                    self.obs - self.state_in_obsspace) + penalty
        x = self.beta * self.state_prior + \
                         (1.0 - self.beta) * x + self.beta * dx
        report = f"    Condition number of (cyyi + R) is {conn}"

        # debug
        if self._debug:
            debug_dict = {
                'GN': gauss_newton_matrix, 'pen': penalty, 'inv': inv,
                'cyyi': cyyi, 'Hxp': hxp, 'xp': xp}
            self._save_debug(debug_dict, f'{i}_')
        return x, report


class EnKF_MDA(InverseMethod):
    """ Implementation of the ensemble Kalman Filter-Multi data
    assimilaton (EnKF-MDA).

    It inherits most methods from parent class (``InverseMethodDAFI``),
    but replaces the ``analysis`` method to use EnKF-MDA for the
    data-assimilation step.

    The EnKF-MDA is updated by:
    ``xa = xf + K_mda*(obs - Hx - err_mda)`` where *xf* is the
    forecasted state vector (by the dynamic model),
    *xa* is the updated vector after data-assimilation, *K_mda* is the
    modified Kalman gain matrix, *obs* is the observation vector, and
    *Hx* is the forwarded state vector in observation space, 'err_mda'
    is inflated error. See the documentation for more information.
    """

    def __init__(self, nsamples, t_interval, t_end, max_iterations,
                 model, input_dict,):
        """ Parse input file and assign values to class attributes.

        Note
        ----
        See InverseMethodDAFI.__init__ for details.
        """
        super(self.__class__, self).__init__(
            nsamples, t_interval, t_end, max_iterations, model,
            input_dict)
        self.name = 'Ensemble Kalman Filter-Multi Data Assimilation (EnKF-MDA)'
        self.alpha = self.max_iterations
        self.convergence_option = 'max'
        self.convergence_residual = None

    def analysis(self,):
        """ Correct the propagated ensemble (filtering step) using
        EnKF_MDA.
        """
        # calculate the Kalman gain matrix
        x = self.state_forecast.copy()
        xp = _mean_subtracted_matrix(self.state_forecast)
        hxp = _mean_subtracted_matrix(self.state_in_obsspace)
        coeff = (1.0 / (self.nsamples - 1.0))
        pht = coeff * np.dot(xp, hxp.T)
        hpht = coeff * hxp.dot(hxp.T)
        conn =  _check_condition_number(hpht + self.alpha * self.obs_error)
        inv = la.inv(hpht + self.alpha * self.obs_error)
        kalman_gain_matrix = pht.dot(inv)
        # analysis step
        obs = self.obs - self.obs_perturbation
        obs_mda = obs + np.sqrt(self.alpha) * self.obs_perturbation
        dx = np.dot(kalman_gain_matrix, obs_mda - self.state_in_obsspace)
        x += dx
        report = f"    Condition number of (HPHT + aR) is {conn}\n"
        # debug
        if self._debug:
            debug_dict = {
                'K': kalman_gain_matrix, 'inv': inv, 'HPHT': hpht, 'PHT': pht,
                'Hxp': hxp, 'xp': xp}
            self._save_debug(debug_dict, f'{i}_')
        return x, report


class REnKF(InverseMethod):
    """ Implementation of the regularized ensemble Kalman Filter
    (REnKF).

    It inherits most methods from parent class (``InverseMethodDAFI``),
    but replaces the ``analysis`` method to use penalized EnKF for
    the data-assimilation step.

    For detail on the method see
    [cite paper].

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

    def __init__(self, nsamples, t_interval, t_end, max_iterations,
                 model, input_dict,):
        """ Parse input file and assign values to class attributes.

        Note
        ----
        See InverseMethodDAFI.__init__ for details.
        """
        super(self.__class__, self).__init__(
            nsamples, t_interval, t_end, max_iterations, model,
            input_dict)
        self.name = 'Regularized Ensemble Kalman Filter (REnKF)'
        # load penalties
        pfile = input_dict['penalties_python_file']
        sys.path.append(os.path.dirname(pfile))
        penalties = getattr(importlib.import_module(
            os.path.splitext(os.path.basename(pfile))[0]), 'penalties')
        self.penalties = penalties(self)


    def analysis(self,):
        """ Correct the propagated ensemble (filtering step) using
        regularized EnKF.
        """
        # calculate the Kalman gain matrix
        xp = _mean_subtracted_matrix(self.state_forecast)
        hxp = _mean_subtracted_matrix(self.state_in_obsspace)
        coeff = (1.0 / (self.nsamples - 1.0))
        pht = coeff * np.dot(xp, hxp.T)
        hpht = coeff * hxp.dot(hxp.T)
        conn =  _check_condition_number(hpht + self.obs_error)
        report = f"    Condition number of (HPHT + R) is {conn}"
        inv = la.inv(hpht + self.obs_error)
        kalman_gain_matrix = pht.dot(inv)
        # calculate the "K2" matrix
        hxx = np.dot(hxp, xp.T)
        k2_gain_matrix = \
            coeff * np.dot(kalman_gain_matrix, hxx) - coeff*np.dot(xp, xp.T)
        # calculate penalty matrix
        penalty_mat = np.zeros([len(self.state_forecast), self.nsamples])
        for ipenalty in self.penalties:
            w_mat = ipenalty['weight_matrix']
            lamb = ipenalty['lambda']
            lamda = lamb(self.i_iteration)
            func_penalty = ipenalty['penalty']
            func_gradient = ipenalty['gradient']
            for isamp in range(self.nsamples):
                istate = self.state_forecast[:, isamp]
                gpw = np.dot(func_gradient(istate).T, w_mat)
                gpwg = np.dot(gpw, func_penalty(istate))
                penalty_mat[:, isamp] += lamda * gpwg
        # analysis step
        dx1 = np.dot(kalman_gain_matrix, self.obs - self.state_in_obsspace)
        dx2 = np.dot(k2_gain_matrix, penalty_mat)
        state_analysis = self.state_forecast + dx1 + dx2

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
            self._save_debug(debug_dict)
        return state_analysis, report


# functions
def _create_dir(dir,):
    """ Create directory if does not exist. """
    if not os.path.exists(dir):
        os.makedirs(dir)

def _iteration_residual(list, iter,):
    """ Calculate the residual at a given iteration. """
    if iter > 0:
        iterative_residual = abs(list[iter] - list[iter-1]) / abs(list[0])
    else:
        iterative_residual = np.nan
    return iterative_residual

def _vec_to_mat(vec, ncol,):
    "Tile a vector ncol times to form a matrix"
    return np.tile(vec, (ncol, 1)).T

def _perturb_vec(mean, cov, nsamps,):
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
    return np.tile(mean, (nsamps, 1)).T + perturb, perturb

def _cov_to_std(cov,):
    nvars = cov.shape[0]
    std = np.sqrt(np.diag(cov))
    std_norm = la.norm(std) / nvars
    return std_norm

def _check_condition_number(mat, eps=1e16,):
    con_inv = la.cond(mat)
    if (con_inv > eps):
        message = "The matrix is singular, inverse will fail."
        warnings.warn(message, RuntimeWarning)
    return con_inv

def _mean_subtracted_matrix(mat, samp_axis=1,):
    nsamps = mat.shape[samp_axis]
    mean_vec = np.array([np.mean(mat, axis=samp_axis)])
    mean_vec = mean_vec.T
    mean_vec = np.tile(mean_vec, (1, nsamps))
    mean_sub_mat = mat - mean_vec
    return mean_sub_mat
