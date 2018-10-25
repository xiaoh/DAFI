# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Collection of different filtering techniques. """

# standard library imports
import ast
import warnings
import os
import sys

# third party imports
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt


# parent classes (templates)
class DAFilter(object):
    """ Parent class for data assimilation filtering techniques.

    Use this as a template to write new filtering classes.
    The required methods are summarized below.
    """

    def __init__(self, nsamples, da_interval, t_end, forward_model,
                 input_dict):
        """ Parse input file and assign values to class attributes.

        Parameters
        ----------
        nsamples : int
            Ensemble size.
        da_interval : float
            Iteration interval between data assimilation steps.
        t_end : float
            Final time.
        forward_model : DynModel
            Dynamic model.
        input_dict : dict[str]
            All filter-specific inputs.
        """
        self.dyn_model = forward_model

    def __str__(self):
        str_info = 'An empty data assimilation filtering technique.'
        return str_info

    def solve(self):
        """ Implement the filtering technique. """
        pass

    def report(self):
        """ Report summary information. """
        try:
            self.dyn_model.report()
        except:
            pass

    def plot(self):
        """ Create any relevant plots. """
        try:
            self.dyn_model.plot()
        except:
            pass

    def clean(self):
        """ Perform any necessary cleanup at completion. """
        try:
            self.dyn_model.clean()
        except:
            pass

    def save(self):
        """ Save any important results to text files. """
        try:
            self.dyn_model.save()
        except:
            pass


class DAFilter2(DAFilter):
    """ Parent class for DA filtering techniques.

    This class includes more methods than the DAFilter class.
    The DAFilter class is a barebones template. DAFilter2 contains
    several methods (e.g error checking, plotting, reporting) as well
    as a framework for the main self.solve() method. This can be used
    to quickly create new filter classes if some of the same methods
    are desired.
    """

    def __init__(self, nsamples, da_interval, t_end, forward_model,
                 input_dict):
        """ Parse input file and assign values to class attributes.

        Parameters
        ----------
        nsamples : int
            Ensemble size.
        da_interval : float
            Iteration interval between data assimilation steps.
        t_end : float
            Final time.
        forward_model : DynModel
            Dynamic model.
        input_dict : dict[str]
            All filter-specific inputs.

        Note
        ----
        Inputs in ``input_dict``:
            * **save_folder** (``string``, ``./results_da``) -
              Folder where to save results.
            * **debug_flag** (``bool``, ``False``) -
              Save extra information for debugging.
            * **debug_folder** (``string``, ``./debug``) -
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
            * **convergence_residual** (``float``) -
              Residual value for convergence if ``reach_max_flag`` is
              ``False`` and ``convergence_option`` is ``residual``.
            * **convergence_norm** (``int, inf``,``2``) -
              Order of norm to use for convergence criteria. can be an
              integer or ``np.inf``. Default is L2 norm.
        """
        self.name = 'Generic DA filtering technique'
        self.short_name = None
        self.dyn_model = forward_model
        self.nsamples = nsamples  # number of samples in ensemble
        self.nstate = self.dyn_model.nstate  # number of states
        self.nstate_obs = self.dyn_model.nstate_obs  # number of observations
        self.da_interval = da_interval  # DA time step interval
        self.t_end = t_end  # total run time

        # filter-specific inputs
        try:
            self.stationary_flag = ast.literal_eval(
                input_dict['stationary_flag'])
        except:
            self.stationary_flag = True
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
            self._debug_folder = input_dict['debug_folder']
        except:
            self._debug_folder = os.curdir + os.sep + 'debug'
        if self._debug_flag:
            self._create_folder(self._debug_folder)
        try:
            self._save_folder = input_dict['save_folder']
        except:
            self._save_folder = os.curdir + os.sep + 'results_da'
        try:
            self._verb = int(input_dict['verbosity'])
        except:
            self._verb = int(1)
        try:
            self.max_pseudo_time = int(
                input_dict['max_pseudo_time'])
        except:
            self.max_pseudo_time = 1

        try:
            self.forward_interval = float(
                input_dict['forward_interval'])
        except:
            self.forward_interval = 1

        # initialize iteration array
        self.time_array = np.arange(0.0, self.t_end, self.da_interval)
        self.pseudo_time_array = np.arange(
            0.0, self.max_pseudo_time, self.forward_interval)
        # initialize states: these change at every iteration
        self.time = 0.0  # current time
        self.pseudo_time = 0.0  # pseudo_time step for each real time
        self.da_step = int(0)  # current DA step
        self.forward_step = int(0)  # current forward step
        # ensemble matrix (nsamples, nstate)
        self.state_vec_prior = np.zeros(
            [self.dyn_model.nstate, self.nsamples])
        self.state_vec_forecast = np.zeros(
            [self.dyn_model.nstate, self.nsamples])
        self.state_vec_analysis = np.zeros(
            [self.dyn_model.nstate, self.nsamples])
        # ensemble matrix projected to observed space (nsamples, nstateSample)
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
            * self.state_vec
            * self.model_obs
            * self.obs
            * self.obs_error
            * self.state_vec_analysis_all
            * self.state_vec_forecast_all
        """
        # Generate initial state Ensemble
        self.state_vec_analysis, self.model_obs = \
            self.dyn_model.generate_ensemble()

        # sensitivity only
        if self._sensitivity_only:
            self.state_vec_forecast, self.model_obs = \
                self.dyn_model.forecast_to_time(
                    self.state_vec_analysis, self.da_interval)
            if self.ver >= 1:
                print("\nSensitivity study completed.")
            sys.exit(0)
        # main DA loop
        early_stop = False
        for time in self.time_array:
            self.time = time + self.da_interval
            self.da_step += 1
            if self._verb >= 1:
                print("\nData-assimilation step: {}".format(self.da_step) +
                      "\n  Time: {}".format(self.time))
            # dyn_model: propagate the state ensemble to, and
            #   get observations at, next DA time.
            self.state_vec_forecast = \
                self.dyn_model.forecast_to_time(
                    self.state_vec_analysis.copy(), self.time)
            self.state_vec_prior = self.state_vec_forecast.copy()
            self.forward_step = 0
            for pseudo_time in self.pseudo_time_array:
                self.pseudo_time = pseudo_time + self.forward_interval
                self.forward_step += 1
                if self._verb >= 1:
                    print("\nforward step: {}".format(self.forward_step) +
                          "\n Pseudo Time: {}".format(self.pseudo_time))
                self.state_vec_forecast, self.model_obs = \
                    self.dyn_model.forward(
                        self.state_vec_forecast, self.pseudo_time)
                self.obs, self.obs_perturb, self.obs_error = \
                    self.dyn_model.get_obs(self.time)
                # data assimilation
                self._correct_forecasts()
                self._calc_misfits()
                self.state_vec_forecast = self.state_vec_analysis.copy()
                if self.stationary_flag:
                    self.save_report()
                conv_var, conv_res = self._check_convergence()
                if self.convergence_option is 'variance':
                    conv = conv_var
                else:
                    conv = conv_res
                if conv and not self.reach_max_flag:
                    early_stop = True
                    break

            if not self.stationary_flag:
                self.save_report()
            if self._verb >= 1:
                if early_stop:
                    print("\nDA Filtering completed: convergence early stop.")
                else:
                    print("\nDA Filtering completed: Max iteration reached.")

    def save_report(self):
        # save results
        self.obs_all.append(self.obs.copy())
        self.model_obs_all.append(self.model_obs.copy())
        self.obs_error_all.append(self.obs_error.copy())
        self.state_vec_analysis_all.append(self.state_vec_analysis.copy())
        self.state_vec_forecast_all.append(self.state_vec_forecast.copy())
        # report
        if self._verb >= 2:
            print(self._report())

    def correct_forecasts(self):
        """ Correct the propagated ensemble (filtering step).

        **Updates:**
            * self.state_vec
        """
        raise NotImplementedError(
            "Needs to be implemented in the child class!")

    def plot(self):
        """ Plot iteration convergence. """
        self._create_folder(self._save_folder)
        fig, ax = plt.subplots()
        iter = np.arange(self.da_step) + 1
        ax.plot(iter, self._misfit_norm, '.-', label='norm(obs-HX)')
        ax.plot(iter, self._sigma_hx_norm, '.-', label='norm(std(HX))')
        ax.plot(iter, self._sigma_obs_norm, '.-', label='norm(std(obs))')
        ax.set_title('Iteration Convergence')
        ax.set_xlabel('Data-Assimilation Step')
        ax.legend(loc='upper right')
        fig.savefig(self._save_folder + os.sep + 'iteration_convergence.pdf')
        try:
            self.dyn_model.plot()
        except:
            pass

    def report(self):
        """ Report summary information. """
        str_report = '\n\nInverse Modeling Report\n' + '='*23
        str_report += '\nDA_step: {}\nTime: {}'.format(self.da_step, self.time)
        str_report += self._report()
        print(str_report)
        print('\n\nForward Modeling Report\n' + '='*23)
        try:
            self.dyn_model.report()
        except:
            pass

    def clean(self):
        """ Cleanup before exiting. """
        try:
            self.dyn_model.clean()
        except:
            pass

    def save(self):
        """ Saves results to text files. """
        self._create_folder(self._save_folder)
        np.savetxt(self._save_folder + os.sep + 'misfit_norm', np.array(
            self._misfit_norm))
        np.savetxt(self._save_folder + os.sep + 'sigma_HX', np.array(
            self._sigma_hx_norm))
        np.savetxt(self._save_folder + os.sep + 'sigma_obs', np.array(
            self._sigma_obs_norm))
        for da_step, value in enumerate(self.state_vec_analysis_all):
            np.savetxt(self._save_folder + os.sep + 'Xa_{}'.format(da_step+1),
                       value)
        for da_step, value in enumerate(self.state_vec_forecast_all):
            np.savetxt(self._save_folder + os.sep + 'Xf_{}'.format(da_step+1),
                       value)
        for da_step, value in enumerate(self.obs_all):
            np.savetxt(self._save_folder + os.sep + 'obs_{}'.format(da_step+1),
                       value)
        for da_step, value in enumerate(self.model_obs_all):
            np.savetxt(self._save_folder + os.sep + 'HX_{}'.format(da_step+1),
                       value)
        for da_step, value in enumerate(self.obs_error_all):
            np.savetxt(self._save_folder + os.sep + 'R_{}'.format(da_step+1),
                       value)
        try:
            self.dyn_model.save()
        except:
            pass

    # private methods
    def _create_folder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_debug(self, debug_dict):
        if self._debug_flag:
            for key, value in debug_dict.items():
                fname = self._debug_folder + os.sep + key + \
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
        # store values
        self._misfit_norm.append(misfit_norm)
        self._sigma_hx_norm.append(sigma_hx_norm)
        self._sigma_obs_norm.append(sigma_obs_norm)

    def _iteration_residual(self, list, iter):
        if iter > 0:
            iterative_residual = abs(list[iter] - list[iter-1]) / abs(list[0])
        else:
            iterative_residual = None
        return iterative_residual

    def _check_convergence(self):
        # Check iteration convergence.
        conv_variance = self._misfit_norm[self.da_step - 1] < \
            self._sigma_obs_norm[self.da_step - 1]
        residual = self._iteration_residual(self._misfit_norm, self.da_step-1)
        if self.convergence_residual is None:
            conv_residual = False
        else:
            conv_residual = residual < self.convergence_residual
        return conv_variance, conv_residual

    def _report(self):
        """ Report at each iteration. """
        residual = self._iteration_residual(self._misfit_norm, self.da_step-1)
        conv_var, conv_res = self._check_convergence()
        str_report = ''
        str_report += "  Norm of standard deviation of HX: {}".format(
            self._sigma_hx_norm[self.da_step - 1]) + \
            "\n  Norm of standard deviation of observation: {}".format(
            self._sigma_obs_norm[self.da_step - 1])
        str_report += "\n  Norm of misfit: {}".format(
            self._misfit_norm[self.da_step - 1])
        str_report += "\n  Convergence (variance): {}".format(conv_var)
        str_report += "\n  Convergence (residual): {}".format(conv_res) \
            + "\n    Relative iterative residual: {}".format(residual) \
            + "\n    Relative convergence criterion: {}".format(
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

    def __init__(self, nsamples, da_interval, t_end, forward_model,
                 input_dict):
        """ Parse input file and assign values to class attributes.

        Note
        ----
        See DAFilter2.__init__ for details.
        """
        super(self.__class__, self).__init__(
            nsamples, da_interval, t_end, forward_model, input_dict)
        self.name = 'Ensemble Kalman Filter'
        self.short_name = 'EnKF'

    def _correct_forecasts(self):
        """ Correct the propagated ensemble (filtering step) using EnKF

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

        # calculate the Kalman gain matrix
        xp = _mean_subtracted_matrix(self.state_vec_forecast)
        hxp = _mean_subtracted_matrix(self.model_obs)
        coeff = (1.0 / (self.nsamples - 1.0))
        pht = coeff * np.dot(xp, hxp.T)
        hpht = coeff * hxp.dot(hxp.T)
        _check_condition_number(hpht)
        inv = la.inv(hpht + self.obs_error)
        inv = inv.A  # convert np.matrix to np.ndarray
        kalman_gain_matrix = pht.dot(inv)
        # analysis step
        dx = np.dot(kalman_gain_matrix, self.obs - self.model_obs)
        self.state_vec_analysis = self.state_vec_forecast + dx
        # debug
        debug_dict = {
            'K': kalman_gain_matrix, 'inv': inv, 'HPHT': hpht, 'PHT': pht,
            'HXP': hxp, 'XP': xp}
        self._save_debug(debug_dict)


# child classes (specific filtering techniques)
class EnRML(DAFilter2):
    """ Implementation of the ensemble Randomized Maximal Likelihood (EnRML).

    It inherits most methods from parent class (``DAFIlter2``), but
    replaces the ``correct_forecasts`` method to use EnKF for the
    data-assimilation step.

    The EnRML is updated by: ``Xa = Xf + GN*(obs - HX)+P`` 
    where *Xf* is the forecasted state vector (by the forward model), 
    *Xa* is the updated vector after data-assimilation, *GN* is the 
    Gauss-Newton matrix, *obs* is the observation vector, and *HX* is the 
    forecasted state vector in observation space, *P* is Penalty matrix. 
    See the documentation for more information.
    """

    def __init__(self, nsamples, da_interval, t_end, forward_model,
                 input_dict):
        """ Parse input file and assign values to class attributes.

        Note
        ----
        See DAFilter2.__init__ for details.
        """
        super(self.__class__, self).__init__(
            nsamples, da_interval, t_end, forward_model, input_dict)
        self.name = 'Ensemble Randomized Maximal Likelihood'
        self.short_name = 'EnRML'
        self.beta = float(input_dict['beta'])
        self.criteria = ast.literal_eval(
            input_dict['criteria'])

    def _correct_forecasts(self):
        """ Correct the propagated ensemble (filtering step) using EnKF

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
            1.0-self.beta) * x + self.beta*dx

        self.state_vec_analysis = x.copy()
        # debug
        debug_dict = {
            'GN': gauss_newton_matrix, 'pen': penalty, 'inv': inv,
            'cyyi': cyyi, 'HXP': hxp, 'XP': xp}
        self._save_debug(debug_dict)


# child classes (specific filtering techniques)
class EnKF_MDA(DAFilter2):
    """ Implementation of the ensemble Kalman Filter-Multi data 
    assimilaton (EnKF-MDA).

    It inherits most methods from parent class (``DAFIlter2``), but
    replaces the ``correct_forecasts`` method to use EnKF for the
    data-assimilation step.

    The EnKF-MDA is updated by: ``Xa = Xf + K_mda*(obs - HX - err_mda)
    `` where *Xf* is the forecasted state vector (by the dynamic model), 
    *Xa* is the updated vector after data-assimilation, *K_mda* is the 
    modified Kalman gain matrix, *obs* is the observation vector, and 
    *HX* is the forwarded state vector in observation space, 'err_mda' 
    is inflated error. See the documentation for more information.
    """

    def __init__(self, nsamples, da_interval, t_end, forward_model,
                 input_dict):
        """ Parse input file and assign values to class attributes.

        Note
        ----
        See DAFilter2.__init__ for details.
        """
        super(self.__class__, self).__init__(
            nsamples, da_interval, t_end, forward_model, input_dict)
        self.name = 'Ensemble Kalman Filter-Multi Data Assimilation'
        self.short_name = 'EnKF-MDA'

    def _correct_forecasts(self):
        """ Correct the propagated ensemble (filtering step) using EnKF

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

        alpha = self.max_pseudo_time/self.forward_interval
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


# child classes (specific filtering techniques)
# developing
class EnKF_Lasso(DAFilter2):
    """ Implementation of the ensemble Kalman Filter with Lasso (EnKF-Lasso).

    It inherits most methods from parent class (``DAFIlter2``), but
    replaces the ``correct_forecasts`` method to use EnKF for the
    data-assimilation step.

    The EnKF_lasso is updated by: ``Xa = Xf + K*(obs - HX) - penalty`` 
    where *Xf* is the forecasted state vector (by the forward model), 
    *Xa* is the updated vector after data-assimilation, *K* is the 
    Kalman gain matrix, *obs* is the observation vector, and *HX* is 
    the forecasted state vector in observation space. See the 
    documentation for more information.
    """

    def __init__(self, nsamples, da_interval, t_end, forward_model,
                 input_dict):
        """ Parse input file and assign values to class attributes.

        Note
        ----
        See DAFilter2.__init__ for details.
        """
        super(self.__class__, self).__init__(
            nsamples, da_interval, t_end, forward_model, input_dict)
        self.name = 'Ensemble Kalman Filter - Lasso'
        self.short_name = 'EnKF-Lasso'

    def _correct_forecasts(self):
        """ Correct the propagated ensemble (filtering step) using EnKF

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

        # calculate the Kalman gain matrix
        xp = _mean_subtracted_matrix(self.state_vec_forecast)
        hxp = _mean_subtracted_matrix(self.model_obs)
        coeff = (1.0 / (self.nsamples - 1.0))
        p = coeff * xp.dot(xp.T)
        pht = coeff * np.dot(xp, hxp.T)
        hpht = coeff * hxp.dot(hxp.T)
        _check_condition_number(hpht)
        inv = la.inv(hpht + self.obs_error)
        inv = inv.A  # convert np.matrix to np.ndarray
        kalman_gain_matrix = pht.dot(inv)

        # calculate the lasso penalty
        lamda = 1e-6
        h_mat = hxp.dot(la.pinv(xp))
        inv_obs_error = la.inv(self.obs_error)

        htrh = np.dot(h_mat.T, inv_obs_error.dot(h_mat))
        inv_lasso = la.pinv(htrh) + p  # htrh is singular
        weight_lasso_vec = np.zeros(htrh.shape[0])

        for i in range(self.nstate):
            if i >= 3000*3-1 and i <= self.nstate - 3:
                for j in range(3):
                    weight_lasso_vec[j+i] = (int((i-8999))/3+1)**2

        weight_lasso_mat = np.tile(weight_lasso_vec, (self.nsamples, 1)).T
        penalty_lasso = inv_lasso.dot(weight_lasso_mat)
        # analysis step
        dx = np.dot(kalman_gain_matrix, self.obs - self.model_obs)
        self.state_vec_analysis = self.state_vec_forecast + dx - lamda * penalty_lasso
        # debug
        debug_dict = {
            'K': kalman_gain_matrix, 'inv': inv, 'HPHT': hpht, 'PHT': pht,
            'HXP': hxp, 'XP': xp}
        self._save_debug(debug_dict)
