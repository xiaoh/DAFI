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

# parent classes (templates)
class DAFilter(object):
    """ Parent class for data assimilation filtering techniques.

    Use this as a template to write new filtering classes.
    The required attributes and methods are summarized below.

    methods:
        solve()
        report()
        plot()
        clean()
    """

    def __init__(
        self, nsamples, da_interval, t_end, forward_model, input_dict
        ):
        """ Parse input file and assign values to class attributes.

        Args:
            nsamples: Ensemble size. [int]
            da_interval: Iteration interval between data assimilation
                        steps. [float]
            t_end: Final time. [float]
            forward_model: Dynamic model. [DynModel]
            input_dict: All filter-specific inputs. [dictionary of str]
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
        try: self.dyn_model.report()
        except: pass

    def plot(self):
        """ Create any relevant plots. """
        try: self.dyn_model.plot()
        except: pass

    def clean(self):
        """ Perform any necessary cleanup at completion. """
        try: self.dyn_model.clean()
        except: pass

    def save(self):
        """ Save any important results to text files. """
        try: self.dyn_model.save()
        except: pass

class DAFilter2(DAFilter):
    """ Parent class for DA filtering techniques.

    This class includes more methods than the DAFilter class.
    The DAFilter class is a barebones template. DAFilter2 contains
    several methods (e.g error checking, plotting, reporting) as well
    as a framework for the main self.solve() method. This can be used
    to quickly create new filter classes if some of the same methods
    are desired.
    """

    def __init__(
        self, nsamples, da_interval, t_end, forward_model, input_dict
        ):
        """ Parse input file and assign values to class attributes.

        Args:
            nsamples: Ensemble size. [int]
            da_interval: Iteration interval between data assimilation
                        steps. [float]
            t_end: Final time. [float]
            forward_model: Dynamic model. [DynModel]
            input_dict: All filter-specific inputs. [dictionary of str]
        """
        self.name = 'Generic DA filtering technique'
        self.short_name = None
        self.dyn_model = forward_model
        self.nsamples = nsamples # number of samples in ensemble
        self.nstate = self.dyn_model.nstate # number of states
        self.nstate_obs = self.dyn_model.nstate_obs # number of observations
        self.da_interval = da_interval # DA time step interval
        self.t_end = t_end # total run time

        # filter-specific inputs
        try:
            self.reach_max_flag = ast.literal_eval(
                input_dict['reach_max_flag'])
        except: self.reach_max_flag = True
        if not self.reach_max_flag:
            self.convergence_residual = float(
                input_dict['convergence_residual'])
        else:
            self.convergence_residual = None
        # private attributes
        try:
            self._sensitivity_only = ast.literal_eval(
                input_dict['sensitivity_only'])
        except:
            self._sensitivity_only = False
        try: self._debug_flag = ast.literal_eval(input_dict['debug_flag'])
        except: self._debug_flag = False
        try: self._debug_folder = input_dict['debug_folder']
        except: self._debug_folder = os.curdir + os.sep + 'debug'
        if self._debug_flag and not os.path.exists(self._debug_folder):
            os.makedirs(self._debug_folder)
        try: self._save_folder = input_dict['save_folder']
        except: self._save_folder = os.curdir + os.sep + 'results_da'
        try: self._verb = int(input_dict['verbosity'])
        except: self._verb = 1

        # initialize iteration array
        self.time_array = np.arange(0.0, self.t_end, self.da_interval)

        # initialize states: these change at every iteration
        self.time = 0.0 # current time
        self.iter = int(0) # current iteration
        self.da_step = int(0) # current DA step
        # ensemble matrix (nsamples, nstate)
        self.state_vec = np.zeros([self.dyn_model.nstate, self.nsamples])
        # ensemble matrix projected to observed space (nsamples, nstateSample)
        self.model_obs = np.zeros([self.dyn_model.nstate_obs, self.nsamples])
        # observation matrix (nstate_obs, nsamples)
        self.obs = np.zeros([self.dyn_model.nstate_obs, self.nsamples])
        # observation covariance matrix (nstate_obs, nstate_obs)
        self.obs_error = np.zeros(
            [self.dyn_model.nstate_obs, self.dyn_model.nstate_obs])

        # initialize misfit: these grow each iteration, but all values stored.
        self._misfit_x_l1norm = []
        self._misfit_x_l2norm = []
        self._misfit_x_inf = []
        self._obs_sigma = []
        self._sigma_hx = []
        self._misfit_max = []

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

        Updates:
            time
            iter
            da_step
            state_vec
            model_obs
            obs
            obs_error
        """
        # Generate initial state Ensemble
        self.state_vec, self.model_obs = self.dyn_model.generate_ensemble()
        # sensitivity only
        if self._sensitivity_only:
            next_end_time = 2.0 * self.da_interval + self.time_array[0]
            self.state_vec, self.model_obs = self.dyn_model.forecast_to_time(
                self.state_vec, next_end_time)
            if self.ver>=1:
                print "\nSensitivity study completed."
            sys.exit(0)
        # main DA loop
        early_stop = False
        for time in self.time_array:
            self.time = time
            next_end_time = 2.0 * self.da_interval + self.time
            self.da_step = (next_end_time - self.da_interval) / \
                self.da_interval
            if self._verb>=1:
                print("\nData Assimilation step: {}".format(self.da_step))
            # dyn_model: propagate the state ensemble to next DA time
            # and get observations at next DA time.
            self.state_vec, self.model_obs = self.dyn_model.forecast_to_time(
                self.state_vec, next_end_time)
            self.obs, self.obs_error = self.dyn_model.get_obs(next_end_time)
            # data assimilation
            debug_dict = {
                'Xf':self.state_vec, 'HX':self.model_obs, 'obs':self.obs,
                'R':self.obs_error}
            self._save_debug(debug_dict)
            self._correct_forecasts()
            self._calc_misfits()
            debug_dict = {'Xa':self.state_vec}
            self._save_debug(debug_dict)
            # check convergence and report
            conv, conv_all = self._check_convergence()
            if self._verb>=2:
                self._report(conv_all)
            if conv and not self.reach_max_flag:
                early_stop = True
                break
            self.iter = self.iter + 1
        if self._verb>=1:
            if early_stop:
                print("\n\nDA Filtering completed: convergence early stop.")
            else:
                print("\n\nDA Filtering completed: Max iteration reached.")

    def plot(self):
        """ Plot iteration convergence. """
        # TODO: os.system('plotIterationConvergence.py')
        try: self.dyn_model.plot()
        except: pass

    def report(self):
        """ Report summary information. """
        # TODO: Report DA results
        try: self.dyn_model.report()
        except: pass

    def clean(self):
        """ Cleanup before exiting. """
        try: self.dyn_model.clean()
        except: pass

    def save(self):
        """ Saves results to text files. """
        if self._save_flag:
            np.savetxt(self._save_folder + os.sep + 'misfit_L1', np.array(
                self._misfit_x_l1norm))
            np.savetxt(self._save_folder + os.sep + 'misfit_L2', np.array(
                self._misfit_x_l2norm))
            np.savetxt(self._save_folder + os.sep + 'misfit_inf', np.array(
                self._misfit_x_inf))
            np.savetxt(self._save_folder + os.sep + '_obs_sigma', np.array(
                self._obs_sigma))
            np.savetxt(self._save_folder + os.sep + 'sigma_HX', np.array(
                self._sigma_hx))
        try: self.dyn_model.save()
        except: pass

    # private methods
    def _correct_forecasts(self):
        # Correct the propagated ensemble (filtering step).
        #
        # Updates:
        #     self.state_vec
        raise NotImplementedError(
            "Needs to be implemented in the child class!")

    def _save_debug(self, debug_dict):
        if self._debug_flag:
            for key, value in debug_dict.items():
                np.savetxt(
                    self._debug_folder + os.sep + key +
                        '_{}'.format(self.da_step),
                    value)

    def _calc_misfits(self):
        # Calculate the misfit.
        #
        # Updates:
        #     _misfit_x_l1norm
        #     _misfit_x_l2norm
        #     _misfit_x_inf
        #     _sigma_hx
        #     _misfit_max
        #     _obs_sigma
        #

        # calculate misfits
        nnorm = self.dyn_model.nstate_obs * self.nsamples
        diff = abs(self.obs - self.model_obs)
        misfit_l1norm = np.sum(diff) / nnorm
        misfit_l2norm = np.sqrt(np.sum(diff**2.0)) / nnorm
        misfit_inf = la.norm(diff, np.inf)
        misfit_max = np.max([misfit_l1norm, misfit_l2norm, misfit_inf])
        sigma_hx = np.std(self.model_obs, axis=1)
        sigma_hx_norm = la.norm(sigma_hx) / self.dyn_model.nstate_obs
        sigma = (la.norm(np.sqrt(np.diag(self.obs_error))) / \
            self.dyn_model.nstate_obs)

        self._misfit_x_l1norm.append(misfit_l1norm)
        self._misfit_x_l2norm.append(misfit_l2norm)
        self._misfit_x_inf.append(misfit_inf)
        self._sigma_hx.append(sigma_hx_norm)
        self._misfit_max.append(misfit_max)
        self._obs_sigma.append(sigma)

    def _check_convergence(self):
        # Check iteration convergence.

        def _convergence_1():
            if self.iter>0:
                iterative_residual = \
                    (abs(self._misfit_x_l2norm[self.iter]
                    - self._misfit_x_l2norm[self.iter-1])) \
                    / abs(self._misfit_x_l2norm[0])
                conv = iterative_residual  < self.convergence_residual
            else:
                iterative_residual = None
                conv = False
            return iterative_residual, conv

        def _convergence_2():
            sigma_inf = np.max(np.diag(np.sqrt(self.obs_error)))
            conv = self._misfit_max[self.iter] > sigma_inf
            return sigma_inf, conv

        iterative_residual, conv1 = _convergence_1()
        sigma_inf, conv2 = _convergence_2()
        conv_all = (iterative_residual, conv1, sigma_inf, conv2)
        conv = conv1
        return conv, conv_all

    def _report(self, info):
        # report at each iteration
        (iterative_residual, conv1, sigma_inf, conv2) = info

        str_report = '\n' + "#"*80
        str_report += "\nStandard deviation of ensemble: {}".format(
            self._sigma_hx[self.iter]) + \
            "\nStandard deviation of observation error: {}".format(
            self._obs_sigma[self.iter])
        str_report += "\n\nMisfit between the predicted Q and observed " + \
            "quantities of interest (QoI)." + \
            "\n  L1 norm = {}\n  L2 norm = {}\n  Inf norm = {}\n\n".format(
            self._misfit_x_l1norm[self.iter], self._misfit_x_l2norm[self.iter],
            self._misfit_x_inf[self.iter])

        str_report += "\n\nConvergence criteria 1:" + \
            "\n  Relative iterative residual: {}".format(iterative_residual) \
            + "\n  Relative convergence criterion: {}".format(
            self.convergence_residual)
        if conv1:
            str_report += "\n  Status: Not yet converged."
        else:
            str_report += "\n  Status: Converged."

        str_report += "\n\nConvergence criteria 2:" + \
            "\n  Infinite misfit: {}".format(self._misfit_max[self.iter]) + \
            "\n  Infinite norm  of observation error: {}".format(
                sigma_inf)
        if  conv2:
            str_report += "\n  Status: Not yet converged."
        else:
            str_report += "\n  Status: Converged."

        str_report += '\n\n'
        print(str_report)

# child classes (specific filtering techniques

class EnKF(DAFilter2):
    """ Implementation of the ensemble Kalman Filter (EnKF).
    """
    # TODO: add more detail to the docstring. E.g. what is EnKF.

    def __init__(
        self, nsamples, da_interval, t_end, forward_model, input_dict
        ):
        """ Parse input file and assign values to class attributes.

        See DAFilter2.__init__ for details.
        """
        super(self.__class__, self).__init__(
            nsamples, da_interval, t_end, forward_model, input_dict)
        self.name = 'Ensemble Kalman Filter'
        self.short_name = 'EnKF'

    def _correct_forecasts(self):
        # Correct the propagated ensemble (filtering step) using EnKF
        #
        # Updates:
        #     self.state_vec

        def _check_condition_number(hpht):
            conInv = la.cond(hpht + self.obs_error)
            print("Conditional number of (hpht + R) is {}".format(conInv))
            if (conInv > 1e16):
                warnings.warn(
                    "The matrix (hpht + R) is singular, inverse will fail."
                    ,RuntimeWarning)

        def _mean_subtracted_matrix(mat, samps=self.nsamples):
            mean_vec = np.array([ np.mean(mat, axis=1) ])
            mean_vec = mean_vec.T
            mean_vec = np.tile(mean_vec, (1, samps))
            mean_sub_mat = mat - mean_vec
            return mean_sub_mat

        # calculate the Kalman gain matrix
        xp = _mean_subtracted_matrix(self.state_vec)
        hxp = _mean_subtracted_matrix(self.model_obs)
        coeff =  (1.0 / (self.nsamples - 1.0))
        pht = coeff * np.dot(xp, hxp.T)
        hpht = coeff * hxp.dot(hxp.T)
        _check_condition_number(hpht)
        inv = la.inv(hpht + self.obs_error)
        inv = inv.A #convert np.matrix to np.ndarray
        kalman_gain_matrix = pht.dot(inv)
        # analysis step
        dx = np.dot(kalman_gain_matrix, self.obs - self.model_obs)
        self.state_vec += dx
        # debug
        debug_dict = {
            'K':kalman_gain_matrix, 'inv':inv, 'HPHT':hpht, 'PHT':pht,
            'HXP':hxp, 'XP':xp}
        self._save_debug(debug_dict)
