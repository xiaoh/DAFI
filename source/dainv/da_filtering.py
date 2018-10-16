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
        s = 'An empty data assimilation filtering technique.'
        return s

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

    Includes more methods than the DAFilter class. The DAFilter class
    is a barebones template. DAFilter2 contains several methods (e.g
    error checking, plotting, reporting) as well as a framework for the
    main self.solve() method. This can be used to quickly create new
    filter classes if all the same methods are desired.
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
        self.time_array = np.arange(0, self.t_end, self.da_interval)

        # initialize states: these change at every iteration
        self.time = 0 # current time
        self.iter = 0 # current iteration
        self.da_step = 0 # current DA step
        # ensemble matrix (nsamples, nstate)
        self.X = np.zeros([self.dyn_model.nstate, self.nsamples])
        # ensemble matrix projected to observed space (nsamples, nstateSample)
        self.HX = np.zeros([self.dyn_model.nstate_obs, self.nsamples])
        # observation matrix (nstate_obs, nsamples)
        self.obs = np.zeros([self.dyn_model.nstate_obs, self.nsamples])
        # observation covariance matrix (nstate_obs, nstate_obs)
        self.R_obs = np.zeros(
            [self.dyn_model.nstate_obs, self.dyn_model.nstate_obs])

        # initialize misfit: these grow each iteration, but all values stored.
        self._misfit_X_L1 = []
        self._misfit_X_L2 = []
        self._misfit_X_inf = []
        self._obs_sigma = []
        self._sigma_HX = []
        self._misfit_max = []

    def __str__(self):
        s = self.name + \
            '\n   Number of samples: {}'.format(self.nsamples) + \
            '\n   Run time:          {}'.format(self.t_end) +  \
            '\n   DA interval:       {}'.format(self.da_interval) + \
            '\n   Forward model:     {}'.format(self.dyn_model.name)
        return s

    def solve(self):
        """ Solve the parameter estimation problem.

        Updates:
            time
            iter
            da_step
            X
            HX
            obs
            R_obs
        """
        # Generate initial state Ensemble
        self.X, self.HX = self.dyn_model.generate_ensemble()
        # sensitivity only
        if self._sensitivity_only:
            next_end_time = 2 * self.da_interval + self.time_array[0]
            self.X, self.HX = self.dyn_model.forecast_to_time(
                self.X, next_end_time)
            if self.ver>=1:
                print "\nSensitivity study completed."
            sys.exit(0)
        # main DA loop
        early_stop = False
        for time in self.time_array:
            self.time = time
            next_end_time = 2 * self.da_interval + self.time
            self.da_step = (next_end_time - self.da_interval) / \
                self.da_interval
            if self._verb>=1:
                print("\nData Assimilation step: {}".format(self.da_step))
            # dyn_model: propagate the state ensemble to next DA time
            # and get observations at next DA time.
            self.X, self.HX = self.dyn_model.forecast_to_time(
                self.X, next_end_time)
            self.obs, self.R_obs = self.dyn_model.get_obs(next_end_time)
            # data assimilation
            debug_dict = {
                'Xf':self.X, 'HX':self.HX, 'obs':self.obs, 'R':self.R_obs}
            self._save_debug(debug_dict)
            self._correct_forecasts()
            self._calc_misfits()
            debug_dict = {'Xa':self.X}
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
        """ Report summary information at completion. """
        # TODO: Report DA results
        try: self.dyn_model.report()
        except: pass

    def clean(self):
        """ Cleanup before exiting. """
        try: self.dyn_model.clean()
        except: pass

    def save(self,save_dir):
        if self._save_flag:
            np.savetxt(elf._debug_folder + os.sep + 'misfit_L1', np.array(
                self._misfit_X_L1))
            np.savetxt(elf._debug_folder + os.sep + 'misfit_L2', np.array(
                self._misfit_X_L2))
            np.savetxt(elf._debug_folder + os.sep + 'misfit_inf', np.array(
                self._misfit_X_inf))
            np.savetxt(elf._debug_folder + os.sep + '_obs_sigma', np.array(
                self._obs_sigma))
            np.savetxt(elf._debug_folder + os.sep + '_sigma_HX', np.array(
                self._sigma_HX))
        try: self.dyn_model.save()
        except: pass

    # private methods
    def _correct_forecasts(self):
        # Correct the propagated ensemble (filtering step).
        #
        # Updates:
        #     self.X
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
        #     _misfit_X_L1
        #     _misfit_X_L2
        #     _misfit_X_inf
        #     _sigma_HX
        #     _misfit_max
        #     _obs_sigma
        #

        # calculate misfits
        Nnorm = self.dyn_model.nstate_obs * self.nsamples
        diff = abs(self.obs - self.HX)
        misFit_L1 = np.sum(diff) / Nnorm
        misFit_L2 = np.sqrt(np.sum(diff**2)) / Nnorm
        misFit_Inf = la.norm(diff, np.inf)
        misfit_max = np.max([misFit_L1, misFit_L2, misFit_Inf])
        _sigma_HX = np.std(self.HX, axis=1)
        _sigma_HX_norm = la.norm(_sigma_HX) / self.dyn_model.nstate_obs
        sigma = (la.norm(np.sqrt(np.diag(self.R_obs))) / self.dyn_model.nstate_obs)

        self._misfit_X_L1.append(misFit_L1)
        self._misfit_X_L2.append(misFit_L2)
        self._misfit_X_inf.append(misFit_Inf)
        self._sigma_HX.append(_sigma_HX_norm)
        self._misfit_max.append(misfit_max)
        self._obs_sigma.append(sigma)

    def _check_convergence(self):
        # Check iteration convergence.

        def _convergence_1():
            if self.iter>0:
                iterative_residual = \
                    (abs(self._misfit_X_L2[self.iter]
                    - self._misfit_X_L2[self.iter-1])) \
                    / abs(self._misfit_X_L2[0])
                conv = iterative_residual  < self.convergence_residual
            else:
                iterative_residual = None
                conv = False
            return iterative_residual, conv

        def _convergence_2():
            sigma_inf = np.max(np.diag(np.sqrt(self.R_obs)))
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

        s = '\n' + "#"*80
        s += "\nStandard deviation of ensemble: {}".format(
            self._sigma_HX[self.iter]) + \
            "\nStandard deviation of observation error: {}".format(
            self._obs_sigma[self.iter])
        s += "\n\nMisfit between the predicted QoI and the observed QoI." + \
            "\nL1 norm = {}\nL2 norm = {}\nInf norm = {}\n\n".format(
            self._misfit_X_L1[self.iter], self._misfit_X_L2[self.iter],
            self._misfit_X_inf[self.iter])

        s += "\n\nConvergence criteria 1:" + \
            "\n  Relative iterative residual: {}".format(iterative_residual) \
            + "\n  Relative convergence criterion: {}".format(
            self.convergence_residual)
        if conv1:
            s += "\n  Status: Not yet converged."
        else:
            s += "\n  Status: Converged."

        s += "\n\nConvergence criteria 2:" + \
            "\n  Infinite misfit: {}".format(self._misfit_max[self.iter]) + \
            "\n  Infinite norm  of observation error: {}".format(
                sigma_inf)
        if  conv2:
            s += "\n  Status: Not yet converged."
        else:
            s += "\n  Status: Converged."

        s += '\n\n'
        print(s)

# child classes (specific filtering techniques

class EnKF(DAFilter2):
    """ Implementation of the ensemble Kalman Filter (EnKF).


    """
    # TODO: add more detail to the docstring

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
        #     self.X

        def _check_condition_number(HPHT):
            conInv = la.cond(HPHT + self.R_obs)
            print("Conditional number of (HPHT + R) is {}".format(conInv))
            if (conInv > 1e16):
                warnings.warn(
                    "The matrix (HPHT + R) is singular, inverse will fail."
                    ,RuntimeWarning)

        def _mean_subtracted_matrix(mat, samps=self.nsamples):
            mean_vec = np.array([ np.mean(mat, axis=1) ])
            mean_vec = mean_vec.T
            mean_vec = np.tile(mean_vec, (1, samps))
            mean_sub_mat = mat - mean_vec
            return mean_sub_mat

        # calculate the Kalman gain matrix
        XP = _mean_subtracted_matrix(self.X)
        HXP = _mean_subtracted_matrix(self.HX)
        coeff =  (1.0 / (self.nsamples - 1.0))
        PHT = coeff * np.dot(XP, HXP.T)
        HPHT = coeff * HXP.dot(HXP.T)
        _check_condition_number(HPHT)
        inv = la.inv(HPHT + self.R_obs)
        inv = inv.A #convert np.matrix to np.ndarray
        kalman_gain_matrix = PHT.dot(inv)
        # analysis step
        dX = np.dot(kalman_gain_matrix, self.obs - self.HX)
        self.X += dX
        # debug
        debug_dict = {
            'K':kalman_gain_matrix, 'inv':inv, 'HPHT':HPHT, 'PHT':PHT,
            'HXP':HXP, 'XP':XP}
        self._save_debug(debug_dict)
