# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Collection of different filtering techniques. """

# standard library imports
import ast
import warnings
import os
import sys
import pdb

# third party imports
import numpy as np
from numpy import linalg as la

# parent classes (templates)
class DAFilter(object):
    """ Parent class for data assimilation filtering techniques.

    Use this as a template to write new filtering classes.
    The required methods are summarized below.
    """

    def __init__(
        self, nsamples, da_interval, t_end, forward_model, input_dict
        ):
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
            * **convergence_residual** (``float``) -
              Residual value for convergence.
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
        try: self.beta = float(input_dict['beta'])
        except: self.beta = 0.5
        try: self.const_beta_flag = ast.literal_eval(
                input_dict['const_beta_flag'])
        except: self.const_beta_flag = False 
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
        except: self._verb = int(1)

        # initialize iteration array
        self.time_array = np.arange(0.0, self.t_end, self.da_interval)

        # initialize states: these change at every iteration
        self.time = 0.0 # current time
        self.iter = int(0) # current iteration
        self.da_step = int(0) # current DA step
        # ensemble matrix (nsamples, nstate)
        self.state_vec_forecast = np.zeros([self.dyn_model.nstate, self.nsamples])
        self.state_vec_analysis = np.zeros([self.dyn_model.nstate, self.nsamples])
        self.state_vec_prior = np.zeros([self.dyn_model.nstate, self.nsamples])
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
        self._misfit_max = []
        self._sigma_obs = []
        self._sigma_hx = []

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
        """
        # Generate initial state Ensemble
        self.state_vec_prior, self.model_obs = self.dyn_model.generate_ensemble()
        self.state_vec_analysis = self.state_vec_prior.copy()
        # sensitivity only
        if self._sensitivity_only:
            next_end_time = 2.0 * self.da_interval + self.time_array[0]
            self.state_vec_forecast, self.model_obs = self.dyn_model.forecast_to_time(
                self.state_vec_analysis, next_end_time)
            if self.ver>=1:
                print "\nSensitivity study completed."
            sys.exit(0)
        # main DA loop
        early_stop = False
        for time in self.time_array:
	    pdb.set_trace()
            self.time = time
            next_end_time = 2.0 * self.da_interval + self.time
            self.da_step = (next_end_time - self.da_interval) / \
                self.da_interval
            if self._verb>=1:
                print("\nData Assimilation step: {}".format(self.da_step))
            # dyn_model: propagate the state ensemble to next DA time
            # and get observations at next DA time.
            self.state_vec_forecast, self.model_obs = self.dyn_model.forecast_to_time(
                self.state_vec_analysis.copy(), next_end_time)
            self.obs, self.obs_error = self.dyn_model.get_obs(next_end_time)
            # data assimilation
            self._correct_forecasts()
            self._calc_misfits()
            debug_dict = {
                'Xf':self.state_vec_forecast, 'Xa':self.state_vec_analysis, 'HX':self.model_obs, 
                'obs':self.obs, 'R':self.obs_error}
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
            np.savetxt(self._save_folder + os.sep + '_sigma_obs', np.array(
                self._sigma_obs))
            np.savetxt(self._save_folder + os.sep + 'sigma_HX', np.array(
                self._sigma_hx))
        try: self.dyn_model.save()
        except: pass

    # private methods
    def _correct_forecasts(self):
        """ Correct the propagated ensemble (filtering step).

        **Updates:**
            * self.state_vec
        """
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
        """ Calculate the misfit.

        **Updates:**
            * self._misfit_x_l1norm
            * self._misfit_x_l2norm
            * self._misfit_x_inf
            * self._sigma_hx
            * self._misfit_max
            * self._sigma_obs
        """

        # calculate misfits
        nnorm = self.dyn_model.nstate_obs * self.nsamples
        # TODO: Why are we dividing the norm by this? And why not the inf norm?
        diff = abs(self.obs - self.model_obs)
        # np.sum(diff) / nnorm
        # np.sqrt(np.sum(diff**2.0)) / nnorm
        misfit_l1norm = la.norm(diff, 1) / nnorm
        misfit_l2norm = la.norm(diff, 2) / nnorm
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
        self._sigma_obs.append(sigma)

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
        """ Report at each iteration. """
        (iterative_residual, conv1, sigma_inf, conv2) = info

        str_report = '\n' + "#"*80
        str_report += "\nStandard deviation of ensemble: {}".format(
            self._sigma_hx[self.iter]) + \
            "\nStandard deviation of observation error: {}".format(
            self._sigma_obs[self.iter])
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

# child classes (specific filtering techniques)

class EnKF(DAFilter2):
    """ Implementation of the ensemble Kalman Filter (EnKF).
    """
    # TODO: add more detail to the docstring. E.g. what is EnKF.

    def __init__(
        self, nsamples, da_interval, t_end, forward_model, input_dict
        ):
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
            * self.state_vec
        """

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
        xp = _mean_subtracted_matrix(self.state_vec_forecast)
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
        self.state_vec_analysis = self.state_vec_forecast + dx
        # debug
        debug_dict = {
            'K':kalman_gain_matrix, 'inv':inv, 'HPHT':hpht, 'PHT':pht,
            'HXP':hxp, 'XP':xp}
        self._save_debug(debug_dict)


# ensemble randomized maximum likelihood techniques (child classes)
class EnRML(DAFilter2):
    """ Implementation of Ensemble Randomized Maximum Likelihood(EnRML).

    """

    def __init__(
        self, nsamples, da_interval, t_end, forward_model, input_dict
        ):
        """ Parse input file and assign values to class attributes.

        See DAFilter2.__init__ for details.
        """
        super(self.__class__, self).__init__(
            nsamples, da_interval, t_end, forward_model, input_dict)
        self.name = 'Ensemble Randomized Maximum Likelihood'
        self.short_name = 'EnRML'

    def _correct_forcast(self):

        # analysis step
        dx = - GNGainMatrix.dot(self.model_obs-self.obs) + penalty
        self.state_vec_analysis = self.beta*self.state_vec_prior + (1.0-\
                            self.beta)*self.state_vec_forecast + self.beta*dx


    def solve(self):
        """ Solves the parameter estimation problem.
        """
        # Generate initial state Ensemble
        (self.Xa, self.HX) = self.dynModel.generateEnsemble()
        if(self._sensitivityOnly):
            print "Sensitivity study completed."
            sys.exit(0)
        ii = 0
        self.Xpr = self.Xa
        Obs = self.dynModel.Observe(self.DAInterval)

        for t in self.T:
            nextEndTime = 2 * self.DAInterval + t
            DAstep = (nextEndTime - self.DAInterval) / self.DAInterval
            print "#######################################################################"
            print "\n Data Assimilation step = ", DAstep, "\n"   
            
            # propagate the state ensemble to next DA time
            self.Xf, self.HX = self.dynModel.forecastToTime(self.Xa, nextEndTime)
            if (self._iDebug):
                np.savetxt(self._debugFolderName+'X_'+ str(DAstep) + '.txt', self.Xf)
                np.savetxt(self._debugFolderName+'HX_'+ str(DAstep) + '.txt', self.HX)
            
            # correct the propagated results
            self.Xa, sigmaHXNorm, misfitMax = self._correctForecasts(self.Xpr, self.Xf, self.HX,Obs, nextEndTime)
            #Check iteration convergence and report misfits (Todo:move to report function)                      
            Robs = self.dynModel.get_Robs()
            sigmaInf = np.max(np.diag(np.sqrt(Robs)));
            sigma = ( LA.norm(np.sqrt(np.diag(Robs))) / self.dynModel.NstateObs)
            print "Std of ensemble = ", sigmaHXNorm
            print "Std of observation error", sigma
            self.obsSigma.append(sigma)
            self.obsSigmaInf.append(sigmaInf)
            if  misfitMax > sigmaInf:
                print "infinit misfit(", misfitMax, ") is larger than Inf norm of observation error(", sigmaInf, "), so iteration continuing\n\n"
            else:
                print "infinit misfit of ensemble reaches Inf norm of observation error, considering converge\n\n"
                self.misfitX_L1 = np.array(self.misfitX_L1)
                self.misfitX_L2 = np.array(self.misfitX_L2)
                self.misfitX_inf = np.array(self.misfitX_inf)
                self.obsSigma = np.array(self.obsSigma)
            if ii > 0:
                iterativeResidual = (abs(self.misfitX_L2[ii] - self.misfitX_L2[ii-1])) /abs(self.misfitX_L2[0])    
                print "relative Iterative residual = ", iterativeResidual
                print "relative Convergence criterion = ", self.convergenceResi 
        
                if iterativeResidual  < self.convergenceResi:
                    if self.reachmaxiteration:
                        pass
                    else:
                        print "Iteration converges, finished \n\n"
                        break
                
                np.savetxt('./misfit_L1.txt', self.misfitX_L1)
                np.savetxt('./misfit_L2.txt', self.misfitX_L2)
                np.savetxt('./misfit_inf.txt', self.misfitX_inf)
                np.savetxt('./obsSigma.txt', self.obsSigma)
                np.savetxt('./obsSigmaInf.txt', self.obsSigmaInf)
                if self.constbeta:
                    print "Use constont step length beta = " ,self.beta
                    pass
                else:
                    print "Use self-adjusting step length \n"
                    if self.misfitX_L2[ii] < self.misfitX_L2[ii-1]:
                        self.beta = 1.2*self.beta
                        print "Iteration Converging, increase step length beta = ", self.beta
                    else:
                        self.beta = 0.8*self.beta
                        self.Xa = self.Xf
                        print "Iteration Diverging, reduce step length beta = ", self.beta
            self.steplength.append(self.beta)
            ii = ii + 1
        # Save misfits
        self.misfitX_L1 = np.array(self.misfitX_L1)
        self.misfitX_L2 = np.array(self.misfitX_L2)
        self.misfitX_inf = np.array(self.misfitX_inf)
        self.obsSigma = np.array(self.obsSigma)
        self.sigmaHX = np.array(self.sigmaHX)
        self.steplength = np.array(self.steplength)

        np.savetxt('./misfit_L1.txt', self.misfitX_L1)
        np.savetxt('./misfit_L2.txt', self.misfitX_L2)
        np.savetxt('./misfit_inf.txt', self.misfitX_inf)
        np.savetxt('./obsSigma.txt', self.obsSigma)
        np.savetxt('./sigmaHX.txt', self.sigmaHX)
        np.savetxt('./steplength.txt',self.steplength)

    def report(self):
        """ Report summary information at each step
        """
        raise NotImplementedError

    def clean(self):
        """ Call the dynamic model to do any necessary cleanup before exiting.
        """
        self.dynModel.clean()

    ############################ Priviate function ################################    
    def _correctForecasts(self, Xpr, X, HX, Obs,nextEndTime):

        """ Filtering step: Correct the propagated ensemble X 
            via EnRML filtering procedure
            
            Arg:
            Xpr: Prior state ensemble matrix (Nstate by Ns)
            X: state ensemble matrix (Nstate by Ns)
            HX: state ensemble matrix projected to observed space
            Obs: Observation
            nextEndTime: next DA time spot
            
            Return:
            X: state ensemble matrix
        """
        DAstep = (nextEndTime - self.DAInterval) / self.DAInterval
        # get XMean and tile XMeanVec to a full matrix (Nsate by Ns)
        XMeanVec = np.mean(X, axis=1) # vector mean of X
        XMeanVec = np.array([XMeanVec])
        XMeanVec = XMeanVec.T
        XMean = np.tile(XMeanVec, (1, self.Ns))
        # get prior XMean and tile XMeanVec to a full matrix (Nsate by Ns)
        XprMeanVec = np.mean(Xpr, axis=1) # vector mean of X
        XprMeanVec = np.array([XprMeanVec])
        XprMeanVec = XprMeanVec.T
        XprMean = np.tile(XprMeanVec, (1, self.Ns))
        # get coveriance matrix
        XP0 = Xpr - XprMean
        XP =  X - XMean
        
        (GNGainMatrix, penalty) = self.dynModel.getGaussNewtonVars(HX, X,Xpr,XP, XP0, Obs,nextEndTime)

        # analyze step
        Robs = self.dynModel.get_Robs()
        Cdd = np.diag(np.diag(np.sqrt(Robs)))
        #dX = - np.dot(Cdd.dot(LA.inv(Hess)),(X-Xpr)) - GNGainMatrix.dot(HX-Obs)
        dX = - GNGainMatrix.dot(HX-Obs) + penalty
        X = self.beta*Xpr + (1.0-self.beta)* X+ self.beta*dX
        if (self._iDebug):
            np.savetxt(self._debugFolderName+'dX_'+str(DAstep), dX)
            np.savetxt(self._debugFolderName+'updateX_'+str(DAstep), X) 
            np.savetxt(self._debugFolderName+'Obs_'+str(DAstep), Obs)  
        
        # calculate misfits
        Nnorm = self.dynModel.NstateObs * self.Ns
        diff = abs(Obs - HX)
        misFit_L1 = np.sum(diff) / Nnorm
        misFit_L2 = np.sqrt(np.sum(diff**2)) / Nnorm
        misFit_Inf = LA.norm(diff, np.inf)                
        misFitMax = np.max([misFit_L1, misFit_L2, misFit_Inf])
        sigmaHX = np.std(HX, axis=1)
        sigmaHXNorm = LA.norm(sigmaHX) / self.dynModel.NstateObs
        
        self.misfitX_L1.append(misFit_L1)
        self.misfitX_L2.append(misFit_L2)
        self.misfitX_inf.append(misFit_Inf)
        self.sigmaHX.append(sigmaHXNorm)
        
        print "\nThe misfit between predicted QoI with observed QoI is:"
        print "L1 norm = ", misFit_L1,", \nL2 norm = ", misFit_L2, "\nInf norm = ", misFit_Inf, "\n\n"
    
        return X, sigmaHXNorm, misFitMax
        

 # ensemble kalman filtering -Multi Data assimilation techniques (child classes)

