# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Dynamic model for solving the Lorenz attractor. """

# standard library imports
import ast
import sys

# third party imports
import numpy as np
from numpy import linalg as la
import scipy.sparse as sp
from scipy.integrate import ode

# local import
from dainv.dyn_model import DynModel
from dainv.utilities import read_input_data

class Solver(DynModel):
    """
        Dynamic forward model: Lorenz 63
        The state variable include: x, y, z
        The parameters need to be augmented: coefficients for rho, beta, sigma
        The Observation include: x, y, z
    """
    #TODO: Fix docstrings

    def __init__(self, Ns, DAInterval, Tend,  ModelInput):
        """
            Initialization
        """
        ## Extract forward Model Input parameters
        paramDict = read_input_data(ModelInput)

        self.name = 'Lorenz63'
        self.Ns = Ns
        self.DAInterval = DAInterval

        self.Tend = Tend
        self.caseName = paramDict['caseName']
        #Hyperparameters
        self.DtInterval = float(paramDict['DtInterval'])
        self.x = float(paramDict['x'])
        self.y = float(paramDict['y'])
        self.z = float(paramDict['z'])
        self.rho = float(paramDict['rho'])
        self.beta = float(paramDict['beta'])
        self.sigma = float(paramDict['sigma'])
        self.ObsSigma = float(paramDict['ObsSigma'])

        self.xsigma = float(paramDict['xsigma'])
        self.ysigma = float(paramDict['ysigma'])
        self.zsigma = float(paramDict['zsigma'])
        self.rhosigma = float(paramDict['rhosigma'])
        # switch control which components are perturbed
        self.perturbx = ast.literal_eval(paramDict['perturbx'])
        self.perturby = ast.literal_eval(paramDict['perturby'])
        self.perturbz = ast.literal_eval(paramDict['perturbz'])
        self.perturbrho = ast.literal_eval(paramDict['perturbrho'])
        self.perturbbeta = ast.literal_eval(paramDict['perturbbeta'])
        self.perturbsigma = ast.literal_eval(paramDict['perturbsigma'])
        Nsvariate =  4 #np.sum(
                            #self.perturbx, self.perturby,self.perturbz
                            #,self.perturbrho, self.perturbbeta,
                            #self.perturbsigma
                           #)

        self.nstate = Nsvariate
        self.nstate_obs = 3


        # specify initial condition
        self.x_init = [self.x, self.y, self.z, self.rho]

    def __str__(self):
        s = 'An empty dynamic model.'
        return s

    def generate_ensemble(self):
        """ Returns states at the first DA time-step. """

        X = np.zeros([self.nstate, self.Ns])
        HX = np.zeros([self.nstate_obs, self.Ns])
        dxRelStd = 0.1

        for iDim in np.arange(self.nstate):
            dxStd = dxRelStd * self.x_init[iDim]
            X[iDim, :] = self.x_init[iDim] + np.random.normal(0, dxStd, self.Ns)
        '''
        X[0,:] = np.random.normal(self.x_init[0],self.xsigma,self.Ns)
        X[1,:] = np.random.normal(self.x_init[1],self.ysigma,self.Ns)
        X[2,:] = np.random.normal(self.x_init[2],self.zsigma,self.Ns)
        X[3,:] = np.random.normal(self.x_init[3],self.rhosigma,self.Ns)
        '''
        H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).T
        HX = H.dot(X)
        return X, HX

    def Lorenz63(self, t, x):
        """ Define Lorenz 63 system """
        # ODEs
        dx = np.zeros([4, 1])
        dx[0] = self.sigma * (-x[0] + x[1])
        dx[1] = x[3] * x[0] - x[1] - x[0] * x[2]
        dx[2] = x[0] * x[1] - self.beta * x[2]
        dx[3] = 0
        return dx

    def forecast_to_time(self, X, next_end_time):
        """ Returns states at the next end time. """
        newStartTime = next_end_time - self.DAInterval
        timeSeries = np.arange(newStartTime + self.DtInterval, next_end_time + self.DtInterval, self.DtInterval)
        self.solver = ode(self.Lorenz63)
        self.solver.set_integrator('dopri5')
        for i in range(self.Ns):
            self.solver.set_initial_value(X[:,i], newStartTime)

            x = np.empty([len(timeSeries), self.nstate])
            for k in np.arange(len(timeSeries)):
                if not self.solver.successful():
                    print "solver failed"
                    sys.exit(1)
                self.solver.integrate(timeSeries[k])
                x[k] = self.solver.y
            X[:,i] = x[-1]
        H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).T
        HX = H.dot(X)

        return X, HX

    def get_obs(self, next_end_time):
        DAstep = (next_end_time - self.DAInterval) / self.DAInterval
        obs = self.Observe(next_end_time)
        R_obs = self.get_Robs()
        return obs, R_obs

    # def getBackgroundVars(self, HX, XP, next_end_time):
    #     """ Function is to generate observation and get kalman Gain Matrix
    #
    #     Args:
    #     HX: ensemble matrix of whole state in observation space
    #     P: covariance matrix of ensemble
    #     next_end_time: next DA interval
    #
    #     Returns:
    #     Obs: state matrix of observation
    #     KalmanGainMatrix
    #     """
    #     DAstep = (next_end_time - self.DAInterval) / self.DAInterval
    #     Obs = self.Observe(next_end_time)
    #     pdb.set_trace()
    #     H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).T
    #
    #     HXP = H.dot(XP)
    #     PHT = (1.0 / (self.Ns - 1.0)) * np.dot(XP, HXP.T)
    #     HPHT = (1.0 / (self.Ns - 1.0)) * HXP.dot(HXP.T)
    #     conInv = la.cond(HPHT + self.Robs)
    #     print "conditional number of (HPHT + R) is " + str(conInv)
    #
    #     if (conInv > 1e16):
    #         print "!!! warning: the matrix (HPHT + R) are singular, inverse would be failed"
    #     INV = la.inv(HPHT + self.Robs)
    #     INV = INV.A #convert np.matrix to np.ndarray
    #     KalmanGainMatrix = PHT.dot(INV)
    #     pdb.set_trace()
    #     return Obs, KalmanGainMatrix

    def get_Robs(self):
        """ Return the observation covariance. """
        return self.Robs.todense()

    def clean(self):
        """ Perform any necessary cleanup before exiting. """
        pass

    def Observe(self, next_end_time):
        """ Function is to get observation Data from experiment

        Arg:

        Returns:
        Obs: observation matrix
        """
        DAstep = (next_end_time - self.DAInterval) / self.DAInterval
        # timeSeries = np.arange(0., self.Tend+self.DAInterval, self.DAInterval)
        # solver = ode(Lorenz63)
        # solver.set_integrator('dopri5')
        # solver.set_initial_value(x_init, 0.025)
        # # solve ode
        # obs = np.empty([len(timeSeries), 4])
        # obs[0] = [1. 1.2, 1, 28]
        #
        # k = 1
        # while solver.successful() and solver.t < self.Tend:
        #     solver.integrate(timeSeries[k])
        #     obs[k] = solver.y
        #     k += 1
        # np.savetxt('obs.txt',obs)
        if DAstep == 1.0:
            Obsvec = np.loadtxt('obs.txt')[int(DAstep)*10,1:-1]
        else:
            Obsvec = np.loadtxt('obs.txt')[int(DAstep)*10,1:-1]

        obs = np.empty(self.nstate_obs)
        dxStdVec = np.empty(self.nstate_obs)

        for iDim in np.arange(self.nstate_obs):
            dxStd = self.ObsSigma * np.abs(Obsvec[iDim])
            dxStdVec[iDim] = dxStd
            obs[iDim] = Obsvec[iDim] + np.random.normal(0, dxStd, 1)
        obsM = np.empty([self.nstate_obs, self.Ns])
        for i in np.arange(self.Ns):
            obsM[:, i] = obs
        self.Robs = sp.diags(dxStdVec**2,0)
        return obsM
