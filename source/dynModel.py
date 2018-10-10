# Parent class for dynamic model classes.

import numpy as np

class dynModel:
    ''' Parent class for dynamic model classes.
        Required attributes and methods are listed below.

        definitions:
            Ns: ensemble size. [int]
            Nstate: number of states in model
            NstateObs: number of observation states
            DAInterval: time/iteration interval between data assimilation steps. [float]
            Tend: final time. [float]
            ModelInput: input file name. [string]
            X: ensemble matrix of whole states. [Nstate x Ns]
            HX: ensemble matrix of whole states in observation space. [NstateObs x Ns]
            t: next end time. [float]
            Obs: observations. [NstateObs x Ns]
            Robs: observation error/covariance matrix. [NstateObs x NstateObs]
            K: Kalman gain matrix. [Nstate x NstateObs]
            XP: . [Nstate x Ns]

        attributes:
            Nstate
            NstateObs

        methods:
            X, HX  = generateEnsemble()
            X, HX  = forecastToTime(X, t)
            Obs, K = getBackgroundVars(HX, XP, t)
            Robs   = get_Robs()
            clean()
    '''

    def __init__(self, Ns, DAInterval, Tend,  ModelInput):
        ''' Initialize dynamic model. Parse input file.
        '''
        self.Nstate = 0
        self.NstateObs = 0
        self._Ns = Ns
        pass

    def __str__(self):
        s = 'An empty dynamic model.'
        return s
        
    def __repr__(self):
        return self.__str__()

    def generateEnsemble(self):
        ''' Returns states at the first DA time-step.
        '''
        raise NotImplementedError
        X = np.zeros([self.Nstate, self._Ns])
        HX = np.zeros([self.NstateObs, self._Ns])
        return X, HX

    def forecastToTime(self, X, t):
        ''' Returns states at the next end time.
        '''
        raise NotImplementedError
        X = np.zeros([self.Nstate, self._Ns])
        HX = np.zeros([self.NstateObs, self._Ns])
        return X, HX

    def getBackgroundVars(self, HX, XP, t):
        ''' Returns the observation matrix and Kalman gain matrix.
        '''
        raise NotImplementedError
        Obs = np.zeros([self.NstateObs, self._Ns])
        K = np.zeros([Nstate, NstateObs])
        return Obs, K

    def get_Robs(self):
        ''' Returns the observation error/covariance matrix.
        '''
        raise NotImplementedError
        Robs = np.zeros([self.NstateObs, self.NstateObs])
        return Robs

    def clean(self):
        ''' Perform any necessary cleanup before exiting.
        '''
        pass
