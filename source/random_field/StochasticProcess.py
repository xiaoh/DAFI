#!/usr/bin/env python

# description        :Perform KL expansion and related field reconstruction.

# author             :Jian-Xun Wang (vtwjx@vt.edu)
# copyright          :Heng Xiao's Group
# date               :Oct.27, 2015
# revision           :Nov.01, 2015

####################################################################################################

## Import system modules
# sci computing
import numpy as np
import scipy.sparse as sp
# system, file operation
import time
import pdb
# plotting
#import seaborn as sns  # for statistical plotting
import matplotlib.pyplot as plt  # for plotting

## Import local modules

class GaussianProcess:
    """Construct a Gaussian process.

    Description:
        Handle 3D spatially Gaussian process.


    Args:
        xState          :xState = (x1; x2; ..., xn), [n by d]
                         xi is d dimension 
                         e.g. xi = (x, y, z). (xState is spatial coordinate)        

    """
    
    def __init__(self, xState):
        print "Generate a Gaussian Random Field (Gaussian Process)"
        self.xState = xState    # matrix of states
        [self.n, self.d] = xState.shape # number of state and dimension of state
        if self.d == int(3):
            print "Now we are working on a 3D spatial problem"
        elif self.d == int(2):
            print "Now we are working on a 2D spatial problem"
        elif self.d == int(1):
            print "Now we are working on a 1D spatial problem"
        else:
            print "Error: we can only handle 1D, 2D and 3D problems, please check xState is n by (1, 2, or 3)"
            exit(1)
        
    def covGen(self, Arg_covGen):
        """generate covariance matrix for xState.

        Description:
            generate covariance matrix, which can be truncated by ignoring very small
            value (truncateTol) and then saved as a sparse matrix


        Args:
            Arg_covGen      : dictionary contains
                "sigmaField"    : array of sigma (std amplitude)
                "lenXField"     : array of x- length scale
                "lenYField"     : array of y- length scale
                "lenZField"     : array of z- length scale
                "weightField"   : array of weight
                "truncateTol"   : tolerance for truncate small covariance to be 0
                                  default value is truncateTol = 1e-16
        return:
            cov_sparse          : sparse covariance matrix (after truncation)
            cov_sparseWeighted  : sparse covariance matrix (after truncation)
        
        """
        # TODO the parsing argument below need to be move to KLReducedModel class
        # parse the arguments
        # if not Arg_covGen.has_key('nonstationarySigmaFlag'):
            # nonstationarySigmaFlag = False
            # print 'warning: no defined nonstationarySigmaFlag, default is stationary: constant sigma'
        # else:
            # nonstationarySigmaFlag = Arg_covGen['nonstationarySigmaFlag']
            # assert type(nonstationarySigmaFlag).__name__ == 'bool', \
            # "nonstationarySigmaFlag should be True or False only!"        
           # 
        # if not Arg_covGen.has_key('nonstationaryLenFlag'):
            # nonstationaryLenFlag = False
            # print 'warning: no defined nonstationaryLenFlag, default is stationary: constant lenth scale'
        # else:
            # nonstationaryLenFlag = Arg_covGen['nonstationaryLenFlag']
            # assert type(nonstationaryLenFlag).__name__ == 'bool', \
            # "nonstationaryLenFlag should be True or False only!"
        if not 'kernelType' in Arg_covGen:
            kernelType = 'SqExp'
            
        kernelType = Arg_covGen['kernelType']
        ## parse the arguments
        if kernelType=='SqExp':
            # x- length scale array
            if not Arg_covGen.has_key('lenXField'):
                print 'Error: You must give me x- length scale Field array'
                exit(1)
            else:
                lenXField = Arg_covGen['lenXField']
                lenXField = np.array([lenXField])
            # y- length scale array
            if self.d == 2 or self.d == 3:
                if not Arg_covGen.has_key('lenYField'):
                    print 'Error: You must give me y- length scale Field array'
                    exit(1)
                else:
                    lenYField = Arg_covGen['lenYField']
                    lenYField = np.array([lenYField])
            # z- length scale array
            if self.d == 3:
                if not Arg_covGen.has_key('lenZField'):
                    print 'Error: You must give me z- length scale Field array'
                    exit(1)
                else:
                    lenZField = Arg_covGen['lenZField']
                    lenZField = np.array([lenZField])
        elif kernelType=='givenStructure':
            CovStruct = Arg_covGen['CovStruct']
        else:
            print "This kernel type is not supported currently!"
            exit(1)

        # sigmaField array
        print "start to generate a Gaussian covariance matrix (cov and weighted cov)"
        if not Arg_covGen.has_key('sigmaField'):
            print 'Use unit variance for sigma field'
            if Arg_covGen.has_key('lenXField'):
                sigmaField = np.ones(lenXField.shape)
            elif kernelType=='givenStructure':
                sigmaField = np.ones((1, CovStruct.shape[0]))
        else:
            sigmaField = Arg_covGen['sigmaField']
            sigmaField = np.array([sigmaField])
        
        # weight array
        if not Arg_covGen.has_key('weightField'):
            print 'Error: You must give me weight Field array'
            exit(1)
        else:
            weightField = Arg_covGen['weightField']
            weightField = np.array([weightField])

        # torelance for truncating the covariance matrix to be sparse
        if not Arg_covGen.has_key('truncateTol'):
            truncateTol = -np.log(1e-3)
            print 'warning: no defined truncateTol for truncating covariance, default tolerance =', truncateTol
        else:
            truncateTol = Arg_covGen['truncateTol']                               

        # sigma[i, j] matrix
        SIGMA = np.dot(sigmaField.T, sigmaField); SIGMA = np.sqrt(SIGMA)
        # weight matrix
        W = np.dot(weightField.T, weightField); W = np.sqrt(W)

        if kernelType=='SqExp':
            LenX = np.dot(lenXField.T, lenXField); LenX = np.sqrt(LenX)
            LenY = None
            LenZ = None
            if self.d >= 2:
                LenY = np.dot(lenYField.T, lenYField); LenY = np.sqrt(LenY)
            if self.d >= 3:
                LenZ = np.dot(lenZField.T, lenZField); LenZ = np.sqrt(LenZ)
            args = (LenX, LenY, LenZ)
        elif kernelType=='givenStructure':
            args = (CovStruct,)

        [cov_sparse, covWeighted_sparse] = self._kernel(kernelType, args, SIGMA, W, truncateTol)
        return cov_sparse, covWeighted_sparse

    def _kernel(self, kernelType, args, SIGMA, W=None, truncateTol=6.9):
        """define the kernel function .
        NOTE(CM): Modified to also accept a given covariance structure. Need to update this description.

        Description:
            this kernel function can handle non-stationary GP, where the std amplitude
            SIGMA and LenX, LenY and LenZ can be spatially different


        Args:
            SIGMA            : SIGMA[i, j] = (sigmaField[i] * sigmaField[j])^0.5
            LenX             : LenX[i, j] = (lenXField[i] * lenXField[j])^0.5
            LenY             : LenY[i, j] = (lenYField[i] * lenYField[j])^0.5
            LenZ             : LenZ[i, j] = (lenZField[i] * lenZField[j])^0.5
            Corr             : Correlation matrix 


        return:
            cov_sparse          : covariance (sparse matrix)
            covWeighted_sparse  : weighted covariance (sparse matrix)
        """
        if kernelType=='SqExp':
            LenX = args[0]
            LenY = args[1]
            LenZ = args[2]
            # alpha = 2 specifying a Squre exponential kernel
            alpha = 2
            # 1-D
            if self.d == 1:
                [X, XPrime] = np.meshgrid(self.xState[:, 0], self.xState[:, 0])
                mLcov =  (X - XPrime)**alpha / (LenX**alpha)
            # 2-D
            elif self.d ==2:
                [X, XPrime] = np.meshgrid(self.xState[:, 0], self.xState[:, 0])
                [Y, YPrime] = np.meshgrid(self.xState[:, 1], self.xState[:, 1])
                mLcov =  (X - XPrime)**alpha / (LenX**alpha) + \
                         (Y - YPrime)**alpha / (LenY**alpha)
            # 3-D
            elif self.d == 3:
                [X, XPrime] = np.meshgrid(self.xState[:, 0], self.xState[:, 0])
                [Y, YPrime] = np.meshgrid(self.xState[:, 1], self.xState[:, 1])
                [Z, ZPrime] = np.meshgrid(self.xState[:, 2], self.xState[:, 2])
                mLcov =  (X - XPrime)**alpha / (LenX**alpha) + \
                         (Y - YPrime)**alpha / (LenY**alpha) + \
                         (Z - ZPrime)**alpha / (LenZ**alpha)
            else:
                print "Error: we can only handle 1D, 2D and 3D problems, please check xState is n by (1, 2, or 3)"
                exit(1)
            CovStruct = np.exp(-mLcov)
            indicatorM = mLcov < truncateTol
        elif kernelType=='givenStructure':
            CovStruct = args[0]
            indicatorM = CovStruct > np.exp(-truncateTol)

        # For validation purpose
        #np.savetxt('./CovStruct.dat',CovStruct)
        #np.savetxt('./indicatorM.dat',indicatorM)
        
        # logical matrix
        indicatorM = indicatorM.astype(float)
        # truncate the covariance
        SIGMA = SIGMA * indicatorM
        W = W * indicatorM
        # calculate cov and weightedd cov
        cov = (SIGMA**2) * CovStruct
        covWeighted = cov * W
        # convert dense cov to sparse cov
        cov_sparse = sp.coo_matrix(cov)
        covWeighted_sparse = sp.coo_matrix(covWeighted)
        return cov_sparse, covWeighted_sparse

if __name__ == '__main__':
    # Read test data
    testDir = './verificationData/klExpansion/cavity16/' # directory where the test data stored
    
    xState = np.loadtxt(testDir + 'cellCenter3D.dat')
    sigmaField = np.loadtxt(testDir + 'cellSigma3D.dat')
    lenXField = np.ones(sigmaField.shape)
    lenYField = np.ones(sigmaField.shape)
    lenZField = np.ones(sigmaField.shape)
    weightField = np.loadtxt(testDir + 'cellArea3D.dat')
    Arg_covGen = {
                    'sigmaField': sigmaField,
                    'lenXField': lenXField,
                    'lenYField': lenYField,
                    'lenZField': lenZField,
                    'weightField':weightField
                 }
                 
    gp = GaussianProcess(xState) # initial a instance of GaussianProcess class
    [cov_sparse, covWeighted_sparse] = gp.covGen(Arg_covGen)
    pdb.set_trace()    
    
            
