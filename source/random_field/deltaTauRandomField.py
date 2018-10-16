#!/usr/bin/env python

# description        :Generate GP for delta Tau field.

# author             :Jian-Xun Wang (vtwjx@vt.edu)
# copyright          :Heng Xiao's Group
# date               :Nov.01, 2015
# revision           :Nov.01, 2015

####################################################################################################

## Import system modules
# sci computing
import numpy as np
import scipy.sparse as sp
import scipy.stats as ss
import numpy.linalg as LA
# system, file operation
import pdb
import time
import os
# plotting
import matplotlib.pyplot as plt  # for plotting

## Import local modules
import StochasticProcess as StP
import KLExpansion as KL

class randomField:
    """Generate a random field (Kernel generation, KL Expansion, reconstruction of field).

    Description:
        XXX
    
    Arg:
        XXX          :XXX     

    """

    def __init__(self, coord, Arg_covGen, Arg_calModes, instanceName):
        # parse input
        self.coord = coord    # matrix of coordinates
        [self.n, self.d] = coord.shape # number of state and dimension of state
        if self.d not in {int(1), int(2), int(3)}:
            print "Error: we can only handle 1D, 2D and 3D problems, please check xState is n by (1, 2, or 3)"
            exit(1)
        self.Arg_covGen = Arg_covGen # arguments for non-stationary covariance generation

        self.Arg_calModes = Arg_calModes # arguments for KL Modes
        
        self.rfFolderName = 'randomData_'+instanceName 
        if not os.path.exists(self.rfFolderName):
            print "creat klExpansionData3D folder for the data related with KL"    
            os.system('mkdir ' + self.rfFolderName)
                        
    def KLExpansion(self):
        """calculate the eigenvalues and KL modes (normalized eigen-vectors)

        Args:
            Arg_calKLModes  :arguments for calKLModes:
                nKL         :number of modes truncated
                weightField :weightField array (N by 1)

        Return:
            KLModes         :sqrt(eigenvalues_i) * eigenvectors/weight (N by nKL)    

        """            
        tic = time.time()
        # initial a GP class
        gp = StP.GaussianProcess(self.coord)
        # generate covariance for the gp
        cov_sparse, covWeighted_sparse = gp.covGen(self.Arg_covGen)
        toc = time.time()
        print "elapse time for generating covariance matrix = ", toc - tic
        # initial a kl class
        tic = time.time()
        self.kl = KL.klExpansion(cov_sparse, covWeighted_sparse)
        # perform KL expansion
        [eigVal, KLModes] = self.kl.calKLModes(self.Arg_calModes)
        np.savetxt(self.rfFolderName+'/KLModes.dat', KLModes)
        toc = time.time()
        print "elapse time for generate modes = ", toc - tic
        
        return KLModes 
       
    def uncorrUniRandom(self, n, distType = "Gaussian"):
        """
        Generate Uncorrelated random variables (Vector) with unit variance
        
        Args:
            n:          length of the vector (1 by 1 scalar)
            distType:   type of distribution (Char) 
                        (default is normal distribution)
        Return:
            RandomVec: vector with uncorrelated random variables (N by 1) 
        """ 
        
        # Default distribution type
        #np.random.seed(1000);
        if distType == "Gaussian":
            RandomVec = np.random.randn(n, 1)
        
        return RandomVec     
        
    def reconstructField(self, omegaVec, KLModes):
        """reconstruct a random field with truncated KL modes

        Args:            
            omegaVec    :the coefficients for KL modes (nKL by 1)
            KLModes     :sqrt(eigenvalues_i) * eigenvectors/weight (N by nKL) 
        Return:
            recField    :reconstructed field (N by 1)

        """
        self.kl = KL.klExpansion()
        recField = self.kl.reconstructField_Reduced(omegaVec, KLModes)
        
        return recField                
