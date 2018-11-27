# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Perform KL expansion and related field reconstruction. """

# third party imports
import numpy as np
import scipy.sparse as sp
import numpy.linalg as la


class klExpansion:
    """Perform KL expansion and related field reconstruction.

    Description:
        Perform Karhunen-Loeve expansion to represent of a stochastic process (random field).
        KLModes: \sqrt{\lambda_i} * f(x)_i; (\lambda_i, f(x)_i) is an eigenpair.
        field: \sum_{i=1}^{nKL}{omega[i] * KLModes[i]}

    Arg:
        cov          :covariance matrix of stochastic process (N by N)
        covWeihted   :weighted covariance matrix of stochastic process (W*cov*W). (N by N)
        meanField    :array of mean value of stochastic process (default is zero mean) (N by 1)
    Notes: N is number of cells
    """
    # TODO: convert docstring to Numpy format. Follow same format as rest of source code. Check info correct. Improve.
    # TODO: Same for all the methods below.
    # TODO: Variable names + other pep8. General Cleanup.

    def __init__(self, cov=None, covWeighted=None, meanField=None):

        # parse input
        if cov is not None:
            if type(cov).__name__ != 'coo_matrix':
                self.cov = sp.coo_matrix(cov)
                print "klExpansion: The covariance not sparse, converting covariance matrix to sparse (coo_matrix)"
            else:
                self.cov = cov

            if type(covWeighted).__name__ != 'coo_matrix':
                self.covWeighted = sp.coo_matrix(covWeighted)
                print "klExpansion: converting weighted covariance matrix to sparse (coo_matrix)"
            else:
                self.covWeighted = covWeighted

        if meanField is None:
            # print "Zero mean is adopted for random field construction"
            pass
        else:
            print "Non-zero mean is adopted for random field construction"
        self.meanField = meanField

    def calKLModes(self, Arg_calKLModes):
        """calculate the eigenvalues and KL modes (normalized eigen-vectors)

        Args:
            Arg_calKLModes  :arguments for calKLModes:
                nKL         :number of modes truncated
                weightField :weightField array (N by 1)

        Return:
            eigVals         :eigenvalues (trucated)
            KLModes         :sqrt(eigenvalues_i) * eigenvectors/weight (N by nKL)

        """
        # parse arguments for calKLModes
        if 'nKL' not in Arg_calKLModes:
            print 'Error: Please defined nKL (number of modes truncated)!'
            exit(1)
        else:
            nKL = Arg_calKLModes['nKL']
            self.nKL = nKL

        if 'weightField' not in Arg_calKLModes:
            print 'Error: You must give me weight Field array'
            exit(1)
        else:
            weightField = Arg_calKLModes['weightField']
        # calculate trace of covWeighted
        covTrace = sum(self.covWeighted.diagonal())
        # perform the eig-decomposition
        # solving cov * eigVecs[i] = eigVal[i] * eigVecs[i]
        eigVals, eigVecs = sp.linalg.eigsh(self.covWeighted, k=nKL)
        # sort the eig-value and eig-vectors in a descending order
        ascendingOrder = eigVals.argsort()
        descendingOrder = ascendingOrder[::-1]
        eigVals = eigVals[descendingOrder]
        eigVecs = eigVecs[:, descendingOrder]
        # weighted eigVec
        W = np.diag(np.sqrt(weightField))
        eigVecsWeighted = np.dot(la.inv(W), eigVecs)
        # KLModes is eigVecWeighted * sqrt(eigVal)
        KLModes = np.zeros([len(weightField), nKL])
        for i in np.arange(nKL):
            if eigVals[i] >= 0:
                KLModes[:, i] = eigVecsWeighted[:, i] * np.sqrt(eigVals[i])
            else:
                print 'Negative eigenvalue detected at nKL=' + \
                    str(i) + ': number of KL modes might be too large!'
                KLModes[:, i] = eigVecsWeighted[:, i] * 0

        self.KLModes = KLModes
        kLRatio = sum(eigVals) / covTrace
        print nKL, 'KL modes can cover', kLRatio, 'of Random field'

        return eigVals, KLModes

    def KLProjection(self, Field, KLModes, eigVals, weightField):
        """project the random field onto KL modes,
           omega_k = sum{weight(x) * F(x) * KLModes(x)_k}/sqrt{eigVal_k}
        Args:
            Field       : random field realization (N by 1)
            eigVals     : eig values
            KLModes     : KL modes (N by nKL)
            weightField : weightField array (N by 1)
        Return:
            omegaVec    : the coefficients for KL modes (nKL by 1)
        """
        nKL = len(eigVals)  # number of KL modes
        omegaVec = np.zeros([nKL, 1])
        for k in range(nKL):
            #omegaVec[k, 0] = np.sum(weightField * Field * KLModes[:, k]) / np.sqrt(eigVals[k])
            omegaVec[k, 0] = np.sum(
                weightField * Field * KLModes[:, k]) / eigVals[k]
        return omegaVec

    def reconstructField_Reduced(self, omegaVec, KLModes):
        """reconstruct a random field with truncated KL modes

        Args:
            omegaVec    :the coefficients for KL modes (nKL by 1)
            KLModes     :sqrt(eigenvalues_i) * eigenvectors/weight (N by nKL)
        Return:
            recField    :reconstructed field (N by 1)

        """
        [N, nKL] = KLModes.shape  # parse the dimension
        assert len(omegaVec) == nKL, \
            "Lengths of KL coefficient omega (%d) and KL modes (%d) differs!" \
            % (len(omegaVec), nKL)

        if self.meanField is None:
            self.meanField = np.zeros([N, 1])
        recField = self.meanField + np.dot(KLModes, omegaVec)
        return recField
