#! /usr/bin/env python

import numpy as np 
from scipy import interpolate
import matplotlib.pylab as plt
import os
import os.path as ospt

import foamFileOperation as foamOp  # OpenFOAM file operator

def computeSigmaField(scatteredSigmaFile, cellCenters, kernel, lengthScale):
    """
    Compute sigma (variance) at all cell centers
    according to an interpolated function constructed from scattered data
    This function use Radial basis function to do the interpolation

    Input:
        scatteredSigmaFile: path to the file with scatterred data
        meshCoord: mesh coordinate (Ncell x 3)
        lengthScale (optional): lengh scale for Gaussian kernel

    Output:
        sigmaField: containing the sigma at all cells (openfoam ordering)
    """

    sigmaData = np.loadtxt(scatteredSigmaFile)

    if sigmaData.shape[1] == 3:
        sigmaFunc = interpolate.Rbf( \
                sigmaData[:, 0], sigmaData[:, 1], sigmaData[:, 2], \
                function=kernel, epsilon=lengthScale)

        sigmaField = sigmaFunc(cellCenters[:, 0], cellCenters[:, 1])
    elif sigmaData.shape[1] == 4:
        sigmaFunc = interpolate.Rbf( \
                sigmaData[:, 0], sigmaData[:, 1], sigmaData[:, 2], sigmaData[:, 3], \
                function=kernel, epsilon=lengthScale)

        sigmaField = sigmaFunc(cellCenters[:, 0], cellCenters[:, 1], cellCenters[:, 2])
    else:
        print "Please check the dimensions of scatteredSigmaFile."

    # plt.figure(1)
    # plt.clf()
    # skip = 2
    # plt.imshow(sigmaField.reshape((60,74)), extent=[0,9,0,3], cmap=plt.cm.jet)
    # plt.scatter(cellCenters[0:-1:skip, 0], cellCenters[0:-1:skip, 1], 50, sigmaField[0:-1:skip], \
    #            cmap=plt.cm.jet)
    # plt.colorbar()
    #plt.show()

    return sigmaField
    

def checkSigmaField():
    # A utility that can called on an openFoam case to generate sigma field in
    # the constant dir, which can then examined further with paraview
    # Assume that the current dir is an openfoam case
    
    baseCaseDir = ospt.join('0/')
    scatteredSigmaFile = ospt.join('constant', \
                         'scatSigma.dat')

    meshCoord3D = foamOp.readTurbCoordinateFromFile(baseCaseDir)

    rbfLengthScale = 0.6 
    rbfKernel = 'gaussian'
    sigma = computeSigmaField(scatteredSigmaFile, meshCoord3D, \
                                       rbfKernel, rbfLengthScale)
    sigmaFile = ospt.join(os.getcwd(), '0/sigma')
    foamOp.writeScalarToFile(sigma, sigmaFile)

if __name__ == '__main__':

    checkSigmaField()
