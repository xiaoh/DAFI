#!/usr/bin/env python
# copyright: Jianxun Wang (vtwjx@vt.edu)
# Mar. 31, 2015
# Modified by Carlos Michelen-Strofer 2018
#
# Inverse modeling main code

## system modules
import sys
import os
import time
import numpy as np
import importlib

## local modules
from DAInverse import DAFiltering

## functions used in main code
def _printUsage():
    """ Print usage of the program.
    """
    print("Usage: mfuMain.py <MainInput.in>")

def _parseInput():
    ''' Parse the input file.
    '''
    try:
        DAInputFile = sys.argv[1]
    except IndexError, e:
        print(e)
        printUsage()
        sys.exit(1)
    return DAInputFile

## main code
DAInputFile = _parseInput()

# initilize which Filtering you will use
np.random.seed(2000)
inverseModel = DAFiltering.EnKF(DAInputFile)

# solving the inverse problem
startTime = time.time()
inverseModel.solve()
inverseModel.clean()
print("Time spent on solver: {}".format(time.time() - startTime))

# report and plots
# inverseModel.report()
# inverseModel.plot()
