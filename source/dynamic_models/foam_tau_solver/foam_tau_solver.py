# Mar.27, 2015

"""

This module is an Dynamic model interface to specific forward model:
     OpenFoam Reynolds stress Tau Solver

It  consist of 3 founctions:
    1 generateEnsemble: generate ensemble
    2 forcastToTime: evolve ensemble to next time using forward model (TauFOAM)
    3 getBackgroundVars: Get observations and Kalman Gain Matrix

"""
# standard library imports
import sys
import os
import os.path as ospt
import ast
import time

# third party imports
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import scipy.sparse as sp
try:
    import pp
    hasPP = True
    print "Parallel Python imported successfully"
except ImportError as e:
    print e, 'I will not use parallel python in this simulation!'
    hasPP = False
    pass

# local imports
from data_assimilation.dyn_model import DynModel
from data_assimilation.utilities import read_input_data, replace, extract_list_from_string
from random_field_org import deltaTauRandomField as ranF
import foam_utilities as foamOp
from dynamic_models.foam_tau_solver import ReynoldsStressRF as ReRF
from dynamic_models.foam_tau_solver.sigmaFieldOperations import computeSigmaField


class FoamTauSolver(DynModel):
    """
    A particular dynamic forward-model interface: OpenFOAM Tau solver
    The state variable include: U (velocity)
    The parameters need to be augmented: coefficients for deltaXi, deltaEta, deltaLog2k,
                                         deltaVA, deltaVB, deltaVC
    """

    # static variables
    NTensor = 6                         # dimension of physical tensor
    NVec = 3                            # dimension of physical vector
    NScalar = 1                         # dimension of physical scalar

    def __init__(self, Ns, DAInterval, Tend, max_pseudo_time, ModelInput):
        """
        Initialization
        """
        self.name = 'FoamTauSolver'
        # Extract forward Model Input parameters
        paramDict = read_input_data(ModelInput)
        # baseline case folder name
        self.caseName = paramDict['case_name']
        # name of case that generate observation
        self.caseNameObservation = self.caseName + "-observation"
        # foward Model's solver name (example: PisoFoam, SimpleFoam etc.)
        self.caseSolver = paramDict['case_solver']
        # PseudoObs 1: using forward model to generate observation; 0: real
        # experiment data as observation
        self.pseudoObs = int(paramDict['pseudoObs'])
        # The number of cells
        self.Ncell = int(paramDict['Ncell'])
        # The number of cells which are observed
        self.NcellObs = int(paramDict['NcellObs'])
        # Determine if Tau is in the state
        self.TauOnFlag = ast.literal_eval(paramDict['TauOnFlag'])
        # switch control which components are perturbed
        self.perturbXi = ast.literal_eval(paramDict['perturbXi'])
        self.perturbEta = ast.literal_eval(paramDict['perturbEta'])
        self.perturbK = ast.literal_eval(paramDict['perturbK'])
        self.perturbVA = ast.literal_eval(paramDict['perturbVA'])
        self.perturbVB = ast.literal_eval(paramDict['perturbVB'])
        self.perturbVC = ast.literal_eval(paramDict['perturbVC'])
        # control of if bound the baseline
        self.capBaseline = ast.literal_eval(paramDict['capBaseline'])
        # hyperparameter for KL expansion
        # --- kernel type for KL expansion
        self.kernelType = paramDict['kernelType']
        if self.kernelType == 'SqExp':
            # --- length scales
            lenConstFlag = ast.literal_eval(paramDict['lenConstFlag'])
            if lenConstFlag:
                lenVecXi = extract_list_from_string(paramDict['lenVecXi'])
                lenVecEta = extract_list_from_string(paramDict['lenVecEta'])
                lenVecK = extract_list_from_string(paramDict['lenVecK'])
                lenVecV = extract_list_from_string(paramDict['lenVecV'])
                self.lenVecXi = np.array([[float(pn) for pn in lenVecXi]])
                self.lenVecEta = np.array([[float(pn) for pn in lenVecEta]])
                self.lenVecK = np.array([[float(pn) for pn in lenVecK]])
                self.lenVecV = np.array([[float(pn) for pn in lenVecV]])
        # --- variance field (for self.perturbation in Xi, Eta, K, VA, VB, VC)
        sigmaConstFlag = ast.literal_eval(paramDict['sigmaConstFlag'])
        if sigmaConstFlag:
            sigmaVec = extract_list_from_string(paramDict['sigmaVec'])
            self.sigmaVec = np.array([[float(pn) for pn in sigmaVec]])
        else:
            # CM: cleanup. No need to initialize KL if V not perturbed.
            self.sigmaVec = np.array([[0.0, 0.0, 0.0, 0.0]])
        # --- resemble the hyper parameters
        # self.hyperParaXi = np.hstack((self.sigmaVec[0, 0], self.lenVecXi[0, :]))
        # self.hyperParaEta = np.hstack((self.sigmaVec[0, 1], self.lenVecEta[0, :]))
        # self.hyperParaK = np.hstack((self.sigmaVec[0, 2], self.lenVecK[0, :]))
        # self.hyperParaV = np.hstack((self.sigmaVec[0, 3], self.lenVecV[0, :]))
        # --- number of KL modes (use same amount of modes for all components)
        self.Nmodes = int(paramDict['Nmode'])
        self.nModesXi = self.Nmodes
        self.nModesEta = self.Nmodes
        self.nModesK = self.Nmodes
        self.nModesV = self.Nmodes
        # enable paralization
        self.enablePP = ast.literal_eval(paramDict['enablePP']) and hasPP
        # initialize parallel python
        if(self.enablePP):
            print "Parallel Python will be used with ",
            self.job_server = pp.Server()
            if (int(paramDict.get('cpuToUse', 0)) > 0):
                n = int(paramDict['cpuToUse'])
                print n, " cores"
                self.job_server.set_ncpus(n)
            else:
                print "all available cores"
                self.job_server.set_ncpus()  # use all CPUs available
            self.jobs = []

        # Output control levels
        outPutControl = paramDict['OutputControlDict']
        outputDict = read_input_data(outPutControl)
        self.screenOutput = ast.literal_eval(outputDict['screenOutput'])
        self.figureOutput = ast.literal_eval(outputDict['figureOutput'])
        self.txtfileOutput = ast.literal_eval(outputDict['txtfileOutput'])
        self.verboseLevel = int(outputDict['verboseLevel'])
        # debug
        self._iDebug = outputDict['iDebug']
        self._iDebug = 1  # hard coded
        self._debugFolderName = outputDict['debugFolderName']
        if not os.path.exists(self._debugFolderName):
            os.makedirs(self._debugFolderName)

        # Start KL expansion for random fields
        # Read cell center coordinates
        self.baseCaseDir = ospt.join(os.getcwd(), self.caseName, '0/')
        meshCoord3D = foamOp.readTurbCoordinateFromFile(self.baseCaseDir)
        # Read cell area
        areaCoord = foamOp.readTurbCellVolumeFromFile(self.baseCaseDir)
        areaCoord = areaCoord[:, 0]
        # construct variance field sigma(x) for Xi, Eta, K, Vs
        zeroSigmaFile = ospt.join(
            os.getcwd(),
            self.caseName,
            'constant/sigmaZero.dat')
        if ospt.exists(zeroSigmaFile):
            zeroMask = np.loadtxt(zeroSigmaFile, dtype=bool)
        else:
            zeroMask = np.zeros(self.Ncell, dtype=bool)
        if sigmaConstFlag:
            print "\nConstant sigma fields are used"
            print "Xi (", self.sigmaVec[0, 0], "), Eta (", self.sigmaVec[0,
                                                                         1], "), K (", self.sigmaVec[0, 2], "), V (", self.sigmaVec[0, 3], ")"

            self.sigmaXi = np.ones(self.Ncell) * self.sigmaVec[0, 0]
            self.sigmaXi[zeroMask] = 0.0
            sigmaXiFile = ospt.join(os.getcwd(), self.caseName, '0/sigmaXi')
            foamOp.writeScalarToFile(self.sigmaXi, sigmaXiFile)

            self.sigmaEta = np.ones(self.Ncell) * self.sigmaVec[0, 1]
            self.sigmaEta[zeroMask] = 0.0
            sigmaEtaFile = ospt.join(os.getcwd(), self.caseName, '0/sigmaEta')
            foamOp.writeScalarToFile(self.sigmaEta, sigmaEtaFile)

            self.sigmaK = np.ones(self.Ncell) * self.sigmaVec[0, 2]
            self.sigmaK[zeroMask] = 0.0
            sigmaKFile = ospt.join(os.getcwd(), self.caseName, '0/sigmaK')
            foamOp.writeScalarToFile(self.sigmaK, sigmaKFile)

            self.sigmaVA = np.ones(self.Ncell) * self.sigmaVec[0, 3]
            self.sigmaVA[zeroMask] = 0.0
            sigmaVAFile = ospt.join(os.getcwd(), self.caseName, '0/sigmaVA')
            foamOp.writeScalarToFile(self.sigmaVA, sigmaVAFile)

            self.sigmaVB = np.ones(self.Ncell) * self.sigmaVec[0, 3]
            self.sigmaVB[zeroMask] = 0.0
            sigmaVBFile = ospt.join(os.getcwd(), self.caseName, '0/sigmaVB')
            foamOp.writeScalarToFile(self.sigmaVB, sigmaVBFile)

            self.sigmaVC = np.ones(self.Ncell) * self.sigmaVec[0, 3]
            self.sigmaVC[zeroMask] = 0.0
            sigmaVCFile = ospt.join(os.getcwd(), self.caseName, '0/sigmaVC')
            foamOp.writeScalarToFile(self.sigmaVC, sigmaVCFile)
        else:
            print "\nNonstationary sigma fields are used"
            rbfLengthScale = float(paramDict['rbfLengthScale'])
            rbfKernel = paramDict['rbfKernel']

            scatteredSigmaXiFile = ospt.join(
                self.caseName, 'constant', 'scatSigmaXi.dat')
            if ospt.exists(scatteredSigmaXiFile):
                self.sigmaXi = np.absolute(
                    computeSigmaField(
                        scatteredSigmaXiFile,
                        meshCoord3D,
                        rbfKernel,
                        rbfLengthScale))
            else:
                print "WARNING: I cannot find file: scatSigmaXi.dat, constant sigma used for Xi "
                self.sigmaXi = np.ones(self.Ncell) * self.sigmaVec[0, 0]
            self.sigmaXi[zeroMask] = 0.0
            sigmaXiFile = ospt.join(os.getcwd(), self.caseName, '0/sigmaXi')
            foamOp.writeScalarToFile(self.sigmaXi, sigmaXiFile)

            scatteredSigmaEtaFile = ospt.join(
                self.caseName, 'constant', 'scatSigmaEta.dat')
            if ospt.exists(scatteredSigmaEtaFile):
                self.sigmaEta = np.absolute(
                    computeSigmaField(
                        scatteredSigmaEtaFile,
                        meshCoord3D,
                        rbfKernel,
                        rbfLengthScale))
            else:
                print "WARNING: I cannot find file: scatSigmaEta.dat, constant sigma used for Eta "
                self.sigmaEta = np.ones(self.Ncell) * self.sigmaVec[0, 1]
            self.sigmaEta[zeroMask] = 0.0
            sigmaEtaFile = ospt.join(os.getcwd(), self.caseName, '0/sigmaEta')
            foamOp.writeScalarToFile(self.sigmaEta, sigmaEtaFile)

            scatteredSigmaKFile = ospt.join(
                self.caseName, 'constant', 'scatSigmaK.dat')
            if ospt.exists(scatteredSigmaKFile):
                self.sigmaK = np.absolute(
                    computeSigmaField(
                        scatteredSigmaKFile,
                        meshCoord3D,
                        rbfKernel,
                        rbfLengthScale))
            else:
                print "WARNING: I cannot find file: scatSigmaK.dat, constant sigma used for K "
                self.sigmaK = np.ones(self.Ncell) * self.sigmaVec[0, 2]
            self.sigmaK[zeroMask] = 0.0
            sigmaKFile = ospt.join(os.getcwd(), self.caseName, '0/sigmaK')
            foamOp.writeScalarToFile(self.sigmaK, sigmaKFile)

            scatteredSigmaVAFile = ospt.join(
                self.caseName, 'constant', 'scatSigmaVA.dat')
            if ospt.exists(scatteredSigmaVAFile):
                self.sigmaVA = np.absolute(
                    computeSigmaField(
                        scatteredSigmaVAFile,
                        meshCoord3D,
                        rbfKernel,
                        rbfLengthScale))
            else:
                print "WARNING: I cannot find file: scatSigmaVA.dat, constant sigma used for VA "
                self.sigmaVA = np.ones(self.Ncell) * self.sigmaVec[0, 3]
            self.sigmaVA[zeroMask] = 0.0
            sigmaVAFile = ospt.join(os.getcwd(), self.caseName, '0/sigmaVA')
            foamOp.writeScalarToFile(self.sigmaVA, sigmaVAFile)

            scatteredSigmaVBFile = ospt.join(
                self.caseName, 'constant', 'scatSigmaVB.dat')
            if ospt.exists(scatteredSigmaVBFile):
                self.sigmaVB = np.absolute(
                    computeSigmaField(
                        scatteredSigmaVBFile,
                        meshCoord3D,
                        rbfKernel,
                        rbfLengthScale))
            else:
                print "WARNING: I cannot find file: scatSigmaVB.dat, constant sigma used for VB "
                self.sigmaVB = np.ones(self.Ncell) * self.sigmaVec[0, 3]
            self.sigmaVB[zeroMask] = 0.0
            sigmaVBFile = ospt.join(os.getcwd(), self.caseName, '0/sigmaVB')
            foamOp.writeScalarToFile(self.sigmaVB, sigmaVBFile)

            scatteredSigmaVCFile = ospt.join(
                self.caseName, 'constant', 'scatSigmaVC.dat')
            if ospt.exists(scatteredSigmaVCFile):
                self.sigmaVC = np.absolute(
                    computeSigmaField(
                        scatteredSigmaVCFile,
                        meshCoord3D,
                        rbfKernel,
                        rbfLengthScale))
            else:
                print "WARNING: I cannot find file: scatSigmaVC.dat, constant sigma used for VC "
                self.sigmaVC = np.ones(self.Ncell) * self.sigmaVec[0, 3]
            self.sigmaVC[zeroMask] = 0.0
            sigmaVCFile = ospt.join(os.getcwd(), self.caseName, '0/sigmaVC')
            foamOp.writeScalarToFile(self.sigmaVC, sigmaVCFile)

        # load covariance structure
        if self.kernelType == 'givenStructure':
            CovStructFile = ospt.join(
                self.caseName, 'constant', 'CovStruct.dat')
            CovStruct = np.loadtxt(CovStructFile, dtype=float)

        # construct length scale field
        if self.kernelType == 'SqExp':
            if lenConstFlag:
                print "\nstationary length scale fields are used"
                lenXField_Xi = np.ones(self.Ncell) * self.lenVecXi[0, 0]
                lenYField_Xi = np.ones(self.Ncell) * self.lenVecXi[0, 1]
                lenZField_Xi = np.ones(self.Ncell) * self.lenVecXi[0, 2]

                lenXField_Eta = np.ones(self.Ncell) * self.lenVecEta[0, 0]
                lenYField_Eta = np.ones(self.Ncell) * self.lenVecEta[0, 1]
                lenZField_Eta = np.ones(self.Ncell) * self.lenVecEta[0, 2]

                lenXField_K = np.ones(self.Ncell) * self.lenVecK[0, 0]
                lenYField_K = np.ones(self.Ncell) * self.lenVecK[0, 1]
                lenZField_K = np.ones(self.Ncell) * self.lenVecK[0, 2]

                lenXField_V = np.ones(self.Ncell) * self.lenVecV[0, 0]
                lenYField_V = np.ones(self.Ncell) * self.lenVecV[0, 1]
                lenZField_V = np.ones(self.Ncell) * self.lenVecV[0, 2]
            else:
                print "\nNonstationary length scale fields are used"
                scatteredLxFile = ospt.join(
                    self.caseName, 'constant', 'scatLx.dat')
                scatteredLyFile = ospt.join(
                    self.caseName, 'constant', 'scatLy.dat')
                scatteredLzFile = ospt.join(
                    self.caseName, 'constant', 'scatLz.dat')

                if ospt.exists(scatteredLxFile) and ospt.exists(
                        scatteredLyFile) and ospt.exists(scatteredLzFile):

                    lenXField_Xi = np.absolute(
                        computeSigmaField(
                            scatteredLxFile,
                            meshCoord3D,
                            rbfKernel,
                            rbfLengthScale))
                    lenYField_Xi = np.absolute(
                        computeSigmaField(
                            scatteredLyFile,
                            meshCoord3D,
                            rbfKernel,
                            rbfLengthScale))
                    lenYField_Xi = np.absolute(
                        computeSigmaField(
                            scatteredLyFile,
                            meshCoord3D,
                            rbfKernel,
                            rbfLengthScale))
                    lenXField_Eta = lenXField_Xi
                    lenYField_Eta = lenYField_Xi
                    lenZField_Eta = lenXField_Xi

                    lenXField_K = lenXField_Xi
                    lenYField_K = lenYField_Xi
                    lenZField_K = lenXField_Xi

                    lenXField_V = lenXField_Xi
                    lenYField_V = lenYField_Xi
                    lenZField_V = lenXField_Xi
                else:
                    print "WARNING: scatLx.dat or scatLy.dat or scatLz.dat cannot be found!"
                    lenXField_Xi = np.ones(self.Ncell) * self.lenVecXi[0, 0]
                    lenYField_Xi = np.ones(self.Ncell) * self.lenVecXi[0, 1]
                    lenZField_Xi = np.ones(self.Ncell) * self.lenVecXi[0, 2]

                    lenXField_Eta = np.ones(self.Ncell) * self.lenVecEta[0, 0]
                    lenYField_Eta = np.ones(self.Ncell) * self.lenVecEta[0, 1]
                    lenZField_Eta = np.ones(self.Ncell) * self.lenVecEta[0, 2]

                    lenXField_K = np.ones(self.Ncell) * self.lenVecK[0, 0]
                    lenYField_K = np.ones(self.Ncell) * self.lenVecK[0, 1]
                    lenZField_K = np.ones(self.Ncell) * self.lenVecK[0, 2]

                    lenXField_V = np.ones(self.Ncell) * self.lenVecV[0, 0]
                    lenYField_V = np.ones(self.Ncell) * self.lenVecV[0, 1]
                    lenZField_V = np.ones(self.Ncell) * self.lenVecV[0, 2]

        # prepare the argument for generate covariance matrix for all
        # components
        truncateTol = -np.log(1e-10)
        # --- for Xi
        Arg_covGen_Xi = {'kernelType': self.kernelType,
                         'sigmaField': self.sigmaXi,
                         'weightField': areaCoord,
                         'truncateTol': truncateTol
                         }

        if self.kernelType == 'SqExp':
            Arg_covGen_Xi['lenXField'] = lenXField_Xi
            Arg_covGen_Xi['lenYField'] = lenYField_Xi
            Arg_covGen_Xi['lenZField'] = lenZField_Xi

        elif self.kernelType == 'givenStructure':
            Arg_covGen_Xi['CovStruct'] = CovStruct

        Arg_calModes_Xi = {
            'nKL': self.nModesXi,
            'weightField': areaCoord
        }
        # --- for Eta
        Arg_covGen_Eta = {'kernelType': self.kernelType,
                          'sigmaField': self.sigmaEta,
                          'weightField': areaCoord,
                          'truncateTol': truncateTol
                          }

        if self.kernelType == 'SqExp':
            Arg_covGen_Eta['lenXField'] = lenXField_Eta
            Arg_covGen_Eta['lenYField'] = lenYField_Eta
            Arg_covGen_Eta['lenZField'] = lenZField_Eta

        elif self.kernelType == 'givenStructure':
            Arg_covGen_Eta['CovStruct'] = CovStruct

        Arg_calModes_Eta = {
            'nKL': self.nModesEta,
            'weightField': areaCoord
        }
        # --- for K
        Arg_covGen_K = {'kernelType': self.kernelType,
                        'sigmaField': self.sigmaK,
                        'weightField': areaCoord,
                        'truncateTol': truncateTol
                        }
        if self.kernelType == 'SqExp':
            Arg_covGen_K['lenXField'] = lenXField_K
            Arg_covGen_K['lenYField'] = lenYField_K
            Arg_covGen_K['lenZField'] = lenZField_K

        elif self.kernelType == 'givenStructure':
            Arg_covGen_K['CovStruct'] = CovStruct

        Arg_calModes_K = {
            'nKL': self.nModesK,
            'weightField': areaCoord
        }
        # --- for V
        Arg_covGen_VA = {'kernelType': self.kernelType,
                         'sigmaField': self.sigmaVA,
                         'weightField': areaCoord,
                         'truncateTol': truncateTol
                         }

        if self.kernelType == 'SqExp':
            Arg_covGen_VA['lenXField'] = lenXField_V
            Arg_covGen_VA['lenYField'] = lenYField_V
            Arg_covGen_VA['lenZField'] = lenZField_V

        elif self.kernelType == 'givenStructure':
            Arg_covGen_VA['CovStruct'] = CovStruct

        Arg_calModes_VA = {
            'nKL': self.nModesV,
            'weightField': areaCoord
        }

        Arg_covGen_VB = {'kernelType': self.kernelType,
                         'sigmaField': self.sigmaVB,
                         'weightField': areaCoord,
                         'truncateTol': truncateTol
                         }

        if self.kernelType == 'SqExp':
            Arg_covGen_VB['lenXField'] = lenXField_V
            Arg_covGen_VB['lenYField'] = lenYField_V
            Arg_covGen_VB['lenZField'] = lenZField_V

        elif self.kernelType == 'givenStructure':
            Arg_covGen_VB['CovStruct'] = CovStruct

        Arg_calModes_VB = {
            'nKL': self.nModesV,
            'weightField': areaCoord
        }

        Arg_covGen_VC = {'kernelType': self.kernelType,
                         'sigmaField': self.sigmaVC,
                         'weightField': areaCoord,
                         'truncateTol': truncateTol
                         }

        if self.kernelType == 'SqExp':
            Arg_covGen_VC['lenXField'] = lenXField_V
            Arg_covGen_VC['lenYField'] = lenYField_V
            Arg_covGen_VC['lenZField'] = lenZField_V

        elif self.kernelType == 'givenStructure':
            Arg_covGen_VC['CovStruct'] = CovStruct

        Arg_calModes_VC = {
            'nKL': self.nModesV,
            'weightField': areaCoord
        }

        # initialize the KL expansion class
        self.rfXi = ranF.randomField(
            meshCoord3D, Arg_covGen_Xi, Arg_calModes_Xi, 'Xi')
        self.rfEta = ranF.randomField(
            meshCoord3D, Arg_covGen_Eta, Arg_calModes_Eta, 'Eta')
        self.rfK = ranF.randomField(
            meshCoord3D, Arg_covGen_K, Arg_calModes_K, 'K')
        self.rfVA = ranF.randomField(
            meshCoord3D, Arg_covGen_VA, Arg_calModes_VA, 'VA')
        self.rfVB = ranF.randomField(
            meshCoord3D, Arg_covGen_VB, Arg_calModes_VB, 'VB')
        self.rfVC = ranF.randomField(
            meshCoord3D, Arg_covGen_VC, Arg_calModes_VC, 'VC')

        tic = time.time()
        # determine if perform KL expansion
        klCalculate = ast.literal_eval(paramDict['klCalculate'])
        XiFile = os.path.exists('randomData_Xi/KLModes.dat')
        EtaFile = os.path.exists('randomData_Eta/KLModes.dat')
        kFile = os.path.exists('randomData_K/KLModes.dat')

        if (klCalculate):
            print "\nStart KL expansions ......"
            self.KLModesXi = self.rfXi.KLExpansion()
            self.KLModesEta = self.rfEta.KLExpansion()
            self.KLModesK = self.rfK.KLExpansion()
            if (self.perturbVA) or (self.perturbVB) or (self.perturbVC):
                self.KLModesVA = self.rfVA.KLExpansion()
                self.KLModesVB = self.rfVB.KLExpansion()
                self.KLModesVC = self.rfVC.KLExpansion()

        elif ((not XiFile) or (not EtaFile) or (not kFile)):
            print "\nWARNING: Cannot find KL files, performing KL expansion calculation"
            self.KLModesXi = self.rfXi.KLExpansion()
            self.KLModesEta = self.rfEta.KLExpansion()
            self.KLModesK = self.rfK.KLExpansion()
            if (self.perturbVA) or (self.perturbVB) or (self.perturbVC):
                self.KLModesVA = self.rfVA.KLExpansion()
                self.KLModesVB = self.rfVB.KLExpansion()
                self.KLModesVC = self.rfVC.KLExpansion()

        else:
            print "\nLoading KL expansion data, won't perform KL calculation \
                   \nplease make sure:\
                   \n mesh, len, sigma, number of modes are not changed"
            self.KLModesXi = np.loadtxt('randomData_Xi/KLModes.dat')
            self.KLModesEta = np.loadtxt('randomData_Eta/KLModes.dat')
            self.KLModesK = np.loadtxt('randomData_K/KLModes.dat')
            if (self.perturbVA):
                self.KLModesVA = np.loadtxt('randomData_VA/KLModes.dat')
                self.KLModesVB = np.loadtxt('randomData_VB/KLModes.dat')
                self.KLModesVC = np.loadtxt('randomData_VC/KLModes.dat')
        toc = time.time()
        print "\ncollapse time for getting All KL modes is ", toc - tic, "s"

        tic = time.time()
        # initilize Tau mapping class
        if (self.capBaseline):
            self.mapTau = ReRF.ReynoldsStressRF(
                self.baseCaseDir, 'Tau', self.Ncell, 1, True)
            os.system(
                'cp ' +
                self.baseCaseDir +
                'Tau ' +
                self.baseCaseDir +
                'TauCap')
            self.TauBaseline = self.mapTau.tau
            foamOp.writeTurbStressToFile(
                self.mapTau.tau, self.baseCaseDir + 'TauCap')
        else:
            self.mapTau = ReRF.ReynoldsStressRF(
                self.baseCaseDir, 'Tau', self.Ncell, 1)
            self.tauBaseFile = ospt.join(self.baseCaseDir, 'Tau')
            self.TauBaseline = foamOp.readTurbStressFromFile(self.tauBaseFile)

        # Ns Tau for each ensemble member at current DA step
        self.Tau = np.zeros([Ns, self.Ncell, FoamTauSolver.NTensor])
        # Ns Tau for each ensemble member at last DA step
        self.TauOld = np.zeros([Ns, self.Ncell, FoamTauSolver.NTensor])
        # Ns deltaTau = (Tau - Taubasline) for each ensemble member
        self.deltaTau = np.zeros([Ns, self.Ncell, FoamTauSolver.NTensor])
        # 3 * Nmodes unknown parameters
        # need to be part of parameters
        self.omegaXiM = np.zeros([self.nModesXi, Ns])
        # need to be part of parameters
        self.omegaEtaM = np.zeros([self.nModesEta, Ns])
        # need to be part of parameters
        self.omegakM = np.zeros([self.nModesK, Ns])
        # need to be part of parameters
        self.omegaVAM = np.zeros([self.nModesV, Ns])
        # need to be part of parameters
        self.omegaVBM = np.zeros([self.nModesV, Ns])
        # need to be part of parameters
        self.omegaVCM = np.zeros([self.nModesV, Ns])

        # Ns deltaXi (deltaEta or deltak) for each ensemble member
        self.deltaXiM = np.zeros([Ns, self.Ncell, FoamTauSolver.NScalar])
        self.deltaEtaM = np.zeros([Ns, self.Ncell, FoamTauSolver.NScalar])
        self.deltakM = np.zeros([Ns, self.Ncell, FoamTauSolver.NScalar])
        self.deltaVAM = np.zeros([Ns, self.Ncell, FoamTauSolver.NScalar])
        self.deltaVBM = np.zeros([Ns, self.Ncell, FoamTauSolver.NScalar])
        self.deltaVCM = np.zeros([Ns, self.Ncell, FoamTauSolver.NScalar])

        # clear output data figures
        os.system('rm -rf ' + self._debugFolderName + '*')
        os.system('mkdir ' + self._debugFolderName + 'init')
        os.system('mkdir ' + self._debugFolderName + 'init-propagate')
        os.system('mkdir ' + self._debugFolderName + 'init/Tau')
        # Tau for baseline case
        self.deltaTauBaseline = np.zeros(self.TauBaseline.shape)

        # generate deltaXi, deltaEta, deltak, deltaVA, deltaVB, deltaVC fields
        # using these perturbation to generate perturbed Tau ensemble
        tic_in = time.time()
        for i in range(Ns):
            omegaXi = self.rfXi.uncorrUniRandom(
                self.nModesXi)  # time = 1.8e-4s (3000 cells)
            # time = 3.1e-5s (3000 cells)
            self.omegaXiM[:, i] = omegaXi[:, 0]
            self.deltaXiField = self.rfXi.reconstructField(
                omegaXi, self.KLModesXi)  # time = 6.1e-4s (3000 cells)
            self.deltaXiM[i, :, :] = self.deltaXiField

            omegaEta = self.rfEta.uncorrUniRandom(self.nModesEta)
            self.omegaEtaM[:, i] = omegaEta[:, 0]
            self.deltaEtaField = self.rfEta.reconstructField(
                omegaEta, self.KLModesEta)
            self.deltaEtaM[i, :, :] = self.deltaEtaField

            omegak = self.rfK.uncorrUniRandom(self.nModesK)
            self.omegakM[:, i] = omegak[:, 0]
            self.deltakField = self.rfK.reconstructField(omegak, self.KLModesK)
            self.deltakM[i, :, :] = self.deltakField

            if (self.perturbVA) or (self.perturbVB) or (self.perturbVC):
                omegaVA = self.rfVA.uncorrUniRandom(self.nModesV)
                self.omegaVAM[:, i] = omegaVA[:, 0]
                self.deltaVAField = self.rfVA.reconstructField(
                    omegaVA, self.KLModesVA)
                self.deltaVAM[i, :, :] = self.deltaVAField

                omegaVB = self.rfVB.uncorrUniRandom(self.nModesV)
                self.omegaVBM[:, i] = omegaVB[:, 0]
                self.deltaVBField = self.rfVB.reconstructField(
                    omegaVB, self.KLModesVB)
                self.deltaVBM[i, :, :] = self.deltaVBField

                omegaVC = self.rfVC.uncorrUniRandom(self.nModesV)
                self.omegaVCM[:, i] = omegaVC[:, 0]
                self.deltaVCField = self.rfVC.reconstructField(
                    omegaVC, self.KLModesVC)
                self.deltaVCM[i, :, :] = self.deltaVCField

            tic_in = time.time()
            if (not self.perturbXi):
                omegaXi = np.zeros(omegaXi.shape)
                self.omegaXiM[:, i] = omegaXi[:, 0]
                self.deltaXiField = self.rfXi.reconstructField(
                    omegaXi, self.KLModesXi)
                self.deltaXiM[i, :, :] = self.deltaXiField
            if (not self.perturbEta):
                omegaEta = np.zeros(omegaEta.shape)
                self.omegaEtaM[:, i] = omegaEta[:, 0]
                self.deltaEtaField = self.rfEta.reconstructField(
                    omegaEta, self.KLModesEta)
                self.deltaEtaM[i, :, :] = self.deltaEtaField
            if (not self.perturbK):
                omegak = np.zeros(omegak.shape)
                self.omegakM[:, i] = omegak[:, 0]
                self.deltakField = self.rfK.reconstructField(
                    omegak, self.KLModesK)
                self.deltakM[i, :, :] = self.deltakField
            if (self.perturbVA) or (self.perturbVB) or (self.perturbVC):
                if (not self.perturbVA):
                    omegaVA = np.zeros(omegak.shape)
                    self.omegaVAM[:, i] = omegaVA[:, 0]
                    self.deltaVAField = self.rfVA.reconstructField(
                        omegaVA, self.KLModesVA)
                    self.deltaVAM[i, :, :] = self.deltaVAField
                if (not self.perturbVB):
                    omegaVB = np.zeros(omegak.shape)
                    self.omegaVBM[:, i] = omegaVB[:, 0]
                    self.deltaVBField = self.rfVB.reconstructField(
                        omegaVB, self.KLModesVB)
                    self.deltaVBM[i, :, :] = self.deltaVBField
                if (not self.perturbVC):
                    omegaVC = np.zeros(omegak.shape)
                    self.omegaVCM[:, i] = omegaVC[:, 0]
                    self.deltaVCField = self.rfVC.reconstructField(
                        omegaVC, self.KLModesVC)
                    self.deltaVCM[i, :, :] = self.deltaVCField

                deltaVAll = np.hstack(
                    (self.deltaVAField, self.deltaVBField, self.deltaVCField))

            # Perturbing Tau with deltaXi deltaEta, deltaLog2k, deltaVA,
            # deltaVB, deltaVC
            if (self.perturbVA) or (self.perturbVB) or (self.perturbVC):
                if (self.enablePP):
                    self.jobs.append(
                        self.job_server.submit(
                            self.mapTau.perturbTau,
                            (self.deltaXiField,
                             self.deltaEtaField,
                             self.deltakField,
                             deltaVAll),
                            modules=(
                                "numpy",
                            )))
                else:
                    self.Tau[i, :, :] = self.mapTau.perturbTau(
                        self.deltaXiField, self.deltaEtaField, self.deltakField, deltaVAll)
                    self.TauOld[i, :, :] = self.TauBaseline
                    self.deltaTau[i, :, :] = self.Tau[i,
                                                      :, :] - self.TauOld[i, :, :]
            else:
                self.Tau[i, :, :] = self.mapTau.perturbTau(
                    self.deltaXiField, self.deltaEtaField, self.deltakField)
                self.TauOld[i, :, :] = self.TauBaseline
                self.deltaTau[i, :, :] = self.Tau[i, :, :] - \
                    self.TauOld[i, :, :]
        i = 0
        if (self.enablePP):
            if (self.perturbVA) or (self.perturbVB) or (self.perturbVC):
                for job in self.jobs:
                    self.Tau[i, :, :] = job()
                    self.TauOld[i, :, :] = self.TauBaseline
                    self.deltaTau[i, :, :] = self.Tau[i,
                                                      :, :] - self.TauOld[i, :, :]
                    i += 1
            # TODO: warning:collapse time for perturbing Tau above need 1.78s (3000 cells)
            # if 10000 samples for this takes 17800s = 5 hours, 1000 samples
            # for this takes 35 minites (3000 cells)
        for i in range(Ns):
            if (self.txtfileOutput):
                if (self.verboseLevel > 0):
                    np.savetxt(self._debugFolderName + 'init/Tau/' +
                               'Tau_init_sample-' + str(i), self.Tau[i, :, :])
                    np.savetxt(self._debugFolderName +
                               'init/Tau/' +
                               'deltaTau_init_sample-' +
                               str(i), self.deltaTau[i, :, :])

        toc = time.time()
        print "\ncollapse time for initially constructed perturbed Tau field samples ", toc - tic, "s"

        # initial of parameter ensemble
        if (self.perturbVA) or (self.perturbVB) or (self.perturbVC):
            self.XC = np.vstack(
                (self.omegaXiM,
                 self.omegaEtaM,
                 self.omegakM,
                 self.omegaVAM,
                 self.omegaVBM,
                 self.omegaVCM))
        else:
            self.XC = np.vstack((self.omegaXiM, self.omegaEtaM, self.omegakM))

        # the number of variable each cell contain before augmentation (not
        # including parameters)
        self.Ns = Ns   # number of samples in the ensemble
        # specify number of variable in each cell
        if (not self.TauOnFlag):
            print "The state only contains velocity (u, v, w)"
            # number of variables in each cell (not including parameter)
            self.Nvariable = FoamTauSolver.NVec
        else:
            print "The state contains both velocity and Reynolds stress"
            self.Nvariable = FoamTauSolver.NVec + FoamTauSolver.NTensor

        # Specify the # of parameters (unknown)
        if (self.perturbVA) or (self.perturbVB) or (self.perturbVC):
            self.Npara = self.nModesXi + self.nModesEta + self.nModesK + 3 * self.nModesV
        else:
            self.Npara = self.nModesXi + self.nModesEta + self.nModesK
        # Specify # of state variable and observed state
        self.nstate = (self.Ncell * self.Nvariable) + \
            self.Npara  # size of state variables
        # size of observed state variables
        self.nstate_obs = self.Nvariable * self.NcellObs

        # output perturbation of physical compoents
        self._outputDeltaComponents()

        # output physical components
        tic = time.time()
        self._getPhyComponents()
        toc = time.time()
        print "\ncollapse time for getPhyComponents ", toc - tic, "s"

        # state vector without parameters
        self.XU = np.zeros([self.Nvariable * self.Ncell, Ns])
        # observed state vector without parameters
        self.HXU = np.zeros([self.Nvariable * self.NcellObs, Ns])

        # Observation vectors stacked in matrices
        # state matirx of observation: part U (u,v,w)
        self.ObsX = np.zeros([self.Nvariable * self.NcellObs, Ns])

        if (self.txtfileOutput):
            if (self.verboseLevel > 1):
                np.savetxt(
                    self._debugFolderName +
                    'init/' +
                    'omegaXi_ensemble_init',
                    self.omegaXiM)
                np.savetxt(
                    self._debugFolderName +
                    'init/' +
                    'omegaEta_ensemble_init',
                    self.omegaEtaM)
                np.savetxt(
                    self._debugFolderName +
                    'init/' +
                    'omegak_ensemble_init',
                    self.omegakM)
                np.savetxt(
                    self._debugFolderName +
                    'init/' +
                    'Tau-org',
                    self.TauBaseline)

        # Observation error covariance (could be given or default generate)
        # TODO: need specify the R locally
        if self.pseudoObs == 0:
            # This is for real experiment Data
            #Robs = np.loadtxt('Robs.txt')
            #self.Robs = sp.coo_matrix(Robs)

            self.rmu = float(paramDict['ObserveMean'])
            ObsSigmaFixedVec = extract_list_from_string(
                paramDict['ObsSigmaFixedVec'])
            self.ObsSigmaFixedVec = [float(pn) for pn in ObsSigmaFixedVec]

            ObsRelCoeffVec = extract_list_from_string(
                paramDict['ObsRelCoeffVec'])
            self.ObsRelCoeffVec = [float(pn) for pn in ObsRelCoeffVec]

            self.ObsRmaCoeff = float(paramDict['ObsRmaCoeff'])
            self.ObsErrCoeff = float(paramDict['ObsErrCoeff'])

            rsigmaVecSq = [pn**2 for pn in self.ObsSigmaFixedVec]
            self.Robs = sp.diags(np.array(rsigmaVecSq * self.NcellObs), 0)

        if self.pseudoObs == 1:
            self.rmu = float(paramDict['ObserveMean'])
            ObsSigmaFixedVec = extract_list_from_string(
                paramDict['ObsSigmaFixedVec'])
            self.ObsSigmaFixedVec = [float(pn) for pn in ObsSigmaFixedVec]

            ObsRelCoeffVec = extract_list_from_string(
                paramDict['ObsRelCoeffVec'])
            self.ObsRelCoeffVec = [float(pn) for pn in ObsRelCoeffVec]

            self.ObsRmaCoeff = float(paramDict['ObsRmaCoeff'])

            rsigmaVecSq = [pn**2 for pn in self.ObsSigmaFixedVec]
            self.Robs = sp.diags(np.array(rsigmaVecSq * self.NcellObs), 0)

        # TODO: need specified outside
        self.normalizeFlag = False
        self.U0 = 0.5
        self.K0 = 1e-3
        self.forward_interval = float(paramDict['forward_interval'])
        # DA step interval Todo:
        # Used in main driver
        self.iDAStep = 0
        self.TotalDASteps = np.int_(
            np.ceil(max_pseudo_time / self.forward_interval))
        self.tPlot = np.zeros(self.TotalDASteps)
        self.XPlot = np.zeros([self.TotalDASteps, self.nstate, Ns])
        self.XCMeanPlot = np.zeros([self.TotalDASteps, self.Npara])

        self.nextEndTime = self.forward_interval
        self.init_state = np.zeros(self.nstate)

    def __str__(self):
        s = 'FoamTauSolver instance'
        return s

    # required methods

    def generate_ensemble(self):
        """ Generate OF case folders and X, HX.

        Args:
        DAInterval: DA step interval

        Returns:
        X: ensemble matrix of whole states
        HX: ensemble matrix of whole states in observation space
        """
        # initialize OpenFoam environment
        # generate cases folders for each ensemble members (Tau is replaced by perturbed Tau)
        # generate observation folder (benchmark case)
        foamOp.genFolders(self.Npara, self.Ns, self.caseName,
                          self.caseNameObservation, self.forward_interval, self.Tau)
        # make value of forward_interval to e-6 string
        writeInterval = "%.6f" % self.forward_interval

        X = np.zeros([self.nstate, self.Ns])
        HX = np.zeros([self.nstate_obs, self.Ns])

        # generate initial X
        caseCount = np.linspace(1, self.Ns, self.Ns)
        os.system('rm -rf ' + self._debugFolderName + 'init-propagate/*')
        for case in caseCount:
            tmpCaseName = self.caseName + "-tmp_" + str(case)
            if (self.screenOutput):
                print "#", case, "/", self.Ns, " solving PDE equations", "to obtain IC"

            if(self.enablePP):
                self.jobs.append(
                    # invoke the dynamic core
                    self.job_server.submit(
                        foamOp.callFoam, (tmpCaseName, self.caseSolver, self.pseudoObs, False))
                )
            else:
                foamOp.callFoam(
                    tmpCaseName,
                    self.caseSolver,
                    self.pseudoObs,
                    False)

        print "Waiting for all OpenFOAM runs to finish ..."
        # Barrier to make sure all cases are finished before moving on
        if(self.enablePP):
            #part_sum1 = sum([job() for job in self.jobs])
            part_sum1 = [job() for job in self.jobs]
            self.jobs = []

        part_sum1 = 0

        # finished all ensemble propagation. print stats
        if(self.enablePP):
            self.job_server.print_stats()

        idx = 0
        for case in caseCount:
            if (self.screenOutput):
                print "#", case, "/", self.Ns, \
                    " processing results to obtaine IC"

            tmpCaseName = self.caseName + "-tmp_" + str(case)

            # get XU (state without augmentation)
            # get XU##########################################################
            UFile = ospt.join(os.getcwd(), tmpCaseName, writeInterval, "U")
            U = foamOp.readVelocityFromFile(UFile)
            TauFile = ospt.join(os.getcwd(), tmpCaseName, writeInterval, "Tau")
            Tau = foamOp.readTurbStressFromFile(TauFile)
            Tau = Tau.flatten()

            if self.normalizeFlag:
                U = U / self.U0
                Tau = Tau / self.K0

            if (not self.TauOnFlag):
                self.XU[:, idx] = U
            else:
                self.XU[:, idx] = np.hstack((U, Tau))

            deltaXiFile = ospt.join(os.getcwd(), tmpCaseName, '0/deltaXi')
            deltaEtaFile = ospt.join(os.getcwd(), tmpCaseName, '0/deltaEta')
            deltakFile = ospt.join(os.getcwd(), tmpCaseName, '0/deltak')
            deltaVAFile = ospt.join(os.getcwd(), tmpCaseName, '0/deltaVA')
            deltaVBFile = ospt.join(os.getcwd(), tmpCaseName, '0/deltaVB')
            deltaVCFile = ospt.join(os.getcwd(), tmpCaseName, '0/deltaVC')
            deltaTauFile = ospt.join(os.getcwd(), tmpCaseName, '0/deltaTau')

            # Write deltaXi and deltaEta in each case folder
            deltaXi = self.deltaXiM[idx, :, :]
            deltaEta = self.deltaEtaM[idx, :, :]
            deltak = self.deltakM[idx, :, :]
            deltaVA = self.deltaVAM[idx, :, :]
            deltaVB = self.deltaVBM[idx, :, :]
            deltaVC = self.deltaVCM[idx, :, :]

            foamOp.writeScalarToFile(deltaXi, deltaXiFile)
            foamOp.writeScalarToFile(deltaEta, deltaEtaFile)
            foamOp.writeScalarToFile(deltak, deltakFile)
            foamOp.writeScalarToFile(deltaVA, deltaVAFile)
            foamOp.writeScalarToFile(deltaVB, deltaVBFile)
            foamOp.writeScalarToFile(deltaVC, deltaVCFile)
            foamOp.writeTurbStressToFile(
                self.deltaTau[idx, :, :], deltaTauFile)

            idx += 1

        # make state ensemble matrix: X = [u,v,w,k,C1,C2]
        X = np.vstack((self.XU, self.XC))

        if self.pseudoObs == 0:
            H = self._constructHMatrix()
            HX = H.dot(X)
            # print "HX \n", HX.shape
        if self.pseudoObs == 1:
            pass

        print "#", "solving PDE to get pseudo truth"
        foamOp.callFoam(
            self.caseNameObservation,
            self.caseSolver,
            self.pseudoObs)

        if (self.txtfileOutput):
            np.savetxt(self._debugFolderName + 'init-propagate/' + 'X_init', X)
            np.savetxt(
                self._debugFolderName +
                'init-propagate/' +
                'HX_init',
                HX)
        return (X, HX)

    def forecast_to_time(self, X, nextEndTime):
        """ Do nothing (static model). """
        return X

    def state_to_observation(self, X):
        """ Call the dynamic core to evolve the ensemble states to time $tDest$

        Args:
        X: ensemble matrix of whole state at current time
        nextEndTime: target time of this forecast

        Returns:
        X: ensemble matrix of whole states at next DA time
        HX: projection of X (above) in observation space
        """
        # split the X to XU,Xk,XC
        (self.XU, self.XC) = self._SplitEnsemble(X)
        # split the XC to omegaXiM and omegaEtaM
        self._SplitXC(self.XC)

        # Inflation:(TODO: developing)
        # stdXC_max = np.min(np.std(self.XC, axis=0))
        # Robs = self.Robs.todense()
        # stdObsErr_max = np.sqrt(np.max(np.diag(np.sqrt(Robs))));
        # if stdXC_max/stdObsErr_max < 5e-1:
        #     self.XC = self.XC + 0.5*np.randn(self.XC.shape)

        # modify OpenFoam files
        # To rewrite all files that changed for OpenFOAM

        self.nextEndTime += self.forward_interval
        newStartTime = self.nextEndTime - self.forward_interval

        self._modifyOpenFOAM(X, newStartTime)
        self.deltaTau = self.Tau - self.TauOld
        newStopTime = "%.6f" % self.nextEndTime

        # propagate to newStopTime
        caseCount = np.linspace(1, self.Ns, self.Ns)

        for case in caseCount:
            if (self.screenOutput):
                print "#", case, "/", self.Ns, " solving PDE equations"
            tmpCaseName = self.caseName + "-tmp_" + str(case)

            if(self.enablePP):
                self.jobs.append(
                    # invoke the dynamic core
                    self.job_server.submit(
                        foamOp.callFoam, (tmpCaseName, self.caseSolver, self.pseudoObs, False))
                )
            else:
                foamOp.callFoam(
                    tmpCaseName,
                    self.caseSolver,
                    self.pseudoObs,
                    False)

        # Barrier to make sure all cases are finished before moving on
        if(self.enablePP):
            #part_sum1 = sum([job() for job in self.jobs])
            part_sum1 = [job() for job in self.jobs]
            self.jobs = []

        part_sum1 = 0

        # finished all ensemble propagation. print stats
        if(self.enablePP):
            self.job_server.print_stats()

        idx = 0
        # process results from OpenFOAM runs above
        for case in caseCount:
            if (self.screenOutput):
                print "#", case, "/", self.Ns, " processing results", "DAloop"

            tmpCaseName = self.caseName + "-tmp_" + str(case)
            UFile = ospt.join(os.getcwd(), tmpCaseName, newStopTime, "U")
            U = foamOp.readVelocityFromFile(UFile)
            TauFile = ospt.join(os.getcwd(), tmpCaseName, newStopTime, "Tau")
            Tau = foamOp.readTurbStressFromFile(TauFile)
            Tau = Tau.flatten()

            if self.normalizeFlag:
                U = U / self.U0
                Tau = Tau / self.K0

            if (not self.TauOnFlag):
                self.XU[:, idx] = U
            else:
                self.XU[:, idx] = np.hstack((U, Tau))

            # get HX
            # TODO: we need HXS
            if self.pseudoObs == 0:
                # Observation should be experiment data
                pass
            if self.pseudoObs == 1:
                # TODO: for transient case need to use sample
                pass

            idx += 1

        # make state ensemble matrix: X = [u,v,w,omegaXi,omegaEta]
        X = np.vstack((self.XU, self.XC))

        DAstep = (self.nextEndTime - self.forward_interval) / \
            self.forward_interval

        if (self.txtfileOutput):
            np.savetxt(self._debugFolderName + 'XC_' + str(DAstep), self.XC)
        self.XPlot[self.iDAStep] = X
        self.tPlot[self.iDAStep] = self.nextEndTime
        self.iDAStep += self.iDAStep

        if self.pseudoObs == 0:
            #os.system('sample -case ' + self.caseNameObservation)
            H = self._constructHMatrix()
            HX = H.dot(X)
        if self.pseudoObs == 1:
            pass

        print "#", "solving PDE to obtain pseudo truth"
        foamOp.callFoam(
            self.caseNameObservation,
            self.caseSolver,
            self.pseudoObs)

        return HX

    def get_obs(self, next_end_time):
        obs = self._observe(next_end_time)
        R_obs = self.Robs.todense().A
        return obs, R_obs

    def clean(self):
        ''' Perform any necessary cleanup before exiting.
        '''
        os.remove('xcoor.txt')
        os.remove('ycoor.txt')
        os.remove('zcoor.txt')
        os.remove('cellVolumn.txt')
        os.remove('scalarTemp')
        os.remove('tauUpdate')
        os.remove('Utemp')
        os.remove('UM.txt')
        os.remove('tau.txt')

    # internal methods

    def _getPhyComponents(self):
        """
        Output physical components (c1, c2, c3, xi, eta, k, XC1, XC2, VA, VB, VC)
        """
        # Note: for this function, time is linearly increasing with samples,
        # each samples takes 3s, indicating, 1000 samples this function takes
        # 3000s (1 hour)
        self.c1Fields = np.zeros([self.Ns, self.Ncell])
        self.c2Fields = np.zeros([self.Ns, self.Ncell])
        self.c3Fields = np.zeros([self.Ns, self.Ncell])

        self.XC1Fields = np.zeros([self.Ns, self.Ncell])
        self.XC2Fields = np.zeros([self.Ns, self.Ncell])

        self.XiFields = np.zeros([self.Ns, self.Ncell])
        self.EtaFields = np.zeros([self.Ns, self.Ncell])
        self.kFields = np.zeros([self.Ns, self.Ncell])
        self.VAFields = np.zeros([self.Ns, self.Ncell])
        self.VBFields = np.zeros([self.Ns, self.Ncell])
        self.VCFields = np.zeros([self.Ns, self.Ncell])

        # TODO, in ReynoldStressRF.py class, when you use _tau2PhysParams(), the pertubation
        #       will be introduced. This is not what we want.
        mapTau = ReRF.ReynoldsStressRF(
            self.baseCaseDir, 'Tau', self.Ncell, 1, True)

        for isamp in np.arange(self.Ns):
            TauNew = self.Tau[isamp, :, :]
            k, V1, V2, V3, C, NP = mapTau._tau2PhysParams(
                TauNew)  # collapse time = 1.005s (3000 cells)
            X = mapTau._C2X(C)  # collapse time = 0.02s (3000 cells)
            RS = mapTau._phys2Natural(X)  # collapse time = 0.02s (3000 cells)
            # collapse time = 1.005s (3000 cells)
            VA, VB, VC = mapTau.getThetaVABC(TauNew)

            self.c1Fields[isamp, :] = C[:, 0]
            self.c2Fields[isamp, :] = C[:, 1]
            self.c3Fields[isamp, :] = C[:, 2]

            self.XC1Fields[isamp, :] = X[:, 0]
            self.XC2Fields[isamp, :] = X[:, 1]

            self.XiFields[isamp, :] = RS[:, 0]
            self.EtaFields[isamp, :] = RS[:, 1]
            self.kFields[isamp, :] = k[:, 0]
            self.VAFields[isamp, :] = VA[:, 0]
            self.VBFields[isamp, :] = VB[:, 0]
            self.VCFields[isamp, :] = VC[:, 0]

        # baseline
        k, V1, V2, V3, C, NP = mapTau._tau2PhysParams(self.TauBaseline)
        X = mapTau._C2X(C)
        RS = mapTau._phys2Natural(X)
        VA, VB, VC = mapTau.getThetaVABC(self.TauBaseline)

        self.c1Field_base = C[:, 0]
        self.c2Field_base = C[:, 1]
        self.c3Field_base = C[:, 2]

        self.XC1Field_base = X[:, 0]
        self.XC2Field_base = X[:, 1]

        self.XiField_base = RS[:, 0]
        self.EtaField_base = RS[:, 1]
        self.kField_base = k[:, 0]

        self.VAField_base = VA[:, 0]
        self.VBField_base = VB[:, 0]
        self.VCField_base = VC[:, 0]

        # output all components

        np.savetxt(self._debugFolderName + 'init/XC1_s', self.XC1Fields)
        np.savetxt(self._debugFolderName + 'init/XC2_s', self.XC2Fields)
        np.savetxt(self._debugFolderName + 'init/c1_s', self.c1Fields)
        np.savetxt(self._debugFolderName + 'init/c2_s', self.c2Fields)
        np.savetxt(self._debugFolderName + 'init/c3_s', self.c3Fields)
        np.savetxt(self._debugFolderName + 'init/Xi_s', self.XiFields)
        np.savetxt(self._debugFolderName + 'init/Eta_s', self.EtaFields)
        np.savetxt(self._debugFolderName + 'init/TKE_s', self.kFields)
        np.savetxt(self._debugFolderName + 'init/VA_s', self.VAFields)
        np.savetxt(self._debugFolderName + 'init/VB_s', self.VBFields)
        np.savetxt(self._debugFolderName + 'init/VC_s', self.VCFields)

        np.savetxt(self._debugFolderName + 'init/XC1_base', self.XC1Field_base)
        np.savetxt(self._debugFolderName + 'init/XC2_base', self.XC2Field_base)
        np.savetxt(self._debugFolderName + 'init/c1_base', self.c1Field_base)
        np.savetxt(self._debugFolderName + 'init/c2_base', self.c2Field_base)
        np.savetxt(self._debugFolderName + 'init/c3_base', self.c3Field_base)
        np.savetxt(self._debugFolderName + 'init/Xi_base', self.XiField_base)
        np.savetxt(self._debugFolderName + 'init/Eta_base', self.EtaField_base)
        np.savetxt(self._debugFolderName + 'init/TKE_base', self.kField_base)
        np.savetxt(self._debugFolderName + 'init/VA_base', self.VAField_base)
        np.savetxt(self._debugFolderName + 'init/VB_base', self.VBField_base)
        np.savetxt(self._debugFolderName + 'init/VC_base', self.VCField_base)

    def _outputDeltaComponents(self):
        """
        Output perturbation in 6 components
        """
        np.savetxt(self._debugFolderName +
                   'init/deltaXi_s', self.deltaXiM[:, :, 0])
        np.savetxt(self._debugFolderName +
                   'init/deltaEta_s', self.deltaEtaM[:, :, 0])
        np.savetxt(self._debugFolderName +
                   'init/deltaK_s', self.deltakM[:, :, 0])
        np.savetxt(self._debugFolderName +
                   'init/deltaVA_s', self.deltaVAM[:, :, 0])
        np.savetxt(self._debugFolderName +
                   'init/deltaVB_s', self.deltaVBM[:, :, 0])
        np.savetxt(self._debugFolderName +
                   'init/deltaVC_s', self.deltaVCM[:, :, 0])

    def _SplitEnsemble(self, X):
        """ Function is to Split X to XU, XC

        Args:
        X: The matrix which needed to be splited

        Returns:
        XU: component of ensemble matrix of state: (u,v,w)
        XC: component of ensemble matrix of state: parameters
        """
        self.XU = X[0:(self.Nvariable * self.Ncell), :]
        self.XC = X[(self.Nvariable * self.Ncell):, :]

        return self.XU, self.XC

    def _SplitXC(self, XC):
        """ Function is to Split XC to omegas

        Args:
        XC: The parameter matrix which needed to be splited

        Returns:
        omegaXiM:  omegaXi
        omegaEtaM: omegaEta
        """

        if (self.perturbVA) or (self.perturbVB) or (self.perturbVC):
            self.omegaXiM = XC[0:self.nModesXi, :]
            self.omegaEtaM = XC[self.nModesXi:self.nModesXi +
                                self.nModesEta, :]
            self.omegakM = XC[self.nModesXi +
                              self.nModesEta: self.nModesXi +
                              self.nModesEta +
                              self.nModesV, :]
            self.omegaVAM = XC[self.nModesXi +
                               self.nModesEta +
                               self.nModesV: self.nModesXi +
                               self.nModesEta +
                               2 *
                               self.nModesV, :]
            self.omegaVBM = XC[self.nModesXi +
                               self.nModesEta +
                               2 *
                               self.nModesV: self.nModesXi +
                               self.nModesEta +
                               3 *
                               self.nModesV, :]
            self.omegaVCM = XC[self.nModesXi +
                               self.nModesEta + 3 * self.nModesV:, :]
        else:
            self.omegaXiM = XC[0:self.nModesXi, :]
            self.omegaEtaM = XC[self.nModesXi:self.nModesXi +
                                self.nModesEta, :]
            self.omegakM = XC[self.nModesXi + self.nModesEta:, :]

    def _modifyOpenFOAM(self, X, newStartTime):
        """ Function is to replace U and tau file and contrdict in OpenFoam

        Args:
        X: The ensemble matrix of state
        nextEndTime: next DA interval

        Returns:
        None
        """
        # TODO: if we need modify U??? we need think about it!

        # initialize tau mapping class
        if (self.capBaseline):
            mapTau = ReRF.ReynoldsStressRF(
                self.baseCaseDir, 'Tau', self.Ncell, 1, True)
        else:
            mapTau = ReRF.ReynoldsStressRF(
                self.baseCaseDir, 'Tau', self.Ncell, 1)
        # TODO: if transient problem, here self.baseCaseDir need to be propagate
        #      as well
        ii = 0
        caseCount = np.linspace(1, self.Ns, self.Ns)
        DAstep = newStartTime / self.forward_interval
        if (self._iDebug):
            os.system('mkdir ' + self._debugFolderName + 'DA-' + str(DAstep))
            os.system(
                'mkdir ' +
                self._debugFolderName +
                'DA-' +
                str(DAstep) +
                '/Tau/')
        for case in caseCount:

            tic = time.time()
            #tmpCaseName = self.caseName + "-tmp_" + str(case)
            #caseStart = "%.6f"%newStartTime
            #tmpCaseDir = ospt.join(os.getcwd(), tmpCaseName, caseStart)

            ## Modify Tau #####################################################
            # Reconstruct field
            omegaXi = self.omegaXiM[:, ii]
            omegaXi = np.array([omegaXi])
            omegaXi = omegaXi.T
            self.deltaXiField = self.rfXi.reconstructField(
                omegaXi, self.KLModesXi)
            self.deltaXiM[ii, :, :] = self.deltaXiField

            omegaEta = self.omegaEtaM[:, ii]
            omegaEta = np.array([omegaEta])
            omegaEta = omegaEta.T
            self.deltaEtaField = self.rfEta.reconstructField(
                omegaEta, self.KLModesEta)
            self.deltaEtaM[ii, :, :] = self.deltaEtaField

            omegak = self.omegakM[:, ii]
            omegak = np.array([omegak])
            omegak = omegak.T
            self.deltakField = self.rfK.reconstructField(omegak, self.KLModesK)
            self.deltakM[ii, :, :] = self.deltakField
            toc = time.time()
            # print "Time for reconstruction is: ", toc - tic, "s"

            if (self.perturbVA) or (self.perturbVB) or (self.perturbVC):
                omegaVA = self.omegaVAM[:, ii]
                omegaVA = np.array([omegaVA])
                omegaVA = omegaVA.T
                self.deltaVAField = self.rfVA.reconstructField(
                    omegaVA, self.KLModesVA)
                self.deltaVAM[ii, :, :] = self.deltaVAField

                omegaVB = self.omegaVBM[:, ii]
                omegaVB = np.array([omegaVB])
                omegaVB = omegaVB.T
                self.deltaVBField = self.rfVB.reconstructField(
                    omegaVB, self.KLModesVB)
                self.deltaVBM[ii, :, :] = self.deltaVBField

                omegaVC = self.omegaVCM[:, ii]
                omegaVC = np.array([omegaVC])
                omegaVC = omegaVC.T
                self.deltaVCField = self.rfVC.reconstructField(
                    omegaVC, self.KLModesVC)
                self.deltaVCM[ii, :, :] = self.deltaVCField

                deltaVAll = np.hstack(
                    (self.deltaVAField, self.deltaVBField, self.deltaVCField))

            if (self.screenOutput):
                print "#", case, "/", self.Ns, " Modifying OpenFOAM Files "

            # mapping deltaXi deltaEta
            tic = time.time()
            if (self.perturbVA) or (self.perturbVB) or (self.perturbVC):
                if (self.enablePP):
                    self.jobs.append(
                        self.job_server.submit(
                            self.mapTau.perturbTau,
                            (self.deltaXiField,
                             self.deltaEtaField,
                             self.deltakField,
                             deltaVAll),
                            modules=(
                                "numpy",
                            )))
                    TauNew = []
                else:
                    TauNew = self.mapTau.perturbTau(
                        self.deltaXiField, self.deltaEtaField, self.deltakField, deltaVAll)
                    self.Tau[ii, :, :] = TauNew
                    self.deltaTau[ii, :, :] = self.Tau[ii,
                                                       :, :] - self.TauBaseline
            else:
                if (self.enablePP):
                    self.jobs.append(
                        self.job_server.submit(
                            self.mapTau.perturbTau,
                            (self.deltaXiField,
                             self.deltaEtaField,
                             self.deltakField),
                            modules=(
                                "numpy",
                            )))
                    TauNew = []
                else:
                    TauNew = self.mapTau.perturbTau(
                        self.deltaXiField, self.deltaEtaField, self.deltakField)
                    self.Tau[ii, :, :] = TauNew
                    self.deltaTau[ii, :, :] = self.Tau[ii,
                                                       :, :] - self.TauBaseline

            ii += 1

        ii = 0
        # Barrier to make sure all cases are finished before moviing on
        if (self.enablePP):
            for job in self.jobs:
                TauNew.append(job())
                self.Tau[ii, :, :] = job()
                self.deltaTau[ii, :, :] = self.Tau[ii, :, :] - self.TauBaseline
                ii += 1
            ii = 0

        part_sum1 = 0
        for case in caseCount:
            tmpCaseName = self.caseName + "-tmp_" + str(case)
            caseStart = "%.6f" % newStartTime
            tmpCaseDir = ospt.join(os.getcwd(), tmpCaseName, caseStart)
            tauFile = tmpCaseDir + '/Tau'
            foamOp.writeTurbStressToFile(self.Tau[ii, :, :], tauFile)

            os.system('cp ' + self.caseName + '/0/deltaTau ' + tmpCaseDir)
            deltaTauFile = tmpCaseDir + '/deltaTau'
            foamOp.writeScalarToFile(self.deltaTau[ii, :, :], deltaTauFile)

            # write tau component
            os.system('cp ' + self.caseName + '/0/deltaXi ' + tmpCaseDir)
            tmpFile = tmpCaseDir + '/deltaXi'
            replace(tmpFile, '"0"', caseStart)
            foamOp.writeScalarToFile(self.deltaXiM[ii, :, :], tmpFile)

            os.system('cp ' + self.caseName + '/0/deltaEta ' + tmpCaseDir)
            tmpFile = tmpCaseDir + '/deltaEta'
            replace(tmpFile, '"0"', caseStart)
            foamOp.writeScalarToFile(self.deltaEtaM[ii, :, :], tmpFile)

            os.system('cp ' + self.caseName + '/0/deltak ' + tmpCaseDir)
            tmpFile = tmpCaseDir + '/deltak'
            replace(tmpFile, '"0"', caseStart)
            foamOp.writeScalarToFile(self.deltakM[ii, :, :], tmpFile)

            os.system('cp ' + self.caseName + '/0/deltaVA ' + tmpCaseDir)
            tmpFile = tmpCaseDir + '/deltaVA'
            replace(tmpFile, '"0"', caseStart)
            foamOp.writeScalarToFile(self.deltaVAM[ii, :, :], tmpFile)

            os.system('cp ' + self.caseName + '/0/deltaVB ' + tmpCaseDir)
            tmpFile = tmpCaseDir + '/deltaVB'
            replace(tmpFile, '"0"', caseStart)
            foamOp.writeScalarToFile(self.deltaVBM[ii, :, :], tmpFile)

            os.system('cp ' + self.caseName + '/0/deltaVC ' + tmpCaseDir)
            tmpFile = tmpCaseDir + '/deltaVC'
            replace(tmpFile, '"0"', caseStart)
            foamOp.writeScalarToFile(self.deltaVCM[ii, :, :], tmpFile)

            self.XiFields[ii, :] = self.XiField_base + self.deltaXiM[ii, :, 0]
            self.EtaFields[ii, :] = self.EtaField_base + \
                self.deltaEtaM[ii, :, 0]
            self.kFields[ii, :] = self.kField_base * \
                np.exp2(self.deltakM[ii, :, 0])
            self.VAFields[ii, :] = self.VAField_base + self.deltaVAM[ii, :, 0]
            self.VBFields[ii, :] = self.VBField_base + self.deltaVBM[ii, :, 0]
            self.VCFields[ii, :] = self.VCField_base + self.deltaVCM[ii, :, 0]

            os.system('cp ' + self.caseName + '/0/Xi ' + tmpCaseDir)
            tmpFile = tmpCaseDir + '/Xi'
            replace(tmpFile, '"0"', caseStart)
            foamOp.writeScalarToFile(
                np.array([self.XiFields[ii, :]]).T, tmpFile)

            os.system('cp ' + self.caseName + '/0/Eta ' + tmpCaseDir)
            tmpFile = tmpCaseDir + '/Eta'
            replace(tmpFile, '"0"', caseStart)
            foamOp.writeScalarToFile(
                np.array([self.EtaFields[ii, :]]).T, tmpFile)

            os.system('cp ' + self.caseName + '/0/TKE ' + tmpCaseDir)
            tmpFile = tmpCaseDir + '/TKE'
            replace(tmpFile, '"0"', caseStart)
            foamOp.writeScalarToFile(
                np.array([self.kFields[ii, :]]).T, tmpFile)

            os.system('cp ' + self.caseName + '/0/VA ' + tmpCaseDir)
            tmpFile = tmpCaseDir + '/VA'
            replace(tmpFile, '"0"', caseStart)
            foamOp.writeScalarToFile(
                np.array([self.VAFields[ii, :]]).T, tmpFile)

            os.system('cp ' + self.caseName + '/0/VB ' + tmpCaseDir)
            tmpFile = tmpCaseDir + '/VB'
            replace(tmpFile, '"0"', caseStart)
            foamOp.writeScalarToFile(
                np.array([self.VBFields[ii, :]]).T, tmpFile)

            os.system('cp ' + self.caseName + '/0/VC ' + tmpCaseDir)
            tmpFile = tmpCaseDir + '/VC'
            replace(tmpFile, '"0"', caseStart)
            foamOp.writeScalarToFile(
                np.array([self.VCFields[ii, :]]).T, tmpFile)

            ## Modify U and save the original file ############################
            UFile = tmpCaseDir + '/U'
            UorgFile = tmpCaseDir + '/Uorg'
            TauFile = tmpCaseDir + '/Tau'
            TauUpdateFile = tmpCaseDir + '/Tau_update'
            os.system('cp ' + TauFile + ' ' + TauUpdateFile)
            os.system('cp ' + UFile + ' ' + UorgFile)
            os.system('sed -i \'s/U/Uorg/\' ' + UorgFile)
            if self.TauOnFlag:
                XU, XTau = self._splitUTau(self.XU[:, ii])
                UNew = np.reshape(XU, (-1, 3))
                foamOp.writeTurbStressToFile(XTau, TauUpdateFile)
            else:
                UNew = np.reshape(self.XU[:, ii], (-1, 3))

            if self.normalizeFlag:
                UNew = UNew * self.U0

            foamOp.writeVelocityToFile(UNew, UFile)

            toc = time.time()
            ii += 1
        ii = 0

        for case in caseCount:
            if (self.txtfileOutput):

                if (self.verboseLevel > 0):
                    tic = time.time()
                    np.savetxt(self._debugFolderName +
                               'DA-' +
                               str(DAstep) +
                               '/Tau/' +
                               'Tau_sample-' +
                               str(ii), self.Tau[ii, :, :])
                    np.savetxt(self._debugFolderName +
                               'DA-' +
                               str(DAstep) +
                               '/Tau/' +
                               'deltaTau_sample-' +
                               str(ii), self.deltaTau[ii, :, :])

                toc = time.time()
                # print "Time for diagnosis output is: ", toc - tic, "s"
                ii += 1

        if (self.txtfileOutput):
            if (self.verboseLevel > 0):
                np.savetxt(
                    self._debugFolderName +
                    'DA-' +
                    str(DAstep) +
                    '/XC1_s',
                    self.XC1Fields)
                np.savetxt(
                    self._debugFolderName +
                    'DA-' +
                    str(DAstep) +
                    '/XC2_s',
                    self.XC2Fields)

                np.savetxt(
                    self._debugFolderName +
                    'DA-' +
                    str(DAstep) +
                    '/' +
                    'XiField',
                    self.XiFields)
                np.savetxt(
                    self._debugFolderName +
                    'DA-' +
                    str(DAstep) +
                    '/' +
                    'EtaField',
                    self.EtaFields)
                np.savetxt(
                    self._debugFolderName +
                    'DA-' +
                    str(DAstep) +
                    '/' +
                    'TKEField',
                    self.kFields)
                np.savetxt(
                    self._debugFolderName +
                    'DA-' +
                    str(DAstep) +
                    '/' +
                    'VAField',
                    self.VAFields)
                np.savetxt(
                    self._debugFolderName +
                    'DA-' +
                    str(DAstep) +
                    '/' +
                    'VBField',
                    self.VBFields)
                np.savetxt(
                    self._debugFolderName +
                    'DA-' +
                    str(DAstep) +
                    '/' +
                    'VCField',
                    self.VCFields)

                np.savetxt(self._debugFolderName + 'DA-' +
                           str(DAstep) + '/deltaXi_s', self.deltaXiM[:, :, 0])
                np.savetxt(self._debugFolderName + 'DA-' + str(DAstep) +
                           '/deltaEta_s', self.deltaEtaM[:, :, 0])
                np.savetxt(self._debugFolderName + 'DA-' +
                           str(DAstep) + '/deltaK_s', self.deltakM[:, :, 0])
                np.savetxt(self._debugFolderName + 'DA-' +
                           str(DAstep) + '/deltaVA_s', self.deltaVAM[:, :, 0])
                np.savetxt(self._debugFolderName + 'DA-' +
                           str(DAstep) + '/deltaVB_s', self.deltaVBM[:, :, 0])
                np.savetxt(self._debugFolderName + 'DA-' +
                           str(DAstep) + '/deltaVC_s', self.deltaVCM[:, :, 0])

    def _splitUTau(self, X):
        """

        :param XU:
        :return:
        """
        XU = X[0: (FoamTauSolver.NVec * self.Ncell)]
        XTau = X[(FoamTauSolver.NVec * self.Ncell):]

        return XU, XTau

    def _splitUTauObs(self, XObs):
        """

        :param XU:
        :return:
        """
        UObs = XObs[0: (FoamTauSolver.NVec * self.NcellObs)]
        TauObs = XObs[(FoamTauSolver.NVec * self.NcellObs):]

        return UObs, TauObs

    def _observe(self, nextEndTime):
        """ Function is to get observation Data from experiment

        Arg:

        Returns:
        Obs: observation matrix
        """
        if self.TauOnFlag:
            ObsFile = 'obsX'
            absErr = np.zeros(9)
            absErrSigma = np.zeros(9)
            relErr = np.zeros(9)
            relErrSigma = np.zeros(9)
        else:
            ObsFile = 'obsVelocity'
            absErr = np.zeros(3)
            absErrSigma = np.zeros(3)
            relErr = np.zeros(3)
            relErrSigma = np.zeros(3)

        ObsVec = np.loadtxt(
            ospt.join(
                self.caseNameObservation,
                'observationData/' +
                ObsFile))
        obs_perturb_vec = np.zeros(ObsVec.shape)
        obs_perturb = np.zeros([self.nstate_obs, self.Ns])
        iobs = 0
        smallVal = 1e-10
        rSigmaVec = np.zeros(self.nstate_obs)
        if self.TauOnFlag:
            for i in range(self.nstate_obs / 9):
                for j in range(9):
                    # absolute error
                    absErrSigma[j] = self.ObsSigmaFixedVec[j]
                    absErr[j] = (
                        np.random.normal(
                            self.rmu,
                            absErrSigma[j],
                            1))[0]

                    # relative error
                    relErrSigma[j] = abs(
                        self.ObsRelCoeffVec[j] * ObsVec[iobs]) + smallVal
                    relErr[j] = (
                        np.random.normal(
                            self.rmu,
                            relErrSigma[j],
                            1))[0]

                    ObsVec[iobs + j] = ObsVec[iobs + j] + \
                        (absErr[j] + relErr[j]) * self.ObsErrCoeff
                    obs_perturb_vec[iobs + j] = relErrSigma[j] + absErrSigma[j]
                    rSigmaVec[iobs + j] = (relErrSigma[j] + absErrSigma[j])**2
                iobs = iobs + 9
        else:
            for i in range(self.nstate_obs / 3):
                for j in range(3):
                    # absolute error
                    absErrSigma[j] = self.ObsSigmaFixedVec[j]
                    absErr[j] = (
                        np.random.normal(
                            self.rmu,
                            absErrSigma[j],
                            1))[0]

                    # relative error
                    relErrSigma[j] = abs(
                        self.ObsRelCoeffVec[j] * ObsVec[iobs]) + smallVal
                    relErr[j] = (
                        np.random.normal(
                            self.rmu,
                            relErrSigma[j],
                            1))[0]

                    ObsVec[iobs + j] = ObsVec[iobs + j] + \
                        (absErr[j] + relErr[j]) * self.ObsErrCoeff
                    rSigmaVec[iobs + j] = (relErrSigma[j] + absErrSigma[j])**2
                iobs = iobs + 3

        rSigmaVec = self.ObsRmaCoeff * rSigmaVec
        if self.TauOnFlag:
            UObs, TauObs = self._splitUTauObs(ObsVec)
            USigma, TauSigma = self._splitUTauObs(rSigmaVec)
            if self.normalizeFlag:
                UObs = UObs / self.U0
                TauObs = TauObs / self.K0
                USigma = USigma / self.U0 / self.U0
                TauSigma = TauSigma / self.K0 / self.K0
                ObsVec = np.hstack((UObs, TauObs))
                rSigmaVec = np.hstack((USigma, TauSigma))
        else:
            if self.normalizeFlag:
                ObsVec = ObsVec / self.U0
                rSigmaVec = rSigmaVec / self.U0 / self.U0

        self.Robs = sp.diags(rSigmaVec, 0)

        # Assemble vector to matrix ObsU
        irow = 0
        while True:
            self.ObsX[:, irow] = ObsVec
            obs_perturb[:, irow] = obs_perturb_vec
            irow = irow + 1
            if irow == self.Ns:
                break
            Obs = self.ObsX
        return ObsVec

    def _constructHMatrix(self):

        NVec = 3
        Ntensor = 6
        idx = np.loadtxt(
            self.caseNameObservation +
            "/constant/indexH.txt")  # index of H Matrix
        # weight (element of H Matrix)
        weight = np.zeros((idx.shape[0], 1))
        weight[:, 0] = np.loadtxt(
            self.caseNameObservation + "/constant/weightH.txt")

        m, n = idx.shape
        idx3 = np.zeros((m * NVec, n))
        weight3 = np.zeros((m * NVec, 1))

        currentI = 0
        for i in range(int(idx[:, 0].max()) + 1):  # for each block
            rg = np.where(idx[:, 0] == i)[0]
            start, duration = rg[0], len(rg)
            idxBlock = np.copy(idx[start:start + duration, :])

            for ii in range(duration):

                idxBlock[ii, 1] = idxBlock[ii, 1] * NVec

            wgtBlock = np.copy(weight[start:start + duration, :])
            idxBlock[:, 0] = currentI

            idxBlock1 = np.copy(idxBlock)
            idxBlock1[:, 0] += 1
            idxBlock1[:, 1] += 1

            idxBlock2 = np.copy(idxBlock)
            idxBlock2[:, 0] += 2
            idxBlock2[:, 1] += 2

            idx3[NVec * start:NVec * (start + duration),
                 :] = np.vstack((idxBlock, idxBlock1, idxBlock2))
            weight3[NVec *
                    start:NVec *
                    (start +
                     duration), :] = np.vstack((wgtBlock, wgtBlock, wgtBlock))

            currentI += NVec
        # construct if Tau is in the state

        if self.TauOnFlag:
            idx6 = np.zeros((m * Ntensor, n))
            weight6 = np.zeros((m * Ntensor, 1))
            idx6base_1 = currentI
            idx6base_2 = NVec * self.Ncell
            currentI = 0
            for i in range(int(idx[:, 0].max()) + 1):  # for each block
                rg = np.where(idx[:, 0] == i)[0]
                start, duration = rg[0], len(rg)
                idxBlock = np.copy(idx[start:start + duration, :])

                for ii in range(duration):
                    idxBlock[ii, 1] = idxBlock[ii, 1] * Ntensor

                wgtBlock = np.copy(weight[start:start + duration, :])
                idxBlock[:, 0] = currentI

                idxBlock1 = np.copy(idxBlock)
                idxBlock1[:, 0] += 1
                idxBlock1[:, 1] += 1

                idxBlock2 = np.copy(idxBlock)
                idxBlock2[:, 0] += 2
                idxBlock2[:, 1] += 2

                idxBlock3 = np.copy(idxBlock)
                idxBlock3[:, 0] += 3
                idxBlock3[:, 1] += 3

                idxBlock4 = np.copy(idxBlock)
                idxBlock4[:, 0] += 4
                idxBlock4[:, 1] += 4

                idxBlock5 = np.copy(idxBlock)
                idxBlock5[:, 0] += 5
                idxBlock5[:, 1] += 5

                idx6[Ntensor * start:Ntensor * (start + duration), :] = np.vstack(
                    (idxBlock, idxBlock1, idxBlock2, idxBlock3, idxBlock4, idxBlock5))
                weight6[Ntensor * start:Ntensor * (start + duration), :] = np.vstack(
                    (wgtBlock, wgtBlock, wgtBlock, wgtBlock, wgtBlock, wgtBlock))

                currentI += Ntensor
            idx6[:, 0] = idx6[:, 0] + idx6base_1
            idx6[:, 1] = idx6[:, 1] + idx6base_2
            idx9 = np.append(idx3, idx6, axis=0)
            weight9 = np.append(weight3, weight6, axis=0)
            H = sp.coo_matrix((weight9.flatten(1), (idx9[:, 0], idx9[:, 1])), shape=(
                self.nstate_obs, self.nstate))
        else:
            H = sp.coo_matrix((weight3.flatten(1), (idx3[:, 0], idx3[:, 1])), shape=(
                self.nstate_obs, self.nstate))
        return H
