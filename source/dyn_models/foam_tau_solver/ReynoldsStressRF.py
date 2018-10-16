import numpy as np
import numpy
import matplotlib.pylab as plt
import os.path as osp
import pdb
from mpl_toolkits.mplot3d import Axes3D
from foamFileOperation import *

class ReynoldsStressRF:

    """
    A class representing Reynolds stress as random fields.

    Note: Symmetric tensor fields \\tau_{ij} are represented as numpy array of
    dimension N x 6. since numpy stores data in row major order.

    1. Individual instances of perturbed Reynolds stresses are not stored as
    data member.  Instead, their values are written to OpenFOAM file and
    their workspace cleaned.

    Data Members:

        casePath: path of OpenFOAM case associated with tauOrg (string)
        tauFile: name of the file of Reynolds stress (string)
        N: # of cells in field (consistent to OpenFOAM cell ordering)
        m: # of of samples (perturbed instances)
        deltaXi: normalized perturbation/error coordinates,
                 horizontal. (N x m), in [-1, 1]
        deltaEta: normalized perturbation/error coordinates,
                 vertical. (N x m), in [-1, 1]
        xcs: coordinates of the triangle (1 x 3)
        tauOrg: original Reynolds stress field (N x 6)
        kOrg: TKE field obtained from eigendecomposition of tauOrg (N x 1)
        v1Org, v2Org, v3Org: eigenvectors of tauOrg (N x 3)
        COrg: Baycentric coordinates associated with tauOrg
        NPOrg: If if the matrix is right-handed (1) or left-handed (-1)
    """

    def __init__(self,casePath, tauFile, nMesh, nSample, correctInitTau = 'False'):

        """
            __init__(): constructor.

            Construct by reading from file or input argument. Also accept
            argument about kernel function of deltaXi and deltaEta
            (e.g., SqExp, which is Gaussian)
        """


        self.casePath = casePath        # path where tau is
        self.tauFile = tauFile          # name of tau file
        self.indexDict = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
        self.rotFactor = [1,-1,-1,1,1,-1]

        if correctInitTau == 'False':
            self.correctInitTau = False
        else:
            self.correctInitTau = True
        
        #TODO: the path here may cause some problem
        if casePath == 'None':
            self.tau = tauFile
        else:
            self.tau = readTurbStressFromFile(osp.join(casePath,tauFile))

        self.nMesh = self.tau.shape[0]  # number of cells

        self.nSample = nSample          # number of samples 
        self.deltaXi = np.zeros((self.nMesh,self.nSample))
        self.deltaEta = np.zeros((self.nMesh,self.nSample))
        self.xcs = np.array([[1,0],[0,0],[0.5,(3**0.5/2.0)]])
        self.triangle = np.array([[0,0],[1,0],
                                  [0.5,3**0.5/2.0],[0.5,3**0.5/2.0]])

        self.C = np.zeros((self.nMesh,3))
        self.V1 = np.zeros((self.nMesh,3))
        self.V2 = np.zeros((self.nMesh,3))
        self.V3 = np.zeros((self.nMesh,3))
        self.k = np.zeros((self.nMesh,1))
        self.NP = np.zeros((self.nMesh,1))

        self.COrg = np.zeros((self.nMesh,3))
        self.V1Org = np.zeros((self.nMesh,3))
        self.V2Org = np.zeros((self.nMesh,3))
        self.V3Org = np.zeros((self.nMesh,3))
        self.kOrg = np.zeros((self.nMesh,1))
        self.NPOrg = np.zeros((self.nMesh,1))

        self.kOrg,self.V1Org,self.V2Org,self.V3Org,self.COrg,self.NPOrg = \
            self._tau2PhysParams(self.tau)


        self.k,self.V1,self.V2,self.V3,self.C,self.NP = self._tau2PhysParams(self.tau)
        self.tauOrg = self.tau
        self.tau = self._C2Tau(self.C)

    def perturbTau(self,deltaXi,deltaEta,*deltaKT):

        """
        Perturb Reynolds stresses according to deltaXi, deltaEta, and deltaLog2K
        Write them to OpenFOAM instances.  First call _perturbXiEta() to
        perturb on \\xi, \eta plane, and map to Reynolds stresses

        deltaXi and deltaEta are based on the absolute value in the natural
        domain
        if deltaLog2K is activated, k will be treated as k*(2**deltaLog2K)
        where deltaLog2K = log2(K.new/K.old)
        """
        X = self._C2X(self.COrg)
        self.modifyError(deltaXi,deltaEta)
        RSs = self._phys2Natural(X)
        XNew = self._natural2Phys(RSs)
        self.C = self._X2C(XNew)
        if len(deltaKT) >= 2:
            tau = self._C2Tau(self.C,deltaKT[1:])
        else:
            tau = self._C2Tau(self.C)


        if len(deltaKT) > 0:
            tau = tau*numpy.exp2(deltaKT[0])
        return tau


    def perturbTauInBary(self,deltaXbary,deltaYbary,*deltaKT):

        """
        Perturb Reynolds stresses according to deltaXi, deltaEta, and deltaLog2K
        Write them to OpenFOAM instances.  First call _perturbXiEta() to
        perturb on \\xi, \eta plane, and map to Reynolds stresses

        deltaXi and deltaEta are based on the absolute value in the natural
        domain
        if deltaLog2K is activated, k will be treated as k*(2**deltaLog2K)
        where deltaLog2K = log2(K.new/K.old)
        """
        X = self._C2X(self.COrg)
        XNew = X
        XNew[:,0] = X[:,0] + deltaXbary.T
        XNew[:,1] = X[:,1] + deltaYbary.T       
        Cs = self._X2C(XNew)
        if len(deltaKT) == 2:
            tau = self._C2Tau(Cs,deltaKT[1:])
        else:
            tau = self._C2Tau(Cs)

        if len(deltaKT) > 0:
            tau = tau*numpy.exp2(deltaKT[0])
        return tau


    def getXiFactor(self):
        """
        Get the shrink factor of Xi from Xi-Eta plane to the Barycentric Triangle.

        This shrink factor is always smaller or equals to 1, as long as the point is within
        Barycentric Triangle.
        """
        k, V1, V2, V3, C, NP = self._tau2PhysParams(self.tau)
        X = self._C2X(C)
        factor = (self.xcs[2,1] - X[:,1]) / self.xcs[2,1]
        return factor

    def getDeltaXiEtaLog2K(self,tauOld,tauNew):

        """
        Get deltaXi deltaEta from two Tau fields

        where deltaLog2K = log2(K.new/K.old)
        """

        kOld,V1Old,V2Old,V3Old,COld,NPOld = self._tau2PhysParams(tauOld)
        kNew,V1New,V2New,V3New,CNew,NPNew = self._tau2PhysParams(tauNew)

        XOld = self._C2X(COld)
        XNew = self._C2X(CNew)

        RSOld = self._phys2Natural(XOld)
        RSNew = self._phys2Natural(XNew)

        Xi = np.zeros((self.nMesh,1))
        Eta = np.zeros((self.nMesh,1))
        deltaLog2K = np.zeros((self.nMesh,1))

        tiny = np.finfo(np.float).tiny

        for i in range(self.nMesh):
            deltaXi = RSNew[i,0]-RSOld[i,0]
            signXi = abs(deltaXi)/(deltaXi + tiny)
            deltaEta = RSNew[i,1]-RSOld[i,1]
            signEta = abs(deltaEta)/(deltaEta + tiny)
            dK = np.log2(kNew[i]/(kOld[i] + tiny))

            # relative xi and eta
            # Xi[i] = signXi*deltaXi/(1.0*signXi - RSOld[i,0] + tiny);
            # Eta[i] = signEta*deltaEta/(1.0*signEta - RSOld[i,1] + tiny);

            # absolute xi and eta
            Xi[i] = deltaXi
            Eta[i] = deltaEta
            deltaLog2K[i] = dK

        return Xi, Eta, deltaLog2K

    def getDeltaXYbaryLog2K(self,tauOld,tauNew):

        """
        Get deltaXbary deltaYbary from two Tau fields

        where deltaLog2K = log2(K.new/K.old)
        """

        kOld,V1Old,V2Old,V3Old,COld,NPOld = self._tau2PhysParams(tauOld)
        kNew,V1New,V2New,V3New,CNew,NPNew = self._tau2PhysParams(tauNew)

        XOld = self._C2X(COld)
        XNew = self._C2X(CNew)

        RSOld = self._phys2Natural(XOld)
        RSNew = self._phys2Natural(XNew)

        Xbary = np.zeros((self.nMesh,1))
        Ybary = np.zeros((self.nMesh,1))
        deltaLog2K = np.zeros((self.nMesh,1))

        tiny = np.finfo(np.float).tiny

        for i in range(self.nMesh):
            deltaXbary = XNew[i,0]-XOld[i,0]
            deltaYbary = XNew[i,1]-XOld[i,1]
            dK = np.log2(kNew[i]/(kOld[i] + tiny))

            # absolute xi and eta
            Xbary[i] = deltaXbary
            Ybary[i] = deltaYbary
            deltaLog2K[i] = dK

        return Xbary, Ybary, deltaLog2K

    def getXYinBary(self,tauOld):

        """
        Get deltaXi deltaEta from two Tau fields

        where deltaLog2K = log2(K.new/K.old)
        """

        kOld,V1Old,V2Old,V3Old,COld,NPOld = self._tau2PhysParams(tauOld)

        XOld = self._C2X(COld)

        Xbary = np.zeros((self.nMesh,1))
        Ybary = np.zeros((self.nMesh,1))

        for i in range(self.nMesh):
            Xbary[i] = XOld[i,0]
            Ybary[i] = XOld[i,1]
        return Xbary, Ybary

    def getXYKinBary(self,tauOld):

        """
        Get deltaXi deltaEta from two Tau fields

        where deltaLog2K = log2(K.new/K.old)
        """

        kOld,V1Old,V2Old,V3Old,COld,NPOld = self._tau2PhysParams(tauOld)

        XOld = self._C2X(COld)

        Xbary = np.zeros((self.nMesh,1))
        Ybary = np.zeros((self.nMesh,1))
        Log2K = np.zeros((self.nMesh,1))

        for i in range(self.nMesh):
            Xbary[i] = XOld[i,0]
            Ybary[i] = XOld[i,1]
            Log2K[i] = np.log2(kOld[i])
        return Xbary, Ybary, Log2K

    def getDeltaThetaVABC(self,tauOld,tauNew, *indexList):

        """
        Get deltaThetaVA deltaThetaVB deltaThetaVC from two Tau fields
        The theory is according to the defination of Eulerian Angles

        """

        ThetaVAOld, ThetaVBOld, ThetaVCOld = self.getThetaVABC(tauOld)
        if np.shape(indexList)[0] == 0:
            ThetaVANew, ThetaVBNew, ThetaVCNew = self.getThetaVABC(tauNew)
        else:
            ThetaVANew, ThetaVBNew, ThetaVCNew = self.getThetaVABC(tauNew, indexList)

        ThetaVA = np.zeros((self.nMesh,1))
        ThetaVB = np.zeros((self.nMesh,1))
        ThetaVC = np.zeros((self.nMesh,1))

        for i in range(self.nMesh):

            angleA = ThetaVANew[i] - ThetaVAOld[i]
            angleB = ThetaVBNew[i] - ThetaVBOld[i]
            angleC = ThetaVCNew[i] - ThetaVCOld[i]

            ThetaVA[i] = self._adjustThetaAngle2(angleA)
            ThetaVB[i] = self._adjustThetaAngle2(angleB)
            ThetaVC[i] = self._adjustThetaAngle2(angleC)

        return ThetaVA, ThetaVB, ThetaVC

    def getDeltaThetaVABC_direct(self,tau,tauNew,*indexList):

        """
        get the euler angle with respect to the coordinate system
        (1 0 0), (0 1 0) and (0 0 1)
        """

        k,V1,V2,V3,C,NP = self._tau2PhysParams(tau)
        kNew,V1New,V2New,V3New,CNew,NPNew = self._tau2PhysParams(tauNew)

        ThetaVAList = np.zeros((self.nMesh,1))
        ThetaVBList = np.zeros((self.nMesh,1))
        ThetaVCList = np.zeros((self.nMesh,1))

        indexDict = self.indexDict

        if np.shape(indexList)[0] > 0:
            indexList = indexList[0][0]
        
        for i in range(self.nMesh):

            # The rotation matrix for each basis is themself
            if np.shape(indexList)[0] == 0:
                VMatrix = np.array([V1[i],V2[i],V3[i]])
            else:
                VMatrixOrg = np.array([V1[i],V2[i],V3[i]])
                VMatrixNew = np.array([V1New[i],V2New[i],V3New[i]])
                VMatrix = np.dot(VMatrixNew,np.transpose(VMatrixOrg))
            #ThetaVB = np.arccos(VMatrix[2,2])
            ThetaVB = np.arccos(np.min([1,VMatrix[2,2]]))
            
            if np.sin(ThetaVB) < 1e-6:
                aAndC = np.arctan2(VMatrix[0,1],VMatrix[0,0])
                if aAndC < 0:
                    aAndC = aAndC + 2*np.pi

                #ThetaVA = 0.5*aAndC
                #ThetaVC = 0.5*aAndC
                ThetaVA = 0
                ThetaVC = aAndC

            else:
                ThetaVA = np.arctan2(VMatrix[2,0],-VMatrix[2,1])
                ThetaVC = np.arctan2(VMatrix[0,2],VMatrix[1,2])

            ThetaVA = self._adjustThetaAngle(ThetaVA)
            ThetaVB = self._adjustThetaAngle(ThetaVB)
            ThetaVC = self._adjustThetaAngle(ThetaVC)

            #if ThetaVA < -1e-3:
            #    ThetaVA = ThetaVA + np.pi
            #    ThetaVB = - ThetaVB

            ThetaVAList[i] = ThetaVA
            #ThetaVAList[i] = np.log10(np.absolute(np.tan(ThetaVA)))
            ThetaVBList[i] = ThetaVB
            ThetaVCList[i] = ThetaVC
            #ThetaVCList[i] = np.log10(np.absolute(np.tan(ThetaVC)))

            #ThetaVAList[i] = self._adjustThetaAngle2(ThetaVAList[i])
            #ThetaVBList[i] = self._adjustThetaAngle2(ThetaVBList[i])
            #ThetaVCList[i] = self._adjustThetaAngle2(ThetaVCList[i])

        return ThetaVAList, ThetaVBList, ThetaVCList



    def getThetaVABC(self,tau, *indexList):

        """
        get the euler angle with respect to the coordinate system
        (1 0 0), (0 1 0) and (0 0 1)
        """

        k,V1,V2,V3,C,NP = self._tau2PhysParams(tau)

        ThetaVAList = np.zeros((self.nMesh,1))
        ThetaVBList = np.zeros((self.nMesh,1))
        ThetaVCList = np.zeros((self.nMesh,1))

        indexDict = self.indexDict

        if np.shape(indexList)[0] > 0:
            indexList = indexList[0][0]
        
        for i in range(self.nMesh):

            # The rotation matrix for each basis is themself
            if np.shape(indexList)[0] == 0:
                VMatrix = np.array([V1[i],V2[i],V3[i]])
            else:
                VMatrixOrg = np.array([V1[i],V2[i],V3[i]])
                ## mutiply by -1 if switching vectors violate the right-hang rule
                VMatrix = np.array([self.rotFactor[int(indexList[i])] * \
                                    VMatrixOrg[indexDict[int(indexList[i])][0]], \
                                    VMatrixOrg[indexDict[int(indexList[i])][1]], \
                                    VMatrixOrg[indexDict[int(indexList[i])][2]]])
            ThetaVB = np.arccos(VMatrix[2,2])
            
            if np.sin(ThetaVB) < 1e-6:
                aAndC = np.arctan2(VMatrix[0,1],VMatrix[0,0])
                if aAndC < 0:
                    aAndC = aAndC + 2*np.pi

                ThetaVA = 0.5*aAndC
                ThetaVC = 0.5*aAndC
                #ThetaVA = aAndC
                #if ThetaVA > np.pi/2.0:
                #    ThetaVA = ThetaVA - np.pi
                #ThetaVC = 0

            else:
                ThetaVA = np.arctan2(VMatrix[2,0],-VMatrix[2,1])
                ThetaVC = np.arctan2(VMatrix[0,2],VMatrix[1,2])

            ThetaVAList[i] = ThetaVA
            ThetaVBList[i] = ThetaVB
            ThetaVCList[i] = ThetaVC

        return ThetaVAList, ThetaVBList, ThetaVCList

    def evalVecNorm(self, VMOld, VMNew):

        """
        Calculate the marker of discrepancy between two eigenvectors systems
        """

        rotM = np.zeros([3,3])
        for iterN in range(3):
            for iterP in range(3):
                rotM[iterN,iterP] = np.inner(VMOld[iterN], VMNew[iterP])
        deltaV = np.linalg.norm(rotM - np.diag(np.diag(rotM))) / np.linalg.norm(rotM)

        return deltaV


    def deltaVec(self, tauOld, tauNew):

        """
        Get the minimum rotation pattern from old eigenvectors system to the new one.
        """
        
        kOld,V1Old,V2Old,V3Old,COld,NPOld = self._tau2PhysParams(tauOld)
        kNew,V1New,V2New,V3New,CNew,NPNew = self._tau2PhysParams(tauNew)

        deltaVecList = np.zeros((self.nMesh,1))
        indexList = np.zeros((self.nMesh,1))
        indexDict = self.indexDict

        for i in range(self.nMesh):
            VMOld = np.array([V1Old[i],V2Old[i],V3Old[i]])
            VMNew = np.array([V1New[i],V2New[i],V3New[i]])
            deltaV = np.zeros(len(indexDict))

            for iterI in range(len(indexDict)):
                VMNew_rot = np.array([VMNew[indexDict[iterI][0]], \
                                     VMNew[indexDict[iterI][1]], \
                                     VMNew[indexDict[iterI][2]]])
                deltaV[iterI] = self.evalVecNorm(VMOld, VMNew_rot)

            deltaVecList[i] = np.amin(deltaV)
            indexList[i] = np.argmin(deltaV)

        return deltaVecList, indexList


    def plotEigenSystem(self, rotM):

        colorList = ['r', 'b', 'k']

        fig = plt.figure()
        ax = Axes3D(fig)
        for iterN in range(3):
            ax.plot([0, rotM[iterN,0]], [0, rotM[iterN,1]], [0, rotM[iterN,2]], colorList[iterN])
        plt.show()
        return 0


    def getDeltaCosine(self, tauOld, tauNew, indexList):

        """
        Calculate e11, e12, e22 for the rotation matrix of eigenvectors system.
        """
        #kOld,V1Old,V2Old,V3Old,COld,NPOld = self._tau2PhysParams(tauOld)
        #kNew,V1New,V2New,V3New,CNew,NPNew = self._tau2PhysParams(tauNew)
        V1Old, V2Old, V3Old = self._eigenVectors(tauOld)
        V1New, V2New, V3New = self._eigenVectors(tauNew)

        indexDict = self.indexDict

        e11 = np.zeros((self.nMesh,1))
        e12 = np.zeros((self.nMesh,1))
        e13 = np.zeros((self.nMesh,1))
        e22 = np.zeros((self.nMesh,1))
        e23 = np.zeros((self.nMesh,1))
        e33 = np.zeros((self.nMesh,1))

        E = np.zeros((self.nMesh,9))

        for i in range(self.nMesh):
            VMOld = np.array([V1Old[i],V2Old[i],V3Old[i]])
            #print VMOld
            VMNew = np.array([V1New[i],V2New[i],V3New[i]])
            deltaV = np.zeros(len(indexDict))

            #VMNew_rot = np.array([self.rotFactor[int(indexList[i])] * \
            #                      VMNew[indexDict[int(indexList[i])][0]], \
            #                      VMNew[indexDict[int(indexList[i])][1]], \
            #                      VMNew[indexDict[int(indexList[i])][2]]])
            VMNew_rot = np.array([VMNew[indexDict[int(indexList[i])][0]], \
                                  VMNew[indexDict[int(indexList[i])][1]], \
                                  VMNew[indexDict[int(indexList[i])][2]]])
            #print VMNew_rot
            rotM = np.zeros([3,3])

            # Temporary test of flipping
            if np.inner(VMOld[0], VMNew_rot[0]) < 0:
                VMNew_rot[0] = VMNew_rot[0] * (-1)
                #VMNew_rot[2] = VMNew_rot[2] * (-1)
            if np.inner(VMOld[1], VMNew_rot[1]) < 0:
                VMNew_rot[1] = VMNew_rot[1] * (-1)
                #VMNew_rot[2] = VMNew_rot[2] * (-1)
            if np.inner(VMOld[2], VMNew_rot[2]) < 0:
                VMNew_rot[2] = VMNew_rot[2] * (-1)

            #print VMNew_rot

            for iterN in range(3):
                for iterP in range(3):
                    rotM[iterN,iterP] = np.inner(VMOld[iterN], VMNew_rot[iterP])
            e11[i] = rotM[0, 0]
            e12[i] = rotM[0, 1]
            e13[i] = rotM[0, 2]
            e22[i] = rotM[1, 1]
            e23[i] = rotM[1, 2]
            e33[i] = rotM[2, 2]

            #E[i, 0] = rotM[0, 0]
            #E[i, 1] = rotM[0, 1]
            #E[i, 2] = rotM[0, 2]
            #E[i, 3] = rotM[1, 1]
            #E[i, 4] = rotM[1, 2]
            #E[i, 5] = rotM[2, 2]
            E[i, :] = rotM.reshape(9,)

            #self.plotEigenSystem(rotM)
            #pdb.set_trace()
            #print rotM
        return e11, e12, e13, e22, e23, e33
        #return E

    def getQuaternion(self, tauOld, tauNew, indexList):

        """
        Calculate e11, e12, e22 for the rotation matrix of eigenvectors system.
        """
        kOld,V1Old,V2Old,V3Old,COld,NPOld = self._tau2PhysParams(tauOld)
        kNew,V1New,V2New,V3New,CNew,NPNew = self._tau2PhysParams(tauNew)
        #V1Old, V2Old, V3Old = self._eigenVectors(tauOld)
        #V1New, V2New, V3New = self._eigenVectors(tauNew)

        indexDict = self.indexDict

        theta = np.zeros((self.nMesh,1))
        vx = np.zeros((self.nMesh,1))
        vy = np.zeros((self.nMesh,1))
        vz = np.zeros((self.nMesh,1))

        for i in range(self.nMesh):
            VMOld = np.array([V1Old[i],V2Old[i],V3Old[i]])
            #print VMOld
            VMNew = np.array([V1New[i],V2New[i],V3New[i]])
            deltaV = np.zeros(len(indexDict))

            #VMNew_rot = np.array([self.rotFactor[int(indexList[i])] * \
            #                      VMNew[indexDict[int(indexList[i])][0]], \
            #                      VMNew[indexDict[int(indexList[i])][1]], \
            #                      VMNew[indexDict[int(indexList[i])][2]]])
            VMNew_rot = np.array([VMNew[indexDict[int(indexList[i])][0]], \
                                  VMNew[indexDict[int(indexList[i])][1]], \
                                  VMNew[indexDict[int(indexList[i])][2]]])
            rotM = np.zeros([3,3])

            # Temporary test of flipping
            if np.inner(VMOld[0], VMNew_rot[0]) < 0:
                VMNew_rot[0] = VMNew_rot[0] * (-1)
                #VMNew_rot[2] = VMNew_rot[2] * (-1)
            if np.inner(VMOld[1], VMNew_rot[1]) < 0:
                VMNew_rot[1] = VMNew_rot[1] * (-1)
                #VMNew_rot[2] = VMNew_rot[2] * (-1)
            if np.inner(VMOld[2], VMNew_rot[2]) < 0:
                VMNew_rot[2] = VMNew_rot[2] * (-1)

            #print VMNew_rot

            for iterN in range(3):
                for iterP in range(3):
                    rotM[iterN,iterP] = np.inner(VMOld[iterN], VMNew_rot[iterP])
            quat = self._transform2quat(rotM)
            #quat = self._mat2quat(rotM)
            theta[i] = quat[0]
            vx[i] = quat[1]
            vy[i] = quat[2]
            vz[i] = quat[3]
            #vz[i] = np.linalg.det(VMNew_rot)
        return theta, vx, vy, vz

    def quadraticSolver(self, bb, ee, hh, cc):

        """
        """

        #pdb.set_trace()
        if hh == 0:
            ff = -bb * cc / ee
            if 1 - cc**2 - ff**2 > 0:
                ii = np.sqrt(1 - cc**2 - ff**2)
            else:
                ii = 0.0
        else:
            ff = np.roots([1 + ee**2 / (hh**2), \
                           2*bb*cc*ee/(hh**2), \
                           cc**2 + (bb*cc)**2/(hh**2) - 1])
            ii = bb*cc + ee*ff / hh
            ii = np.sqrt(1 - cc**2 - ff**2) 
            #for iterN in range(2):
            #    if abs(bb*cc+ee*ff[iterN]+hh*ii[iterN]) > abs(bb*cc+ee*ff[iterN]-hh*ii[iterN]):
            #        ii[iterN] = -1*ii[iterN]

            #if abs(ff[0]) > 1 or abs(ii[0]) > 1:
            #    ff = ff[1]
            #    ii = ii[1]
            #else:
            #    ff = ff[1]
            #    ii = ii[1]
        return ff, ii

    def checkOrthogonal(self, e11, e12, e13, e21List, e22, e23List, e31List, e32, e33List):
        
        innerPro = 1
        indexColumn1 = 0
        indexColumn2 = 0
        for iterN1 in range(2):
            for iterN2 in range(2):
                E = np.array([e11, e12, e13, \
                              e21List[iterN2], e22, e23List[iterN1], \
                              e31List[iterN2], e32, e33List[iterN1]])
                E = E.reshape(3,3)
                if np.inner(E[1], E[2]) < innerPro:
                    innerPro = np.inner(E[1], E[2])
                    indexColumn1 = iterN1
                    indexColumn2 = iterN2

        return indexColumn1, indexColumn2

        return np.inner(E[:,0],E[:,2])

    def adjustSign(self, rotM):

        rotM_adjust = np.zeros(rotM.shape)
        rotM_adjust[:,:] = rotM[:,:]
        rotM_adjust[2,0] = rotM_adjust[2,0] * (-1)
        if np.inner(rotM_adjust[:,0],rotM[:,2]) < np.inner(rotM[:,0],rotM[:,2]):
            return rotM_adjust
        else:
            return rotM

    #def genRotationMatrix(self, e11, e12, e22):

    #    """
    #    Calculate the rotation matrix from three elements e11, e12, e22
    #    """

    #    RMatrix = np.zeros((self.nMesh,9))
    #    for i in range(self.nMesh):
    #        rotM = np.zeros([3,3])
    #        rotM[0, 0] = e11[i]
    #        rotM[0, 1] = e12[i]
    #        rotM[1, 1] = e22[i]
    #        if 1 - rotM[0, 0]**2 -rotM[0, 1]**2 > 0:
    #            rotM[0, 2] = np.sqrt(1 - rotM[0, 0]**2 -rotM[0, 1]**2)
    #        else: 
    #            rotM[0, 2] = 0.0
    #        if 1 - rotM[0, 1]**2 -rotM[1, 1]**2 > 0:
    #            rotM[2, 1] = np.sqrt(1 - rotM[0, 1]**2 -rotM[1, 1]**2)
    #        else: 
    #            rotM[2, 1] = 0.0
    #        #rotM[1, 2], rotM[2, 2] = self.quadraticSolver(rotM[0, 1], rotM[1, 1], \
    #        #                                              rotM[2, 1], rotM[0, 2])
    #        #rotM[1, 0], rotM[2, 0] = self.quadraticSolver(rotM[0, 1], rotM[1, 1], \
    #        #                                              rotM[2, 1], rotM[0, 0])
    #        e23, e33 = self.quadraticSolver(rotM[0, 1], rotM[1, 1], \
    #                                                      rotM[2, 1], rotM[0, 2])
    #        e21, e31 = self.quadraticSolver(rotM[0, 1], rotM[1, 1], \
    #                                                      rotM[2, 1], rotM[0, 0])
    #        index1, index2 = self.checkOrthogonal(rotM[0,0], rotM[0,1], rotM[0,2], \
    #                                              e21, rotM[1,1], e23, \
    #                                              e31, rotM[2,1], e33)
    #        rotM[1, 2], rotM[2, 2] = e23[index1], e33[index1]
    #        rotM[1, 0], rotM[2, 0] = e21[index2], e31[index2]
    #        rotM = self.adjustSign(rotM)
    #        if np.inner(np.cross(rotM[0], rotM[1]), rotM[2]) < 0:
    #            print "The ", i, "th rotation matrix violate the right-hand rule"
    #            rotM[0] = rotM[0] * (-1)
    #        print rotM
    #        RMatrix[i, :] = rotM.reshape(9,)
    #    return RMatrix

    def crossSolver(self, rotM, e23, e33):
        
        rotM_copy = np.zeros(rotM.shape)
        rotM_copy[:,:] = rotM[:,:]
        innerProduct = 1
        for iterN in range(2):
            rotM_C1 = np.cross(rotM[:,1],[rotM[0,2],e23[iterN],e33[iterN]]) 
            rotM_C1 = rotM_C1 / np.sign(rotM_C1[0]) * np.sign(rotM[0,0])
            rotM_copy[:,0] = rotM_C1
            rotM_copy[1,2] = e23[iterN]
            rotM_copy[2,2] = e33[iterN]
            if np.inner(rotM_copy[1], rotM_copy[2]) < innerProduct:
                innerProduct = np.inner(rotM_copy[1], rotM_copy[2])
                rotM[:,:] = rotM_copy[:,:]
        return rotM

    #def genRotationMatrix(self, e11, e12, e22):

    #    """
    #    Calculate the rotation matrix from three elements e11, e12, e22
    #    """

    #    RMatrix = np.zeros((self.nMesh,9))
    #    for i in range(self.nMesh):
    #        rotM = np.zeros([3,3])
    #        rotM[0, 0] = e11[i]
    #        rotM[0, 1] = e12[i]
    #        rotM[1, 1] = e22[i]
    #        if 1 - rotM[0, 0]**2 -rotM[0, 1]**2 > 0:
    #            rotM[0, 2] = np.sqrt(1 - rotM[0, 0]**2 -rotM[0, 1]**2)
    #            #rotM[0, 2] = -1*np.sqrt(1 - rotM[0, 0]**2 -rotM[0, 1]**2)
    #        else: 
    #            rotM[0, 2] = 0.0
    #        if 1 - rotM[0, 1]**2 -rotM[1, 1]**2 > 0:
    #            rotM[2, 1] = np.sqrt(1 - rotM[0, 1]**2 -rotM[1, 1]**2)
    #            #rotM[2, 1] = -1*np.sqrt(1 - rotM[0, 1]**2 -rotM[1, 1]**2)
    #        else: 
    #            rotM[2, 1] = 0.0
    #        #rotM[1, 2], rotM[2, 2] = self.quadraticSolver(rotM[0, 1], rotM[1, 1], \
    #        #                                              rotM[2, 1], rotM[0, 2])
    #        #rotM[1, 0], rotM[2, 0] = self.quadraticSolver(rotM[0, 1], rotM[1, 1], \
    #        #                                              rotM[2, 1], rotM[0, 0])
    #        e23, e33 = self.quadraticSolver(rotM[0, 1], rotM[1, 1], \
    #                                                      rotM[2, 1], rotM[0, 2])
    #        self.crossSolver(rotM, e23, e33)
    #        if np.inner(np.cross(rotM[0], rotM[1]), rotM[2]) < 0:
    #            print "The ", i, "th rotation matrix violate the right-hand rule"
    #            rotM[0] = rotM[0] * (-1)
    #        #pdb.set_trace()
    #        self.plotEigenSystem(rotM)
    #        print rotM
    #        RMatrix[i, :] = rotM.reshape(9,)
    #    return RMatrix

    def setSign1(self, rotM):

        sign02List = [1,1,-1,-1,1,1,-1,-1]
        sign12List = [1,-1,1,-1,1,-1,1,-1]
        sign21List = [1,1,1,1,-1,-1,-1,-1]

        for iterN in range(len(sign02List)):
            rotM[0,2] = rotM[0,2] * sign02List[iterN]
            rotM[1,2] = rotM[1,2] * sign12List[iterN]
            rotM[2,1] = rotM[2,1] * sign21List[iterN]
            rotM[:,0] = np.cross(rotM[:,1], rotM[:,2])
            if abs(np.inner(rotM[1],rotM[1])-1) < 1e-6 and abs(np.inner(rotM[2],rotM[2])-1) < 1e-6:
                break

        return rotM

    #def genRotationMatrix(self, e11, e12, e22, e33):

    #    """
    #    Calculate the rotation matrix from three elements e11, e12, e22
    #    """

    #    RMatrix = np.zeros((self.nMesh,9))
    #    for i in range(self.nMesh):
    #        rotM = np.zeros([3,3])
    #        rotM[0, 0] = e11[i]
    #        rotM[0, 1] = e12[i]
    #        rotM[1, 1] = e22[i]
    #        rotM[2, 2] = e33[i]
    #        if 1 - rotM[0, 0]**2 -rotM[0, 1]**2 > 0:
    #            rotM[0, 2] = np.sqrt(1 - rotM[0, 0]**2 -rotM[0, 1]**2)
    #            #rotM[0, 2] = -1*np.sqrt(1 - rotM[0, 0]**2 -rotM[0, 1]**2)
    #        else: 
    #            rotM[0, 2] = 0.0
    #        if 1 - rotM[0, 1]**2 -rotM[1, 1]**2 > 0:
    #            rotM[2, 1] = np.sqrt(1 - rotM[0, 1]**2 -rotM[1, 1]**2)
    #            #rotM[2, 1] = -1*np.sqrt(1 - rotM[0, 1]**2 -rotM[1, 1]**2)
    #        else: 
    #            rotM[2, 1] = 0.0
    #        if 1 - rotM[0, 2]**2 -rotM[2, 2]**2 > 0:
    #            rotM[1, 2] = np.sqrt(1 - rotM[0, 2]**2 -rotM[2, 2]**2)
    #            #rotM[2, 1] = -1*np.sqrt(1 - rotM[0, 1]**2 -rotM[1, 1]**2)
    #        else: 
    #            rotM[1, 2] = 0.0
    #        rotM[:,0] = np.cross(rotM[:,1], rotM[:,2])
    #        self.plotEigenSystem(rotM)
    #        rotM = self.setSign1(rotM)
    #        #pdb.set_trace()
    #        print rotM
    #        RMatrix[i, :] = rotM.reshape(9,)
    #    return RMatrix

    def setSign2(self, rotM):

        sign02List = [1,-1,1,-1]
        sign21List = [1,1,-1,-1]
        #sign02List = [-1]
        #sign21List = [1]

        if rotM[1,1] == 1.0:
            rotM[0,1] = 0.0
            rotM[1,2] = 0.0
            rotM[2,1] = 0.0

        signIndex = 0
        detRef = 1

        rotM_copy = np.zeros(rotM.shape)

        for iterN in range(len(sign02List)):
            rotM_copy[:,:] = rotM[:,:]
            rotM_copy[0,2] = rotM_copy[0,2] * sign02List[iterN]
            rotM_copy[2,1] = rotM_copy[2,1] * sign21List[iterN]
            rotM_copy[:,0] = (-1) * np.cross(rotM_copy[:,1], rotM_copy[:,2])
            if np.sign(rotM_copy[0,0]) == np.sign(rotM[0,0]):
                det = abs(np.linalg.det(np.linalg.inv(rotM_copy)-np.transpose(rotM_copy)))
                if det < detRef:
                    signIndex = iterN
                    detRef = det 

        rotM[0,2] = rotM[0,2] * sign02List[signIndex]
        rotM[2,1] = rotM[2,1] * sign21List[signIndex]
        rotM[:,0] = (-1) * np.cross(rotM[:,1], rotM[:,2])

        return rotM

    #def genRotationMatrix(self, e11, e12, e22, e23, e33):

    #    """
    #    Calculate the rotation matrix from three elements e11, e12, e22
    #    """

    #    RMatrix = np.zeros((self.nMesh,9))
    #    for i in range(self.nMesh):
    #        rotM = np.zeros([3,3])
    #        rotM[0, 0] = e11[i]
    #        rotM[0, 1] = e12[i]
    #        rotM[1, 1] = e22[i]
    #        rotM[1, 2] = e23[i]
    #        rotM[2, 2] = e33[i]
    #        if 1 - rotM[0, 0]**2 -rotM[0, 1]**2 > 0:
    #            rotM[0, 2] = np.sqrt(1 - rotM[0, 0]**2 -rotM[0, 1]**2)
    #            #rotM[0, 2] = -1*np.sqrt(1 - rotM[0, 0]**2 -rotM[0, 1]**2)
    #        else: 
    #            #rotM[0, 2] = 0.0
    #            rotM[0, 2] = 1e-20
    #        if 1 - rotM[0, 1]**2 -rotM[1, 1]**2 > 0:
    #            rotM[2, 1] = np.sqrt(1 - rotM[0, 1]**2 -rotM[1, 1]**2)
    #            #rotM[2, 1] = -1*np.sqrt(1 - rotM[0, 1]**2 -rotM[1, 1]**2)
    #        else: 
    #            #rotM[2, 1] = 0.0
    #            rotM[2, 1] = 1e-20
    #        #self.plotEigenSystem(rotM)
    #        rotM = self.setSign2(rotM)
    #        #pdb.set_trace()
    #        print rotM
    #        RMatrix[i, :] = rotM.reshape(9,)
    #    return RMatrix

    def setSign3(self, rotM):

        sign21List = [1,-1]
        #sign02List = [-1]
        #sign21List = [1]

        signIndex = 0
        detRef = 1

        rotM_copy = np.zeros(rotM.shape)

        for iterN in range(len(sign21List)):
            rotM_copy[:,:] = rotM[:,:]
            rotM_copy[2,1] = rotM_copy[2,1] * sign21List[iterN]
            rotM_copy[:,0] = np.cross(rotM_copy[:,1], rotM_copy[:,2])
            if np.sign(rotM_copy[0,0]) == np.sign(rotM[0,0]):
                det = abs(np.linalg.det(np.linalg.inv(rotM_copy)-np.transpose(rotM_copy)))
                if det < detRef:
                    signIndex = iterN
                    detRef = det 

        rotM[2,1] = rotM[2,1] * sign21List[signIndex]
        rotM[:,0] = np.cross(rotM[:,1], rotM[:,2])

        if np.linalg.det(rotM) < 0:
            rotM[0] = rotM[0]*-1

        return rotM


    def genRotationMatrix(self, e11, e12, e22, e33, e13Sign, e23Sign):

        """
        Calculate the rotation matrix from three elements e11, e12, e22
        """

        RMatrix = np.zeros((self.nMesh,9))
        for i in range(self.nMesh):
            rotM = np.zeros([3,3])
            rotM[0, 0] = e11[i]
            rotM[0, 1] = e12[i]
            rotM[1, 1] = e22[i]
            rotM[2, 2] = e33[i]
            if 1 - rotM[0, 0]**2 -rotM[0, 1]**2 > 0:
                rotM[0, 2] = e13Sign[i] * np.sqrt(1 - rotM[0, 0]**2 -rotM[0, 1]**2)
                #rotM[0, 2] = -1*np.sqrt(1 - rotM[0, 0]**2 -rotM[0, 1]**2)
            else: 
                rotM[0, 2] = 0.0
            if 1 - rotM[0, 1]**2 -rotM[1, 1]**2 > 0:
                rotM[2, 1] = np.sqrt(1 - rotM[0, 1]**2 -rotM[1, 1]**2)
                #rotM[2, 1] = -1*np.sqrt(1 - rotM[0, 1]**2 -rotM[1, 1]**2)
            else: 
                rotM[2, 1] = 0.0
            if 1 - rotM[0, 2]**2 -rotM[2, 2]**2 > 0:
                rotM[1, 2] = e23Sign[i] * np.sqrt(1 - rotM[0, 2]**2 -rotM[2, 2]**2)
                #rotM[2, 1] = -1*np.sqrt(1 - rotM[0, 1]**2 -rotM[1, 1]**2)
            else: 
                rotM[1, 2] = 0.0
            rotM[:,0] = np.cross(rotM[:,1], rotM[:,2])
            #pdb.set_trace()
            rotM = self.setSign3(rotM)
            #print rotM
            RMatrix[i, :] = rotM.reshape(9,)
        return RMatrix

    def reconstructTau(self,TauOld,deltaXi,deltaEta,deltaKT,RMatrix,indexList):

        """
        Perturb Reynolds stresses according to deltaXi, deltaEta, and deltaLog2K
        Write them to OpenFOAM instances.  First call _perturbXiEta() to
        perturb on \\xi, \eta plane, and map to Reynolds stresses

        deltaXi and deltaEta are based on the absolute value in the natural
        domain
        if deltaLog2K is activated, k will be treated as k*(2**deltaLog2K)
        where deltaLog2K = log2(K.new/K.old)
        """
        X = self._C2X(self.COrg)
        self.modifyError(deltaXi,deltaEta)
        RSs = self._phys2Natural(X)
        XNew = self._natural2Phys(RSs)
        Cs = self._X2C(XNew)

        V1Old, V2Old, V3Old = self._eigenVectors(TauOld)

        indexDict = self.indexDict

        tau = np.zeros((self.nMesh,6))

        for i in range(self.nMesh):
            lam1 = Cs[i,0] + Cs[i,1]/2.0 + Cs[i,2]/3.0 - 1.0/3.0
            lam2 = Cs[i,1]/2.0 + Cs[i,2]/3.0 - 1.0/3.0
            lam3 = Cs[i,2]/3.0 - 1.0/3.0
            Lambda = np.diag([lam3,lam2,lam1])
            rotM = RMatrix[i].reshape(3,3)
            V = np.array([V1Old[i], V2Old[i], V3Old[i]])
            #pdb.set_trace()
            Vorg = np.dot(np.transpose(rotM), V)
            V[indexDict[int(indexList[i])][0],:] = Vorg[0,:]
            V[indexDict[int(indexList[i])][1],:] = Vorg[1,:]
            V[indexDict[int(indexList[i])][2],:] = Vorg[2,:]

            #print V

            t = 2*self.k[i][0]\
                *(np.eye(3)/3.0 + np.dot(np.dot(np.transpose(V),Lambda),V))
            tau[i,:] = np.array([t[0,0],t[0,1],t[0,2],t[1,1],t[1,2],t[2,2]])

        if len(deltaKT) > 0:
            tau = tau*numpy.exp2(deltaKT)
        return tau

    def reconstructTauFromQuat(self,TauOld,deltaXi,deltaEta,deltaKT,vx,vy,vz,theta,indexList):

        """
        Perturb Reynolds stresses according to deltaXi, deltaEta, and deltaLog2K
        Write them to OpenFOAM instances.  First call _perturbXiEta() to
        perturb on \\xi, \eta plane, and map to Reynolds stresses

        deltaXi and deltaEta are based on the absolute value in the natural
        domain
        if deltaLog2K is activated, k will be treated as k*(2**deltaLog2K)
        where deltaLog2K = log2(K.new/K.old)
        """
        X = self._C2X(self.COrg)
        self.modifyError(deltaXi,deltaEta)
        RSs = self._phys2Natural(X)
        XNew = self._natural2Phys(RSs)
        Cs = self._X2C(XNew)

        #V1Old, V2Old, V3Old = self._eigenVectors(TauOld)
        kOld,V1Old,V2Old,V3Old,COld,NPOld = self._tau2PhysParams(TauOld)

        indexDict = self.indexDict

        tau = np.zeros((self.nMesh,6))

        for i in range(self.nMesh):
            lam1 = Cs[i,0] + Cs[i,1]/2.0 + Cs[i,2]/3.0 - 1.0/3.0
            lam2 = Cs[i,1]/2.0 + Cs[i,2]/3.0 - 1.0/3.0
            lam3 = Cs[i,2]/3.0 - 1.0/3.0
            Lambda = np.diag([lam3,lam2,lam1])
            rotM = self._quat2transform([vx[i],vy[i],vz[i],theta[i]])
            V = np.array([V1Old[i], V2Old[i], V3Old[i]])
            #pdb.set_trace()
            Vorg = np.dot(np.transpose(rotM), V)
            V[indexDict[int(indexList[i])][0],:] = Vorg[0,:]
            V[indexDict[int(indexList[i])][1],:] = Vorg[1,:]
            V[indexDict[int(indexList[i])][2],:] = Vorg[2,:]

            #print V

            t = 2*self.k[i][0]\
                *(np.eye(3)/3.0 + np.dot(np.dot(np.transpose(V),Lambda),V))
            tau[i,:] = np.array([t[0,0],t[0,1],t[0,2],t[1,1],t[1,2],t[2,2]])

        if len(deltaKT) > 0:
            tau = tau*numpy.exp2(deltaKT)
        return tau

    def reconstructTauFromQuatInBary(self,TauOld,deltaXbary,deltaYbary,deltaKT,vx,vy,vz,theta,indexList):

        """
        Perturb Reynolds stresses according to deltaXi, deltaEta, and deltaLog2K
        Write them to OpenFOAM instances.  First call _perturbXiEta() to
        perturb on \\xi, \eta plane, and map to Reynolds stresses

        deltaXi and deltaEta are based on the absolute value in the natural
        domain
        if deltaLog2K is activated, k will be treated as k*(2**deltaLog2K)
        where deltaLog2K = log2(K.new/K.old)
        """
        
        X = self._C2X(self.COrg)
        XNew = X
        XNew[:,0] = X[:,0] + deltaXbary.T
        XNew[:,1] = X[:,1] + deltaYbary.T               
        Cs = self._X2C(XNew)

        #V1Old, V2Old, V3Old = self._eigenVectors(TauOld)
        kOld,V1Old,V2Old,V3Old,COld,NPOld = self._tau2PhysParams(TauOld)

        indexDict = self.indexDict

        tau = np.zeros((self.nMesh,6))

        for i in range(self.nMesh):
            lam1 = Cs[i,0] + Cs[i,1]/2.0 + Cs[i,2]/3.0 - 1.0/3.0
            lam2 = Cs[i,1]/2.0 + Cs[i,2]/3.0 - 1.0/3.0
            lam3 = Cs[i,2]/3.0 - 1.0/3.0
            Lambda = np.diag([lam3,lam2,lam1])
            rotM = self._quat2transform([vx[i],vy[i],vz[i],theta[i]])
            V = np.array([V1Old[i], V2Old[i], V3Old[i]])
            #pdb.set_trace()
            Vorg = np.dot(np.transpose(rotM), V)
            V[indexDict[int(indexList[i])][0],:] = Vorg[0,:]
            V[indexDict[int(indexList[i])][1],:] = Vorg[1,:]
            V[indexDict[int(indexList[i])][2],:] = Vorg[2,:]

            #print V

            t = 2*self.k[i][0]\
                *(np.eye(3)/3.0 + np.dot(np.dot(np.transpose(V),Lambda),V))
            tau[i,:] = np.array([t[0,0],t[0,1],t[0,2],t[1,1],t[1,2],t[2,2]])

        if len(deltaKT) > 0:
            tau = tau*numpy.exp2(deltaKT)
        return tau


    def _quat2transform(self, q):
        """
        Transform a unit quaternion into its corresponding rotation matrix (to
        be applied on the right side).
        
        :returns: transform matrix
        :rtype: numpy array
        
        """
        x, y, z, w = q
        xx2 = 2 * x * x
        yy2 = 2 * y * y
        zz2 = 2 * z * z
        xy2 = 2 * x * y
        wz2 = 2 * w * z
        zx2 = 2 * z * x
        wy2 = 2 * w * y
        yz2 = 2 * y * z
        wx2 = 2 * w * x
        
        rmat = np.empty((3, 3), float)
        rmat[0,0] = 1. - yy2 - zz2
        rmat[0,1] = xy2 - wz2
        rmat[0,2] = zx2 + wy2
        rmat[1,0] = xy2 + wz2
        rmat[1,1] = 1. - xx2 - zz2
        rmat[1,2] = yz2 - wx2
        rmat[2,0] = zx2 - wy2
        rmat[2,1] = yz2 + wx2
        rmat[2,2] = 1. - xx2 - yy2
        
        return rmat

    def plotTauOnBaycentric(self,tau, tauComp = 'None', tauPerturb = 'None', sampleLS = 'o-'):
        """
        Plot a Reynolds stress (the i^th error realization) on Baycentric
        coordinate
        """

        if tauComp == 'None':
            k,V1,V2,V3,C,NP = self._tau2PhysParams(tau)
            X = self._C2X(C)
            plt.clf()
            plt.plot(X[:,0],X[:,1],'ko')
            plt.plot([0,1,0.5,0.5,0],[0,0,3**0.5/2.0,3**0.5/2.0,0],'b-')
            plt.show()
        else:
            k,V1,V2,V3,C,NP = self._tau2PhysParams(tauPerturb)
            X = self._C2X(C)
            p3, = plt.plot(X[:,0],X[:,1], sampleLS, markeredgecolor='grey', markerfacecolor='None', alpha = 0.5)
            plt.plot(X[0,0],X[0,1],'bo', markeredgecolor='b', markerfacecolor='None', alpha = 0.5)
            plt.plot(X[-1,0],X[-1,1],'bo', alpha = 0.5)

            k,V1,V2,V3,C,NP = self._tau2PhysParams(tau)
            X = self._C2X(C)

            p1, = plt.plot(X[:,0],X[:,1],'rs', markeredgecolor='r', markerfacecolor='None')
            plt.plot(X[0,0],X[0,1],'bs', markeredgecolor='b', markerfacecolor='None')
            plt.plot(X[-1,0],X[-1,1],'bs')

            k,V1,V2,V3,C,NP = self._tau2PhysParams(tauComp)
            X = self._C2X(C)
            p2, = plt.plot(X[:,0],X[:,1],'k^', markeredgecolor='k', markerfacecolor='None')
            plt.plot(X[1,0],X[1,1],'b^', markeredgecolor='b', markerfacecolor='None')
            plt.plot(X[-1,0],X[-1,1],'b^')

            plt.plot([0,1,0.5,0.5,0],[0,0,3**0.5/2.0,3**0.5/2.0,0],'b-')
	return p1, p2, p3

    def plotXPerturb(self,deltaXi,deltaEta,fileName):

        X = self._C2X(self.COrg)
        plt.clf()
        plt.plot(X[:,0],X[:,1],'ko')
        plt.plot([0,1,0.5,0.5,0],[0,0,3**0.5/2.0,3**0.5/2.0,0],'b-')

        self.perturbTau(deltaXi,deltaEta)

        XNew = self._C2X(self.C)

        plt.plot(XNew[:,0],XNew[:,1],'r+')

        plt.savefig(fileName)

    def plotRSPerturb(self,deltaXi,deltaEta,fileName):

        X = self._C2X(self.COrg)
        plt.clf()

        # Draw the points in RS coordinate before "perturb"
        RS = self._phys2Natural(X)
        plt.plot(RS[:,0],RS[:,1],'ko')
        plt.plot([-1,1,1,-1,-1],[-1,-1,1,1,-1],'b-')

        self.perturbTau(deltaXi,deltaEta)

        # Draw the points in RS coordinate after "perturb"
        XNew = self._C2X(self.C)
        RSNew = self._phys2Natural(XNew)
        plt.plot(RSNew[:,0],RSNew[:,1],'r+')

        plt.savefig(fileName)

    def plotSquare(self):
        plt.plot([-1,1,1,-1,-1],[-1,-1,1,1,-1],'b-')

    def plotOnRS(self, tau, cl, alpha):
        k, V1, V2, V3, C, NP = self._tau2PhysParams(tau)
        X = self._C2X(C)
        RS = self._phys2Natural(X)
        p, = plt.plot(RS[0,0],RS[0,1],'o', color = cl, alpha = alpha)
        return p, RS[0,0], RS[0,1]

    def plotTriangle(self):
        plt.plot([0,1,0.5,0.5,0],[0,0,3**0.5/2.0,3**0.5/2.0,0],'b-')

    def plotOnTriangle(self,tau,cl,alpha):
        """
        Plot a Reynolds stress (the i^th error realization) on Baycentric
        coordinate
        """
        k,V1,V2,V3,C,NP = self._tau2PhysParams(tau)
        X = self._C2X(C)
 
        p, = plt.plot(X[0,0],X[0,1], 'o', color = cl, alpha = alpha)
        return p, X[0,0], X[0,1]

    def modifyError(self,deltaXi,deltaEta):
        """
        == Mutation functions ==

        modifyError():

        *   Modify deltaXi and deltaEta from given values (either specified or
        obtained from KL summation)
        """
        self.deltaXi = deltaXi
        self.deltaEta = deltaEta
    
    ##############################  Priviate  #################################
    def _eigenVectors(self,tauArray):
        """
        Convert tau to physical parameters (k, C1, C2, C3, v1, v2, v3)
        """
        self.nMesh = tauArray.shape[0]
        ks = np.zeros((self.nMesh,1))
        Cs = np.zeros((self.nMesh,3))
        NPs = np.zeros((self.nMesh,1))
        V1s = np.zeros((self.nMesh,3))
        V2s = np.zeros((self.nMesh,3))
        V3s = np.zeros((self.nMesh,3))

        for i in range(self.nMesh):
            tau = np.zeros((3,3))
            tauData = tauArray[i,:]

            tau[0,0] = tauData[0]
            tau[0,1] = tauData[1]
            tau[0,2] = tauData[2]
            tau[1,1] = tauData[3]
            tau[1,2] = tauData[4]
            tau[2,2] = tauData[5]

            tau[1,0] = tau[0,1]
            tau[2,0] = tau[0,2]
            tau[2,1] = tau[1,2]

            tiny = np.finfo(np.float).tiny
            # find the k, aij, and the eigenvalue
            k = np.trace(tau)/2.0
            aij = tau/2.0/(k + tiny) - (np.eye(3) + tiny)/3.0
            w,v = np.linalg.eig(aij)
            max,min = w.argmax(),w.argmin()
            mid = 3 - max - min
            w[0],w[1],w[2] = w[min],w[mid],w[max]

            vContainer = np.zeros((3,3))
            vContainer[:,0],vContainer[:,1],vContainer[:,2] =\
                v[:,min],v[:,mid],v[:,max]

            ks[i,0] = k

            vContainer = np.transpose(vContainer)

            V1s[i,:] = (vContainer[0,:])
            V2s[i,:] = (vContainer[1,:])
            V3s[i,:] = (vContainer[2,:])
        return V1s, V2s, V3s

    def _tau2PhysParams(self,tauArray):
        """
        Convert tau to physical parameters (k, C1, C2, C3, v1, v2, v3)
        """
        self.nMesh = tauArray.shape[0]
        ks = np.zeros((self.nMesh,1))
        Cs = np.zeros((self.nMesh,3))
        NPs = np.zeros((self.nMesh,1))
        V1s = np.zeros((self.nMesh,3))
        V2s = np.zeros((self.nMesh,3))
        V3s = np.zeros((self.nMesh,3))

        for i in range(self.nMesh):
            tau = np.zeros((3,3))
            tauData = tauArray[i,:]

            tau[0,0] = tauData[0]
            tau[0,1] = tauData[1]
            tau[0,2] = tauData[2]
            tau[1,1] = tauData[3]
            tau[1,2] = tauData[4]
            tau[2,2] = tauData[5]

            tau[1,0] = tau[0,1]
            tau[2,0] = tau[0,2]
            tau[2,1] = tau[1,2]

            tiny = np.finfo(np.float).tiny
            # find the k, aij, and the eigenvalue
            k = np.trace(tau)/2.0
            aij = tau/2.0/(k + tiny) - (np.eye(3) + tiny)/3.0
            w,v = np.linalg.eig(aij)
            max,min = w.argmax(),w.argmin()
            mid = 3 - max - min
            w[0],w[1],w[2] = w[min],w[mid],w[max]

            vContainer = np.zeros((3,3))
            vContainer[:,0],vContainer[:,1],vContainer[:,2] =\
                v[:,min],v[:,mid],v[:,max]

            ks[i,0] = k

            vContainer = np.transpose(vContainer)

            detV = np.linalg.det(vContainer)
            NPs[i,0] = 1
            if detV < 0:
                vContainer = np.dot(np.array([[-1,0,0],[0,1,0],[0,0,1]]),vContainer)
                NPs[i,0] = -1

            ThetaVB = np.arccos(vContainer[2,2])
            if ThetaVB == 0:
                print "WARNING: ThetaVB == 0"
                aAndC = np.arctan2(vContainer[0,1],vContainer[0,0])
                if aAndC < 0:
                    aAndC = aAndC + 2*np.pi

                ThetaVA = 0.5*aAndC
                ThetaVC = 0.5*aAndC
            else:
                #ThetaVA = np.arctan2(vContainer[2,0]/np.sin(ThetaVB),-vContainer[2,1]/np.sin(ThetaVB))
                #ThetaVC = np.arctan2(vContainer[0,2]/np.sin(ThetaVB), vContainer[1,2]/np.sin(ThetaVB))
                ThetaVA = np.arctan2(vContainer[2,0],-vContainer[2,1])
                ThetaVC = np.arctan2(vContainer[0,2], vContainer[1,2])

            signA = np.sin(ThetaVA)/np.abs(np.sin(ThetaVA))
            signC = np.sin(ThetaVC)/np.abs(np.sin(ThetaVC))

            #if signA < 0 and abs(vContainer[2,0]) > 1e-1:
            if signA < 0 and abs(ThetaVA) > 5e-3*np.pi:
                vContainer = np.dot(np.array([[-1,0,0],[0,1,0],[0,0,-1]]),vContainer)

                ThetaVB2 = np.arccos(vContainer[2,2])
                #ThetaVC2 = np.arctan2(vContainer[0,2]/np.sin(ThetaVB2), vContainer[1,2]/np.sin(ThetaVB2))
                ThetaVC2 = np.arctan2(vContainer[0,2], vContainer[1,2])
                signC = np.sin(ThetaVC2)/np.abs(np.sin(ThetaVC2))

            ## JL: Only enable flipping sign if vContainer[0,2] is not small
            #if signC < 0 and abs(vContainer[0,2]) > 1e-1:
            if signC < 0 and abs(ThetaVC) > 5e-3*np.pi:
                vContainer = np.dot(np.array([[-1,0,0],[0,-1,0],[0,0,1]]),vContainer)

            # ThetaVB2 = np.arccos(vContainer[2,2])
            # ThetaVA2 = np.arctan2(vContainer[2,0]/np.sin(ThetaVB2),-vContainer[2,1]/np.sin(ThetaVB2))
            # ThetaVC2 = np.arctan2(vContainer[0,2]/np.sin(ThetaVB2), vContainer[1,2]/np.sin(ThetaVB2))

            V1s[i,:] = (vContainer[0,:])
            V2s[i,:] = (vContainer[1,:])
            V3s[i,:] = (vContainer[2,:])

            c1 = w[2] - w[1]
            c2 = 2*(w[1] - w[0])
            c3 = 3*w[0] + 1

            Cs[i,:] = np.array([c1,c2,c3])
            #pdb.set_trace()
            
        if self.correctInitTau == True:
            X = self._C2X(Cs)

            RSs = self._phys2Natural(X)
            XNew = self._natural2Phys(RSs,True)
            Cs = self._X2C(XNew)

        return ks,V1s,V2s,V3s,Cs,NPs


    def _C2Tau(self,Cs,*deltaThetaV):
        """
        Mapping physical parameters (k, C1, C2, C3, v1, v2, v3) to
        Reynolds stress. Note that only C can be modified.
        """
        taus = np.zeros((self.nMesh,6))
        ThetaVAOld, ThetaVBOld, ThetaVCOld = self.getThetaVABC(self.tau)

        indexDict = self.indexDict

        for i in range(self.nMesh):
            lam1 = Cs[i,0] + Cs[i,1]/2.0 + Cs[i,2]/3.0 - 1.0/3.0
            lam2 = Cs[i,1]/2.0 + Cs[i,2]/3.0 - 1.0/3.0
            lam3 = Cs[i,2]/3.0 - 1.0/3.0
            Lambda = np.diag([lam3,lam2,lam1])
            V = np.zeros((3,3))

            #if np.shape(deltaThetaV)[0] > 0:
            if len(deltaThetaV) > 0:
                thetaVA = (deltaThetaV[0][0][i,0] + ThetaVAOld[i])[0]
                thetaVB = (deltaThetaV[0][0][i,1] + ThetaVBOld[i])[0]
                thetaVC = (deltaThetaV[0][0][i,2] + ThetaVCOld[i])[0]

                thetaVA = self._adjustThetaAngle(thetaVA)
                thetaVB = self._adjustThetaAngle(thetaVB)
                thetaVC = self._adjustThetaAngle(thetaVC)

                RA = np.array([[np.cos(thetaVA),np.sin(thetaVA),0],\
                               [-np.sin(thetaVA),np.cos(thetaVA),0],\
                               [0,0,1]])
                RB = np.array([[1,0,0],\
                               [0,np.cos(thetaVB),np.sin(thetaVB)],\
                               [0,-np.sin(thetaVB),np.cos(thetaVB)]])
                RC = np.array([[np.cos(thetaVC),np.sin(thetaVC),0],\
                               [-np.sin(thetaVC),np.cos(thetaVC),0],\
                               [0,0,1]])
                RMatrix = np.dot(np.dot(RC,RB),RA)

                if len(deltaThetaV[0]) == 1:
                    V[0,:] = RMatrix[0,:]
                    V[1,:] = RMatrix[1,:]
                    V[2,:] = RMatrix[2,:]
                else:
                    V[indexDict[int(deltaThetaV[0][1][i])][0],:] = RMatrix[0,:]
                    V[indexDict[int(deltaThetaV[0][1][i])][1],:] = RMatrix[1,:]
                    V[indexDict[int(deltaThetaV[0][1][i])][2],:] = RMatrix[2,:]

            else:
                V[0,:] = self.V1[i,:]
                V[1,:] = self.V2[i,:]
                V[2,:] = self.V3[i,:]
                

            t = 2*self.k[i][0]\
                *(np.eye(3)/3.0 + np.dot(np.dot(np.transpose(V),Lambda),V))
            taus[i,:] = np.array([t[0,0],t[0,1],t[0,2],t[1,1],t[1,2],t[2,2]])
        return taus


    def _C2X(self,Cs):
        """
        Conversion between Baycentric coordinates and their global
        physical coordinates used to plot Baycentric triangle
        """
        Xs = np.zeros((self.nMesh,2))

        for i in range(self.nMesh):
            Xs[i,:] = Cs[i,0]*self.xcs[0,:] + Cs[i,1]*self.xcs[1,:]\
                      + Cs[i,2]*self.xcs[2,:]

        return Xs


    def _X2C(self,Xs):

        Cs = np.zeros((self.nMesh,3))
        Cmatrix = np.zeros((3,3))
        Cmatrix[:,0] = self.xcs[:,0]
        Cmatrix[:,1] = self.xcs[:,1]
        Cmatrix[:,2] = np.array([[1,1,1]])
        Cmatrix = np.transpose(Cmatrix)

        for i in range(self.nMesh):
            Cs[i,:] = np.dot(np.linalg.inv(Cmatrix),
                             np.array([Xs[i,0],Xs[i,1],1]))

        return Cs

    def _phys2Natural(self, XYs):

        """
        Mapping from physical coordinate to natural (normalized)
        coordinates \\xi and \eta

        # TODO: make sure the point is in the triangle
        # make sure the triangle is [[0,0],[1,0],[0,3**0.5/2.0],[0,3**0.5/2.0]]

        # According to Chongyu Hua's derivation in "An inverse transformation
        # for quadrilateral isoparametric elements: Analysis and application"
        # in "Finite Elements in Analysis and Design", 1990
        """

        a1,a2 = -1,0
        b1,b2 = 1,0
        c1,c2 = 0,3**0.5

        x1,y1 = self.triangle[0,0],self.triangle[0,1]
        x2,y2 = self.triangle[1,0],self.triangle[1,1]
        x3,y3 = self.triangle[2,0],self.triangle[2,1]
        x4,y4 = self.triangle[3,0],self.triangle[3,1]

        RSs = np.zeros((self.nMesh,2))
        for i in range(self.nMesh):
            px,py = XYs[i,0],XYs[i,1]

            d1 = 4*px - x1 - x2 - x3 - x4
            d2 = 4*py - y1 - y2 - y3 - y4

            RSs_r = (d1*c2 - c1*d2)/(a1*d2 + b1*c2 + 0.0)
            RSs_s = d2/(c2 + 0.0)

            RSs[i,:] = np.array([RSs_r,RSs_s])

        return RSs

    def _natural2Phys(self, RSs, capFlag = 'False'):

        """
        Mapping from natural coordinates to physcial coordinates

        # TODO: make sure the point is in the triangle
        """

        x1,y1 = self.triangle[0,0],self.triangle[0,1]
        x2,y2 = self.triangle[1,0],self.triangle[1,1]
        x3,y3 = self.triangle[2,0],self.triangle[2,1]
        x4,y4 = self.triangle[3,0],self.triangle[3,1]

        XYs = np.zeros((self.nMesh,2))

        for i in range(self.nMesh):
            pr0,ps0 = RSs[i,0],RSs[i,1]
            perturbr,perturbs = self.deltaXi[i],self.deltaEta[i]
            pr2Max, ps2Max = 1 - pr0, 1 - ps0
            pr2Min, ps2Min = 1 + pr0, 1 + ps0

            tiny = np.finfo(np.float).tiny

            # Adding relative perturbation
            # pr = pr0 + (np.abs(perturbr)/(tiny + perturbr) + 1)\
            #            /2.0*perturbr*pr2Max\
            #          - (np.abs(perturbr)/(tiny + perturbr) - 1)\
            #            /2.0*perturbr*pr2Min
            # ps = ps0 + (np.abs(perturbs)/(tiny + perturbs) + 1)\
            #            /2.0*perturbs*ps2Max\
            #          - (np.abs(perturbs)/(tiny+perturbs) - 1)\
            #            /2.0*perturbs*ps2Min

            # Adding absolute perturbation
            pr = pr0 + perturbr
            ps = ps0 + perturbs

            # ADD by JX TODO temporarily shut down capping
            # Capping the value of r&s when the absolute value is large than 1
            if capFlag == "False":
                pr = pr
                ps = ps
            else:
                if np.abs(pr) > 1:
                    pr = np.abs(pr)/(tiny + pr)

                if np.abs(ps) > 1:
                    ps = np.abs(ps)/(tiny + ps)

            Nxy1 = 1.0/4.0*(1 - pr)*(1 - ps)
            Nxy2 = 1.0/4.0*(1 + pr)*(1 - ps)
            Nxy3 = 1.0/4.0*(1 + pr)*(1 + ps)
            Nxy4 = 1.0/4.0*(1 - pr)*(1 + ps)

            XYs_x = Nxy1*x1 + Nxy2*x2 + Nxy3*x3 + Nxy4*x4
            XYs_y = Nxy1*y1 + Nxy2*y2 + Nxy3*y3 + Nxy4*y4

            # TODO:check[0]
            XYs[i,:] = np.array([XYs_x[0],XYs_y[0]])
        return XYs

    def _adjustThetaAngle(self, angle):

        """
        Adjust the theta angle to [0, pi)

        """
        
        angleAdjusted = angle
        if angle >= np.pi: 
            angleAdjusted = angle - np.pi
        elif angle < 0:
            angleAdjusted = angle + np.pi

        return angleAdjusted

    def _adjustThetaAngle2(self, angle):

        """
        Adjust the theta angle to [-pi/2, pi/2)

        """
        
        angleAdjusted = angle
        if angle >= np.pi/2.0: 
            while angleAdjusted >= np.pi/2.0:
                angleAdjusted = angleAdjusted - np.pi
        elif angle < -np.pi/2.0:
            while angleAdjusted < -np.pi/2.0:
                angleAdjusted = angleAdjusted + np.pi

        return angleAdjusted

    def _transform2quat( self, T ):
       """Construct quaternion from the transform/rotation matrix 
       :returns: quaternion formed from transform matrix
       :rtype: numpy array
       """

       # Code was copied from perl PDL code that uses backwards index ordering
       T = T.transpose()  
       den = np.array([ 1.0 + T[0,0] - T[1,1] - T[2,2],
                        1.0 - T[0,0] + T[1,1] - T[2,2],
                        1.0 - T[0,0] - T[1,1] + T[2,2],
                        1.0 + T[0,0] + T[1,1] + T[2,2]])
       
       max_idx = np.flatnonzero(den == max(den))[0]

       q = np.zeros(4)
       q[max_idx] = 0.5 * np.sqrt(max(den))
       denom = 4.0 * q[max_idx]
       if (max_idx == 0):
          q[1] =  (T[1,0] + T[0,1]) / denom 
          q[2] =  (T[2,0] + T[0,2]) / denom 
          q[3] = -(T[2,1] - T[1,2]) / denom 
       if (max_idx == 1):
          q[0] =  (T[1,0] + T[0,1]) / denom 
          q[2] =  (T[2,1] + T[1,2]) / denom 
          q[3] = -(T[0,2] - T[2,0]) / denom 
       if (max_idx == 2):
          q[0] =  (T[2,0] + T[0,2]) / denom 
          q[1] =  (T[2,1] + T[1,2]) / denom 
          q[3] = -(T[1,0] - T[0,1]) / denom 
       if (max_idx == 3):
          q[0] = -(T[2,1] - T[1,2]) / denom 
          q[1] = -(T[0,2] - T[2,0]) / denom 
          q[2] = -(T[1,0] - T[0,1]) / denom 

       return q

    def _mat2quat(self, M):
        ''' Calculate quaternion corresponding to given rotation matrix
    
        Parameters
        ----------
        M : array-like
          3x3 rotation matrix
    
        Returns
        -------
        q : (4,) array
          closest quaternion to input matrix, having positive q[0]
    
        Notes
        -----
        Method claimed to be robust to numerical errors in M
    
        Constructs quaternion by calculating maximum eigenvector for matrix
        K (constructed from input `M`).  Although this is not tested, a
        maximum eigenvalue of 1 corresponds to a valid rotation.
    
        A quaternion q*-1 corresponds to the same rotation as q; thus the
        sign of the reconstructed quaternion is arbitrary, and we return
        quaternions with positive w (q[0]).
    
        References
        ----------
        * http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        * Bar-Itzhack, Itzhack Y. (2000), "New method for extracting the
          quaternion from a rotation matrix", AIAA Journal of Guidance,
          Control and Dynamics 23(6):1085-1087 (Engineering Note), ISSN
          0731-5090
    
        Examples
        --------
        >>> import numpy as np
        >>> q = mat2quat(np.eye(3)) # Identity rotation
        >>> np.allclose(q, [1, 0, 0, 0])
        True
        >>> q = mat2quat(np.diag([1, -1, -1]))
        >>> np.allclose(q, [0, 1, 0, 0]) # 180 degree rotn around axis 0
        True
    
        '''
        # Qyx refers to the contribution of the y input vector component to
        # the x output vector component.  Qyx is therefore the same as
        # M[0,1].  The notation is from the Wikipedia article.
        Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
        # Fill only lower half of symmetric matrix
        K = np.array([
            [Qxx - Qyy - Qzz, 0,               0,               0              ],
            [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0              ],
            [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0              ],
            [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
            ) / 3.0
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K)
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[0] < 0:
            q *= -1
        return q


"""
Methods:

    writeTauToOpenFoam():

    * Write one perturbed instance of Reynolds stress to OpenFOAM file

    readTauFromOpenFoam():

    *   Read from Reynolds stress from OpenFOAM file


    == Mapping functions ==
    (Functions starting with underscore are "private")

    _perturbXiEta():

    *   Perturb on the natural coordinate system (\\xi and \eta)
    according to deltaXi and deltaEta, and write to OpenFOAM instances


    == Diagnosis functions ==

    plotTauOnBaycentric():

    *   Plot a Reynolds stress (the i^th error realization) on Baycentric
    coordinate

    plotTauOnSquare():

    *   Plot a Reynolds stress (the i^th error realization) on natural (\\xi,
    \eta) coordinate
"""
