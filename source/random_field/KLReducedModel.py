# KL Reduced model
# Copyright: Jianxun Wang (vtwjx@vt.edu)
# Mar.27, 2015

# system import
import numpy as np
import pdb
import os
import matplotlib
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as pplt
# import matplotlib.tri as tri
from matplotlib.mlab import griddata
import numpy.linalg as LA
# local import
# import pyutils as pyu


class KLReducedModel:
    """    

    Karhunen-Loeve representation of a random field, which is used to
    represent deltaXi and deltaEta, error of Reynolds stress on natural
    coordinates. Essentially a python wrapper for the functionalities in
    UQTk

    Data members:

    m: number of terms/modes used (may be determined adaptively by inspecting
    eigenvalue spectrum)

    N: number of cells in mesh
    
    meshCoords: coordinates (x, y) of the field

    omega: coefficients of KL modes (initialized as realizations of a
    specified distribution, are specified during EnKF iteration); m x 1

    modes: \sqrt{\lambda_i} * f(x)_i;  N x m

    field: field values represented with truncated KL representation,
    i.e. \sum_{i=1}^{m} omegas[i] * mode[i]

    Methods:

    __init__(): constructor

    * Initialize mesh from given (or read from file); Specify kernel
    (e.g., SqExp); call UQTk functions to compute KL modes
    
    recomputeField():

    * whenever the coefficients omega are updated, recompute the field
    value with new coeff and the saved KL modes

    """
    
    def __init__(self, meshCoords, kernelType, hyperPara, 
                 m, KLRatio, sigmaCoords=[], areaCoords = [], instance = 'default'):
        """
        Initialization of data members
        
        Args:
            meshCoords: matrix of mesh coordinates, can be 1D, 2D and 3D
                size of meshCoords matrix
                    1D: N by 1
                    2D: N by 2
                    3D: N by 3
            areaCoords: vector (colum) of cell area corresponding to cell center
                        (for 1D case it not used)            
            KernelType: keyword for kernels ("sqExp" or "Exp")
            hyperPara: hyperparameters for kernel (sigma, len)   
        
        """
        self.meshCoords = meshCoords # matrix of mesh coordinates
        self.areaCoords = areaCoords # vector of cell area
        self.sigmaCoords = sigmaCoords
        #pdb.set_trace()
        (numRow, numColumn) = meshCoords.shape
        assert numRow > numColumn, \
        "The coordinate matrix should be N by (1, 2 or 3), please transpose"
        self.N = numRow  # Number of cells in mesh
        self.dim = numColumn # Dimension of the mesh
        self.kernelType = kernelType # keyword of kernel type (SqExp, or Exp)
        self.hyperPara = hyperPara # hyperparameters for selected kernel
        self.m = m # number of kl modes
        self.KLRatio = KLRatio
        # Creat klexpansionData folder
        # The c++ solver source file is specified relative this this file
        thisFilePath = os.path.dirname(os.path.realpath(__file__))
        self.srcDir = os.path.join(thisFilePath, '../cpp/KLexpansionSolver/klSolver/')
        self.klFolderName = 'klExpansionData'+instance
        self.gridName1D = 'cellCenter1D.dat'
        self.gridName2D = 'cellCenter2D.dat'
        self.gridName3D = 'cellCenter3D.dat'
        self.areaName2D = 'cellArea2D.dat'
        self.sigmaName2D = 'cellSigma2D.dat'
        self.areaName3D = 'cellArea3D.dat'
        self.sigmaName3D = 'cellSigma3D.dat'
        self.instance = instance

        if not os.path.exists(self.klFolderName + '1D'):
            print "creat klExpansionData1D folder for the data related with KL"
            os.system('mkdir ' + self.klFolderName + '1D')
        if not os.path.exists(self.klFolderName + '2D'):
            print "creat klExpansionData2D folder for the data related with KL"    
            os.system('mkdir ' + self.klFolderName + '2D')
        if not os.path.exists(self.klFolderName + '3D'):
            print "creat klExpansionData3D folder for the data related with KL"    
            os.system('mkdir ' + self.klFolderName + '3D')
   
    def calKLModes(self):
        """
        call the KL-solver to solve the eigenvalues and KL modes
        
        Args:
        Return:
        """
        # output the meshgrid as the file can be read by kl_solver
        self._generateMesh()
        
        # for 1D cases
        if self.dim == 1:
            print "\n\nNow we start a 1-D case:\n\n\n"
            os.system(self.srcDir + 'kl_sol1D.x -x '+self.gridName1D 
                      + ' -c '+self.kernelType + ' -n '+str(self.N) 
                      + ' -e '+str(self.m) + ' -s '+str(self.hyperPara[0]) 
                      + ' -l '+str(self.hyperPara[1]))
                      
            os.system('mv *.dat ' + self.klFolderName + '1D')
            
            cov = np.loadtxt(self.klFolderName+'1D/cov.dat')
            eig = np.loadtxt(self.klFolderName+'1D/eig.dat')
            KLmodes = np.loadtxt(self.klFolderName+'1D/KLmodes.dat')
            #TODO KLmode first column is x coordinate
            
            print "\nFinished writing files, case done\n\n"
        
        # for 2D cases
        elif self.dim == 2:
            print "\n\nNow we start a 2-D case:\n\n\n"        
            os.system(self.srcDir + 'kl_solFOAM.x -x'+self.gridName2D
                      + ' -a ' +self.areaName2D + ' -v ' + self.sigmaName2D
                      + ' -c '+self.kernelType
                      + ' -e ' +str(self.m) + ' -s' +str(self.hyperPara[0])
                      + ' -l ' +str(self.hyperPara[1]))
                      
            os.system('mv *.dat ' + self.klFolderName + '2D')
            cov = np.loadtxt(self.klFolderName+'2D/cov.dat')
            eig = np.loadtxt(self.klFolderName+'2D/eig.dat')
            KLmodes = np.loadtxt(self.klFolderName+'2D/KLmodes.dat')
            #TODO KLmode first two column is x and y coordinate
            
            print "\nFinished writing files, case done\n\n"          
        
        # for 3D cases
        elif self.dim == 3:
            #TODO: 3D case need to be developed in the future
            print "\n\nNow we start a 3-D case:\n\n\n"        
            os.system(self.srcDir + 'kl_solFOAM3D.x -g'+self.gridName3D
                      + ' -a ' +self.areaName3D + ' -v ' + self.sigmaName3D
                      + ' -c '+self.kernelType + ' -n ' + str(self.m)
                      + ' -r ' +str(self.KLRatio) + ' -s' +str(self.hyperPara[0])
                      + ' -x ' +str(self.hyperPara[1])
                      + ' -y ' +str(self.hyperPara[2])
                      + ' -z ' +str(self.hyperPara[3]))
                      
            os.system('mv *.dat ' + self.klFolderName + '3D')
            cov = np.loadtxt(self.klFolderName+'3D/cov.dat')
            eig = np.loadtxt(self.klFolderName+'3D/eig.dat')
            KLmodes = np.loadtxt(self.klFolderName+'3D/KLmodes.dat')
            (n, m) = KLmodes.shape
            self.m = m - 2
            #TODO KLmode first two column is x and y coordinate
            
            print "\nFinished writing files, case done\n\n"                 

        return cov, eig, KLmodes, self.m 

    
    def recomputeFieldReduced(self, meanField = 0, omega = None, KLModesDir = None, 
                              KLModes = None):
        """
        reconstruct the Field by finite KL modes 
        
        Args:
            meanField: <F(x, theta)> mean value of the field, now we assume 
                       they are all 0; fieldMean (N by 1 vector)                       
            
            KLModes: KL modes matrix (m+2 by N)
            
            KLModesDir: path of KLmodes matrix
            (Note: KLmMdes can be input or can be read from file(KLMOdesDir))
            
            omega: the coefficients for KL modes (normally it is unit random iid)             
        
        Return:
            recField: reconstructed field (n by 1)
            coord: corresponding coordinate (x, y)
        """
        if KLModes is None:
            pass
        else:
            KLmodes = KLModes
        if KLModesDir is None:
            pass
        else:
            KLmodes = np.loadtxt(KLModesDir)
            
        N, m = np.shape(KLmodes) 
        m = m -2     # N is number of cells, m is number of KL modes
                     # The first two column are x, y coordinates 
        
        KLmodeMatrix = KLmodes[:, 2:]  # only contain KL modes, 
                                       # mode 1 --column 1 ......
        if meanField == 0:
            meanField = np.zeros((N, 1)) 
                                                
        RandomVec = omega
       
        assert len(RandomVec) == m, \
        "Lengths of KL coefficient omega (%d) and modes (%d) differs!"\
        % (len(RandomVec), m)
            
        recField = meanField +  np.dot(KLmodeMatrix, RandomVec)  
        coord = KLmodes[:, 0:2]               
        return recField, coord


    def projectFieldReduced(self, meanField = 0, recField = None, KLModesDir = None, 
                              KLModes = None):
        """
        Project the Field to finite nkl coefficient for modes 
        
        Args:
            meanField: <F(x, theta)> mean value of the field, now we assume 
                       they are all 0; fieldMean (N by 1 vector)                       
            
            KLModes: KL modes matrix (m+2 by N)
            
            KLModesDir: path of KLmodes matrix
            (Note: KLmMdes can be input or can be read from file(KLMOdesDir))
            
            recField: reconstructed field)             
        
        Return:
            omega: coeffcient for modes (n_modes by 1)
        """
        if KLModes is None:
            pass
        else:
            KLmodes = KLModes
        if KLModesDir is None:
            pass
        else:
            KLmodes = np.loadtxt(KLModesDir)
            
        N, m = np.shape(KLmodes) 
        m = m -2     # N is number of cells, m is number of KL modes
                     # The first two column are x, y coordinates 
        
        KLmodeMatrix = KLmodes[:m, 2:]  # only contain KL modes, 
                                       # mode 1 --column 1 ......
        if meanField == 0:
            meanField = np.zeros((N, 1)) 
                                                   
        field = recField - meanField  
        omega = np.dot(LA.inv(KLmodeMatrix), field[:m])
        return omega
        
        
        
    def recomputeFieldFull(self, meanField = 0, fileDir = "none", cov= 0):    
        """
        reconstruct the Field by finite KL modes 
        
        Args:
        Return:
        """
        #TODO: need to be implemented later
        if cov == 0:
            cov = np.loadtxt(fileDir)
        
        N, N = np.shape(cov)
        
        if meanField == 0:
            meanField = np.zeros((N, 1)) 
        
        RandomVec = self.uncorrUniRandom(N)
        pdb.set_trace()
        
        # eigen decomposation
        lamb, B = LA.eig(cov)
        D = np.diag((lamb))
        sqD = np.sqrt(D)
        A = np.dot(B, sqD)
        
        recField = meanField +  np.dot(A, RandomVec)

        return recField
           
        
        
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

               
                
    def plotCov(self, cov):
        """
        plot covariance matrix
        
        Arg: 
            cov: covariance matrix
        return:
            None 
        """  
        vmax = np.array(cov).max()
        fig=plt.figure(figsize=(4, 4))
        ax = fig.add_axes([0.05, 0.05, 0.85, 0.85]) 
        plt.imshow(cov, interpolation='nearest', vmin=-0.5*vmax, vmax=vmax,
                  cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        plt.title("length scale = "+str(self.hyperPara[1]))
        plt.savefig("cov_sigma_"+str(self.hyperPara[0])
                    +"_len"+str(self.hyperPara[1])+".eps")
    
    def plotKLModes1D(self, nkl, fileDir):
        """
        plot Modes for 1D cases
        
        Arg: 
            fileDir: path of KLModes
            nkl: number of modes want to be plotted
        return:
            None 
        """
        #parameters
        lw = 2
        fs = 16

        modeMatrix = np.loadtxt(fileDir)
        xp = modeMatrix[:, 0]

        fig = plt.figure(figsize=(6,4))
        ax=fig.add_axes([0.17, 0.15, 0.75, 0.75]) 
        pleg=[]       
        for i in range(1,nkl+1):
            y = modeMatrix[:, i]
            pleg.append(plt.plot(xp,y,linewidth=lw))       
        plt.xlabel("x",fontsize=fs)
        plt.ylabel(r"$\sqrt{\lambda_k}f_k$",fontsize=fs)        
        ax.set_ylim([-4,4])      

        leg=plt.legend( (pleg[0][0], pleg[1][0], pleg[2][0], pleg[3][0]),
                        (r"$f_1$", r"$f_2$", r"$f_3$", r"$f_4$"),'lower right' )
        plt.savefig("KLmodes1D.eps")
        
    def plotField(self, field, coord, Nres, figName="field.eps"):
        """
        plot contour for a field
        
        Arg:
            field: field data need to be plotted (N by 1)
            coord: coordinate data for field (N by 2) (x, y) pair
            Nres:  number for resolution of the plotting
        return:
        """
        fig = plt.figure()
        
        x = coord[:, 0]
        y = coord[:, 1]
        z = field[:, 0]
        n = len(field)
        xmin = min(x)
        xmax = max(x)
        ymin = min(y)
        ymax = max(y)
        xi = np.linspace(xmin, xmax, Nres)
        yi = np.linspace(ymin, ymax, Nres)
        
        X, Y = np.meshgrid(xi, yi)
        #pdb.set_trace()
        Z = griddata(x, y, z, xi, yi)
        
        surf = plt.contour(X, Y, Z, 15, linewidths = 0.5, colors = 'k')
        plt.pcolormesh(X, Y, Z, cmap = plt.get_cmap('rainbow'))
        #ax.set_zlim3d(np.min(Z), np.max(Z))
        plt.colorbar()
        # plt.show()                       
        plt.savefig(figName)

        
    def plotModes(self, prefix=""):
        """
        Plot the retained modes in separate figures
        """

        modes = np.loadtxt(self.klFolderName+'3D/KLmodes.dat')
        coords = modes[:, 0:2]
        #pdb.set_trace()
        Nres = 30
        for i in range(self.m):
            modeFigName = prefix + 'mode-' + str(i) + '.eps'
            modei = np.array([modes[:, i+2]]).T
            #pdb.set_trace()
            #print i, ': ', modei.shape
            self.plotField(modei, coords, Nres, modeFigName)

        
    def plotFieldSimple(self, field, coord, Nres, label=""):
        """
        plot contour for a field
        
        Arg:
            field: field data need to be plotted (N by 1)
            coord: coordinate data for field (N by 2) (x, y) pair
            Nres:  number for resolution of the plotting
        return:
        """
        
        x = coord[0:-1:2, 0]
        
        y = coord[0:-1:2, 1]
        z = field[0:-1:2, 0]
        #pdb.set_trace()
        plt.plot(y, z);
                
                   
    def _generateMesh(self):
        """
        Write meshgrid file for UQTK to read, each dimension need to be wrote
        as a column to xgrid.dat, ygrid.dat and zgrid.dat, respectively.
        
        Args:
            None
        Return:
            None
        """
        if self.dim == 1:
            np.savetxt('./cellCenter1D.dat', self.meshCoords)
        elif self.dim == 2:
            np.savetxt('./cellCenter2D.dat', self.meshCoords)
            np.savetxt('./cellArea2D.dat', self.areaCoords)
            np.savetxt('./cellSigma2D.dat', self.sigmaCoords)
        elif self.dim == 3:
            np.savetxt('./cellCenter3D.dat', self.meshCoords)        
            np.savetxt('./cellArea3D.dat', self.areaCoords)
            np.savetxt('./cellSigma3D.dat', self.sigmaCoords)
        else:
            print "we cannot handle more than 3 dimensions, please check \
                   meshCoords matrix shape, need to be n by (1, 2, 3)"
        
        
            
