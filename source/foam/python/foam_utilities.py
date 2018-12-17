# Manipulate OpenFOAM files
###############################################################################

# system import
import numpy as np
import os
import os.path as ospt
import shutil
import re
import tempfile
import subprocess

# local import
from data_assimilation import utilities as util


def get_number_cells(foam_case='.'):
    bash_command = "checkMesh -case " + foam_case + \
        " -time '0' | grep '    cells:' > tmp.ncells"
    subprocess.call(bash_command, shell=True)
    file = open('tmp.ncells', 'r')
    line = fin.read()
    file.close()
    os.remove('tmp.ncells')
    return int(line.split()[-1])


def get_cell_coordinates(foam_case='.'):
    bash_command = "writeCellCentres -case " + foam_case + " -time '0' " + \
        "&> /dev/null"
    subprocess.call(bash_command, shell=True)
    coords = read_cell_coordinates(ospt.join(foam_case, '0'))
    os.remove(ospt.join(foam_case, '0', 'ccx'))
    os.remove(ospt.join(foam_case, '0', 'ccy'))
    os.remove(ospt.join(foam_case, '0', 'ccz'))
    return coords


def get_cell_volumes(foam_case='.'):
    bash_command = "writeCellCentres -case " + foam_case + " -time '0' " + \
        "&> /dev/null"
    subprocess.call(bash_command, shell=True)
    os.rename(ospt.join(foam_case, '0', 'V'), ospt.join(foam_case, '0', 'cv'))
    vol = read_cell_volumes(ospt.join(foam_case, '0'))
    os.remove(ospt.join(foam_case, '0', 'cv'))
    os.remove(ospt.join(foam_case, '0', 'ccx'))
    os.remove(ospt.join(foam_case, '0', 'ccy'))
    os.remove(ospt.join(foam_case, '0', 'ccz'))
    return vol


def read_cell_coordinates(file_dir):
    """

    Arg:
    file_dir: The directory path of file of ccx, ccy, and ccz

    Regurn:
    coordinate: matrix of (x, y, z)
    """
    coorX = ospt.join(file_dir, "ccx")
    coorY = ospt.join(file_dir, "ccy")
    coorZ = ospt.join(file_dir, "ccz")

    resMidx = extract_scalar(coorX)
    resMidy = extract_scalar(coorY)
    resMidz = extract_scalar(coorZ)

    # write it in Tautemp
    fout = open('xcoor.txt', 'w')
    glob_patternx = resMidx.group()
    glob_patternx = re.sub(r'\(', '', glob_patternx)
    glob_patternx = re.sub(r'\)', '', glob_patternx)
    fout.write(glob_patternx)
    fout.close()
    xVec = np.loadtxt('xcoor.txt')
    os.remove('xcoor.txt')

    fout = open('ycoor.txt', 'w')
    glob_patterny = resMidy.group()
    glob_patterny = re.sub(r'\(', '', glob_patterny)
    glob_patterny = re.sub(r'\)', '', glob_patterny)
    fout.write(glob_patterny)
    fout.close()
    yVec = np.loadtxt('ycoor.txt')
    os.remove('ycoor.txt')

    fout = open('zcoor.txt', 'w')
    glob_patternz = resMidz.group()
    glob_patternz = re.sub(r'\(', '', glob_patternz)
    glob_patternz = re.sub(r'\)', '', glob_patternz)
    fout.write(glob_patternz)
    fout.close()
    zVec = np.loadtxt('zcoor.txt')
    os.remove('zcoor.txt')

    coordinate = np.vstack((xVec, yVec, zVec))
    coordinate = coordinate.T
    return coordinate


def read_cell_volumes(file_dir):
    """

    Arg:
    file_dir: The directory path of file of cv

    Regurn:
    coordinate: vector of cell area
    """
    cellVolume = ospt.join(file_dir, "cv")
    resMid = extract_scalar(cellVolume)

    # write it in Tautemp
    fout = open('cellVolume.txt', 'w')
    glob_patternx = resMid.group()
    glob_patternx = re.sub(r'\(', '', glob_patternx)
    glob_patternx = re.sub(r'\)', '', glob_patternx)
    fout.write(glob_patternx)
    fout.close()
    cvVec = np.loadtxt('cellVolume.txt')
    os.remove('cellVolume.txt')
    cvVec = np.array([cvVec])
    cellVolume = cvVec.T
    return cellVolume


def extract_scalar(scalar_file):
    """ subFunction of readTurbStressFromFile
        Using regular expression to select scalar value out

    Args:
    scalar_file: The directory path of file of scalar

    Returns:
    resMid: scalar selected;
            you need use resMid.group() to see the content.
    """
    fin = open(scalar_file, 'r')  # need consider directory
    line = fin.read()  # line is k file to read
    fin.close()
    # select k as ()pattern (Using regular expression)
    patternMid = re.compile(r"""
        \(                                                   # match"("
        \n                                                   # match next line
        (
        [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
        \n                                                   # match next line
        )+                                                   # search greedly
        \)                                                   # match")"
    """, re.DOTALL | re.VERBOSE)
    resMid = patternMid.search(line)

    return resMid


def read_scalar_from_file(file_name):
    """

    Arg:
    file_name: The file name of scalar file in OpenFOAM form

    Regurn:
    scalar file in vector form
    """
    resMid = extract_scalar(file_name)

    # write it in Tautemp
    fout = open('temp.txt', 'w')
    glob_patternx = resMid.group()
    glob_patternx = re.sub(r'\(', '', glob_patternx)
    glob_patternx = re.sub(r'\)', '', glob_patternx)
    fout.write(glob_patternx)
    fout.close()
    scalarVec = np.loadtxt('temp.txt')
    os.remove('temp.txt')
    return scalarVec


def write_scalar_to_file(Scalar, ScalarFile):
    """Write the modified scalar to the scalar the OpenFOAM file

    Args:
    Scalar: E.g. DeltaXi or DeltaEta
    ScalarFile: path of the Scalar file in OpenFOAM

    Returns:
    None

    """

    # Find openFoam scalar file's pattern'
    (resStartk, resEndk) = extractTensorPattern(ScalarFile)

    tempFile = 'scalarTemp'
    np.savetxt('scalarTemp', Scalar)

    # read scalar field
    fin = open(tempFile, 'r')
    field = fin.read()
    fin.close()
    # revise k

    fout = open(ScalarFile, 'w')
    fout.write(resStartk.group())
    fout.write("\n")
    fout.write(field)
    fout.write(resEndk.group())
    fout.close()

    os.remove('scalarTemp')


def read_vector_from_file(UFile):
    """
    """
    resMid = extractVector(UFile)
    fout = open('Utemp', 'w')
    glob_pattern = resMid.group()
    glob_pattern = re.sub(r'\(', '', glob_pattern)
    glob_pattern = re.sub(r'\)', '', glob_pattern)
    fout.write(glob_pattern)
    fout.close()
    vector = np.loadtxt('Utemp')
    os.remove('Utemp')
    return vector


def readTurbStressFromFile(tauFile):
    """

    Arg:
    tauFile: The directory path of file of tau

    Regurn:
    tau: Matrix of Reynolds stress (sysmetric tensor)
    """
    resMid = extractSymmTensor(tauFile)

    # write it in Tautemp
    fout = open('tau.txt', 'w')
    glob_pattern = resMid.group()
    glob_pattern = re.sub(r'\(', '', glob_pattern)
    glob_pattern = re.sub(r'\)', '', glob_pattern)

    tau = glob_pattern
    fout.write(tau)
    fout.close()

    tau = np.loadtxt('tau.txt')
    return tau


def readTensorFromFile(tauFile):
    """

    Arg:
    tauFile: The directory path of file of tau

    Regurn:
    tau: Matrix of Reynolds stress (sysmetric tensor)
    """
    resMid = extractTensor(tauFile)

    # write it in Tautemp
    fout = open('tau.txt', 'w')
    glob_pattern = resMid.group()
    glob_pattern = re.sub(r'\(', '', glob_pattern)
    glob_pattern = re.sub(r'\)', '', glob_pattern)

    tau = glob_pattern
    fout.write(tau)
    fout.close()

    tau = np.loadtxt('tau.txt')
    return tau


def readVelocityFromFile(UFile):
    """ Function is to get value of U from the openFoam U files

    Args:
    UFile: directory of U file in OpenFoam

    Returns:
    U: as vector (u1,v1,w1,u2,v2,w2,....uNcell,vNcell,wNcell)
    """
    resMid = extractVector(UFile)

    # write it in Utemp
    fout = open('Utemp', 'w')
    fout.write(resMid.group())
    fout.close()

    # write it in UM with the pattern that numpy.load txt could read
    fin = open('Utemp', 'r')
    fout = open('UM.txt', 'w')

    while True:
        line = fin.readline()
        line = line[1:-2]
        fout.write(line)
        fout.write(" ")
        if not line:
            break
    fin.close()
    fout.close()
    # to convert UM as U vector: (u1,v1,w1,u2,v2,w2,....uNcell,vNcell,wNcell)
    U = np.loadtxt('UM.txt')
    return U


def readTurbCoordinateFromFile(fileDir):
    """

    Arg:
    fileDir: The directory path of file of ccx, ccy, and ccz

    Regurn:
    coordinate: matrix of (x, y, z)
    """
    coorX = fileDir + "ccx"
    coorY = fileDir + "ccy"
    coorZ = fileDir + "ccz"

    resMidx = extractScalar(coorX)
    resMidy = extractScalar(coorY)
    resMidz = extractScalar(coorZ)

    # write it in Tautemp
    fout = open('xcoor.txt', 'w')
    glob_patternx = resMidx.group()
    glob_patternx = re.sub(r'\(', '', glob_patternx)
    glob_patternx = re.sub(r'\)', '', glob_patternx)
    fout.write(glob_patternx)
    fout.close()
    xVec = np.loadtxt('xcoor.txt')

    fout = open('ycoor.txt', 'w')
    glob_patterny = resMidy.group()
    glob_patterny = re.sub(r'\(', '', glob_patterny)
    glob_patterny = re.sub(r'\)', '', glob_patterny)
    fout.write(glob_patterny)
    fout.close()
    yVec = np.loadtxt('ycoor.txt')

    fout = open('zcoor.txt', 'w')
    glob_patternz = resMidz.group()
    glob_patternz = re.sub(r'\(', '', glob_patternz)
    glob_patternz = re.sub(r'\)', '', glob_patternz)
    fout.write(glob_patternz)
    fout.close()
    zVec = np.loadtxt('zcoor.txt')

    coordinate = np.vstack((xVec, yVec, zVec))
    coordinate = coordinate.T
    return coordinate


def readTurbCellAreaFromFile(fileDir):
    """

    Arg:
    fileDir: The directory path of file of cv, dz.dat

    Regurn:
    coordinate: vector of cell area
    """
    cellVolume = fileDir + "cv"
    dvfile = fileDir + "dz.dat"
    resMid = extractScalar(cellVolume)

    # write it in Tautemp
    fout = open('cellVolume.txt', 'w')
    glob_patternx = resMid.group()
    glob_patternx = re.sub(r'\(', '', glob_patternx)
    glob_patternx = re.sub(r'\)', '', glob_patternx)
    fout.write(glob_patternx)
    fout.close()
    cvVec = np.loadtxt('cellVolume.txt')
    cvVec = np.array([cvVec])
    dz = np.loadtxt(dvfile)
    cellArea = cvVec / dz
    cellArea = cellArea.T
    return cellArea


def readScalarFromFile(fileName):
    """

    Arg:
    fileName: The file name of scalar file in OpenFOAM form

    Regurn:
    scalar file in vector form
    """
    resMid = extractScalar(fileName)

    # write it in Tautemp
    fout = open('temp.txt', 'w')
    glob_patternx = resMid.group()
    glob_patternx = re.sub(r'\(', '', glob_patternx)
    glob_patternx = re.sub(r'\)', '', glob_patternx)
    fout.write(glob_patternx)
    fout.close()
    scalarVec = np.loadtxt('temp.txt')
    return scalarVec


def extractSymmTensor(tensorFile):
    """ subFunction of readTurbStressFromFile
        Using regular expression to select tau value out (sysmetric tensor)
        Requiring the tensor to be 6-components tensor, and output is with
        Parentheses.

    Args:
    tensorFile: The directory path of file of tensor

    Returns:
    resMid: the tau as (tau11, tau12, tau13, tau22, tau23, tau33);
            you need use resMid.group() to see the content.
    """

    fin = open(tensorFile, 'r')  # need consider directory
    line = fin.read()  # line is U file to read
    fin.close()

    # select U as (X X X)pattern (Using regular expression)
    patternMid = re.compile(r"""
    (
    \(                                                   # match(
    [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
    (\ )                                                 # match space
    [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
    (\ )                                                 # match space
    [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
    (\ )                                                 # match space
    [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
    (\ )                                                 # match space
    [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
    (\ )                                                 # match space
    [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
    \)                                                   # match )
    (\n|\ )                                              # match next line
    )+                                                   # search greedly
    """, re.DOTALL | re.VERBOSE)

    resMid = patternMid.search(line)

    return resMid


def extractTensor(tensorFile):
    """ subFunction of readTurbStressFromFile
        Using regular expression to select tau value out (general tensor)
        Requiring the tensor to be 9-components tensor, and output is with
        Parentheses.

    Args:
    tensorFile: The directory path of file of tensor

    Returns:
    resMid: the tau as (tau11, tau12, tau13, tau22, tau23, tau33);
            you need use resMid.group() to see the content.
    """

    fin = open(tensorFile, 'r')  # need consider directory
    line = fin.read()  # line is U file to read
    fin.close()

    # select U as (X X X)pattern (Using regular expression)
    patternMid = re.compile(r"""
    (
    \(                                                   # match(
    [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
    (\ )                                                 # match space
    [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
    (\ )                                                 # match space
    [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
    (\ )                                                 # match space
    [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
    (\ )                                                 # match space
    [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
    (\ )                                                 # match space
    [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
    (\ )                                                 # match space
    [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
    (\ )                                                 # match space
    [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
    (\ )                                                 # match space
    [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
    \)
    (\n|\ )                                              # match next line
    )+                                                   # search greedly
    """, re.DOTALL | re.VERBOSE)

    resMid = patternMid.search(line)

    return resMid


def extractVector(vectorFile):
    """ Function is using regular expression select Vector value out

    Args:
    UFile: The directory path of file: U

    Returns:
    resMid: the U as (Ux1,Uy1,Uz1);(Ux2,Uy2,Uz2);........
    """

    fin = open(vectorFile, 'r')  # need consider directory
    line = fin.read()  # line is U file to read
    fin.close()
    # select U as (X X X)pattern (Using regular expression)
    patternMid = re.compile(r"""
    (
    \(                                                   # match(
    [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
    (\ )                                                 # match space
    [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
    (\ )                                                 # match space
    [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
    \)                                                   # match )
    \n                                                   # match next line
    )+                                                   # search greedly
    """, re.DOTALL | re.VERBOSE)
    resMid = patternMid.search(line)
    return resMid


def extractScalar(scalarFile):
    """ subFunction of readTurbStressFromFile
        Using regular expression to select scalar value out

    Args:
    scalarFile: The directory path of file of scalar

    Returns:
    resMid: scalar selected;
            you need use resMid.group() to see the content.
    """
    fin = open(scalarFile, 'r')  # need consider directory
    line = fin.read()  # line is k file to read
    fin.close()
    # select k as ()pattern (Using regular expression)
    patternMid = re.compile(r"""
        \(                                                   # match"("
        \n                                                   # match next line
        (
        [\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
        \n                                                   # match next line
        )+                                                   # search greedly
        \)                                                   # match")"
    """, re.DOTALL | re.VERBOSE)
    resMid = patternMid.search(line)

    return resMid


def extractTensorPattern(tensorFile):
    """ Function is using regular expression select OpenFOAM tensor files pattern

    Args:
    tensorFile: directory of file U in OpenFoam, which you want to change

    Returns:
    resStart: Upper Pattern
    resEnd:  Lower Pattern
    """
    fin = open(tensorFile, 'r')
    line = fin.read()
    fin.close()
    patternStart = re.compile(r"""
        .                        # Whatever except next line
        +?                       # Match 1 or more of preceding-Non-greedy
        internalField            # match interanlField
        [a-zA-Z\ \n]+            # match class contained a-z A-Z space and \n
        <((vector)|(symmTensor)|(scalar))>    # match '<vector>' or '<scalar>'
        ((\ )|(\n))+?            # space or next line--non greedy
        [0-9]+                   # match 0-9
        ((\ )|(\n))+?            # match space or next line
        \(                       # match (
    """, re.DOTALL | re.VERBOSE)
    resStart = patternStart.search(line)

    patternEnd = re.compile(r"""
        \)                       # match )
        ((\ )|;|(\n))+?          # match space or nextline or ;
        boundaryField            # match boundaryField
        ((\ )|(\n))+?            # match space or nextline
        \{                       # match {
        .+                       # match whatever in {}
        \}                       # match }
    """, re.DOTALL | re.VERBOSE)
    resEnd = patternEnd.search(line)
    return resStart, resEnd


def _extractLocPattern(locFile):
    """ Function is using regular expression select OpenFOAM Location files pattern

    Args:
    tensorFile: directory of Locations in OpenFoam, which you want to change

    Returns:
    resStart: Upper Pattern
    resEnd:  Lower Pattern
    """
    fin = open(locFile, 'r')
    line = fin.read()
    fin.close()
    patternStart = re.compile(r"""
        .                        # Whatever except next line
        +?                       # Match 1 or more of preceding-Non-greedy
        \(                       # match (
    """, re.DOTALL | re.VERBOSE)
    resStart = patternStart.search(line)

    patternEnd = re.compile(r"""
        \)                       # match )
        ((\ )|;|(\n))+          # match space or nextline or ;
        ((\ )|;|(\n))+          # match space or nextline or ;
    """, re.DOTALL | re.VERBOSE)
    resEnd = patternEnd.search(line)
    return resStart, resEnd


def writeLocToFile(coords, locFile):
    """Write the coords to the locFile

    Args:
    coords: locations coordinates
    locFile: path of the location file in OpenFOAM

    Returns:
    None
    """
    # add parentheses to tensor
    tempFile = 'loctemp'
    np.savetxt(tempFile, coords)
    os.system("sed -i -e 's/^/(/g' " + tempFile)
    os.system(r"sed -i -e 's/\($\)/)/g' " + tempFile)
    # read tensor out
    fin = open(tempFile, 'r')
    field = fin.read()
    fin.close()
    # read patterns
    (resStart, resEnd) = _extractLocPattern(locFile)
    fout = open(locFile, 'w')
    fout.write(resStart.group())
    fout.write("\n")
    fout.write(field)
    fout.write(resEnd.group())
    fout.close()


def writeTurbStressToFile(tau, tauFile):
    """Write the modified tau to the tauFile

    Args:
    tau: modified reynolds stress
    tauFile: path of the tau

    Returns:
    None
    """
    # add parentheses to tensor
    tempFile = 'tauUpdate'
    np.savetxt(tempFile, tau)
    os.system("sed -i -e 's/^/(/g' " + tempFile)
    os.system(r"sed -i -e 's/\($\)/)/g' " + tempFile)

    # read tensor out
    fin = open(tempFile, 'r')
    field = fin.read()
    fin.close()
    # read patterns
    (resStartTau, resEndTau) = extractTensorPattern(tauFile)

    fout = open(tauFile, 'w')
    fout.write(resStartTau.group())
    fout.write("\n")
    fout.write(field)
    fout.write(resEndTau.group())
    fout.close()


def writeVelocityToFile(U, UFile):
    """Write the modified tau to the tauFile

    Args:
    U: modified velocity
    UFile: path of the U file in OpenFOAM

    Returns:
    None
    """
    # add parentheses to tensor
    tempFile = 'Utemp'
    np.savetxt(tempFile, U)
    os.system("sed -i -e 's/^/(/g' " + tempFile)
    os.system(r"sed -i -e 's/\($\)/)/g' " + tempFile)

    # read tensor out
    fin = open(tempFile, 'r')
    field = fin.read()
    fin.close()
    # read patterns
    (resStartU, resEndU) = extractTensorPattern(UFile)

    fout = open(UFile, 'w')
    fout.write(resStartU.group())
    fout.write("\n")
    fout.write(field)
    fout.write(resEndU.group())
    fout.close()


def org_genFolders(Npara, Ns, caseName, caseNameObservation, DAInterval, Tau):
    """ Function:to generate case folders

    Args:
    Npara: number of parameters
    Ns: number of parameters
    caseName: templecase name(string)
    caseNameObservation: Observationcase name
    DAInterval: data assimilation interval

    Returns:
    None
    """
    # remove previous ensemble case files
    os.system('rm -fr ' + caseName + '-tmp_*')
    os.system('rm -fr ' + caseName + '_benchMark')
    writeInterval = "%.6f" % DAInterval
    ii = 0
    caseCount = np.linspace(1, Ns, Ns)
    for case in caseCount:

        print "#", case, "/", Ns, " Creating folder for Case = ", case

        tmpCaseName = caseName + "-tmp_" + str(case)

        if(ospt.isdir(tmpCaseName)):  # see if tmpCaseName's'directory is existed
            shutil.rmtree(tmpCaseName)
        shutil.copytree(caseName, tmpCaseName)  # copy

        # Replace Tau ensemble for cases ensemble
        tauTemp = Tau[ii, :, :]
        tauFile = './' + tmpCaseName + '/0/Tau'
        writeTurbStressToFile(tauTemp, tauFile)

        # Replace case writeInterval
        rasFile = ospt.join(os.getcwd(), tmpCaseName, "system", "controlDict")
        util.replace(rasFile, "<writeInterval>", writeInterval)

        ii += 1

    # generate observation folder
    if(ospt.isdir(caseNameObservation)):
        shutil.rmtree(caseNameObservation)
    shutil.copytree(caseName, caseNameObservation)  # copy
    # prepare case directory
    rasFile = ospt.join(
        os.getcwd(),
        caseNameObservation,
        "system",
        "controlDict")
    util.replace(rasFile, "<writeInterval>", writeInterval)


def callFoam(ensembleCaseName, caseSolver, pseudoObs, parallel=False):
    """ Function is to call myPisoFoam and sampling (solved to next DAInterval)

    Args:
    ensembleCaseName: name of openFoam ensemble case folder

    Returns:
    None
    """
    if(parallel):
        # run pisoFoam (or other OpenFOAM solver as appropriate)
        os.system('mpirun -np 4 pisoFoam -case ' + ensembleCaseName +
                  ' -parallel > ' + ensembleCaseName + '/log')
        # extract value at observation location by "sample"_pseudo matrix H)
        os.system('mpirun -np 4 sample -case ' + ensembleCaseName +
                  ' -latestTime -parallel > ' + ensembleCaseName + '/log')
    else:  # same as above, but for single processor runs
        os.system(caseSolver + ' -case ' + ensembleCaseName + ' &>>' +
                  ensembleCaseName + '/log')
        os.system('sample -case ' + ensembleCaseName + ' -latestTime > ' +
                  ensembleCaseName + '/sample.log')

        # os.system('myPisoFoam -case ' + ensembleCaseName)

        # extract value at observation location
        if pseudoObs == 1:
            os.system('sample -case ' + ensembleCaseName + ' -time 0 >> ' +
                      ensembleCaseName + '/log')
            os.system('sample -case ' + ensembleCaseName + ' -latestTime >> ' +
                      ensembleCaseName + '/log')
        else:
            pass
