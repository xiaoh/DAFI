# Copyright 2018 Virginia Polytechnic Institute and State University.
""" OpenFOAM file manipulation. """

# standard library imports
import numpy as np
import os
import os.path as ospt
import shutil
import re
import tempfile
import subprocess

# local import
from data_assimilation import utilities as util


# global variables
NSCALAR = 1
NVECTOR = 3
NSYMMTENSOR = 6
NTENSOR = 9


# get mesh properties
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
    cellVolume = np.loadtxt('cellVolume.txt')
    os.remove('cellVolume.txt')
    return cellVolume


# read and write fields
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
    (resStartk, resEndk) = extract_foam_pattern(ScalarFile)

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
    resMid = extract_vector(UFile)
    fout = open('Utemp', 'w')
    glob_pattern = resMid.group()
    glob_pattern = re.sub(r'\(', '', glob_pattern)
    glob_pattern = re.sub(r'\)', '', glob_pattern)
    fout.write(glob_pattern)
    fout.close()
    vector = np.loadtxt('Utemp')
    os.remove('Utemp')
    return vector


def extract_vector(vectorFile):
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


def read_tensor_from_file(tauFile):
    """

    Arg:
    tauFile: The directory path of file of tau

    Regurn:
    tau: Matrix of Reynolds stress (sysmetric tensor)
    """
    resMid = extract_tensor(tauFile)

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


def extract_symm_tensor(tensorFile):
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


def extract_tensor(tensorFile):
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


# file header/footer
def extract_foam_pattern(tensorFile):
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
        internalField            # match internalField
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


def foam_header(name, location, foamclass, version, format='ascii',
                website='www.OpenFOAM.org'):
    header = '/*--------------------------------*- C++ -*---------------' + \
        '-------------------*\\\n'
    header += '| =========                 |                            ' + \
        '                     |\n'
    header += '| \\\\      /  F ield         | OpenFOAM: The Open Source' + \
        ' CFD Toolbox           |\n'
    header += '|  \\\\    /   O peration     | Version:  ' + \
        '{}.x                                 |\n'.format(version)
    header += '|   \\\\  /    A nd           | Web:      ' + \
        '{}                      |\n'.format(website)
    header += '|    \\\\/     M anipulation  |                          ' + \
        '                       |\n'
    header += '\\*------------------------------------------------------' + \
        '---------------------*/\n'
    header += 'FoamFile\n{\n'
    header += '    version     {};\n'.format(version)
    header += '    format      {};\n'.format(format)
    header += '    class       {};\n'.format(foamclass)
    header += '    location    "{}";\n'.format(location)
    header += '    object      {};\n'.format(name) + '}\n'
    header += '// * * * * * * * * * * * * * * * * * * * * * * * * * * * ' + \
        '* * * * * * * * * * //\n'
    return header
