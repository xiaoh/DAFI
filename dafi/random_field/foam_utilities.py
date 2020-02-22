# Copyright 2020 Virginia Polytechnic Institute and State University.
""" OpenFOAM file manipulation. """

# standard library imports
import numpy as np
import os
import re
import tempfile
import subprocess


# global variables
NSCALAR = 1
NVECTOR = 3
NSYMMTENSOR = 6
NTENSOR = 9


# get mesh properties by running OpenFOAM shell utilities
def get_number_cells(foam_case='.'):
    bash_command = "checkMesh -case " + foam_case + \
        " -time '0' | grep '    cells:' > tmp.ncells"
    cells = subprocess.check_output(bash_command, shell=True)
    cells = cells.decode("utf-8").replace('\n', '').split(':')[1].strip()
    return int(cells)


def get_cell_coordinates(foam_case='.', timedir='0', keep_file=False):
    bash_command = "simpleFoam -postProcess -func writeCellCentres " + \
        "-case " + foam_case + f" -time '{timedir}' " + "&> /dev/null"
    subprocess.call(bash_command, shell=True)
    os.remove(os.path.join(foam_case, timedir, 'Cx'))
    os.remove(os.path.join(foam_case, timedir, 'Cy'))
    os.remove(os.path.join(foam_case, timedir, 'Cz'))
    file = os.path.join(foam_case, timedir, 'C')
    coords = read_cell_coordinates(file, group='internalField')
    if not keep_file:
        os.remove(file)
    return coords


def get_cell_volumes(foam_case='.', timedir='0', keep_file=False):
    bash_command = "simpleFoam -postProcess -func writeCellVolumes " + \
        "-case " + foam_case + f" -time '{timedir}' " + "&> /dev/null"
    subprocess.call(bash_command, shell=True)
    file = os.path.join(foam_case, timedir, 'V')
    vol = read_cell_volumes(file)
    if not keep_file:
        os.remove(file)
    return vol


# read fields
def read_field(file, ndim, group='internalField'):
    # read file
    with open(file, 'r') as f:
        content = f.read()
    # keep file portion after specified group
    content = content.partition(group)[2]
    # data structure
    whole_number = r"([+-]?[\d]+)"
    decimal = r"([\.][\d]*)"
    exponential = r"([Ee][+-]?[\d]+)"
    floatn = f"{whole_number}{{1}}{decimal}?{exponential}?"
    if ndim==1:
        data_structure = f"({floatn}\\n)+"
    else:
        data_structure = r'(\(' + f"({floatn}" + r"(\ ))" + \
                         f"{{{ndim-1}}}{floatn}" + r"\)\n)+"
    # extract data
    pattern = r'\(\n' + data_structure + r'\)'
    data_str = re.compile(pattern).search(content).group()
    # convert to numpy array
    data_str = data_str.replace('(', '').replace(')', '').replace('\n', ' ')
    data_str = data_str.strip()
    data = np.fromstring(data_str, dtype=float, sep=' ')
    if ndim > 1:
        data.reshape([-1, ndim])
    return data


def read_scalar_field(file, group='internalField'):
    return read_field(file, ndim=NSCALAR, group=group)


def read_vector_field(file, group='internalField'):
    return read_field(file, ndim=NVECTOR, group=group)


def read_symmTensor_field(file, group='internalField'):
    return read_field(file, ndim=NSYMMTENSOR, group=group)


def read_tensor_field(file, group='internalField'):
    return read_field(file, ndim=NTENSOR, group=group)


def read_cell_coordinates(file='C', group='internalField'):
    return read_field(file, ndim=NVECTOR, group=group)


def read_cell_volumes(file='V'):
    return read_field(file, ndim=NSCALAR, group='internalField')


# TODO: Read properties (version, name, loc, boundary names and types, etc.)
# i.e. everything needed to then write it again using write_field.
# Use: to modify the file by rewritting with modified field values.
def read_field_info(file):
    # foam_version
    # object_name
    # foam_class
    # location (optional)
    # dimension
    # internalField: uniform/nonuniform
    # boundaries: type and value(optional): value can be uniform/nonuniform scalar/(multi)
    raise NotImplementedError

# write fields
def write_p(foam_version, internal_field, boundaries, website, location=None, file=None):
    name = 'p'
    ofclass = 'scalar'
    dimension = 'p'
    write_field(foam_version, name, ofclass, location, dimension,
            internal_field, boundaries, website, file=None)



def get_info(fieldname):
    def get_foam_class(fieldname):
        scalarlist = ['p', 'k', 'epsilon', 'omega', 'nut', 'Cx', 'Cy',
            'Cz', 'V']
        if fieldname in scalarlist:
            foam_class = 'scalar'
        elif fieldname in ['U', 'C']:
            foam_class = 'vector'
        elif fieldname == 'Tau':
            foam_class = 'symmTensor'
        elif fieldname == 'grad(U)':
            foam_class = 'tensor'
        return foam_class

    foam_class = get_foam_class(fieldname)


def write_field(name, ofclass, dimension, internal_field,
        boundaries, foam_version, website, location=None, file=None):
    """ e.g.
    write_field(foam_version='1912', name='p', ofclass='scalar', location='0',
        dimension='p', internal_field={'uniform': True, 'value':0},
        boundaries=[{'name': 'top', 'type': 'zeroGradient',
                       'value': {'uniform':True, 'data': 0}},
                    {'name': 'bottom', 'type': 'zeroGradient'},
                    {'name': 'left', 'type': 'cyclic',
                       'value': {'uniform': False, 'data': [1, 2, 3, 4, 5]}},
                    {'name': 'right', 'type': 'cyclic']):
    """
    def _foam_sep():
        return '\n// ' + '* '*37 + '//'

    def _foam_header_logo(version):

        def header_line(str1, str2):
            return f'\n| {str1:<26}| {str2:<48}|'

        # def get_website(website):
        #     if website == 'org':
        #         website = 'www.openfoam.org'
        #     elif website == 'com':
        #         website = 'www.openfoam.com'
        #     return website
        #
        # website = get_website(website)

        header_start = '/*' + '-'*32 + '*- C++ -*' + '-'*34 + '*\\'
        header_end = '\n\\*' +'-'*75 + '*/'
        logo = ['=========',
                r'\\      /  F ield',
                r' \\    /   O peration',
                r'  \\  /    A nd',
                r'   \\/     M anipulation',
                ]
        info = ['',
                'OpenFOAM: The Open Source CFD Toolbox',
                f'Version:  {version}',
                f'Website:  {website}',
                '',
                ]
        # create header
        header = header_start
        for l,i in zip(logo, info):
            header += header_line(l,i)
        header += header_end
        return header

    def _foam_header_info(object, foamclass, location=None):
        def header_line(str1, str2):
            return f'\n    {str1:<12}{str2};'

        VERSION = '2.0'
        FORMAT = 'ascii'
        foamclass = 'vol' + foamclass[0].capitalize() + foamclass[1:] + 'Field'
        # create header
        header = 'FoamFile\n{'
        header += header_line('version', VERSION)
        header += header_line('format', FORMAT)
        header += header_line('class', foamclass)
        if location is not None:
            header += header_line('location', f'"{location}"')
        header += header_line('object', object)
        header += '\n}'
        return header

    def _foam_dimensions(fieldname_or_foamdim):
        #  1 - Mass (kg)
        #  2 - Length (m)
        #  3 - Time (s)
        #  4 - Temperature (K)
        #  5 - Quantity (mol)
        #  6 - Current (A)
        #  7 - Luminous intensity (cd)
        err_msg = 'Input must be string or list of ints length 7.'
        if isinstance(fieldname_or_foamdim, str):
            if fieldname_or_foamdim == 'U':
                dimensions = '[ 0 1 -1 0 0 0 0]'
            elif fieldname_or_foamdim in ['p', 'k', 'Tau']:
                dimensions = '[0 2 -2 0 0 0 0]'
            elif fieldname_or_foamdim == 'phi':
                dimensions = '[0 3 -1 0 0 0 0]'
            elif fieldname_or_foamdim == 'epsilon':
                dimensions = '[0 2 -3 0 0 0 0]'
            elif fieldname_or_foamdim in ['omega', 'grad(U)']:
                dimensions = '[0 0 -1 0 0 0 0]'
            elif fieldname_or_foamdim == 'nut':
                dimensions = '[0 2 -1 0 0 0 0]'
            elif fieldname_or_foamdim in ['C', 'Cx', 'Cy', 'Cz']:
                dimensions = '[0 1 0 0 0 0 0]'
            elif fieldname_or_foamdim == 'V':
                dimensions = '[0 3 0 0 0 0 0]'
            else:
                raise ValueError('Unkown field "{fieldname_or_foamdim}"')
        elif isinstance(fieldname_or_foamdim, list):
            if len(fieldname_or_foamdim) == 7:
                dimensions = fieldname_or_foamdim
            elif len(fieldname_or_foamdim) == 3:
                dimensions = fieldname_or_foamdim.append([0, 0, 0, 0])
            else:
                raise ValueError(err_msg)
        else:
            raise ValueError(err_msg)
        return f'dimensions      {dimensions};'

    def _list_to_foamstr(inlist):
        outstr = ''
        for l in inlist:
            outstr += f'{l} '
        return outstr.strip()

    def _foam_field(uniform, value, foamclass=None):
        def _foam_nonuniform(data):
            field = f'{len(data)}\n('
            if data.ndim == 1:
                for d in data:
                    field += f'\n{d}'
            elif data.ndim == 2:
                for d in data:
                    field += f'\n({_list_to_foamstr(d)})'
            else:
                raise ValueError('"data" cannot have more than 2 dimensions.')
            field += '\n)'
            return field

        if uniform:
            # scalar type
            if np.issubdtype(type(value), np.number) :
                data = str(value)
            # list type
            elif isinstance(value, (list, np.ndarray)):
                if isinstance(value, np.ndarray):
                    value = np.squeeze(value)
                    if value.ndim != 1:
                        raise ValueError('Uniform data should have one dimension.')
                data = f'({_list_to_foamstr(value)})'
            field = f'uniform {data}'
        else:
            if foamclass is None:
                raise ValueError('foamclass required for nonuniform data.')
            value = np.squeeze(value)
            field = f'nonuniform List<{foamclass}>'
            field += '\n' + _foam_nonuniform(value)
        return field

    # create string
    file_str = _foam_header_logo(foam_version)
    file_str += '\n' + _foam_header_info(name, ofclass, location)
    file_str += '\n' + _foam_sep() + '\n'*2 + _foam_dimensions(dimension)
    file_str += '\n'*2 + 'internalField   '
    file_str += _foam_field(
        internal_field['uniform'], internal_field['value'], ofclass) + ';'
    file_str += '\n\nboundaryField\n{'
    for bc in boundaries:
        file_str += '\n' + ' '*4 + bc["name"] + '\n' + ' '*4 + '{'
        file_str += '\n' + ' '*8 + 'type' + ' '*12 + bc["type"] + ';'
        write_value = False
        if 'value' in bc:
            if bc['value'] is not None:
                write_value = True
        if write_value:
            data = _foam_field(
                bc['value']['uniform'], bc['value']['data'], ofclass)
            file_str += '\n' + ' '*8 + 'value' + ' '*11 + data + ';'
        file_str += '\n' + ' '*4 + '}'
    file_str += '\n}\n' + _foam_sep()

    # write to file
    if file is None:
        file = name
    with open(file, 'w') as f:
        f.write(file_str)
    return os.path.abspath(file), file_str
