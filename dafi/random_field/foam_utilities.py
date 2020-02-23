# Copyright 2020 Virginia Polytechnic Institute and State University.
""" OpenFOAM file manipulation. """

# standard library imports
import numpy as np
import os
import re
import tempfile
import subprocess


# global variables
NDIM = {'scalar': 1,
        'vector': 3,
        'symmTensor': 6,
        'tensor': 9}


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
        data = data.reshape([-1, ndim])
    return data


def read_scalar_field(file, group='internalField'):
    return read_field(file, NDIM['scalar'], group=group)


def read_vector_field(file, group='internalField'):
    return read_field(file, NDIM['vector'], group=group)


def read_symmTensor_field(file, group='internalField'):
    return read_field(file, NDIM['symmTensor'], group=group)


def read_tensor_field(file, group='internalField'):
    return read_field(file, NDIM['tensor'], group=group)


def read_cell_coordinates(file='C', group='internalField'):
    return read_vector_field(file, group=group)


def read_cell_volumes(file='V'):
    return read_scalar_field(file, group='internalField')



# read entire field file
def read_field_file(file):
    with open(file, 'r') as f:
        content = f.read()
    info = {}
    # read logo
    def _read_logo(pat):
        pattern = pat + r":\s+\S+"
        data_str = re.compile(pattern).search(content).group()
        return data_str.split(':')[1].strip()

    info['foam_version'] = _read_logo('Version')
    info['website'] = _read_logo('Website')

    # read header
    def _read_header(pat):
        pattern = pat + r"\s+\S+;"
        data_str = re.compile(pattern).search(content).group()
        return data_str.split(pat)[1][:-1].strip()

    foam_class = _read_header('class').split('Field')[0].split('vol')[1]
    info['foam_class'] = foam_class[0].lower() + foam_class[1:]
    info['name'] = _read_header('object')
    try:
        info['location'] = _read_header('location')
    except AttributeError:
        info['location'] = None

    # dimension
    pattern = r"dimensions\s+.+"
    data_str = re.compile(pattern).search(content).group()
    info['dimensions'] = data_str.split('dimensions')[1][:-1].strip()

    # internalField: uniform/nonuniform
    internal = {}
    pattern = r'internalField\s+\S+\s+.+'
    data_str = re.compile(pattern).search(content).group()
    if data_str.split()[1] == 'uniform':
        internal['uniform'] = True
        internal['value'] = data_str.split('uniform')[1].strip()[:-1]
    else:
        internal['uniform'] = False
        internal['value'] = read_field(file, NDIM[info['foam_class']])
    info['internal_field'] = internal

    # boundaries: type and value(optional): value can be uniform/nonuniform scalar/(multi)
    boundaries = []
    bcontent = content.split('boundaryField')[1].strip()[1:].strip()
    pattern = r'\w+' + r'[\s\n]*' + r'\{' + r'[\w\s\n\(\);\.\<\>\-+]+' + r'\}'
    boundaries_raw = re.compile(pattern).findall(bcontent)
    for bc in boundaries_raw:
        ibc = {}
        # name
        pattern = r'[\w\s\n]+' + r'\{'
        name = re.compile(pattern).search(bc).group()
        name = name.replace('{','').strip()
        ibc['name'] = name
        # type
        pattern = r'type\s+\w+;'
        type = re.compile(pattern).search(bc).group()
        type = type.split('type')[1].replace(';','').strip()
        ibc['type'] = type
        # value
        if 'value' in bc:
            value = {}
            v = bc.split('value')[1]
            if v.split()[0]=='uniform':
                value['uniform'] = True
                v = v.split('uniform')[1]
                value['data'] = v.replace('}','').replace(';','').strip()
            else:
                value['uniform'] = False
                value['data'] = read_field(
                    file, NDIM[info['foam_class']], group=ibc['name'])
        else:
            value = None
        ibc['value'] = value
        boundaries.append(ibc)
    info['boundaries'] = boundaries
    return info


# write fields
def write_field_file(name, foam_class, dimensions, internal_field,
        boundaries, foam_version, website, location=None, file=None):
    """ e.g.
    write_field(foam_version='1912', name='p', foam_class='scalar', location='0',
        dimensions='p', internal_field={'uniform': True, 'value':0},
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

    def _foam_field(uniform, value, foamclass=None):
        def _list_to_foamstr(inlist):
            outstr = ''
            for l in inlist:
                outstr += f'{l} '
            return outstr.strip()

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
            # list type
            if isinstance(value, (list, np.ndarray)):
                if isinstance(value, np.ndarray):
                    value = np.squeeze(value)
                    if value.ndim != 1:
                        err_msg = 'Uniform data should have one dimension.'
                        raise ValueError(err_msg)
                value = f'({_list_to_foamstr(value)})'
            field = f'uniform {value}'
        else:
            if foamclass is None:
                raise ValueError('foamclass required for nonuniform data.')
            value = np.squeeze(value)
            field = f'nonuniform List<{foamclass}>'
            field += '\n' + _foam_nonuniform(value)
        return field

    def _foam_dimensions(dimensions):
        if isinstance(dimensions, list):
            if len(dimensions) == 3:
                dimensions = dimensions.append([0, 0, 0, 0])
            elif len(dimensions) != 7:
                raise ValueError('Dimensions must be length 3 or 7.')
            str = ''
            for idim in dimensions:
                str += f'{idim} '
            return f'[{str.strip()}]'

    # create string
    file_str = _foam_header_logo(foam_version)
    file_str += '\n' + _foam_header_info(name, foam_class, location)
    file_str += '\n' + _foam_sep()
    file_str += '\n'*2 + f'dimensions      {dimensions}'
    file_str += '\n'*2 + 'internalField   '
    file_str += _foam_field(
        internal_field['uniform'], internal_field['value'], foam_class) + ';'
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
                bc['value']['uniform'], bc['value']['data'], foam_class)
            file_str += '\n' + ' '*8 + 'value' + ' '*11 + data + ';'
        file_str += '\n' + ' '*4 + '}'
    file_str += '\n}\n' + _foam_sep()

    # write to file
    if file is None:
        file = name
    with open(file, 'w') as f:
        f.write(file_str)
    return os.path.abspath(file), file_str


def write(version, fieldname, internal, boundaries, location=None, file=None):
    def field_info(fieldname):
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

        def get_dimensions(fieldname):
            #  1 - Mass (kg)
            #  2 - Length (m)
            #  3 - Time (s)
            #  4 - Temperature (K)
            #  5 - Quantity (mol)
            #  6 - Current (A)
            #  7 - Luminous intensity (cd)
            if fieldname == 'U':
                dimensions = '[ 0 1 -1 0 0 0 0]'
            elif fieldname in ['p', 'k', 'Tau']:
                dimensions = '[0 2 -2 0 0 0 0]'
            elif fieldname == 'phi':
                dimensions = '[0 3 -1 0 0 0 0]'
            elif fieldname == 'epsilon':
                dimensions = '[0 2 -3 0 0 0 0]'
            elif fieldname in ['omega', 'grad(U)']:
                dimensions = '[0 0 -1 0 0 0 0]'
            elif fieldname == 'nut':
                dimensions = '[0 2 -1 0 0 0 0]'
            elif fieldname in ['C', 'Cx', 'Cy', 'Cz']:
                dimensions = '[0 1 0 0 0 0 0]'
            elif fieldname == 'V':
                dimensions = '[0 3 0 0 0 0 0]'
            return dimensions

        field = {'name': fieldname,
                 'class': get_foam_class(fieldname),
                 'dimensions': get_dimensions(fieldname)}
        return field

    def version_info(version):
        # string ('7' or '1912')
        website = 'www.openfoam.'
        if len(version)==1:
            version += '.x'
            website += 'org'
        elif len(version)==4:
            version = 'v' + version
            website += 'com'
        foam = {'version': version,
                'website': website}
        return foam

    foam = version_info(version)
    field = field_info(fieldname)

    file_path, file_str = write_field_file(
        foam_version=foam['version'],
        website=foam['website'],
        name=field['name'],
        foam_class=field['class'],
        dimensions=field['dimensions'],
        internal_field=internal,
        boundaries=boundaries,
        location=location,
        file=file)
    return file_path, file_str


def write_p(version, internal, boundaries, location=None, file=None):
    write(version, 'p', internal, boundaries, location, file)


def write_U(version, internal, boundaries, location=None, file=None):
    write(version, 'U', internal, boundaries, location, file)


def write_Tau(version, internal, boundaries, location=None, file=None):
    write(version, 'Tau', internal, boundaries, location, file)


def write_nut(version, internal, boundaries, location=None, file=None):
    write(version, 'nut', internal, boundaries, location, file)


def write_k(version, internal, boundaries, location=None, file=None):
    write(version, 'k', internal, boundaries, location, file)


def write_epsilon(version, internal, boundaries, location=None, file=None):
    write(version, 'epsilon', internal, boundaries, location, file)


def write_omega(version, internal, boundaries, location=None, file=None):
    write(version, 'omega', internal, boundaries, location, file)
