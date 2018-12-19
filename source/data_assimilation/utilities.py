# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Collection of I/O functions and other utilities. """


# standard library imports
import tempfile
import os
import shutil
import sys


def replace(filename, pattern, subst):
    """ Replace all instances of a pattern in a file with a new pattern.

    Parameters
    ----------
    file : str
        The file to modify.
    pattern : str
        The old character string which you want replace.
    subst : str
        The new character string.
    """
    # create temp file
    fh, abs_path = tempfile.mkstemp()
    new_file = open(abs_path, 'w')
    old_file = open(filename)
    for line in old_file:
        new_file.write(line.replace(pattern, subst))
    # close temp file
    new_file.close()
    os.close(fh)
    old_file.close()
    # remove original file
    os.remove(filename)
    # move new file
    shutil.move(abs_path, filename)


def read_input_data(input_file):
    """ Parse an input file into a dictionary.

    The input file should contain two entries per line separated by any
    number of white space characters or ':' or '='. Comments are
    indicated with '#'. Each non-empty, non-comment, line is stored in
    the output dictionary with the first entry as the key (input name)
    and the second as the argument(input value). Both keys and
    arguments are strings.

    Parameters
    ----------
    input_file : str
        Name of the input file.

    Returns
    -------
    param_dict : dict
        Dictionary containing input names and values.
    """
    param_dict = {}
    try:
        file = open(input_file, "r")
    except IOError:
        print('Cannot open file: {}'.format(input_file))
        sys.exit(1)
    else:
        for line in file:
            # remove comments and trailing/leading whitespace
            line = line.split('#', 1)[0]
            line = line.strip()
            # ignore empty or comment lines
            if not line:
                continue
            # allow ":" and "=" in input file
            line = line.replace(':', ' ').replace('=', ' ')
            # parse into dictionary
            param, value = line.split(None, 1)
            param_dict[param] = value
        file.close()
    return param_dict


def extract_list_from_string(string, sep=',', type_func=str):
    """ Extract a list from a string.

    Parameters
    ----------
        string: str
            String containing the list.
        sep: str
            The separator in the string.
        type_func: Function to convert the string to desired type.

    Returns
    -------
        value_list : list
            List of sub-strings extracted from input string.

    Example
    -------
        >>> extract_list_from_string('1; 2;3.6 ; 10', ';', int)
        [1, 2, 3, 10]
    """
    value_list = string.split(sep)
    value_list = [type_func(pn.strip()) for pn in value_list]
    return value_list


def str2bool(input_str):
    """ Convert string to bool. """
    return input_str in ("True", "1")


def create_dir(directory):
    """ Create a directory if does not exist. """
    if not os.path.exists(directory):
        os.makedirs(directory)
