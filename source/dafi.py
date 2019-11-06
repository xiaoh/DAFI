#!/usr/bin/env python3
# Copyright 2018 Virginia Polytechnic Institute and State University.

""" Inverse modeling main executable.

Example
-------
    >>> dafi.py <input_file>

Note
----
Required inputs:
    * **dyn_model** (``str``) -
      Name of dynamic model module/package.
    * **dyn_model_input** (``str``) -
      Path to input file for the dynamic model.
    * **da_filter** (``str``) -
      Name of filter from dainv.da_filtering module.
    * **max_da_iteration** (``int``, ``1``) -
      Maximum number of DA iterations at each timestep.
    * **nsamples** (``int``) -
      Number of samples for ensemble.

Note
----
Optional inputs:
    * **t_end** (``float``, ``1``) -
      Final time step.
    * **da_t_interval** (``float``, ``1``) -
      Time interval to perform data assimilation.
    * **plot_flag** (``bool``, ``False``) -
      Call the filter's plot method.
    * **save_flag** (``bool``, ``True``) -
      Call the filter's save method.
    * **rand_seed_flag** (``bool``, ``False``) -
      Use fixed random seed, for debugging.
    * **rand_seed** (``float``, ``1``) -
      Seed for numpy.random.

Note
----
Other inputs:
    * As required by the chosen filter.
"""

# standard library imports
import sys
import os
import time
import importlib
import subprocess

# third party imports
import numpy as np

# local imports
import dafi.utilities as util
# user-specified inverse model filter imported later with importlib
# user-specified dynamic model imported later with importlib


def _print_usage():
    """ Print usage of the program. """
    print("Usage: dafi.py <input_file>")


def _get_input():
    """ Get the input file. """
    try:
        input_file = sys.argv[1]
    except IndexError as e:
        print(e)
        _print_usage()
        sys.exit(1)
    return input_file


def get_code_version():
    """ Save the Git version of DAFI. """
    git_dir = os.path.dirname(os.path.realpath(__file__))
    cwd = os.getcwd()
    file = os.path.join(cwd, '.dafi_ver')
    bash_command = "cd {}; ".format(git_dir)
    bash_command += "git rev-parse HEAD > {} 2> /dev/null; ". format(file)
    bash_command += "cd {}".format(cwd)
    subprocess.call(bash_command, shell=True)


def main():
    """ Run DAFI. """
    # required inputs
    input_file_da = _get_input()
    get_code_version()
    param_dict = util.read_input_data(input_file_da)
    dyn_model = param_dict['dyn_model']
    input_file_dm = param_dict['dyn_model_input']
    da_filter = param_dict['da_filter']
    nsamples = int(param_dict['nsamples'])
    max_da_iteration = int(param_dict['max_da_iteration'])
    # optional inputs - set default if not specified.
    # dynamic (time progressing) problems
    try:
        t_end = float(param_dict['t_end'])
    except:
        t_end = 1.0
    try:
        da_t_interval = float(param_dict['da_t_interval'])
    except:
        da_t_interval = 1.0
    # output flags
    try:
        plot_flag = util.str2bool(param_dict['plot_flag'])
    except:
        plot_flag = False
    try:
        save_flag = util.str2bool(param_dict['save_flag'])
    except:
        save_flag = True
    # debug flags
    try:
        rand_seed_flag = util.str2bool(param_dict['rand_seed_flag'])
    except:
        rand_seed_flag = False
    if rand_seed_flag:
        try:
            rand_seed = int(param_dict['rand_seed'])
        except:
            rand_seed = 1.0
    # remove all the inputs meant for this file, dafi.py.
    # what is left are inputs meant for the specific DA filter method used.
    main_inputs = [
        'dyn_model', 'dyn_model_input', 'da_filter', 't_end', 'da_t_interval',
        'nsamples', 'max_da_iteration', 'plot_flag',
        'save_flag', 'rand_seed_flag', 'rand_seed']
    for inp in main_inputs:
        try:
            _ = param_dict.pop(inp)
        except:
            pass

    # import and initialize forward and inverse models
    # random seed: do this before importing local modules that use np.random
    if rand_seed_flag:
        np.random.seed(rand_seed)
    # dynamic model
    DynModel = getattr(
        importlib.import_module('dafi.dynamic_models.' + dyn_model), 'Solver')
    dynamic_model = DynModel(nsamples, da_t_interval, t_end, max_da_iteration,
                             input_file_dm)
    # inverse model
    InvFilter = getattr(
        importlib.import_module('dafi.filters'), da_filter)
    inverse_model = InvFilter(nsamples, da_t_interval, t_end, max_da_iteration,
                              dynamic_model, param_dict)

    # solve the inverse problem
    print("Solving the inverse problem:" +
          "\n  Model:  {}".format(dynamic_model.name) +
          "\n  Filter: {}".format(inverse_model.name))
    start_time = time.time()
    inverse_model.solve()
    print("\nTime spent on solver: {}s".format(time.time() - start_time))

    # plot and save
    if plot_flag:
        print('\nPlotting ...')
        inverse_model.plot()
    if save_flag:
        print('\nSaving ...')
        inverse_model.save()
    inverse_model.clean()
    print('\nDone.')


if __name__ == "__main__":
    main()
