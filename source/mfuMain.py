#!/usr/bin/env python2
# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Inverse modeling main executable.

Usage:
    >>> mfu_main.py <input_file>

Required inputs:
    dyn_model: (string)
        name of dynamic model module/package
    dyn_model_input: (string)
        path to input file for the dynamic model
    da_filter: (string)
        name of filter from dainv.da_filtering module
    t_end: (float)
        final time step
    da_interval: (float)
        time interval to perform data assimilation
    nsamples: (int)
        number of samples for ensemble

Optional inputs:
    report_flag: (bool, False)
        call the filter's report method
    plot_flag: (bool, False)
        call the filter's plot method
    save_flag: (bool, False)
        call the filter's save method
    rand_seed_flag: (bool, False)
        use fixed random seed, for debugging
    rand_seed: (float, 1)
        seed for numpy.random

Other inputs:  as required by the chosen filter.
"""

# standard library imports
import sys
import os
import time
import ast
import importlib

# third party imports
import numpy as np

# local imports
from dainv.utilities import read_input_data
# user-specified inverse model filter imported later with importlib
# user-specified dynamic model imported later with importlib

# parse input file

def _print_usage():
    # Print usage of the program.
    print("Usage: mfu_main.py <input_file>")

def _parse_input():
    # Parse the input file. #
    try:
        input_file = sys.argv[1]
    except IndexError, e:
        print(e)
        _print_usage()
        sys.exit(1)
    return input_file

# required inputs
input_file_da = _parse_input()
param_dict = read_input_data(input_file_da)
dyn_model = param_dict['dyn_model']
input_file_dm = param_dict['dyn_model_input']
da_filter = param_dict['da_filter']
t_end = float(param_dict['t_end'])
da_interval = float(param_dict['da_interval'])
nsamples = int(param_dict['nsamples'])
# optional inputs - set default if missing.
try: report_flag = ast.literal_eval(param_dict['report_flag'])
except: report_flag = False
try: plot_flag = ast.literal_eval(param_dict['plot_flag'])
except: plot_flag = False
try: save_flag = ast.literal_eval(param_dict['save_flag'])
except: save_flag = False
try rand_seed_flag = ast.literal_eval(param_dict['rand_seed_flag'])
except rand_seed_flag = False
if rand_seed_flag:
    try rand_seed = float(param_dict['rand_seed'])
    except rand_seed = 1.0
# remove all the inputs meant for this file, mfu_main.py.
# what is left are inputs meant for the specific DA filter method used.
main_inputs = [
    'dyn_model', 'dyn_model_input', 'da_filter', 't_end', 'da_interval',
    'nsamples', 'report_flag', 'plot_flag', 'save_flag', 'rand_seed_fag',
    'rand_seed']
for inp in main_inputs:
    try: param_dict.pop(inp);
    except: pass

# import and initialize forward and inverse models
# random seed: do this before importing local modules that use np.random
if rand_seed_flag:
    np.random.seed(rand_seed)
# forward model
DynModel = getattr(
    importlib.import_module('dyn_models.' + dyn_model), 'Solver')
forward_model = DynModel(nsamples, da_interval, t_end, input_file_dm)
# inverse model
InvFilter = getattr(importlib.import_module('dainv.da_filtering'), da_filter)
inverse_model = InvFilter(
    nsamples, da_interval, t_end, forward_model, param_dict)

# solve the inverse problem
print(
    "Solving the inverse problem:\n  Model:  {}".format(forward_model.name) +
    "\n  Filter: {}\n".format(inverse_model.name) )
start_time = time.time()
inverse_model.solve()
print("Time spent on solver: {}s".format(time.time() - start_time))

# report and plot
if report_flag:
    inverse_model.report()
if plot_flag:
    inverse_model.plot()
if save_flag:
    inverse_model.save()
inverse_model.clean()
