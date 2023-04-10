#!/usr/bin/env python3

"""
Executable.
Save a trained neural network so that it can be used as a fixed turbulence model.
"""

import os
import argparse

import numpy as np
import yaml

import neuralnet


## Parse input file
parser = argparse.ArgumentParser(description='Train Neural Network.')
parser.add_argument('input_file', help='Name (path) of input file')
args =  parser.parse_args()
with open(args.input_file, 'r') as f:
    input_dict = yaml.load(f, yaml.SafeLoader)
# architecture
ninputs = input_dict['nscalar_invariants']
noutputs =  input_dict['nbasis_tensors']
nhlayers =  input_dict['nhlayers']
nnodes =  input_dict['nnodes']
# weights
wfile = input_dict['weights_file']
# stats
theta_min_file = input_dict['min_file']
theta_max_file = input_dict['max_file']
# save directory
save_dir = input_dict['save_dir']

## Create neural network
th_min = np.loadtxt(theta_min_file)
th_max = np.loadtxt(theta_max_file)
nn = neuralnet.cpp_build_nn(ninputs, noutputs, nhlayers, nnodes, th_min, th_max)

# load & set weights
import pdb; pdb.set_trace()
w = np.loadtxt(wfile)#.mean(axis=1)
w_shapes = neuralnet.weights_shape(nn.trainable_variables)

w = neuralnet.reshape_weights(w, w_shapes)
neuralnet.cpp_set_weights(nn, w)

## Save
nn.save(save_dir)
