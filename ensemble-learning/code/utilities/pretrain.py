#!/usr/bin/env python3

"""
Executable.
Run pre-training for case with ...
"""

import os
import time
import argparse

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import yaml

import neuralnet as neuralnet
import gradient_descent as gd


NBASISTENSORS = 10
NSCALARINVARIANTS = 5

## PARSE INPUT FILE
# input file
parser = argparse.ArgumentParser(description='Pre-train Neural Network.')
parser.add_argument('input_file', help='Name (path) of input file')
args =  parser.parse_args()
with open(args.input_file, 'r') as f:
    input_dict = yaml.load(f, yaml.SafeLoader)

# training data
gtruth = np.array(input_dict['gtruth'])
ndata = input_dict['ndata']
data_lim = input_dict['data_lim']
amplitude = input_dict['amplitude']

# architecture
nscalar_invariants = input_dict.get('nscalar_invariants', NSCALARINVARIANTS)
nbasis_tensors = input_dict.get('nbasis_tensors', NBASISTENSORS)
nhlayers = input_dict.get('nhlayers', 10)
nnodes = input_dict.get('nnodes', 10)

# optimization
opt_algorithm = input_dict.get('opt_algorithm', 'GradientDescent')
opt_parameters = input_dict.get('opt_parameters', None)
opt_restart = input_dict.get('opt_restart', None)
opt_steps = input_dict.get('opt_steps')


## I/O: LOAD DATA, DIRECTORY STRUCTURE
# create save_dir
save_dir = input_dict.get('save_dir', 'results')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# create inputs & truth
theta = np.linspace(data_lim[0], data_lim[1], ndata)
assert len(gtruth) == nbasis_tensors
if nscalar_invariants == 1:
    # inputs
    training_scalar = np.expand_dims(theta, -1)
    # truth
    truth = np.zeros([ndata, nbasis_tensors])
    for i in range(nbasis_tensors):
        truth[:, i] = amplitude * \
            np.sin(2 * np.pi * training_scalar[:, 0]) + gtruth[i]
elif nscalar_invariants == 2:
    # inputs
    training_scalar = np.zeros([ndata*ndata, nscalar_invariants])
    THETA1, THETA2 = np.meshgrid(theta, theta)
    training_scalar[:, 0] = THETA1.flatten()
    training_scalar[:, 1] = THETA2.flatten()
    # truth
    truth = np.zeros([ndata*ndata, nbasis_tensors])
    for i in range(nbasis_tensors):
        TRUTH = 0.5 * amplitude * \
            (np.sin(2 * np.pi * THETA1) +
             np.sin(2 * np.pi * THETA2)) + gtruth[i]
        truth[:, i] = TRUTH.flatten()
elif nscalar_invariants == 3:
    # inputs
    training_scalar = np.zeros([ndata*ndata*ndata, nscalar_invariants])
    THETA1, THETA2, THETA3 = np.meshgrid(theta, theta, theta)
    training_scalar[:, 0] = THETA1.flatten()
    training_scalar[:, 1] = THETA2.flatten()
    training_scalar[:, 2] = THETA3.flatten()
    # truth
    truth = np.zeros([ndata*ndata*ndata, nbasis_tensors])
    for i in range(nbasis_tensors):
        TRUTH = 0.5 * amplitude * \
            (np.sin(2 * np.pi * THETA1) +
            np.sin(2 * np.pi * THETA2) +
             np.sin(2 * np.pi * THETA3)) + gtruth[i]
        truth[:, i] = TRUTH.flatten()
elif nscalar_invariants == 4:
    training_scalar = np.zeros([ndata*ndata*ndata*ndata, nscalar_invariants])
    THETA1, THETA2, THETA3, THETA4 = np.meshgrid(theta, theta, theta, theta)
    training_scalar[:, 0] = THETA1.flatten()
    training_scalar[:, 1] = THETA2.flatten()
    training_scalar[:, 2] = THETA3.flatten()
    training_scalar[:, 3] = THETA4.flatten()
    # truth
    truth = np.zeros([ndata*ndata*ndata*ndata, nbasis_tensors])
    for i in range(nbasis_tensors):
        TRUTH = 0.5 * amplitude * \
            (np.sin(2 * np.pi * THETA1) +
             np.sin(2 * np.pi * THETA2) +
             np.sin(2 * np.pi * THETA3) +
             np.sin(2 * np.pi * THETA4)) + gtruth[i]
        truth[:, i] = TRUTH.flatten()
elif nscalar_invariants == 5:
    training_scalar = np.zeros([ndata*ndata*ndata*ndata*ndata, nscalar_invariants])
    THETA1, THETA2, THETA3, THETA4, THETA5 = np.meshgrid(theta, theta, theta, theta, theta)
    training_scalar[:, 0] = THETA1.flatten()
    training_scalar[:, 1] = THETA2.flatten()
    training_scalar[:, 2] = THETA3.flatten()
    training_scalar[:, 3] = THETA4.flatten()
    training_scalar[:, 4] = THETA5.flatten()
    # truth
    truth = np.zeros([ndata*ndata*ndata*ndata*ndata, nbasis_tensors])
    for i in range(nbasis_tensors):
        TRUTH = 0.5 * amplitude * \
            (np.sin(2 * np.pi * THETA1) +
             np.sin(2 * np.pi * THETA2) +
             np.sin(2 * np.pi * THETA3) +
             np.sin(2 * np.pi * THETA4) +
             np.sin(2 * np.pi * THETA5)) + gtruth[i]
        truth[:, i] = TRUTH.flatten()
elif nscalar_invariants == 6:
    training_scalar = np.zeros([ndata*ndata*ndata*ndata*ndata*ndata, nscalar_invariants])
    THETA1, THETA2, THETA3, THETA4, THETA5, THETA6 = np.meshgrid(theta, theta, theta, theta, theta, theta)
    training_scalar[:, 0] = THETA1.flatten()
    training_scalar[:, 1] = THETA2.flatten()
    training_scalar[:, 2] = THETA3.flatten()
    training_scalar[:, 3] = THETA4.flatten()
    training_scalar[:, 4] = THETA5.flatten()
    training_scalar[:, 5] = THETA6.flatten()
    # truth
    truth = np.zeros([ndata*ndata*ndata*ndata*ndata*ndata, nbasis_tensors])
    for i in range(nbasis_tensors):
        TRUTH = 0.5 * amplitude * \
            (np.sin(2 * np.pi * THETA1) +
            np.sin(2 * np.pi * THETA2) +
            np.sin(2 * np.pi * THETA3) +
            np.sin(2 * np.pi * THETA4) +
            np.sin(2 * np.pi * THETA5) +
             np.sin(2 * np.pi * THETA6)) + gtruth[i]
        truth[:, i] = TRUTH.flatten()
else:
    raise NotImplementedError()
np.savetxt(os.path.join(save_dir, 'theta_input'), training_scalar)
gtruth = truth
np.savetxt(os.path.join(save_dir, 'g_truth'), gtruth)

## Create Neural Network
input_shape = training_scalar.shape
nn = neuralnet.NN(nscalar_invariants, nbasis_tensors, nhlayers, nnodes)

# initial weights
w_init = np.array([])
for iw in nn.trainable_variables:
    w_init = np.concatenate([w_init, iw.numpy().flatten()])
w_shapes = neuralnet.weights_shape(nn.trainable_variables)

# print NN summary
print('\n' + '#'*80 + '\nCreated NN:' +
    f'\n  Field dimension: {ndata}' +
    f'\n  Number of scalar invariants: {nscalar_invariants}' +
    f'\n  Number of basis tensors: {nbasis_tensors}' +
    f'\n  Number of trainable parameters: {nn.count_params()}' +
    '\n' + '#'*80)


## Cost function
if opt_restart is None:
    iter = 0
else:
    iter = opt_restart

def cost(w):
    global iter
    print(f'\nEvaluating Cost Function: {iter}')

    # set weights
    w = neuralnet.reshape_weights(w, w_shapes)
    nn.set_weights(w)

    # evaluate NN: cost and gradient
    with tf.GradientTape() as tape:
        tape.watch(nn.trainable_variables)
        g = nn(training_scalar)
        J = tf.norm(g-gtruth)
    dJdw = tape.gradient(J, nn.trainable_variables)
    # flatten
    dJdw_flat = np.empty([1, nn.count_params()])
    i = 0
    for idJdw in dJdw:
        idJdw = idJdw.numpy().flatten()
        di = len(idJdw)
        dJdw_flat[0, i:i+di] = idJdw
        i += di

    # save
    np.savetxt(os.path.join(save_dir, f'g.{iter}'), g.numpy())
    np.savetxt(os.path.join(save_dir, f'J.{iter}'), [J.numpy()])

    # summary
    print(f'Cost: {J}')
    iter += 1
    return J.numpy(), np.squeeze(dJdw_flat)



## Optimization
tstart_opt = time.time()
optimization = getattr(gd, opt_algorithm)
optimization = optimization(objective=cost, restart=opt_restart, x=w_init,
    parameters=opt_parameters, save=True, save_directory=save_dir)
optimization.optimize(opt_steps)
tend_opt = time.time()

# summary
J0 = np.loadtxt(os.path.join(save_dir, 'J.0'))
J = np.loadtxt(os.path.join(save_dir, f'J.{iter-1}'))
print(f'\n\nInitial Cost: {J0}\nFinal Cost: {J}')
print(f'Time: {int(tend_opt-tstart_opt):d} s')
