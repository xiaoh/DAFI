#!/usr/bin/env python3

"""
Executable.
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

from dafi import random_field as rf
from get_inputs import get_inputs

NBASISTENSORS = 2
NSCALARINVARIANTS = 2
VECTORDIM = 3
TENSORSQRTDIM = 3
TENSORDIM = 9
DEVSYMTENSORDIM = 6
DEVSYMTENSOR_INDEX = [0,1,2,4,5,8]

nscalar_invariants = 2
nbasis_tensors = 2

## PARSE INPUT FILE
# input file
parser = argparse.ArgumentParser(description='Pre-train Neural Network.')
parser.add_argument('input_file', help='Name (path) of input file')
args =  parser.parse_args()
with open(args.input_file, 'r') as f:
    input_dict = yaml.load(f, yaml.SafeLoader)

# training data
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
gradU_0 = rf.foam.read_tensor_field('inputs/case_1p0_komega/10000/grad(U)')
k_0 = rf.foam.read_scalar_field('inputs/case_1p0_komega/10000/k')
omega_0 = rf.foam.read_scalar_field('inputs/case_1p0_komega/10000/omega')
time_scale_0 = 1 / omega_0 / 0.09
input_scalars, input_tensors = get_inputs(gradU_0, time_scale_0)
ncells = len(k_0)
# input_scalars = input_scalars_t.copy()

input_scalars_scale = np.zeros(input_scalars.shape)

gmax = 1
gmin = 0
xmin, xmax = input_scalars[:, 0].min(), input_scalars[:,0].max()
input_scalars_scale[:, 0] = (input_scalars[:, 0] - xmin)/(xmax-xmin) * (gmax - gmin) + gmin
xmin, xmax = input_scalars[:, 1].min(), input_scalars[:,1].max()
input_scalars_scale[:, 1] = (input_scalars[:, 1] - xmin)/(xmax-xmin) * (gmax - gmin) + gmin
training_scalar = input_scalars_scale[:, :2] #.copy()

def Tau2a(Tau, tke):
    a = Tau.copy()
    a[:,0] -= 2./3.*tke
    a[:,3] -= 2./3.*tke
    a[:,5] -= 2./3.*tke
    return a
def a2b(a, tke):
    return a / (2 * tke[:, None])

Ub = 0.028
tau_t = rf.foam.read_symmTensor_field('inputs/data/dns/fine/1/Tau')
k_t = 0.5 * (tau_t[:, 0] + tau_t[:, 3] + tau_t[:, 5]) 
a_t = Tau2a(tau_t, k_t)
b_dns = a2b(a_t, k_t)
g1_prior = -0.09
g2_prior = 0.0

lamda1 = 10 * np.ones(ncells) #
lamda2 = 10 * np.ones(ncells) #

gtruth = np.zeros((ncells, 2))

# get tensor basis T_ij
def get_tensor_basis(gradU, time_scale):
    ncells = len(time_scale)
    T = np.zeros([ncells, DEVSYMTENSORDIM, nbasis_tensors])
    for i, (igradU, it) in enumerate(zip(gradU, time_scale)):
        igradU = igradU.reshape([TENSORSQRTDIM, TENSORSQRTDIM])
        S = it * 0.5*(igradU + igradU.T)
        R = it * 0.5*(igradU - igradU.T)
        T1 = S
        T2 = S @ R - R @ S
        T3 = minus_thirdtrace(S @ S)
        # T4 = minus_thirdtrace(R @ R)
        for j, iT in enumerate([T1, T2]): #, T3]):
            iT = iT.reshape([TENSORDIM])
            T[i, :, j] = iT[DEVSYMTENSOR_INDEX]
    return T

def minus_thirdtrace(x):
    return x - 1./3.*np.trace(x)*np.eye(TENSORSQRTDIM)

T = get_tensor_basis(gradU_0, time_scale_0)
data_index = []
count = 0
for i in range(ncells):  
    b_T = b_dns[i, 0] * T[i, 0, 0] + 2 * b_dns[i, 1] * T[i, 1, 0] + 2 * b_dns[i, 2] * T[i, 2, 0] + b_dns[i, 3] * T[i, 3, 0] + 2 * b_dns[i, 4] * T[i, 4, 0] + b_dns[i, 5] * T[i, 5, 0]
    T2_1 = T[i, 0, 0] * T[i, 0, 0] + T[i, 3, 0] * T[i, 3, 0] + T[i, 5, 0] * T[i, 5, 0] + 2 * T[i, 1, 0] * T[i, 1, 0] + 2 * T[i, 2, 0] * T[i, 2, 0] + 2 * T[i, 4, 0] * T[i, 4, 0]
    gtruth[i, 0] = (b_T + lamda1[i] * g1_prior) / (T2_1 + lamda1[i])
    if gtruth[i, 0] > 0: gtruth[i, 0] = 0

    b_T = b_dns[i, 0] * T[i, 0, 1] + 2 * b_dns[i, 1] * T[i, 1, 1] + 2 * b_dns[i, 2] * T[i, 2, 1] + b_dns[i, 3] * T[i, 3, 1] + 2 * b_dns[i, 4] * T[i, 4, 1] + b_dns[i, 5] * T[i, 5, 1]
    T2_2 = T[i, 0, 1] * T[i, 0, 1] + T[i, 3, 1] * T[i, 3, 1] + T[i, 5, 1] * T[i, 5, 1] + 2 * T[i, 1, 1] * T[i, 1, 1] + 2 * T[i, 2, 1] * T[i, 2, 1] + 2 * T[i, 4, 1] * T[i, 4, 1]
    gtruth[i, 1] = (b_T + lamda2[i] * g2_prior) / (T2_2 + lamda2[i])
    if gtruth[i, 1] < 0: gtruth[i, 1] = 0

np.savetxt(os.path.join(save_dir, 'theta_input'), training_scalar)
np.savetxt(os.path.join(save_dir, 'g_truth'), gtruth)

g_data_list=[]
nbasis = 2
for ibasis in range(nbasis):
    g_file = os.path.join('foam_base_truth', '0', f'g{ibasis+1}')
    g_data = rf.foam.read_field_file(g_file)
    g_data['file'] = os.path.join('foam_base_truth', '0', f'g{ibasis+1}')
    g_data_list.append(g_data)

foam_dir='foam_base_truth'
for i, g_data in enumerate(g_data_list):
        g_data['internal_field']['value'] = gtruth[:, i]
        if foam_dir is not None:
            g_data['file'] = os.path.join(foam_dir, str(0), f'g{i+1}')
        _ = rf.foam.write_field_file(**g_data)


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
    con_flag = 0
    # set weights
    w = neuralnet.reshape_weights(w, w_shapes)
    nn.set_weights(w)

    # evaluate NN: cost and gradient
    J = 0
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
    # if J < 0.15:
    #     print(f'converged')
    #     con_flag = 1
    iter += 1
    return J.numpy(), np.squeeze(dJdw_flat)#, con_flag



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

# write files
g_data_list=[]
nbasis = 2
for ibasis in range(nbasis):
    g_file = os.path.join('foam_base', '0', f'g{ibasis+1}')
    g_data = rf.foam.read_field_file(g_file)
    g_data['file'] = os.path.join('foam_base', '0', f'g{ibasis+1}')
    g_data_list.append(g_data)

g = nn(training_scalar)

foam_dir='foam_base'
for i, g_data in enumerate(g_data_list):
        g_data['internal_field']['value'] = g[:, i]
        if foam_dir is not None:
            g_data['file'] = os.path.join(foam_dir, str(0), f'g{i+1}')
        _ = rf.foam.write_field_file(**g_data)
