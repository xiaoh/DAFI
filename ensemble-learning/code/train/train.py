#!/usr/bin/env python3

"""
Executable.
Use:
    >> train.py <input_file>

Sample input file:
    ../input_template.yaml
"""
import os
import time
import argparse

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import yaml

import neuralnet
import gradient_descent as gd
import regularization as reg
import data_preproc as preproc
import cost
from get_inputs import get_inputs

TENSORDIM = 9
TENSORSQRTDIM = 3
DEVSYMTENSORDIM = 5
DEVSYMTENSOR_INDEX = [0,1,2,4,5]
NBASISTENSORS = 10
NSCALARINVARIANTS = 5


## PARSE INPUT FILE
# input file
parser = argparse.ArgumentParser(description='Train Neural Network.')
parser.add_argument('input_file', help='Name (path) of input file')
args =  parser.parse_args()
with open(args.input_file, 'r') as f:
    input_dict = yaml.load(f, yaml.SafeLoader)

# architecture
nscalar_invariants = input_dict.get('nscalar_invariants', NSCALARINVARIANTS)
nbasis_tensors = input_dict.get('nbasis_tensors', NBASISTENSORS)
nhlayers = input_dict.get('nhlayers', 10)
nnodes = input_dict.get('nnodes', 10)
alpha = input_dict.get('alpha', 0.0)

# Modify the pre-trained network without having to pre-train again
g_init = np.array(input_dict.get('g_init', [0.0]*nbasis_tensors))
g_scale = input_dict.get('g_scale', 1.0)

# optimization
opt_algorithm = input_dict.get('opt_algorithm', 'GradientDescent')
opt_parameters = input_dict.get('opt_parameters', None)
opt_restart = input_dict.get('opt_restart', None)
opt_steps = input_dict.get('opt_steps', 100)

# data pre-processing
preproc_class = input_dict.get('preproc_class', None)

# regularization
regularization = input_dict.get('regularization', None)
reg_coeff = input_dict.get('reg_coeff', 1.0)

# debug
save_gradients = input_dict.get('save_gradients', False)
fixed_inputs  = input_dict.get('fixed_inputs', False)

# create save_dir
save_dir = input_dict.get('save_dir', 'results')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# training cases
nflows = input_dict.get('nflows', 1)
parallel = input_dict.get('parallel', True)
flows = []
for i in range(nflows):
    iflow = input_dict[f'flow_{i}']
    measurements = []
    for j in range(iflow['nmeasurements']):
        imeasurement = input_dict[f'flow_{i}_measurement_{j}']
        measurements.append(imeasurement)
    iflow['measurements'] = measurements
    iflow['name'] = f'flow_{i}'
    flows.append(iflow)


## CREATE NN
nn = neuralnet.NN(nscalar_invariants, nbasis_tensors, nhlayers, nnodes, alpha)

# call Tensorflow to get initilazation messages out of the way
with tf.GradientTape(persistent=True) as tape:
    gtmp = nn(np.zeros([1, nscalar_invariants]))
_ = tape.jacobian(gtmp, nn.trainable_variables, experimental_use_pfor=False)

# initial weights
w_init = np.array([])
for iw in nn.trainable_variables:
    w_init = np.concatenate([w_init, iw.numpy().flatten()])
w_shapes = neuralnet.weights_shape(nn.trainable_variables)

# print NN summary
print('\n' + '#'*80 + '\nCreated NN:' +
    f'\n  Number of scalar invariants: {nscalar_invariants}' +
    f'\n  Number of basis tensors: {nbasis_tensors}' +
    f'\n  Number of trainable parameters: {nn.count_params()}' +
    '\n' + '#'*80)


## Gradient: analytic dTau/dg
def get_dadg(tensors, tke):
    tke = np.expand_dims(np.squeeze(tke), axis=(1, 2))
    return 2.0*tke*tensors


## Cost function
# create one cost function per flow, each can handle multiple measurements
cost_list = []
for iflow in flows:
    if iflow['gradient_method'] == 'adjoint':
        cost_list.append(
            cost.CostAdjoint(iflow, nbasis=nbasis_tensors, restart=opt_restart, restart_dir=save_dir))
    elif iflow['gradient_method'] == 'ensemble':
        cost_list.append(
            cost.CostEnsemble(iflow, nbasis=nbasis_tensors, restart=opt_restart, restart_dir=save_dir))
# get the preprocesing class
if preproc_class is not None:
    PreProc = getattr(preproc, preproc_class)


# get fixed inputs (debug)
if fixed_inputs:
    input_scalars_list = []
    dadg_list = []
    for iflow, icost in enumerate(cost_list):
        # scalars
        input_scalars = np.load(icost.flow['input_scalars'])
        input_scalars = input_scalars[:, :nscalar_invariants]
        input_scalars_list.append(input_scalars)
        if preproc_class is not None:
            preprocess_data.update_stats(input_scalars)
        # tensors
        input_tensors = np.load(icost.flow['input_tensors'])
        input_tensors = input_tensors[:, :, :nbasis_tensors]
        # dadg
        tke = np.load(icost.flow['input_tke'])
        dadg = get_dadg(input_tensors, tke)
        dadg_list.append(dadg)
        # TODO: Write the tensors to OpenFOAM files
    # scale the input scalars
    if preproc_class is not None:
        for iflow, input_scalars in enumerate(input_scalars_list):
            input_scalars_list[iflow] = preprocess_data.scale(
                input_scalars, preprocess_data.stats)
        # save stats
        for i, stat in enumerate(preprocess_data.stats):
            file = os.path.join(save_dir, f'input_preproc_stat_{i}_{iter}')
            np.savetxt(file, np.atleast_1d(stat))


def cost(w):
    tstart = time.time()
    global iter, cost_list
    print(f'\n\nEvaluating Cost Function: {iter}')
    J = 0.0
    dJdw = np.zeros([1, nn.count_params()])

    # set weights
    w0 = w.copy()
    w = neuralnet.reshape_weights(w, w_shapes)
    nn.set_weights(w)

    # calculate inputs
    # initialize preprocessing instance
    if (preproc_class is not None) and (iter != first_iter):
        preprocess_data = PreProc()
    # get inputs and calculate statistics over all flows
    if not fixed_inputs:
        input_scalars_list = []
        for iflow, icost in enumerate(cost_list):
            input_scalars, _ = get_inputs(icost.gradU, icost.time_scale)
            input_scalars = input_scalars[:, :nscalar_invariants]
            input_scalars_list.append(input_scalars)
            if (preproc_class is not None) and (iter != first_iter):
                preprocess_data.update_stats(input_scalars)
        # scale the inputs
        if (preproc_class is not None) and (iter != first_iter):
            for iflow, input_scalars in enumerate(input_scalars_list):
                input_scalars_list[iflow] = preprocess_data.scale(
                    input_scalars, preprocess_data.stats)
            # save stats
            for i, stat in enumerate(preprocess_data.stats):
                file = os.path.join(save_dir, f'input_preproc_stat_{i}_{iter}')
                np.savetxt(file, np.atleast_1d(stat))

    # calculate cost and gradient for each flow
    # TODO: Make parallel
    for iflow, icost in enumerate(cost_list):
        print(f'    Flow {iflow}')

        # evaluate NN: cost and gradient
        ts = time.time()
        with tf.GradientTape(persistent=True) as tape:
            g = nn(input_scalars_list[iflow])*g_scale + g_init
        print(f'      TensorFlow forward ... {time.time()-ts:.2f}s')
        ts = time.time()
        dgdw_list = tape.jacobian(
            g, nn.trainable_variables, experimental_use_pfor=False)
        dgdw = neuralnet.jacobian_cellwise_submatrices(dgdw_list)
        print(f'      TensorFlow backward ... {time.time()-ts:.2f}s')

        # evaluate J, dJda
        ts = time.time()
        iJ, dJda, cost_vars = icost.cost(g.numpy())
        if not fixed_inputs:
            icost.gradU = cost_vars['gradU']
            icost.tke = cost_vars['tke']
            icost.time_scale = cost_vars['timeScale']
        print(f'    PDEs gradient ...            {time.time()-ts:.2f}s')

        # evaluate dadg analytically
        if fixed_inputs:
            dadg = dadg_list[iflow]
        else:
            _, input_tensors = get_inputs(icost.gradU, icost.time_scale)
            input_tensors = input_tensors[:, :, :nbasis_tensors]
            dadg = get_dadg(input_tensors, icost.tke)

        # calculate gradient
        dadw = dadg @ dgdw
        dadw = dadw.reshape([DEVSYMTENSORDIM*icost.ncells, nn.count_params()])
        idJdw = np.squeeze(dJda @ dadw)

        # update
        J += iJ
        dJdw += idJdw

        # save
        np.savetxt(os.path.join(save_dir, f'J.flow_{iflow}.{iter}'), [iJ])
        for name, val in cost_vars.items():
            np.savetxt(os.path.join(save_dir, name+f'.flow_{iflow}.{iter}'), val)
        if save_gradients:
            np.savetxt(os.path.join(save_dir, f'dJdw.flow_{iflow}.{iter}'), idJdw)
            np.savetxt(os.path.join(save_dir, f'dadw.flow_{iflow}.{iter}'), dadw)
            np.savetxt(os.path.join(save_dir, f'dJda.flow_{iflow}.{iter}'), dJda)

    # regularization
    if regularization != None:
        J0 = J.copy()
        dJdw0 = dJdw.copy()
        reg_method = getattr(reg, regularization)
        J_reg, dJdw_reg = reg_method(w.copy())
        J += reg_coeff * J_reg
        dJdw += reg_coeff * dJdw_reg

    # save
    np.savetxt(os.path.join(save_dir, f'J.{iter}'), [J])
    np.savetxt(os.path.join(save_dir, f'g.{iter}'), g.numpy())
    if regularization != None:
        np.savetxt(os.path.join(save_dir, f'J_org.{iter}'), [J0])
    if save_gradients:
        np.savetxt(os.path.join(save_dir, f'dJdw.{iter}'), dJdw)
        if regularization != None:
            np.savetxt(os.path.join(save_dir, f'dJdw_org.{iter}'), dJdw0)

    # summary
    print(f'Cost: {J}')
    if regularization != None:
        print(f'Cost noreg: {J0}')
    print(f'Time: {time.time()-tstart:.2f}s')
    iter += 1
    return J, dJdw


## Optimization
tstart_opt = time.time()
if (opt_restart is None) or (opt_restart == 'pretrain'):
    iter = 0
    first_iter = 0
else:
    iter = opt_restart
    first_iter = -1
optimization = getattr(gd, opt_algorithm)
optimization = optimization(objective=cost, restart=opt_restart, x=w_init,
    parameters=opt_parameters, save=True, save_directory=save_dir)
optimization.optimize(opt_steps)
tend_opt = time.time()

# clean
for icost in cost_list:
    try:
        icost.clean()
    except:
        pass

# summary
J0 = np.loadtxt(os.path.join(save_dir, 'J.0'))
J = np.loadtxt(os.path.join(save_dir, f'J.{iter-1}'))
print(f'\n\nInitial Cost: {J0}\nFinal Cost: {J}')
if regularization != None:
    J0 = np.loadtxt(os.path.join(save_dir, 'J_org.0'))
    J = np.loadtxt(os.path.join(save_dir, f'J_org.{iter-1}'))
    print(f'Initial Cost (no reg): {J0}\nFinal Cost (no reg): {J}')
print(f'Time: {int(tend_opt-tstart_opt):d} s')
