#!/usr/bin/env python3

""" 
Executable. 
Run case for range of C_\mu. No neural network, only one parameter. 
Get cost (J) and gradient (dJ/dCmu). 
"""

import os
import time
import argparse

import numpy as np
import yaml

import cost
from get_inputs import get_inputs

DEVSYMTENSORDIM = 5
NBASISTENSORS = 1
NCMU = 1


# TODO: For simplicity, removed the following options. Can be added if needed. 
#     * fixed inputs 
#     * regularized cost function
#     * multiple flows

## PARSE INPUT FILE
# input file
parser = argparse.ArgumentParser(description='Train Neural Network.')
parser.add_argument('input_file', help='Name (path) of input file')
args =  parser.parse_args()
with open(args.input_file, 'r') as f:
    input_dict = yaml.load(f, yaml.SafeLoader)


# create save_dir
save_dir = input_dict.get('save_dir', 'results')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Cmu grid 
cmu_grid = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
cmu_grid += [0.058, 0.0585, 0.059, 0.0595, 0.06, 0.061, 0.062, 0.063, 0.064]
cmu_grid += [0.07, 0.08]
cmu_grid += [0.085, 0.087, 0.089, 0.091, 0.093, 0.095]
cmu_grid += [0.1, 0.11, 0.12, 0.13, 0.14, 0.16, 0.17, 0.18, 0.19, 0.2] 

# flow
i = 0
flow = input_dict[f'flow_{i}']
measurements = []
for j in range(flow['nmeasurements']):
    imeasurement = input_dict[f'flow_{i}_measurement_{j}']
    measurements.append(imeasurement)
flow['measurements'] = measurements
flow['name'] = f'flow_{i}'


## Gradient: analytic dTau/dg
def get_dadg(tensors, tke):
    tke = np.expand_dims(np.squeeze(tke), axis=(1, 2))
    return 2.0*tke*tensors


## Cost function
if flow['gradient_method'] == 'adjoint':
    icost = cost.CostAdjoint(flow, nbasis=NBASISTENSORS)
elif flow['gradient_method'] == 'ensemble':
    icost = cost.CostEnsemble(flow, nbasis=NBASISTENSORS)

def cost(cmu):
    tstart = time.time()
    global icost, iter
    print(f'\n\nEvaluating Cost Function: Cmu = {cmu}')

    g = np.ones([icost.ncells, NBASISTENSORS]) * -cmu
    dgdCmu = np.ones([icost.ncells, NBASISTENSORS, NCMU])

    # evaluate J, dJda
    J, dJda, cost_vars = icost.cost(g)

    # evaluate dadg analytically
    icost.gradU = cost_vars['gradU']
    icost.tke = cost_vars['tke']
    icost.time_scale = cost_vars['timeScale']
    _, input_tensors = get_inputs(icost.gradU, icost.time_scale)
    input_tensors = input_tensors[:, :, :NBASISTENSORS]
    dadg = get_dadg(input_tensors, icost.tke)

    # calculate gradient
    dadCmu = dadg @ dgdCmu
    dadCmu = dadCmu.reshape([DEVSYMTENSORDIM*icost.ncells, NCMU])
    dJdCmu = np.squeeze(dJda @ dadCmu)

    # save 
    for name, val in cost_vars.items():
        np.savetxt(os.path.join(save_dir, name+f'.{iter}'), val)

    # summary
    print(f'  Cost: {J}')
    print(f'  Grad: {dJdCmu}')
    print(f'  Time: {time.time()-tstart:.2f}s')

    iter += 1
    return J, dJdCmu


## Grid eval
cost_list = []
grad_list = [] 
iter = 0
for jcmu in cmu_grid:
    jcost, jgrad = cost(jcmu)
    cost_list.append(jcost)
    grad_list.append(jgrad)
postproc = lambda x: np.squeeze(np.array(x))
np.savetxt(os.path.join(save_dir, 'cmu'), postproc(cmu_grid))
np.savetxt(os.path.join(save_dir, 'cost'), postproc(cost_list))
np.savetxt(os.path.join(save_dir, 'grad'), postproc(grad_list))
