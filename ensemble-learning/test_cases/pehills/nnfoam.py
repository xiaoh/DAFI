# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Dynamic model for OpenFOAM eddy viscosity nutFoam solver. """

# standard library imports
import os
import shutil
import subprocess
import multiprocessing

# third party imports
import numpy as np
import scipy.sparse as sp
import yaml

# local imports
from dafi import PhysicsModel
from dafi import random_field as rf
from dafi.random_field import foam


import time
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

import neuralnet
import gradient_descent as gd
import regularization as reg
import data_preproc as preproc
import cost
from get_inputs import get_inputs


import pdb

TENSORDIM = 9
TENSORSQRTDIM = 3
DEVSYMTENSORDIM = 5
DEVSYMTENSOR_INDEX = [0,1,2,4,5]
NBASISTENSORS = 10
NSCALARINVARIANTS = 5

VECTORDIM = 3

class Model(PhysicsModel):
    """ Dynamic model for OpenFoam Reynolds stress nutFoam solver.

    The eddy viscosity field (nu_t) is infered by observing the
    velocity field (U). Nut is modeled as a random field with lognormal
    distribution and median value equal to the baseline (prior) nut
    field.
    """

    def __init__(self, inputs_dafi, inputs_model):
        # get required DAFI inputs.
        self.nsamples = inputs_dafi['nsamples']
        max_iterations = inputs_dafi['max_iterations']
        self.analysis_to_obs = inputs_dafi['analysis_to_obs']

        # read input file
        self.foam_case = inputs_model['foam_case']
        iteration_nstep = inputs_model['iteration_nstep']
        self.foam_timedir = str(iteration_nstep)

        nweights = inputs_model.get('nweights', None)
        self.ncpu = inputs_model.get('ncpu', 20)
        self.rel_stddev = inputs_model.get('rel_stddev', 0.5)
        self.abs_stddev = inputs_model.get('abs_stddev', 0.5)
        self.obs_rel_std = inputs_model.get('obs_rel_std', 0.001)
        self.obs_abs_std = inputs_model.get('obs_abs_std', 0.0001)

        obs_file = inputs_model['obs_file']
        # obs_err_file = inputs_model['obs_err_file']
        # obs_mat_file = inputs_model['obs_mat_file']

        weight_baseline_file = inputs_model['weight_baseline_file']

        # required attributes
        self.name = 'NN parameterized RANS model'

        # results directory
        self.results_dir = 'results_ensemble'

        # counter
        self.da_iteration = -1

        iteration_step_length = 1.0 / iteration_nstep

        # control dictionary
        self.timeprecision = 6
        self.control_list = {
            'application': 'simpleFoam',
            'startFrom': 'latestTime',
            'startTime': '0',
            'stopAt': 'nextWrite',
            'endTime': f'{max_iterations}',
            'deltaT': f'{iteration_step_length}',
            'writeControl': 'runTime',
            'writeInterval': '1',
            'purgeWrite': '2',
            'writeFormat': 'ascii',
            'writePrecision': f'{self.timeprecision}',
            'writeCompression': 'off',
            'timeFormat': 'fixed',
            'timePrecision': '0',
            'runTimeModifiable': 'true',
        }

        nut_base_foamfile = inputs_model['nut_base_foamfile']
        self.foam_info = foam.read_header(nut_base_foamfile)
        self.foam_info['file'] = os.path.join(
            self.foam_case, 'system', 'controlDict')

        # NN architecture
        self.nscalar_invariants = inputs_model.get('nscalar_invariants', NSCALARINVARIANTS)
        self.nbasis_tensors = inputs_model.get('nbasis_tensors', NBASISTENSORS)
        nhlayers = inputs_model.get('nhlayers', 10)
        nnodes = inputs_model.get('nnodes', 10)
        alpha = inputs_model.get('alpha', 0.0)

        # initial g
        self.g_init  = np.array(inputs_model.get('g_init', [0.0]*self.nbasis_tensors))
        self.g_scale = inputs_model.get('g_scale', 1.0)

        # data pre-processing
        self.preproc_class = inputs_model.get('preproc_class', None)

        # debug
        self.fixed_inputs  = inputs_model.get('fixed_inputs', True)

        parallel = inputs_model.get('parallel', True)

        ## CREATE NN
        self.nn = neuralnet.NN(self.nscalar_invariants, self.nbasis_tensors,
            nhlayers, nnodes, alpha)

        # call Tensorflow to get initilazation messages out of the way
        with tf.GradientTape(persistent=True) as tape:
            gtmp = self.nn(np.zeros([1, self.nscalar_invariants]))
        _ = tape.jacobian(gtmp, self.nn.trainable_variables, experimental_use_pfor=False)

        # initial weights
        self.w_init = np.loadtxt(weight_baseline_file)
        # self.w_init = np.loadtxt('./results_dafi/t_0/xf/xf_27') # np.array([])

        self.nbasis = self.nbasis_tensors
        self.nstate = len(self.w_init)

        self.g_data_list=[]
        for ibasis in range(self.nbasis):
            g_file = os.path.join(self.foam_case, '0.orig', f'g{ibasis+1}')
            g_data = rf.foam.read_field_file(g_file)
            g_data['file'] = os.path.join(self.foam_case, '0.orig', f'g{ibasis+1}')
            self.g_data_list.append(g_data)
        self.ncells = len(g_data['internal_field']['value'])

        # for iw in self.nn.trainable_variables:
        #     self.w_init = np.concatenate([self.w_init, iw.numpy().flatten()])

        self.w_shapes = neuralnet.weights_shape(self.nn.trainable_variables)

        # print NN summary
        print('\n' + '#'*80 + '\nCreated NN:' +
            f'\n  Number of scalar invariants: {self.nscalar_invariants}' +
            f'\n  Number of basis tensors: {self.nbasis_tensors}' +
            f'\n  Number of trainable parameters: {self.nn.count_params()}' +
            '\n' + '#'*80)

        foam.write_controlDict(
            self.control_list, self.foam_info['foam_version'],
            self.foam_info['website'], ofcase=self.foam_case)

        # get the preprocesing class
        if self.preproc_class is not None:
            self.PreProc = getattr(preproc, self.preproc_class)

        # calculate inputs
        # initialize preprocessing instance
        if os.path.isdir(self.results_dir):
            shutil.rmtree(self.results_dir)
        os.makedirs(self.results_dir)

        # observations
        # read observations
        norm_truth = 5e-6 / 0.00017857142857142857

        u1 = np.loadtxt(obs_file + '/dns/mapped/postProcessing/sampleDict/0/' + 'line_x1_U.xy')[:, 1]
        u2 = np.loadtxt(obs_file + '/dns/mapped/postProcessing/sampleDict/0/' + 'line_x3_U.xy')[:, 1]
        u3 = np.loadtxt(obs_file + '/dns/mapped/postProcessing/sampleDict/0/' + 'line_x5_U.xy')[:, 1]
        u4 = np.loadtxt(obs_file + '/dns/mapped/postProcessing/sampleDict/0/' + 'line_x7_U.xy')[:, 1]
        Ux = np.hstack([u1, u2, u3, u4])

        Uy = []
        v1 = np.loadtxt(obs_file + '/dns/mapped/postProcessing/sampleDict/0/' + 'line_x1_U.xy')[:, 2] #'/UyFullField') * 3
        v2 = np.loadtxt(obs_file + '/dns/mapped/postProcessing/sampleDict/0/' + 'line_x3_U.xy')[:, 2] #'/UyFullField') * 3
        v3 = np.loadtxt(obs_file + '/dns/mapped/postProcessing/sampleDict/0/' + 'line_x5_U.xy')[:, 2] #'/UyFullField') * 3
        v4 = np.loadtxt(obs_file + '/dns/mapped/postProcessing/sampleDict/0/' + 'line_x7_U.xy')[:, 2] #'/UyFullField') * 3
        Uy = np.hstack([v1, v2, v3, v4])


        # pdb.set_trace()
        self.obs = np.concatenate([Ux, Uy]) / norm_truth
        self.obs_error = np.diag(self.obs_rel_std * abs(self.obs) + self.obs_abs_std)
        self.nstate_obs = len(self.obs)

        # create sample directories
        sample_dirs = []
        for isample in range(self.nsamples):
            sample_dir = self._sample_dir(isample)
            sample_dirs.append(sample_dir)
            # TODO foam.copyfoam(, post='') - copies system, constant, 0
            shutil.copytree(self.foam_case, sample_dir)
            foam.write_controlDict(
                self.control_list, self.foam_info['foam_version'],
                self.foam_info['website'], ofcase=sample_dir)
        self.sample_dirs = sample_dirs
        # pdb.set_trace()

    def __str__(self):
        return 'Dynamic model for nutFoam eddy viscosity solver.'

    # required methods
    def generate_ensemble(self):
        """ Return states at the first data assimilation time-step.

        Creates the OpenFOAM case directories for each sample, creates
        samples of eddy viscosity (nut) based on samples of the KL modes
        coefficients (state) and writes nut field files. Returns the
        coefficients of KL modes for each sample.
        """

        # update X (nut)
        w = np.zeros([self.nstate, self.nsamples])
        for i in range(self.nstate):
            w[i, :] = self.w_init[i] + np.random.normal(0,
                abs(self.w_init[i] * self.rel_stddev + self.abs_stddev)
                , self.nsamples)
        return w

    def state_to_observation(self, state_vec):
        """ Map the states to observation space (from X to HX).

        Modifies the OpenFOAM cases to use nu_t reconstructed from the
        specified coeffiecients. Runs OpenFOAM, and returns the
        velocities at the observation locations.
        """
        self.da_iteration += 1

        # set weights
        w = state_vec.copy()
        time_dir = f'{self.da_iteration:d}'
        gsamps = []
        self.preprocess_data = self.PreProc()
        ts = time.time()
        for isamp in range(self.nsamples):
            gradU_file = os.path.join(self._sample_dir(isamp),
                str(self.da_iteration), 'grad(U)')
            tke_file = os.path.join(self._sample_dir(isamp), str(self.da_iteration), 'k')
            time_scale_file = os.path.join(
                self._sample_dir(isamp), str(self.da_iteration), 'timeScale')
            gradU = rf.foam.read_tensor_field(gradU_file)
            tke = rf.foam.read_scalar_field(tke_file)
            time_scale = rf.foam.read_scalar_field(time_scale_file)
            input_scalars, input_tensors = get_inputs(gradU, time_scale)
            input_scalars = input_scalars[:, :self.nscalar_invariants]
            input_tensors = input_tensors[:, :, :self.nbasis_tensors]
            dadg = _get_dadg(input_tensors, tke)

            # update min and max of input scalars
            # self.preprocess_data.update_stats(input_scalars)
            self.preprocess_data.update_stats(input_scalars)
            # scale the inputs
            input_scalars_scale = self.preprocess_data.scale(
                input_scalars, self.preprocess_data.stats)
            # save stats
            for i, stat in enumerate(self.preprocess_data.stats):
                file = os.path.join(self.results_dir,
                    f'input_preproc_stat_{i}_{self.da_iteration}')
                np.savetxt(file, np.atleast_1d(stat))

            w_reshape = neuralnet.reshape_weights(w[:, isamp], self.w_shapes)
            self.nn.set_weights(w_reshape)
            # evaluate NN: cost and gradient
            with tf.GradientTape(persistent=True) as tape:
                g = self.nn(input_scalars_scale) * self.g_scale + self.g_init


            dgdw_list = tape.jacobian(
                g, self.nn.trainable_variables, experimental_use_pfor=False)
            dgdw = neuralnet.jacobian_cellwise_submatrices(dgdw_list)
            # print(f'      TensorFlow backward ... {time.time()-ts:.2f}s')

            g = np.array(g)
            for i in range(g.shape[0]):
                for j in range(g.shape[1]):
                    # if g[i, j] < -0.5: g[i, j] = -0.5
                    # if j != 0 and g[i, j] > 0.01: g[i, j] = 0.01
                    # if j != 0 and g[i, j] < -0.01: g[i, j] = -0.01
                    if j == 0 and g[i, j] > -0.0: g[i, j] = -0.0

            gsamps.append(g)

        print(f'      TensorFlow ... {time.time()-ts:.2f}s')

        # write sample
        for i in range(self.nsamples):
            ig = np.zeros(g.shape)
            for j in range(self.nbasis):
                ig[:, j] = gsamps[i][:, j] # gsamps[j][:, i] # TODO:
            self._modify_foam_case(ig, self.da_iteration, foam_dir=self._sample_dir(i))

        parallel = multiprocessing.Pool(self.ncpu)
        inputs = [
            (self._sample_dir(i), self.da_iteration,
                self.timeprecision) for i in range(self.nsamples)]
        _ = parallel.starmap(_run_foam, inputs)
        parallel.close()

        # get HX
        state_in_obs = np.empty([self.nstate_obs, self.nsamples])
        for isample in range(self.nsamples):

            file = os.path.join(self._sample_dir(isample), 'postProcessing', 'sampleDict', time_dir)
            u1 = np.loadtxt(file + '/line_x1_U.xy')[:, 1]
            u2 = np.loadtxt(file + '/line_x3_U.xy')[:, 1]
            u3 = np.loadtxt(file + '/line_x5_U.xy')[:, 1]
            u4 = np.loadtxt(file + '/line_x7_U.xy')[:, 1]
            Ux = np.hstack([u1, u2, u3, u4])

            v1 = np.loadtxt(file + '/line_x1_U.xy')[:, 2] #'/UyFullField') * 3
            v2 = np.loadtxt(file + '/line_x3_U.xy')[:, 2] #'/UyFullField') * 3
            v3 = np.loadtxt(file + '/line_x5_U.xy')[:, 2] #'/UyFullField') * 3
            v4 = np.loadtxt(file + '/line_x7_U.xy')[:, 2] #'/UyFullField') * 3
            Uy = np.hstack([v1, v2, v3, v4])

            state_in_obs[:, isample] = np.concatenate([Ux, Uy])

        return state_in_obs

    def get_obs(self, time):
        """ Return the observation and error matrix.
        """
        return self.obs, self.obs_error

    def clean(self, loop):
        if loop == 'iter' and self.analysis_to_obs:
            for isamp in range(self.nsamples):
                dir = os.path.join(self._sample_dir(isamp),
                                   f'{self.da_iteration + 1:d}')
                shutil.rmtree(dir)

    # internal methods
    def _sample_dir(self, isample):
        "Return name of the sample's directory. "
        return os.path.join(self.results_dir, f'sample_{isample:d}')

    def _modify_foam_case(self, g, step, foam_dir=None):
        for i, g_data in enumerate(self.g_data_list):
            g_data['internal_field']['value'] = g[:, i]
            if foam_dir is not None:
                g_data['file'] = os.path.join(foam_dir, str(step), f'g{i+1}')
            _ = rf.foam.write_field_file(**g_data)

## Gradient: analytic dTau/dg
def _get_dadg(tensors, tke):
    tke = np.expand_dims(np.squeeze(tke), axis=(1, 2))
    return 2.0*tke*tensors

def _clean_foam(foam_dir):
    bash_command = './clean > /dev/null'
    bash_command = 'cd ' + foam_dir + ';' + bash_command + '; cd -'
    return subprocess.call(bash_command, shell=True, stdout=subprocess.DEVNULL)

# def _run_foam_init(foam_dir, iteration, timeprecision):
#     bash_command = './run > /dev/null'
#     bash_command = 'cd ' + foam_dir + ';' + bash_command + '; cd -'
#     return subprocess.call(bash_command, shell=True, stdout=subprocess.DEVNULL)

def _run_foam(foam_dir, iteration, timeprecision):

    # run foam
    solver = 'simpleFoam'
    logfile = os.path.join(foam_dir, solver + '.log')
    bash_command = f'{solver} -case {foam_dir} > {logfile}'
    subprocess.call(bash_command, shell=True)

    logfile = os.path.join(foam_dir, 'gradU.log')
    bash_command = f"postProcess -func 'grad(U)' -case {foam_dir}" + \
         f"> {logfile}" # 2>&1'
    subprocess.call(bash_command, shell=True)

    logfile = os.path.join(foam_dir, 'log.R')
    bash_command = f"{solver} -postProcess -func 'turbulenceFields(R)' -case {foam_dir}" + \
         f"> {logfile}" # 2>&1'
    subprocess.call(bash_command, shell=True)

    # bash_command = './run > /dev/null'
    # bash_command = 'cd ' + foam_dir + ';' + bash_command + '; cd -'
    # process = subprocess.call(bash_command, shell=True, stdout=subprocess.DEVNULL)

    # move results from i to i-1 directory
    tsrc = f'{iteration + 1:d}' # + '0'*timeprecision
    src = os.path.join(foam_dir, tsrc)
    dst = os.path.join(foam_dir, f'{iteration + 1:d}')
    shutil.move(src, dst)
    for field in ['U', 'p', 'phi', 'grad(U)', 'timeScale', 'k']:
        src = os.path.join(foam_dir, f'{iteration + 1:d}', field)
        dst = os.path.join(foam_dir, f'{iteration:d}', field)
        shutil.copyfile(src, dst)

    logfile = os.path.join(foam_dir, 'sample.log')
    bash_command = f"postProcess -func 'sampleDict' -case {foam_dir}" + \
         f"> {logfile}" # 2>&1'
    subprocess.call(bash_command, shell=True)
