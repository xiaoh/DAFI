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

class Model(PhysicsModel):
    """ Dynamic model for OpenFoam Reynolds stress nutFoam solver.

    The eddy viscosity field (nu_t) is infered by observing the
    velocity field (U). Nut is modeled as a random field with lognormal
    distribution and median value equal to the baseline (prior) nut
    field.
    """

    def __init__(self, inputs_dafi, inputs_model):
        """ Initialize the nutFOAM solver.

        Note
        ----
        Inputs in ``model_input`` dictionary:
            * **forward_interval** (``int``) -
              Number of simulation steps for forward model.
            * **nkl_modes** (``int``) -
              Number of KL modes to use.
            * **enable_parallel** (``bool``, ``False``) -
              Whether to run in parallel.
            * **ncpu** (``int``, ``0``) -
              Number of CPUs to use if ``enable_parallel`` is True. Set
              to zero to use all available CPUs.
            * **foam_base_dir** (``str``) -
              OpenFOAM case directory to be copied to run each sample.
            * **mesh_dir** (``str``) -
              Directory containing the OpenFOAM mesh coordinates and
              volume information files (ccx, ccy, ccz, cv).
            * **obs_file** (``str``) -
              File containing the observation values.
            * **obs_err_file** (``str``) -
              File containing the observation error/covariance (R)
              matrix.
            * **kl_modes_file** (``str``) -
              File containing the KL modes for the nu_t covariance.
            * **obs_mat_file** (``str``) -
              File containing the mesh cells and weights at observation
              locations. Generated with ``getObsMatrix`` utility.
            * **transform** (``str``) -
              Tranformation option for nut s.t. T(nut)~GP(0,K). Options:
              linear, log.
        """
        # get required DAFI inputs.
        self.nsamples = inputs_dafi['nsamples']
        max_iterations = inputs_dafi['max_iterations']

        # read input file
        with open(inputs_model['input_file'], 'r') as file:
            input_dict = yaml.load(file, yaml.SafeLoader)
        self.foam_case = input_dict['foam_case']
        iteration_step_length = input_dict.get('iteration_step_length', 1.0)
        iteration_nstep = input_dict['iteration_nstep']
        klmodes_file = input_dict['klmodes_file']
        nut_base_foamfile = input_dict['nut_baseline_foamfile']
        nklmodes = input_dict.get('nklmodes', None)
        self.ncpu = input_dict.get('ncpu', 1)

        # required attributes
        self.name = 'nutFoam Eddy viscosity RANS model'

        # results directory
        self.results_dir = 'results_nutFoam'

        # counter
        self.da_iteration = 0

        # control dictionary
        endTime = max_iterations * iteration_nstep * iteration_step_length
        self.control_list = {
            'application': 'nutFoam',
            'startFrom': 'latestTime',
            'startTime': '0',
            'stopAt': 'nextWrite',
            'endTime': f'{endTime}',
            'deltaT': f'{iteration_step_length}',
            'writeControl': 'runTime',
            'writeInterval': f'{iteration_nstep}',
            'purgeWrite': '2',
            'writeFormat': 'ascii',
            'writePrecision': '6',
            'writeCompression': 'off',
            'timeFormat': 'fixed',
            'timePrecision': '6',
            'runTimeModifiable': 'true',
        }  # TODO: foam.read_controlDict()
        self.foam_info = foam.read_header(os.path.join(
            self.foam_case, 'system', 'controlDict'))

        # initialize the random process
        klmodes = np.readtxt(klmodes_file)
        if nklmodes is not None:
            klmodes = klmodes[:, :nklmodes]
        weights = foam.get_cell_volumes(self.foam_case)
        self.nut_base_data = foam.read_field_file(self.nut_base_foamfile)
        nut_base = self.nut_base_data['internal_field']['value']
        self.rf_nut = rf.LogNormal(self.klmodes, nut_base, weights)

        # observations

        #######################################################################
        ########################### OLD #######################################
        #######################################################################
        obs_file = param_dict['obs_file']
        obs_err_file = param_dict['obs_err_file']
        obs_mat_file = param_dict['obs_mat_file']

        # read observations
        self.obs = np.loadtxt(obs_file)
        self.obs_error = np.loadtxt(obs_err_file)
        self.nstate_obs = len(self.obs)

        # create "H matrix"
        obs_mat = np.loadtxt(obs_mat_file)
        self.obs_vel2obs = _construct_obsmat_vec(
            obs_mat, self.nstate_obs, self.ncells)


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
        # create sample directories
        os.makedirs(self.results_dir)
        for isample in range(self.nsamples):
            sample_dir = _sample_dir(isample)
            # TODO foam.copyfoam()
            shutil.copytree(self.foam_case, sample_dir)
            foam.write_controlDict(
                self.control_list, self.foam_info['foam_version'],
                self.foam_info['website'], ofcase=sample_dir)

        # create samples, modify files, output coefficients (state)
        samps, coeffs = self.rf_nut.sample_func(self.nsamples)
        self._modify_openfoam_scalar(samps, '0', nut_base_data)
        return coeffs

    def state_to_observation(self, state_vec, update=True):
        """ Map the states to observation space (from X to HX).

        Modifies the OpenFOAM cases to use nu_t reconstructed from the
        specified coeffiecients. Runs OpenFOAM, and returns the
        velocities at the observation locations.

        Note
        ----
        See documentation for ``DynModel`` (parent class) for
        information on inputs and outputs.
        """
        # modify nut
        if update:
            self.da_step += 1
            delta_nut = self.delta_nut_rf.reconstruct_kl_reduced(state_vec)
            nut = self.transform_inv(self.baseline_mat + delta_nut)
            time_dir = '{:d}'.format(self.da_step * self.forward_interval)
            self._modify_openfoam_scalar('nut', nut, time_dir)
        # run openFOAM
        self._call_foam(sample=True)
        # get HX
        velocities = np.zeros([self.ncells*foam.NVECTOR, self.nsamples])
        time_dir = '{:d}'.format(self.da_step)
        for isample in range(self.nsamples):
            sample_dir = 'sample_{:d}'.format(isample + 1)
            # get velocities
            file_to_read = os.path.join(sample_dir, time_dir, 'U')
            ivel = foam.read_vector_from_file(file_to_read).flatten('C')
            velocities[:, isample] = ivel
        return self.obs_vel2obs.dot(velocities)

    def get_obs(self, time):
        """ Return the observation and error matrix.

        These do not change for a steady problem like this one.

        Note
        ----
        See documentation for ``DynModel`` (parent class) for
        information on inputs and outputs.
        """
        return self.obs, self.obs_error

    def clean(self):
        """ Cleanup before exiting. """
        util.create_dir(self.dir_results)
        # delete last time folder and move results to results folder
        for isample in range(self.nsamples):
            sample_dir = os.path.join('sample_{:d}'.format(isample + 1))
            shutil.rmtree(os.path.join(
                sample_dir,
                '{:d}'.format((self.da_step + 1) * self.forward_interval)))
            shutil.move(sample_dir, self.dir_results)

    # internal methods
    def _sample_dir(self, isample):
        "Return name of the sample's directory. "
        return os.path.join(self.save_dir, f'sample_{isample:d}')

    def _modify_openfoam_scalar(self, values, timedir, data_dict):
        """ Replace the values of a specific field in all samples.

        Parameters
        ----------
        values : ndarray
            New values for the field.
            ``dtype=float``, ``ndim=2``, ``shape=(ncells, nsamples)``
        timedir : str
            The time directory in which the field should be modified.
        data_dict : dict
            Dictionary with field information.
        """
        for isample in range(self.nsamples):
            file = os.path.join(self._sample_dir(isample), timedir, field_name)
            value = values[:, isample]
            data_dict['file'] = file
            data_dict['internal_field']['value'] = value
            foam.write_field_file(data_dict**)

    def _call_foam(self, sample=False):
        """ Run the OpenFOAM cases for all samples, possibly in
        parallel.

        Parameters
        ----------
        sample : bool
            Whether to run the OpenFOAM ``sample`` utility.
        """
        def _run_foam(solver, da_step, forward_interval, case_dir='.',
                      sample=False):
            """ Run a single instance of OpenFOAM in the specified
            directory.
            """
            # run foam
            bash_command = solver + ' -case ' + case_dir + \
                ' &>> ' + os.path.join(case_dir, solver + '.log')
            subprocess.call(bash_command, shell=True)
            # save directories
            dst_time_dir = '{:d}'.format(da_step)
            dst = os.path.join(case_dir, dst_time_dir)
            if da_step > 0:
                os.makedirs(dst)
                # copy nut from previous time directory
                src_time_dir = '{:d}'.format(
                    da_step * forward_interval)
                src = os.path.join(case_dir, src_time_dir)
                shutil.copyfile(os.path.join(src, 'nut'),
                                os.path.join(dst, 'nut'))
            # copy U and p from current time directory
            src_time_dir = '{:d}'.format(
                (da_step+1) * forward_interval)
            src = os.path.join(case_dir, src_time_dir)
            shutil.copyfile(os.path.join(src, 'U'), os.path.join(dst, 'U'))
            shutil.copyfile(os.path.join(src, 'p'), os.path.join(dst, 'p'))
            # copy other results
            files = ['phi']
            directories = ['uniform']
            for directory in directories:
                try:
                    shutil.copytree(os.path.join(src, directory),
                                    os.path.join(dst, directory))
                except:
                    pass
            for file in files:
                try:
                    shutil.copyfile(os.path.join(src, file),
                                    os.path.join(dst, file))
                except:
                    pass
            # delete directory
            if da_step > 0:
                shutil.rmtree(os.path.join(
                    case_dir, '{:d}'.format((da_step) * forward_interval)))
            # run sample
            if sample:
                bash_command = 'sample -case ' + case_dir + \
                    " -time '" + dst_time_dir + "' >> " + \
                    os.path.join(case_dir, 'sample.log')
                subprocess.call(bash_command, shell=True)

        for isample in range(self.nsamples):
            sample_dir = os.path.join('sample_{:d}'.format(isample + 1))
            if self.ncpu > 1:
                self.jobs.append(self.job_server.submit(
                    func=_run_foam,
                    args=(self.foam_solver, self.da_step,
                          self.forward_interval, sample_dir, sample),
                    depfuncs=(),
                    modules=('os', 'subprocess', 'shutil')))
            else:
                _run_foam(self.foam_solver, self.da_step,
                          self.forward_interval, sample_dir, sample)
        if self.ncpu > 1:
            barrier = [job() for job in self.jobs]
            self.jobs = []
        barrier = 0


def _construct_obsmat_vec(obs_mat, nstate_obs, ncells):
    """ Construct the matrix to go from entire vector field to values
    at the observation locations.

    Parameters
    ----------
    obs_mat : ndarray
        Matrix containing the relevant cells for each observation
        location. The three columns are (1) observation index, (2) cell
        index, (3) cell weight. This matrix should be read from the
        output of the ``getObsMatrix`` utility.
    nstate_obs : int
        Number of observation states.
    ncells : int
        Number of cells in OpenFOAM mesh.
    """
    weight = np.expand_dims(obs_mat[:, 2], 1)
    idx = obs_mat[:, :2]
    nidx0, nidx1 = idx.shape
    idx3 = np.zeros((nidx0 * foam.NVECTOR, nidx1))
    weight3 = np.zeros((nidx0 * foam.NVECTOR, 1))
    # loop
    current_idx = 0
    for iblock in range(int(idx[:, 0].max()) + 1):
        rg = np.where(idx[:, 0] == iblock)[0]
        start, duration = rg[0], len(rg)
        # x velocities
        idx_block = np.copy(idx[start:start + duration, :])
        idx_block[:, 1] *= foam.NVECTOR
        wgtBlock = np.copy(weight[start:start + duration, :])
        idx_block[:, 0] = current_idx
        # y velocities
        idx_block1 = np.copy(idx_block)
        idx_block1[:, 0] += 1
        idx_block1[:, 1] += 1
        # z velocities
        idx_block2 = np.copy(idx_block)
        idx_block2[:, 0] += 2
        idx_block2[:, 1] += 2
        # all
        idx3[foam.NVECTOR * start:foam.NVECTOR * (start + duration), :] = \
            np.vstack((idx_block, idx_block1, idx_block2))
        weight3[foam.NVECTOR * start:foam.NVECTOR * (start + duration), :] = \
            np.vstack((wgtBlock, wgtBlock, wgtBlock))
        current_idx += foam.NVECTOR
    hmat = sp.coo_matrix((weight3.flatten('C'), (idx3[:, 0], idx3[:, 1])),
                         shape=(nstate_obs, ncells*foam.NVECTOR))
    return hmat.tocsr()


# pre-processing - not used by DAFI
def get_vector_at_obs(obs_mat_file, field_file):
    """ Get an OpenFOAM vector field at the specified observation
    locations.

    Parameters
    ----------
    obs_mat_file : ndarray
        Output file of the ``getObsMatrix`` utility.
    field_file : str
        OpenFOAM field file containg the full vector field.
    """
    field = foam.read_vector_from_file(field_file)
    ncells = len(field)
    field = field.flatten('C')
    mesh_mat = np.loadtxt(obs_mat_file)
    npoints = int(mesh_mat[-1, 0]) + 1
    nstate_obs = npoints * foam.NVECTOR
    obsmat = _construct_obsmat_vec(mesh_mat, nstate_obs, ncells)
    return obsmat.dot(field)
