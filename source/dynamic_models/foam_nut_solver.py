# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Dynamic model for OpenFoam Reynolds stress nutFoam solver. """

# standard library imports
import os
import os.path as ospt
import shutil
import subprocess

# third party imports
import numpy as np
import scipy.sparse as sp
try:
    import pp
    has_parallel = True
except ImportError as parallel_error:
    has_parallel = False

# local imports
from data_assimilation.dyn_model import DynModel
import data_assimilation.utilities as util
import foam_utilities as foam
import random_field as rf


# global variables
nvector = 3


class Solver(DynModel):
    """ Dynamic model for OpenFoam Reynolds stress nutFoam solver. """

    def __init__(self, nsamples, da_interval, t_end, max_da_iteration,
                 model_input):
        """ Initialize the nutFOAM solver.

        Parses the input file, if needed creates the pseudo
        observations and KL modes, and initializes all variables.

        Note
        ----
        See documentation for ``DynModel`` (parent class) for
        information on inputs.

        Note
        ----
        Inputs in ``model_input`` dictionary:
            * **forward_interval** (``int``) -
              Number of simulation steps for forward model.
            * **nkl_modes** (``int``) -
              Number of KL modes to use.
            * **verbose** (``int``, ``1``) -
              Level of printed output. Zero for no output.
            * **generate_pseudo_obs** (``bool``, ``False``) -
              Whether to generate pseudo-observations. If True, will
              read the input file for generating observations. Else
              will use the existing data file.
            * **generate_kl** (``bool``, ``False``) -
              Whether to generate the KL modes. If True, will read the
              input file for generating KL modes. Else will use the
              existing KL modes file.
            * **generate_hmat** (``bool``, ``False``) -
              Whether to generate the H matrix for a given set of
              observation points.
            * **enable_parallel** (``bool``, ``False``) -
              Whether to run in parallel.
            * **ncpu** (``int``, ``0``) -
              Number of CPUs to use if ``enable_parallel`` is True. Set
              to zero to use all available CPUs.

        Note
        ----
        Directory structure:
            * dir/
                * nutfoam_kl.in
                * nutfoam_pseudo_obs.in
                * nutfoam_inputs/
                    * observations/
                        * observations
                        * observation_errors
                        * hmat_index
                        * hmat_weight
                        * obs_locations
                    * kl_modes/
                        * kl_modes
                        * covariance
                    * foam_base/
                        * 0/
                            * nut, U, p (baseline RANS results)
                        * constants/
                            * polyMesh/
                                * ...
                            * ...
                        * system/
                            * controlDict (template)
                            * sampleDict
                            * ...
        """
        # TODO: add messages if verbose
        # TODO: error/warning if pp specified but no pp
        # save the main inputs
        self.nsamples = nsamples
        self.max_da_iteration = max_da_iteration
        # self.da_interval = da_interval
        # self.t_end = t_end

        # read input file and set defaults
        param_dict = util.read_input_data(model_input)
        self.forward_interval = int(param_dict['forward_interval'])
        nkl_modes = int(param_dict['nkl_modes'])
        if 'verbose' in param_dict:
            self.verbose = int(param_dict['verbose'])
        else:
            self.verbose = 1
        if 'generate_pseudo_obs' in param_dict:
            generate_pseudo_obs = util.str2bool(
                param_dict['generate_pseudo_obs'])
        else:
            generate_pseudo_obs = False
        if 'generate_kl' in param_dict:
            generate_kl = util.str2bool(param_dict['generate_kl'])
        else:
            generate_kl = False
        if 'generate_hmat' in param_dict:
            generate_hmat = util.str2bool(param_dict['generate_hmat'])
        else:
            generate_hmat = False
        if 'enable_parallel' in param_dict:
            enable_parallel = util.str2bool(param_dict['enable_parallel'])
        else:
            enable_parallel = False
        if enable_parallel:
            if 'ncpu' in param_dict:
                self.ncpu = int(param_dict['ncpu'])
            else:
                self.ncpu = 0
        else:
            self.ncpu = 1

        # directory structure
        self.dir = os.getcwd()
        self.dir_input = ospt.join(self.dir, 'nutfoam_inputs')
        self.dir_foam_base = ospt.join(self.dir_input, 'foam_base')
        self.dir_obs = ospt.join(self.dir_input, 'observations')
        self.dir_kl = ospt.join(self.dir_input, 'kl_modes')
        self.dir_results = ospt.join(self.dir, 'results_nutfoam')
        # files
        file_observation_input = 'nutfoam_pseudo_obs.in'
        file_observation = ospt.join(self.dir_obs, 'observations')
        file_obs_error = ospt.join(self.dir_obs, 'observation_errors')
        file_observation_index = ospt.join(self.dir_obs, 'hmat_index')
        file_observation_weight = ospt.join(self.dir_obs, 'hmat_weight')
        file_obs_location = ospt.join(self.dir_obs, 'obs_locations')
        file_kl_input = 'nutfoam_kl.in'
        file_kl = ospt.join(self.dir_kl, 'kl_modes')
        # create directories if they do not exist
        directories = [self.dir_obs, self.dir_kl, self.dir_results]
        for directory in directories:
            util.create_dir(directory)

        # read/generate observations
        if generate_hmat:
            _generate_hmat(file_obs_location,
                           file_observation_index, file_observation_weight)
        if generate_pseudo_obs:
            pseudo_obs_dict = util.read_input_data(file_observation_input)
            _generate_pseudo_observations(
                pseudo_obs_dict, file_observation, file_obs_error)
        self.obs = np.loadtxt(file_observation)
        self.obs_error = np.loadtxt(file_obs_error)

        # read/generate KL modes
        if generate_kl:
            kl_dict = util.read_input_data(file_kl_input)
            _generate_kl_modes(kl_dict, file_kl)
        kl_modes = np.loadtxt(file_kl, usecols=range(nkl_modes))

        # initiliaze the ln(nu_t) field
        # replace the template fields in controlDict
        src = ospt.join(self.dir_foam_base, "system", "controlDict")
        dst = ospt.join(self.dir_foam_base, "system", "controlDict.template")
        shutil.copyfile(src, dst)
        util.replace(src, "<endTime>", '1')
        util.replace(src, "<writeInterval>", '1')
        # read FOAM
        foam_coords = foam.get_cell_coordinates(self.dir_foam_base)
        foam_volume = foam.get_cell_volumes(self.dir_foam_base)
        mean_file = ospt.join(self.dir_foam_base, '0', 'nut')
        nut_mean = foam.read_scalar_from_file(mean_file)
        # os.remove(src)
        shutil.move(dst, src)
        # create field
        self.nut_rf = rf.GaussianProcess(
            name='nut', mean=nut_mean, coords=foam_coords,
            kl_modes=kl_modes, weight_field=foam_volume)
        self.log_mean = np.log(self.nut_rf.mean)

        # required attributes for DAFI
        self.name = 'FoamNutSolver'
        self.nstate = nkl_modes
        self.nstate_obs = len(self.obs)
        self.init_state = np.zeros(self.nstate)

        # other initialization
        self.da_step = 0
        self.foam_solver = 'nutFoam'
        self.ncells = len(self.nut_rf.mean)

        # create H matrix
        idx = np.loadtxt(file_observation_index)
        weight = np.loadtxt(file_observation_weight)
        self.obs_vel2obs = self._construct_hmat(idx, weight)

        # initialize parallel python
        if(enable_parallel):
            self.job_server = pp.Server()
            if self.ncpu > 0:
                self.job_server.set_ncpus(self.ncpu)
            else:
                # use all cpus
                self.job_server.set_ncpus()
                self.ncpu = self.job_server.get_ncpus()
            self.jobs = []

    def __str__(self):
        str_info = 'Dynamic model for nutFoam solver.'
        return str_info

    # required methods
    def generate_ensemble(self):
        """ Return states at the first data assimilation time-step.

        Creates the OpenFOAM case directories for each sample. Creates
        the perturbed nu_t fields (X0) and runs OpenFOAM to get the
        velocities (HX0).

        Note
        ----
        See documentation for ``DynModel`` (parent class) for
        information on outputs.
        """
        # generate folders
        max_end_time = self.max_da_iteration * self.forward_interval
        for isample in range(self.nsamples):
            sample_dir = ospt.join(self.dir, 'sample_{:d}'.format(isample + 1))
            shutil.copytree(self.dir_foam_base, sample_dir)
            control_dict = ospt.join(sample_dir, "system", "controlDict")
            util.replace(control_dict, "<endTime>", str(max_end_time))
            util.replace(control_dict, "<writeInterval>",
                         str(self.forward_interval))
        # update X (nut)
        nut, coeffs = self.nut_rf.sample_kl_reduced(
            self.nsamples, self.nstate, return_coeffs=True)
        self._modify_openfoam('nut', nut, '0')
        # forward to HX (U)
        model_obs = self.forward(None, False)
        return (coeffs, model_obs)

    def forecast_to_time(self, state_vec_current, end_time):
        """ Return states at the next end time.

        Required by DAFI, but not used for steady problems like this one.

        Note
        ----
        See documentation for ``DynModel`` (parent class) for
        information on inputs and outputs.
        """
        return state_vec_current

    def forward(self, state_vec, update=True):
        """ Forward the states to observation space (from X to HX).

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
            nut = self.nut_rf.reconstruct_kl_reduced(state_vec)
            time_dir = '{:.6f}'.format(self.da_step * self.forward_interval)
            self._modify_openfoam('nut', nut, time_dir)

        # run openFOAM
        self._call_foam(sample=True)
        # get HX
        velocities = np.zeros([self.ncells*nvector, self.nsamples])
        for isample in range(self.nsamples):
            sample_dir = ospt.join(self.dir, 'sample_{:d}'.format(isample + 1))
            # copy forwarded fields
            self._copy_to_previous(sample_dir, 'U')
            self._copy_to_previous(sample_dir, 'p')
            # save the results
            if self.da_step > 0:
                src_time_dir = '{:.6f}'.format(
                    self.da_step * self.forward_interval)
                dst_time_dir = '{:d}'.format(self.da_step)
                src = ospt.join(sample_dir, src_time_dir)
                dst = ospt.join(sample_dir, dst_time_dir)
                shutil.copytree(src, dst)
            # get velocities
            if self.da_step == 0:
                time_dir = '0'
            else:
                time_dir = '{:.6f}'.format(
                    self.da_step * self.forward_interval)
            file_to_read = ospt.join(sample_dir, time_dir, 'U')
            ivel = foam.read_vector_from_file(file_to_read).flatten('C')
            velocities[:, isample] = ivel.flatten('C')
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
        # delete last time folder and move results to results folder
        for isample in range(self.nsamples):
            sample_dir = ospt.join(self.dir, 'sample_{:d}'.format(isample + 1))
            # shutil.rmtree(ospt.join(
            #     sample_dir,
            #     '{:.6f}'.format((self.da_step + 1) * self.forward_interval)))
            shutil.move(sample_dir, self.dir_results)

    # internal methods
    # TODO: finish the docstrings for internal methods.
    def _modify_openfoam(self, field_name, values, timedir):
        """ Replace the values of a specific field in an OpenFOAM case.
        """
        for isample in range(self.nsamples):
            sample_dir = ospt.join(self.dir, 'sample_{:d}'.format(isample + 1))
            field_file = ospt.join(sample_dir, timedir, field_name)
            foam.write_scalar_to_file(values[:, isample], field_file)

    def _call_foam(self, sample=False):
        """ Run the OpenFOAM cases for all samples, possibly in
        parallel.
        """
        def _run_foam(solver, case_dir='.', sample=False):
            """ Run a single instance of OpenFOAM in the specified
            directory.
            """
            bash_command = solver + ' -case ' + case_dir + \
                ' &>> ' + os.path.join(case_dir, solver + '.log')
            subprocess.call(bash_command, shell=True)
            if sample:
                bash_command = 'sample -case ' + case_dir + \
                    ' -latestTime >> ' + os.path.join(case_dir, 'sample.log')
                subprocess.call(bash_command, shell=True)

        for isample in range(self.nsamples):
            sample_dir = ospt.join(self.dir,
                                   'sample_{:d}'.format(isample + 1))
            if self.ncpu > 1:
                self.jobs.append(self.job_server.submit(
                    func=_run_foam,
                    args=(self.foam_solver, sample_dir, sample),
                    depfuncs=(),
                    modules=('os', 'subprocess')))
            else:
                _run_foam(self.foam_solver, sample_dir, sample)
        if self.ncpu > 1:
            barrier = [job() for job in self.jobs]
            self.jobs = []
        barrier = 0

    def _copy_to_previous(self, case_dir, field_name):
        """ Copy the specified field to the previous DA-step directory.
        """
        src_time_dir = '{:.6f}'.format(
            (self.da_step + 1) * self.forward_interval)
        if self.da_step == 0:
            dst_time_dir = '0'
        else:
            dst_time_dir = '{:.6f}'.format(
                self.da_step * self.forward_interval)
        src = ospt.join(case_dir, src_time_dir, field_name)
        dst = ospt.join(case_dir, dst_time_dir, field_name)
        shutil.copyfile(src, dst)
        util.replace(dst, '"' + src_time_dir + '"', '"' + dst_time_dir + '"')

    def _construct_hmat(self, idx, weight):
        """ Construct the matrix to go from all velocities to
        observation velocities.
        """
        weight = np.expand_dims(weight, 1)
        nidx0, nidx1 = idx.shape
        idx3 = np.zeros((nidx0 * nvector, nidx1))
        weight3 = np.zeros((nidx0 * nvector, 1))
        # loop
        current_idx = 0
        for iblock in range(int(idx[:, 0].max()) + 1):
            rg = np.where(idx[:, 0] == iblock)[0]
            start, duration = rg[0], len(rg)
            # x velocities
            idx_block = np.copy(idx[start:start + duration, :])
            idx_block[:, 1] *= nvector
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
            idx3[nvector * start:nvector * (start + duration),
                 :] = np.vstack((idx_block, idx_block1, idx_block2))
            weight3[nvector * start:nvector * (start + duration), :] = \
                np.vstack((wgtBlock, wgtBlock, wgtBlock))
            current_idx += nvector
        hmat = sp.coo_matrix((weight3.flatten('C'), (idx3[:, 0], idx3[:, 1])),
                             shape=(self.nstate_obs, self.ncells*nvector))
        return hmat.tocsr()


# internal functions
# TODO: implement
# TODO: docstrings
def _generate_hmat(infile_loc, outfile_index, outfile_weight):
    pass


def _generate_pseudo_observations(input_dict, outfile_obs, outfile_err):
    pass


def _generate_kl_modes(input_dict, filename):
    pass
