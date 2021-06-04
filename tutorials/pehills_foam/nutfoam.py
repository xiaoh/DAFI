# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Dynamic model for OpenFOAM eddy viscosity nutFoam solver. """

# standard library imports
import os
import shutil
import subprocess
import multiprocessing

# third party imports
import numpy as np

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
        Inputs in ``inputs_model`` dictionary:
            * **foam_case** (``str``) -
              OpenFOAM case directory to be copied to run each sample.
            * **iteration_nstep** (``int``) -
              Number of simulation steps for a forward model (OpenFOAM)
              solve.
            * **klmodes_file** (``str``) -
              File containing the KL modes for the nu_t covariance.
            * **nut_baseline_foamfile** (````) -
              OpenFOAM field file for the baseline eddy viscosity.
            * **nklmodes** (``int``) -
              Number of KL modes to use.
            * **ncpu** (``int``, ``0``) -
              Number of CPUs to use. Set to zero to use all available 
              CPUs.
            * **obs_file** (``str``) -
              File containing the observation values.
            * **foam_rc** (``str``) -
              File used to source OpenFOAM in your system, if needed.
        """
        # get required DAFI inputs.
        self.nsamples = inputs_dafi['nsamples']
        max_iterations = inputs_dafi['max_iterations']
        self.analysis_to_obs = inputs_dafi['analysis_to_obs']

        # read input file
        self.foam_case = inputs_model['foam_case']
        iteration_nstep = inputs_model['iteration_nstep']
        klmodes_file = inputs_model['klmodes_file']
        nut_base_foamfile = inputs_model['nut_baseline_foamfile']
        nklmodes = inputs_model.get('nklmodes', None)
        self.ncpu = inputs_model.get('ncpu', 1)
        obs_file = inputs_model['obs_file']
        self.foam_rc = inputs_model.get('foam_rc', None)

        # required attributes
        self.name = 'nutFoam Eddy viscosity RANS model'

        # results directory
        self.results_dir = 'results_nutFoam'

        # counter
        self.da_iteration = -1

        iteration_step_length = 1.0 / iteration_nstep

        # control dictionary
        self.timeprecision = 6
        self.control_list = {
            'application': 'nutFoam',
            'startFrom': 'latestTime',
            'startTime': '0',
            'stopAt': 'nextWrite',
            'endTime': f'{max_iterations}',
            'deltaT': f'{iteration_step_length}',
            'writeControl': 'runTime',
            'writeInterval': '1',
            'purgeWrite': '2',
            'writeFormat': 'ascii',
            'writePrecision': '6',
            'writeCompression': 'off',
            'timeFormat': 'fixed',
            'timePrecision': f'{self.timeprecision}',
            'runTimeModifiable': 'true',
        }
        self.foam_info = foam.read_header(nut_base_foamfile)
        self.foam_info['file'] = os.path.join(
            self.foam_case, 'system', 'controlDict')

        # initialize the random process
        klmodes = np.loadtxt(klmodes_file)
        if nklmodes is not None:
            klmodes = klmodes[:, :nklmodes]
        weights = foam.get_cell_volumes(self.foam_case, foam_rc=self.foam_rc)
        self.nut_data = foam.read_field_file(nut_base_foamfile)
        nut_base = self.nut_data['internal_field']['value']
        self.rf_nut = rf.LogNormal(klmodes, nut_base, weights)
        nstates = len(nut_base)

        # observations
        obsdata = np.loadtxt(obs_file)
        obsfield = obsdata[:, 3]
        obs_Ux = obsdata[obsfield == 0, 4]
        obs_Uy = obsdata[obsfield == 1, 4]
        obs_Uz = obsdata[obsfield == 2, 4]
        self.obs = np.concatenate([obs_Ux, obs_Uy, obs_Uz])
        self.nobs = len(self.obs)

        # observation error
        std_Ux = obsdata[obsfield == 0, 5]
        std_Uy = obsdata[obsfield == 1, 5]
        std_Uz = obsdata[obsfield == 2, 5]
        std = np.concatenate([std_Ux, std_Uy, std_Uz])
        self.obs_error = np.diag(std**2)

        # create "H matrix"
        connectivity = foam.get_neighbors(self.foam_case)
        coords = foam.get_cell_centres(self.foam_case, foam_rc=self.foam_rc)
        pointsUx = obsdata[obsfield == 0, :3]
        pointsUy = obsdata[obsfield == 1, :3]
        pointsUz = obsdata[obsfield == 2, :3]
        if pointsUx.shape[0] > 0:
            self.H_Ux = rf.inverse_distance_weights(
                coords, connectivity, pointsUx)
        else:
            self.H_Ux = np.empty([0, nstates])
        if pointsUy.shape[0] > 0:
            self.H_Uy = rf.inverse_distance_weights(
                coords, connectivity, pointsUy)
        else:
            self.H_Uy = np.empty([0, nstates])
        if pointsUz.shape[0] > 0:
            self.H_Uz = rf.inverse_distance_weights(
                coords, connectivity, pointsUz)
        else:
            self.H_Uz = np.empty([0, nstates])

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
            sample_dir = self._sample_dir(isample)
            shutil.copytree(self.foam_case, sample_dir)
            foam.write_controlDict(
                self.control_list, self.foam_info['foam_version'],
                self.foam_info['website'], ofcase=sample_dir)

        # create samples, modify files, output coefficients (state)
        _, coeffs = self.rf_nut.sample_func(self.nsamples)
        return coeffs

    def state_to_observation(self, state_vec):
        """ Map the states to observation space (from X to HX).

        Modifies the OpenFOAM cases to use nu_t reconstructed from the
        specified coeffiecients. Runs OpenFOAM, and returns the
        velocities at the observation locations.
        """
        self.da_iteration += 1

        # modify nut
        samps = self.rf_nut.reconstruct_func(state_vec)
        time_dir = f'{self.da_iteration:d}'
        for isample in range(self.nsamples):
            file = os.path.join(self._sample_dir(isample), time_dir, 'nut')
            value = samps[:, isample]
            self.nut_data['file'] = file
            self.nut_data['internal_field']['value'] = value
            foam.write_field_file(**self.nut_data)

        # run openFOAM in parallel
        parallel = multiprocessing.Pool(self.ncpu)
        inputs = [
            ('nutFoam', self._sample_dir(i), self.da_iteration,
                self.timeprecision, self.foam_rc)
            for i in range(self.nsamples)]
        _ = parallel.starmap(_run_foam, inputs)
        parallel.close()

        # get HX
        state_in_obs = np.empty([self.nobs, self.nsamples])
        time_dir = f'{self.da_iteration + 1:d}'
        for isample in range(self.nsamples):
            file = os.path.join(self._sample_dir(isample), time_dir, 'U')
            U = foam.read_vector_field(file)
            Ux = self.H_Ux.dot(U[:, 0])
            Uy = self.H_Uy.dot(U[:, 1])
            Uz = self.H_Uz.dot(U[:, 2])
            state_in_obs[:, isample] = np.concatenate([Ux, Uy, Uz])
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


def _run_foam(solver, case_dir, iteration, timeprecision, foam_rc):
    """ Run an OpenFOAM case.

    Used by state_to_observation but needs to be outside class to run
    in parallel.
    """
    # run foam
    logfile = os.path.join(case_dir, solver + '.log')
    bash_command = foam._bash_source_of(foam_rc)
    bash_command += f'{solver} -case {case_dir} > {logfile}'
    subprocess.call(bash_command, shell=True)

    # move results from i to i-1 directory
    tsrc = f'{iteration + 1:d}.' + '0'*timeprecision
    src = os.path.join(case_dir, tsrc)
    dst = os.path.join(case_dir, f'{iteration + 1:d}')
    shutil.move(src, dst)
    for field in ['U', 'p', 'phi']:
        src = os.path.join(case_dir, f'{iteration + 1:d}', field)
        dst = os.path.join(case_dir, f'{iteration:d}', field)
        shutil.copyfile(src, dst)

    # sample
    bash_command = foam._bash_source_of(foam_rc)
    bash_command += f'postProcess  -case {case_dir} -func sampleDict ' + \
        f'-time {iteration:d} > log.sample 2>&1'
    subprocess.call(bash_command, shell=True)
