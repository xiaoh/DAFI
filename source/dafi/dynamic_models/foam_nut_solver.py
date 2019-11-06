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
from dafi.dyn_model import DynModel
import dafi.utilities as util
import dafi.foam_utilities as foam
import dafi.random_field as rf


class Solver(DynModel):
    """ Dynamic model for OpenFoam Reynolds stress nutFoam solver.

    The eddy viscosity field (nu_t) is infered by observing the
    velocity field (U). Nut is modeled as a random field with lognormal
    distribution and median value equal to the baseline (prior) nut
    field.
    """

    def __init__(self, nsamples, da_interval, t_end, max_da_iteration,
                 model_input):
        """ Initialize the nutFOAM solver.

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
        # save the main inputs. ``da_interval`` and ``t_end`` not used
        self.nsamples = nsamples
        self.max_da_iteration = max_da_iteration

        # read input file and set defaults
        param_dict = util.read_input_data(model_input)
        self.forward_interval = int(param_dict['forward_interval'])
        self.dir_foam_base = param_dict['foam_base_dir']
        dir_mesh = param_dict['mesh_dir']
        obs_file = param_dict['obs_file']
        obs_err_file = param_dict['obs_err_file']
        kl_modes_file = param_dict['kl_modes_file']
        obs_mat_file = param_dict['obs_mat_file']
        if 'nkl_modes' in param_dict:
            nkl_modes = int(param_dict['nkl_modes'])
        else:
            nkl_modes = None  # set later to max
        if 'enable_parallel' in param_dict:
            enable_parallel = util.str2bool(param_dict['enable_parallel'])
        else:
            enable_parallel = False
        if enable_parallel:
            if not has_parallel:
                err_message = 'Parallel Python (pp) not loaded succesfully.'
                raise RuntimeError(err_message)
            if 'ncpu' in param_dict:
                self.ncpu = int(param_dict['ncpu'])
            else:
                self.ncpu = 0
        else:
            self.ncpu = 1
        if 'transform' in param_dict:
            transform = param_dict['transform']
        else:
            transform = 'linear'

        # read baseline nut field
        baseline_file = os.path.join(self.dir_foam_base, '0', 'nut')
        nut_baseline = foam.read_scalar_from_file(baseline_file)
        self.ncells = len(nut_baseline)

        # read mesh
        foam_coords = foam.read_cell_coordinates(dir_mesh)
        foam_volume = foam.read_cell_volumes(dir_mesh)

        # read observations
        self.obs = np.loadtxt(obs_file)
        self.obs_error = np.loadtxt(obs_err_file)
        self.nstate_obs = len(self.obs)

        # create "H matrix"
        obs_mat = np.loadtxt(obs_mat_file)
        self.obs_vel2obs = _construct_obsmat_vec(
            obs_mat, self.nstate_obs, self.ncells)

        # read KL modes
        if nkl_modes is not None:
            kl_modes = np.loadtxt(kl_modes_file, usecols=range(nkl_modes))
        else:
            kl_modes = np.loadtxt(kl_modes_file)
            self.nkl_modes = kl_modes.shape[1]

        # initiliaze the Gaussian process
        if transform == 'linear':
            # nut - nut_baseline ~ GP
            # baseline=mean
            def lin(x):
                return x
            self.transform = lin
            self.transform_inv = lin
        elif transform == 'log':
            # ln(nut) _ ln(nut_baseline) = ln(nut / nut_baseline) ~ GP
            # baseline=median
            self.transform = np.log
            self.transform_inv = np.exp
        self.delta_nut_rf = rf.GaussianProcess(
            name='nut', zero_mean=True, coords=foam_coords,
            kl_modes=kl_modes, weight_field=foam_volume)
        self.baseline = self.transform(nut_baseline)
        self.baseline_mat = np.tile(self.baseline, [self.nsamples, 1]).T

        # required attributes for DAFI
        self.name = 'FoamNutSolver'
        self.nstate = nkl_modes
        self.nstate_obs = len(self.obs)
        self.init_state = np.zeros(self.nstate)

        # other initialization
        self.da_step = 0
        self.foam_solver = 'nutFoam'
        self.dir_results = 'results_nutfoam'

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
            sample_dir = os.path.join('sample_{:d}'.format(isample + 1))
            shutil.copytree(self.dir_foam_base, sample_dir)
            control_dict = os.path.join(sample_dir, "system", "controlDict")
            util.replace(control_dict, "<endTime>", str(max_end_time))
            util.replace(control_dict, "<writeInterval>",
                         str(self.forward_interval))
        # update X (nut)
        delta_nut, coeffs = self.delta_nut_rf.sample_kl_reduced(
            self.nsamples, self.nstate, return_coeffs=True)
        nut = self.transform_inv(self.baseline_mat + delta_nut)
        self._modify_openfoam_scalar('nut', nut, '0')
        # map to HX (U)
        model_obs = self.state_to_observation(None, False)
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
    def _modify_openfoam_scalar(self, field_name, values, timedir):
        """ Replace the values of a specific field in all samples.

        Parameters
        ----------
        field_name : str
            The field file to modify.
        values : ndarray
            New values for the field.
            ``dtype=float``, ``ndim=2``, ``shape=(ncells, nsamples)``
        timedir : str
            The time directory in which the field should be modified.
        """
        for isample in range(self.nsamples):
            sample_dir = os.path.join('sample_{:d}'.format(isample + 1))
            field_file = os.path.join(sample_dir, timedir, field_name)
            foam.write_scalar_to_file(values[:, isample], field_file)

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
