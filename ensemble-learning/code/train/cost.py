
""" 
Collection of different ways to calculate cost and gradient.
Current options: adjoint or ensemble.
"""

import os
import shutil
import subprocess
import multiprocessing
import time

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import Rbf

import dafi.random_field as rf
import ensemble_gradient as ensemble
from get_inputs import get_inputs

SCALARDIM = 1
TENSORDIM = 9
VECTORDIM = 3
DEVSYMTENSORDIM = 5
DEVSYMTENSOR_INDEX = [0,1,2,4,5]


def _clean_foam(foam_dir):
    bash_command = './clean > /dev/null'
    bash_command = 'cd ' + foam_dir + ';' + bash_command + '; cd -'
    return subprocess.call(bash_command, shell=True, stdout=subprocess.DEVNULL)


def _run_foam(foam_dir):
    bash_command = './run > /dev/null'
    bash_command = 'cd ' + foam_dir + ';' + bash_command + '; cd -'
    return subprocess.call(bash_command, shell=True, stdout=subprocess.DEVNULL)


class Cost(object):

    def __init__(self, flow, nbasis, restart=None, restart_dir=None):
        # files and directories
        self.foam_dir = flow['foam_dir']
        self.foam_timedir = str(flow['foam_timedir'])
        self.U_file = os.path.join(self.foam_dir, self.foam_timedir, 'U')
        self.p_file = os.path.join(self.foam_dir, self.foam_timedir, 'p')
        self.gradU_file = os.path.join(self.foam_dir, self.foam_timedir, 'grad(U)')
        self.tke_file = os.path.join(self.foam_dir, self.foam_timedir, 'k')
        self.time_scale_file = os.path.join(
            self.foam_dir, self.foam_timedir, 'timeScale')

        # read OF fields: mesh
        self.cell_volumes = rf.foam.get_cell_volumes(self.foam_dir)
        self.volume = np.sum(self.cell_volumes)
        self.coordinates = rf.foam.get_cell_centres(self.foam_dir)
        self.ncells = len(self.cell_volumes)
        self.connectivity = rf.foam.get_neighbors(self.foam_dir)

        # problem dimensions
        self.nvar = DEVSYMTENSORDIM * self.ncells

        # flow variables
        if (restart is None) or (restart=='pretrain'):
            self.gradU = np.zeros([self.ncells, TENSORDIM])
            self.tke = np.zeros([self.ncells, SCALARDIM])
            self.time_scale = np.zeros([self.ncells, SCALARDIM])
        else: 
            post = f".{flow['name']}.{restart}"
            self.gradU = np.loadtxt(os.path.join(restart_dir, 'gradU' + post))
            self.tke = np.loadtxt(os.path.join(restart_dir, 'tke' + post))
            self.time_scale = np.loadtxt(os.path.join(restart_dir, 'timeScale' + post))

        # read OF fields: tensor function basis
        self.nbasis = nbasis
        self.g_data_list=[]
        for ibasis in range(nbasis):
            g_file = os.path.join(self.foam_dir, '0.orig', f'g{ibasis+1}.orig')
            g_data = rf.foam.read_field_file(g_file)
            g_data['file'] = os.path.join(self.foam_dir, '0.orig', f'g{ibasis+1}')
            self.g_data_list.append(g_data)

        # clean openfoam case just in case
        _clean_foam(self.foam_dir)

        # measurements
        self.nobs = 0
        for i, imeasurement in enumerate(flow['measurements']):
            # load data
            data = np.loadtxt(imeasurement['training_data'])
            if imeasurement['observation_type'] == 'fullfield':
                imeasurement['training_data'] = data
                self.nobs += self.ncells
            elif imeasurement['observation_type'] == 'point':
                imeasurement['point'] = data[:3]
                imeasurement['training_data'] = data[3]
                self.nobs += 1
            flow['measurements'][i] = imeasurement
        self.flow = flow

        # iteration count
        if (restart is None) or (restart=='pretrain'):
            self.iteration = 0
        else:
            self.iteration = restart
        
        # start each new RANS solve from previous solution
        self.start_from_prev = flow.get('start_from_prev', False)

        # copy initial files
        self.turb_quant = flow.get('turb_quant', '')
        self.copyfilenames = ['U', 'p', 'k', self.turb_quant, 'nut']
        for filename in self.copyfilenames:
            file = os.path.join(self.foam_dir, '0.orig', filename)
            shutil.copyfile(file+'.orig', file)


    def _modify_foam_case(self, g, foam_dir=None):
        for i, g_data in enumerate(self.g_data_list):
            g_data['internal_field']['value'] = g[:, i]
            if foam_dir is not None:
                g_data['file'] = os.path.join(foam_dir, '0.orig', f'g{i+1}')
            _ = rf.foam.write_field_file(**g_data)


    def cost(self, g):
        # return J, dJda, cost_vars
        pass


class CostAdjoint(Cost):

    def __init__(self, flow, nbasis, restart=None, restart_dir=None):
        super(self.__class__, self).__init__(flow, nbasis, restart, restart_dir)
        
        # files and directories
        self.foam_dir_a = flow['gradient_options']['foam_dir_a']
        self.foam_timedir_a = str(flow['gradient_options']['foam_timedir_a'])
        self.sensitivity_file = os.path.join(
            self.foam_dir_a, self.foam_timedir_a, 'sensitivity')
        self.logfile_p = os.path.join(self.foam_dir, 'log.primal')
        self.logfile_a = os.path.join(self.foam_dir_a, 'log.adjoint')
        log_dir = flow['gradient_options']['log_dir']
        self.logdir_p = os.path.join(log_dir, 'primal')
        self.logdir_a = os.path.join(log_dir, 'adjoint')
        def mkdir(dir):
            try:
                os.makedirs(dir)
            except:
                pass
        mkdir(self.logdir_p)
        mkdir(self.logdir_a) 

        # read OpenFOAM fields
        forcing_file = os.path.join(self.foam_dir_a, '0.orig', 'UForcing.orig')
        self.forcing_data_U = rf.foam.read_field_file(forcing_file)
        self.forcing_data_U['file'] = os.path.join(
            self.foam_dir_a, '0.orig', 'UForcing')

        forcing_file = os.path.join(self.foam_dir_a, '0.orig', 'pForcing.orig')
        self.forcing_data_p = rf.foam.read_field_file(forcing_file)
        self.forcing_data_p['file'] = os.path.join(
            self.foam_dir_a, '0.orig', 'pForcing')

        U_file_a = os.path.join(self.foam_dir_a, '0.orig', 'U.orig')
        self.U_data_a = rf.foam.read_field_file(U_file_a)
        self.U_data_a['file'] = os.path.join(self.foam_dir_a, '0.orig', 'U')

        # measurement masks
        flow['measurements_fullfield'] = []
        flow['measurements_points'] = []
        for i, imeasurement in enumerate(flow['measurements']):
            if imeasurement['type'] == 'boundary':
                raise NotImplementedError(
                    f"'boundary' measurements not implemented")
            # create observation mask
            if imeasurement['observation_type'] == 'fullfield':
                flow['measurements_fullfield'].append(imeasurement)
            elif imeasurement['observation_type'] == 'point':
                # mask
                mask = np.ones(self.ncells)
                for j in range(VECTORDIM):
                    dx = (self.coordinates[:, j] - imeasurement['point'][j]) / \
                         imeasurement['point_mask_length'][j]
                    mask *= np.exp(-1.0 * (dx**2))
                mask /= rf.integral(mask, self.cell_volumes)
                imeasurement['observation_mask'] = mask
                # H matrix
                imeasurement['Hmat'] = rf.inverse_distance_weights(self.coordinates, self.connectivity, np.atleast_2d(np.array(imeasurement['point'])))
                flow['measurements_points'].append(imeasurement)
            elif imeasurement['observation_type'] == 'integral':
                raise NotImplementedError('Integral measurements not implemented.') 
        self.flow = flow


    def cost(self, g):
        # run primal
        self._modify_foam_case(g)
        ts = time.time()
        success = _run_foam(self.foam_dir)
        if success != 0:
            raise RuntimeError("OpenFOAM primal run not succesful. ")
        print(f'      Primal solve time:  {time.time()-ts} s')
        file = os.path.join(
            self.logdir_p, os.path.basename(self.logfile_p)+f'.{self.iteration}')
        shutil.move(self.logfile_p, file)
        U = rf.foam.read_vector_field(self.U_file)
        p = rf.foam.read_scalar_field(self.p_file)
        gradU = rf.foam.read_tensor_field(self.gradU_file)
        tke = rf.foam.read_scalar_field(self.tke_file)
        time_scale = rf.foam.read_scalar_field(self.time_scale_file)
        if self.start_from_prev:
            for filename in self.copyfilenames:
                src = os.path.join(self.foam_dir, self.foam_timedir, filename)
                dst = os.path.join(self.foam_dir, '0.orig', filename)
                shutil.copyfile(src, dst) 
        _clean_foam(self.foam_dir)
        # intermediate variables to save
        cost_vars = {'U': U, 'p':p, 'gradU':gradU, 'tke':tke, 'timeScale':time_scale}

        J = 0.0
        dJda = np.zeros([1, self.nvar])
        iadj = 0

        ## FULLFIELD ##
        # adjoint is run only once
        # cost and adjoint forcing
        forcing_volume_U = np.zeros([self.ncells, VECTORDIM])
        forcing_volume_p = np.zeros([self.ncells, SCALARDIM])
        variables_volume = {'Ux': U[:, 0], 'Uy': U[:, 1], 'Uz': U[:, 2], 'p': p}
        fullfield = False
        for imeasurement in self.flow['measurements_fullfield']:
            fullfield = True
            # cost
            if imeasurement['type'] == 'volume':
                var = variables_volume[imeasurement['variable']]
            elif imeasurement['type'] == 'boundary':
                raise NotImplementedError(
                    f"'boundary' measurements not implemented")
            var_diff = var - imeasurement['training_data']
            J += 0.5 * imeasurement['scaling'] * rf.integral(var_diff**2, self.cell_volumes)
            # adjoint forcing
            forcing = imeasurement['scaling'] * var_diff
            if imeasurement['variable']=='Ux':
                forcing_volume_U[:, 0] += forcing
            elif imeasurement['variable']=='Uy':
                forcing_volume_U[:, 1] += forcing
            elif imeasurement['variable']=='Uz':
                forcing_volume_U[:, 2] += forcing
            elif imeasurement['variable']=='p':
                forcing_volume_p[:, 0] += forcing
        # adjoint
        if fullfield:
            dJda += self.run_adjoint(forcing_volume_U, forcing_volume_p, U, iadj)
            iadj += 1

        ## POINTS ##
        # adjoint is run once per measurement
        for imeasurement in self.flow['measurements_points']:
            forcing_volume_U = np.zeros([self.ncells, VECTORDIM])
            forcing_volume_p = np.zeros([self.ncells, SCALARDIM])
            variables_volume = {'Ux': U[:, 0], 'Uy': U[:, 1], 'Uz': U[:, 2], 'p': p}
            # cost
            if imeasurement['type'] == 'volume':
                var = variables_volume[imeasurement['variable']] 
            elif imeasurement['type'] == 'boundary':
                raise NotImplementedError(
                    f"'boundary' measurements not implemented")
            var_diff = float(imeasurement['Hmat'].dot(var) - imeasurement['training_data'])
            J += 0.5 * imeasurement['scaling'] * var_diff**2
            # adjoint forcing
            forcing = imeasurement['observation_mask'] * var 
            if imeasurement['variable']=='Ux':
                forcing_volume_U[:, 0] += forcing
            elif imeasurement['variable']=='Uy':
                forcing_volume_U[:, 1] += forcing
            elif imeasurement['variable']=='Uz':
                forcing_volume_U[:, 2] += forcing
            elif imeasurement['variable']=='p':
                forcing_volume_p[:, 0] += forcing
            # adjoint
            dJda += self.run_adjoint(forcing_volume_U, forcing_volume_p, U, iadj) * imeasurement['scaling'] * var_diff
            iadj += 1

        self.iteration += 1
        return J, dJda, cost_vars
   
    
    def run_adjoint(self, forcing_volume_U, forcing_volume_p, U, iadj):
        self.forcing_data_U['internal_field']['value'] = forcing_volume_U
        _ = rf.foam.write_field_file(**self.forcing_data_U)
        self.forcing_data_p['internal_field']['value'] = forcing_volume_p
        _ = rf.foam.write_field_file(**self.forcing_data_p)
        self.U_data_a['internal_field']['value'] = U
        _ = rf.foam.write_field_file(**self.U_data_a)
        ts = time.time()
        success = _run_foam(self.foam_dir_a)
        if success != 0:
            raise RuntimeError("OpenFOAM adjoint run not succesful. ")
        print(f'      Adjoint solve time:  {time.time()-ts} s')
        file = os.path.join(
            self.logdir_a, os.path.basename(self.logfile_a)+f'.{iadj}.{self.iteration}')
        shutil.move(self.logfile_a, file)
        dJda_all = rf.foam.read_tensor_field(self.sensitivity_file)
        _clean_foam(self.foam_dir_a)

        # gradient
        dJda = np.zeros([self.ncells, DEVSYMTENSORDIM])
        dJda[:, 0] = dJda_all[:, 0] - dJda_all[:, 8]
        dJda[:, 1] = dJda_all[:, 1] + dJda_all[:, 3]
        dJda[:, 2] = dJda_all[:, 2] + dJda_all[:, 6]
        dJda[:, 3] = dJda_all[:, 4] - dJda_all[:, 8]
        dJda[:, 4] = dJda_all[:, 5] + dJda_all[:, 7]
        dJda = dJda.reshape([1, self.nvar])

        return dJda


    def clean(self, ):
        remove = lambda x : os.remove(os.path.join(self.foam_dir, '0.orig', x))
        remove('U')
        remove('p')
        remove('k')
        remove(self.turb_quant)
        remove('nut')


class CostEnsemble(Cost):

    def __init__(self, flow, nbasis, restart=None, restart_dir=None):
        super(self.__class__, self).__init__(flow, nbasis, restart, restart_dir)

        self.ncpu = flow['gradient_options']['ncpu']
        self.nsamples = flow['gradient_options']['nsamples']
        lenscale = flow['gradient_options']['lenscale']
        self.stddev_ratio = flow['gradient_options']['stddev_ratio']
        ensemble_method = flow['gradient_options']['ensemble_method']
        self.eps = flow['gradient_options']['eps']
        self.baseline_as_mean = flow['gradient_options'].get('baseline_as_mean', False)
        self.results_dir = flow['gradient_options']['results_dir'] 
        self.stddev_type = flow['gradient_options']['stddev_type']
        self.neumann_boundaries_coordinates = flow['neumann_boundaries_coordinates']
        self.inv_perturb = flow.get('inv_perturb', 0.0)

        # results folder
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # create ensemble folders
        self.ensemble_dir = '_samples'
        try:
            shutil.rmtree(self.ensemble_dir)
        except:
            pass
        os.makedirs(self.ensemble_dir)
        sample_dirs = []
        for isample in range(self.nsamples):
            sample_dir = os.path.join(self.ensemble_dir, f'sample_{isample}')
            sample_dirs.append(sample_dir)
            shutil.copytree(self.foam_dir, sample_dir)
        self.sample_dirs = sample_dirs

        # create covariance matrix
        coverage = 0.99
        cov = rf.covariance.generate_cov(kernel = 'sqrexp',
                                         stddev = 1.0,
                                         sp_tol = 1e-8,
                                         coords = self.coordinates,
                                         length_scales = [lenscale]*VECTORDIM)
        # boundary conditions 
        if self.neumann_boundaries_coordinates is not None:
            # read the locations of the cell faces 
            coordinates_b = np.atleast_2d(np.loadtxt(self.neumann_boundaries_coordinates))
            nb = len(coordinates_b)
            # volume-boundary covariance
            exp = np.zeros([self.ncells, nb])
            for idim in range(VECTORDIM):
                dx= np.subtract.outer(self.coordinates[:, idim], coordinates_b[:, idim])
                exp += (dx / lenscale)**2
            sqrexp = np.exp(-0.5*exp)
            diff = np.zeros([self.ncells, nb])
            for m in range(self.ncells):
                for n in range(nb):
                    i = int(coordinates_b[n, 3])
                    diff[m, n] = (coordinates_b[n, i] - self.coordinates[m, i])
            cov_vb = (1.0/lenscale**2) * np.multiply(sqrexp, diff)

            # boundary-boundary covariance 
            cov_bb =  rf.covariance.generate_cov(kernel = 'sqrexp', 
                                                 stddev = 1.0,
                                                 sp_tol = 1e-8,
                                                 coords = coordinates_b[:, :3],
                                                 length_scales = [lenscale]*VECTORDIM)
            mult = np.zeros([nb, nb])
            for m in range(nb):
                i = int(coordinates_b[m, 3])
                for n in range(nb):
                    j = int(coordinates_b[n, 3])
                    quant = -(1.0/lenscale**2) * (coordinates_b[m, i] - coordinates_b[n, i]) * (coordinates_b[m, j] - coordinates_b[n, j])
                    if i==j:
                        quant += 1.0 
                    mult[m, n] = (1.0/lenscale**2) * quant
            cov_bb = np.multiply(mult, cov_bb.toarray()) 
            # update covariance
            cov -= cov_vb @ np.linalg.inv(cov_bb+np.eye(nb)*self.inv_perturb) @ cov_vb.T
        _, klmodes = rf.calc_kl_modes_coverage(cov,
                                   coverage = coverage,
                                   weight_field = self.cell_volumes,
                                   eps = 1e-8,
                                   max_modes = None,
                                   normalize = False)
        self.klmodes = klmodes
        self.nmodes = self.klmodes.shape[1]
        np.savetxt(os.path.join(self.results_dir, 'klmodes'), self.klmodes)

        # ensemble gradient
        self.ensemble_gradient = getattr(ensemble, ensemble_method)
        self.cell_volumes_devsymtensor = np.repeat(self.cell_volumes, DEVSYMTENSORDIM)


    def cost(self, g):
        # create samples for each g-function
        coeffs = np.random.normal(0.0, 1.0, [self.nmodes, self.nsamples])
        gsamps = []
        for i in range(self.nbasis):
            igmean = np.expand_dims(g[:, i], axis=-1)
            # TODO: stddev
            if self.stddev_type == 'fixed':
                igstddev = self.stddev_ratio
                igperturb = igstddev * rf.field.reconstruct_kl(self.klmodes,  coeffs)
            elif self.stddev_type == 'vector':
                igstddev = self.stddev_ratio * np.abs(igmean)
                igperturb = igstddev * rf.field.reconstruct_kl(self.klmodes,  coeffs)
            elif self.stddev_type == 'norm':
                igstddev = self.stddev_ratio * rf.norm(igmean, self.cell_volumes) / np.sqrt(self.volume)
                igperturb = igstddev * rf.field.reconstruct_kl(self.klmodes,  coeffs)
            elif self.stddev_type == 'new':
                igstddev = self.stddev_ratio
                igperturb = igmean * rf.field.reconstruct_kl(self.klmodes, igstddev * coeffs)
            # import pdb; pdb.set_trace()
            # igperturb = igstddev * rf.field.reconstruct_kl(self.klmodes,  coeffs)
            igsamp = igmean + igperturb
            gsamps.append(igsamp)
            np.savetxt(os.path.join(self.results_dir, f"{self.flow['name']}.samps_g_{i}_iter_{self.iteration}"), igsamp)
        # write sample 
        for i in range(self.nsamples):
            ig = np.zeros(g.shape)
            for j in range(self.nbasis):
                ig[:, j] = gsamps[j][:, i]
            self._modify_foam_case(ig, foam_dir=self.sample_dirs[i])
        
        # run OpenFOAM in parallel
        parallel = multiprocessing.Pool(self.ncpu)
        _ = parallel.map(_run_foam, self.sample_dirs)
        parallel.close()

        # collect results
        Ux_samp = np.zeros([self.ncells, self.nsamples])
        Uy_samp = np.zeros([self.ncells, self.nsamples])
        Uz_samp = np.zeros([self.ncells, self.nsamples])
        p_samp = np.zeros([self.ncells, self.nsamples])
        Tau_samp = np.zeros([self.ncells * DEVSYMTENSORDIM, self.nsamples])
        for isamp, sampledir in enumerate(self.sample_dirs):
            Ufile = os.path.join(sampledir, self.foam_timedir, 'U')
            U_samp = rf.foam.read_vector_field(Ufile)
            Ux_samp[:, isamp] = U_samp[:,0]
            Uy_samp[:, isamp] = U_samp[:,1]
            Uz_samp[:, isamp] = U_samp[:,2]
            pfile = os.path.join(sampledir, self.foam_timedir, 'p')
            p_samp[:, isamp] = rf.foam.read_scalar_field(pfile)
            # get Reynolds stress ensemble
            gradU_file = os.path.join(sampledir, self.foam_timedir, 'grad(U)')
            gradU = rf.foam.read_tensor_field(gradU_file)
            tke_file = os.path.join(sampledir, self.foam_timedir, 'k')
            tke = rf.foam.read_scalar_field(tke_file)
            time_scale_file = os.path.join(sampledir, self.foam_timedir, 'timeScale')
            time_scale = rf.foam.read_scalar_field(time_scale_file)
            _, tensor_basis = get_inputs(gradU, time_scale)
            tensor_basis = tensor_basis[:, :, :self.nbasis]
            gT = np.expand_dims(g, axis=1) * tensor_basis
            kgT = 2.0 *np.expand_dims(tke, axis=(-1, -2)) * gT 
            Tau_samp[:, isamp] = np.sum(kgT.reshape(-1, self.nbasis), axis=1)
            _clean_foam(sampledir)
        samps = {'Ux': Ux_samp, 'Uy': Uy_samp, 'Uz': Uz_samp, 'p': p_samp}

        # run baseline case
        self._modify_foam_case(g, self.foam_dir)
        _run_foam(self.foam_dir)
        U_bl = rf.foam.read_vector_field(self.U_file)
        p_bl = rf.foam.read_scalar_field(self.p_file)
        gradU_bl = rf.foam.read_tensor_field(self.gradU_file)
        tke_bl = rf.foam.read_scalar_field(self.tke_file)
        time_scale_bl = rf.foam.read_scalar_field(self.time_scale_file)
        
        # Start from previous baseline solution
        if self.start_from_prev:
            for sample_dir in self.sample_dirs:
                for filename in self.copyfilenames:
                    src = os.path.join(
                        self.foam_dir, self.foam_timedir, filename)
                    dst = os.path.join(sample_dir, '0.orig', filename)
                    shutil.copyfile(src, dst)

        _clean_foam(self.foam_dir)
        
        baseline = {'Ux': U_bl[:, 0], 'Uy': U_bl[:, 1], 'Uz': U_bl[:, 2], 'p': p_bl}
        _, tensor_basis = get_inputs(gradU_bl, time_scale_bl)
        tensor_basis = tensor_basis[:, :, :self.nbasis]
        gT = np.expand_dims(g, axis=1) * tensor_basis
        kgT = 2.0 *np.expand_dims(tke_bl, axis=(-1, -2)) * gT
        Tau_bl = np.sum(kgT.reshape(-1, self.nbasis), axis=1)

        # cost and gradient
        J = 0 
        dJda = np.zeros([self.ncells * DEVSYMTENSORDIM, 1]) 
        for i, imeasurement in enumerate(self.flow['measurements']):
            # Create the observation ensemble and R matrix 
            if imeasurement['observation_type'] == 'fullfield':
                H_samp = samps[imeasurement['variable']]
                H_bl = baseline[imeasurement['variable']]
                if imeasurement['measurement_stddev'] is None:
                    Rinv = np.diag(self.cell_volumes)
                elif imeasurement['measurement_stddev'] == 'norm':
                    mean_cell_volume = self.volume / self.ncells
                    Rinv = np.diag(self.cell_volumes) / mean_cell_volume 
                else: 
                    Rinv = np.eye(self.ncells) * imeasurement['measurement_stddev'] 
            elif imeasurement['observation_type'] == 'point':
                # TODO: Use sparse matrix (single matrix) rather than creating an interpolation for every sample at every step.
                data = samps[imeasurement['variable']]
                Rinv = np.array([[ 1.0/(imeasurement['measurement_stddev']**2) ]])
                H_samp = np.zeros([1, self.nsamples])
                for jsamp in range(self.nsamples):
                    rbf = Rbf(self.coordinates[:,0], self.coordinates[:,1], self.coordinates[:,2], data[:, jsamp])
                    H_samp[0, jsamp] = rbf(imeasurement['point'][0], imeasurement['point'][1], imeasurement['point'][2])
                data = baseline[imeasurement['variable']]
                rbf = Rbf(self.coordinates[:,0], self.coordinates[:,1], self.coordinates[:,2], data)
                H_bl = rbf(imeasurement['point'][0], imeasurement['point'][1], imeasurement['point'][2]) 
            else:
                raise NotImplementedError("Only 'fullfield' and 'point' measurements implemented.")
            np.savetxt(os.path.join(self.results_dir, f"{self.flow['name']}.samps_H_measurement_{i}_iter_{self.iteration}"), H_samp)
            # cost and gradient
            innovation = H_bl - imeasurement['training_data']
            J += 0.5 * imeasurement['scaling'] * innovation.T @ Rinv @ innovation
            dJda += self.ensemble_gradient(Tau_samp, H_samp, Tau_bl, H_bl, imeasurement['training_data'], Rinv, weights=self.cell_volumes_devsymtensor, baseline_as_mean=self.baseline_as_mean, eps=self.eps)
        J = J.squeeze()
        dJda = dJda.T

        cost_vars = {'U': U_bl, 'p':p_bl, 'gradU':gradU_bl, 'tke':tke_bl, 'timeScale':time_scale_bl}
        self.iteration += 1
        return J, dJda, cost_vars

    def clean(self, ):
        # remove sample directories
        shutil.rmtree(self.ensemble_dir)
        # clean baseline flow
        remove = lambda x : os.remove(os.path.join(self.foam_dir, '0.orig', x))
        remove('U')
        remove('p')
        remove('k')
        remove(self.turb_quant)
        remove('nut')
