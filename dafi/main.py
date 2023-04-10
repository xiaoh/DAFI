# Copyright 2020 Virginia Polytechnic Institute and State University.
""" DAFI: a Python package for data assimilaton and field inversion.
"""

# standard library imports
import importlib
import logging
import warnings
import os
import time as tm

import sys

# third party imports
import numpy as np

# local imports
# user-specified physics model imported later with importlib
# user-specified inverse method imported later with importlib


def run(model_file, inverse_method, nsamples, ntime=None,
        perturb_obs_option='iter', obs_err_multiplier=1.0,
        analysis_to_obs=False, convergence_option='max', max_iterations=1,
        convergence_residual=None, convergence_factor=1.0, save_level=None,
        save_dir='results_dafi', rand_seed=None, verbosity=0,
        inputs_inverse={}, inputs_model={}):
    """ Run DAFI.

    Accesible through ``dafi.run()``.

    Parameters
    ----------
    model_file: str
        Name (path) of dynamic model module/package.
    inverse_method: str
        Name of inverse method from dafi.inverse_methods module.
    nsamples : int
        Number of samples in ensemble.
    ntime : int
        Number of data assimilation times. For stationary use *'1'* or
        *'None'*. Default *'None'*.
    perturb_obs_option: str
      Option on when to perturb observations:
      *'iter'* to perturb at each iteration (inner loop),
      *'time'* to perturb only once each data assimilaton time
      (outer loop),
      *'None'* to not perturb observations. Default *'iter'*.
    obs_err_multiplier: float
      Factor by which to multiply the observation error matrix.
      This is done by some authors in the literature. Default *'1.0'*.
    analysis_to_obs: bool
      Map analysis state to observation space. Default *'False'*.
    convergence_option: str
      Convergence criteria to use: *'discrepancy'* to use the
      discrepancy principle,  *'residual'* to use the iterative
      residual, *'max'* to reach the maximum iterations. Default *'max'*.
    max_iterations: int
      Maximum number of iterations at a given time-step. Default *'1'*.
    convergence_residual: float
      Residual value for convergence if *reach_max* is *'False'* and
      *convergence_option* is *'residual'*. Default *'None'*.
    convergence_factor: float
      Factor used in the discrepancy principle convergence option.
      Default *'1.0'*.
    save_level : str
        Level of results to save: *'None'* to not save results,
        *'time'* to save results at each data assimilation time step,
        *'iter'* to save results at each iteration,
        *'debug'* to save additional intermediate quantities.
        Default *'None'*.
    save_dir: str
      Folder where to save results. Default *'./results_dafi'*.
    rand_seed : float
        Seed for numpy.random. If None random seed not set.
        Default *'None'*.
    verbosity: int
        Logging verbosity level, between -1 and 9 (currently -1-3 used).
        For no logging use *'-1'*. For debug-level logging use
        *'debug'*. Default *'0'*.
    inputs_inverse : dict
        Inverse method specific inputs.
        Default empty dictionary *'{}'*.
    inputs_model : dict
        Physics model specific inputs.
        Default empty dictionary *'{}'*.

    Returns
    -------
    state_analysis_list : list
        List of analysis states at each DA time. Length is number of
        DA time steps. Each entry is an nd.array containing the ensemble
        analysis states (:math:`x_a`) at that time step.
        If only one DA time (e.g. stationary, inversion problem) is the
        single ndarray. Each entry is: *dtype=float*, *ndim=2*,
        *shape=(nstate, nsamples)*.
    """
    # collect inputs
    # the inverse method and physics model are allowed to modify these
    inputs_dafi = {
        'model_file': model_file,
        'inverse_method': inverse_method,
        'nsamples': nsamples,
        'ntime': ntime,
        'perturb_obs_option': perturb_obs_option,
        'obs_err_multiplier': obs_err_multiplier,
        'analysis_to_obs': analysis_to_obs,
        'convergence_option': convergence_option,
        'max_iterations': max_iterations,
        'convergence_residual': convergence_residual,
        'convergence_factor': convergence_factor,
        'save_level': save_level,
        'save_dir': save_dir,
        'rand_seed': rand_seed,
        'verbosity': verbosity,
    }

    # configure logger
    if verbosity == 'debug':
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(message)s', level=_log_level(verbosity))
    logger = logging.getLogger(__name__)

    # random seed
    # this is done before importing local modules that use np.random
    if rand_seed is not None:
        np.random.seed(rand_seed)

    # stationary problem
    if inputs_dafi['ntime'] is None:
        inputs_dafi['ntime'] = 1

    # create save directory
    if save_level != None:
        _create_dir(save_dir)

    # initialize physics model
    sys.path.append(os.path.dirname(os.path.abspath(model_file)))
    model_module = os.path.basename(os.path.splitext(model_file)[0])
    Model = getattr(
        importlib.import_module(model_module), 'Model')
    model = Model(inputs_dafi, inputs_model)

    # initialize inverse method
    Inverse = getattr(
        importlib.import_module('dafi.inverse'), inverse_method)
    inverse = Inverse(inputs_dafi, inputs_inverse)

    # log
    log_message = "Solving the inverse problem" + \
        f"\nPhysics Model:  {model.name}" + \
        f"\nInverse Method: {inverse.name}"
    logger.log(_log_level(0), log_message)

    # solve
    state_analysis_list = _solve(inputs_dafi, inverse, model)

    # log
    logger.log(_log_level(0), 'Done.')

    if inputs_dafi['ntime'] == 1:
        state_analysis_list = state_analysis_list[0]
    return state_analysis_list


def _solve(inputs_dafi, inverse, model):
    """ Solve the inverse problem.

    Implements the general inverse problem consisting of an outer (time)
    loop and inner (iteration) loop.
    """
    logger = logging.getLogger(__name__ + '._solve')
    # time and iteration arrays
    time_array = np.arange(inputs_dafi['ntime'], dtype=int)
    iteration_array = np.arange(inputs_dafi['max_iterations'], dtype=int)

    # initial ensemble
    state_forecast = model.generate_ensemble()

    # outer loop - time marching
    state_analysis_list = []
    for time in time_array:
        log_message = f"\nData assimilation step: {time}"
        logger.log(_log_level(1), log_message)

        # dynamic model - propagate the state ensemble to current DA time.
        if time != 0:
            state_forecast = model.forecast_to_time(state_analysis, time)

        # get observations at current DA time
        obs_vec, obs_error = model.get_obs(time)
        obs_error *= inputs_dafi['obs_err_multiplier']
        obs = np.tile(obs_vec, (inputs_dafi['nsamples'], 1)).T
        if inputs_dafi['perturb_obs_option'] == 'time':
            obs, obs_perturbation = _perturb_vec(
                obs_vec, obs_error, inputs_dafi['nsamples'])

        # prior state for iterative methods that require it
        state_prior = state_forecast

        # inner loop - iterative analysis at fixed DA time.
        misfit_list = []
        noise_list = []
        residual_list = []
        tdir = os.path.join(inputs_dafi['save_dir'], f"t_{time}")
        early_stop = False
        for iteration in iteration_array:
            # log
            log_message = f"\n  Iteration: {iteration}"
            logger.log(_log_level(2), log_message)

            ts = tm.time()
            # map the state vector to observation space
            if iteration != 0:
                state_forecast = state_analysis.copy()
            state_in_obsspace = model.state_to_observation(state_forecast)
            print(f'      Ensemble of forecast ... {tm.time()-ts:.2f}s')

            if iteration == 0:
                state_in_obsspace_prior = state_in_obsspace

            # perturb observations
            if inputs_dafi['perturb_obs_option'] == 'iter':
                obs, obs_perturbation = _perturb_vec(
                    obs_vec, obs_error, inputs_dafi['nsamples'])

            # data assimilaton
            ts = tm.time()
            # inflation
            if inverse.inflation_flag:
                if iteration == 0:
                    inverse.gamma = inverse.gamma
                elif iteration == 1:
                    coeff = 1.0 / np.sqrt(inputs_dafi['nsamples'] - 1.0)
                    Sd = coeff * (state_in_obsspace - state_in_obsspace.mean(axis=1,keepdims=1))
                    Sd_norm = np.sqrt(np.linalg.inv(obs_error)).dot(Sd)
                    inverse.gamma *= np.trace(Sd_norm.dot(Sd_norm.T)) / len(obs_vec)
                elif iteration != 1:
                    inverse.gamma = inverse.gamma * inverse.beta
            # else:
                # inverse.gamma = 1
            print(inverse.gamma)
            state_analysis = inverse.analysis(
                iteration, state_forecast, state_in_obsspace, obs, obs_error,
                obs_vec)
            print(f'      Data assimilation analysis ... {tm.time()-ts:.2f}s')

            # save results
            if inputs_dafi['save_level'] in {'iter', 'debug'}:
                results = {'y': obs,
                           'Hx': state_in_obsspace,
                           'xa': state_analysis,
                           'xf': state_forecast,
                           }
                for key, val in results.items():
                    dir = os.path.join(tdir, key)
                    _create_dir(dir)
                    file = key + f'_{iteration}'
                    np.savetxt(os.path.join(dir, file), val)

            # check convergence
            diff = obs - state_in_obsspace
            misfit_norm = np.linalg.norm(np.mean(diff, axis=1))
            misfit_list.append(misfit_norm)

            # inflation
            if inverse.inflation_flag and misfit_list[iteration] > misfit_list[iteration-1]:
                dir = os.path.join(tdir, 'xf')
                state_forecast = np.loadtxt(dir + '/xf_{}'.format(iteration-1))
                for loop in range(5): # TODO: put 5 in inputfile
                    # log
                    print(f'      Inner iteration: {loop}')
                    inverse.gamma = inverse.gamma * inverse.alpha
                    dir = os.path.join(tdir, 'Hx')
                    state_in_obsspace = np.loadtxt(dir + '/Hx_{}'.format(iteration-1)) 

                    # perturb observations
                    if inputs_dafi['perturb_obs_option'] == 'iter':
                        obs, obs_perturbation = _perturb_vec(
                            obs_vec, obs_error, inputs_dafi['nsamples'])

                    diff = obs - state_in_obsspace
                    misfit_norm = np.linalg.norm(np.mean(diff, axis=1))
                    if misfit_norm < misfit_list[iteration-1]:
                        misfit_list[iteration] = misfit_norm
                        break
                    state_analysis = inverse.analysis(
                        iteration, state_forecast, state_in_obsspace, obs, obs_error,
                        obs_vec)
            print(f'      misfit = ... {misfit_norm}s')

            conv, log_message, (residual, noise) = _convergence(
                misfit_list, obs_error, inputs_dafi['convergence_factor'],
                inputs_dafi['convergence_residual'],
                inputs_dafi['convergence_option'])
            residual_list.append(residual)
            noise_list.append(noise)
            if len(iteration_array) > 1:
                logger.log(_log_level(3), log_message)
            if conv and (iteration < inputs_dafi['max_iterations']):
                early_stop = True
                break

        # save inner loop convergence history
        if inputs_dafi['save_level'] in {'iter', 'debug'}:
            convergence = {'misfit': misfit_list,
                           'min_discrepancy': noise_list,
                           'residual': residual_list,
                           }
            for key, val in convergence.items():
                np.savetxt(os.path.join(tdir, key), val)
            key, val = ('iteration', np.array([iteration], dtype=int))
            np.savetxt(os.path.join(tdir, key), val, fmt='%i')

        # log inner loop summary
        if early_stop:
            message = "convergence, early stop."
        else:
            message = "max iteration reached."
        log_message = "\n  Inversion completed: " + message
        logger.log(_log_level(2), log_message)

        # map analysis state to observation space and save
        if inputs_dafi['analysis_to_obs']:
            log_message = "\n  Mapping final analysis states " + \
                "to observation space."
            logger.log(_log_level(2), log_message)
            state_in_obsspace = model.state_to_observation(state_analysis)
            if inputs_dafi['save_level'] in {'iter', 'debug'}:
                dir = os.path.join(tdir, 'Hx')
                file = 'Hxa'
                np.savetxt(os.path.join(dir, file), state_in_obsspace)

        # save outer loop
        if inputs_dafi['save_level'] in {'time', 'iter', 'debug'}:
            results = {'y': obs,
                       'Hx': state_in_obsspace_prior,
                       'R': obs_error,
                       'xa': state_analysis,
                       'xf': state_prior
                       }
            if inputs_dafi['analysis_to_obs']:
                results['Hxa'] = state_in_obsspace
            for key, val in results.items():
                dir = os.path.join(inputs_dafi['save_dir'], key)
                _create_dir(dir)
                file = key + f'_{time}'
                np.savetxt(os.path.join(dir, file), val)

        # model cleanup
        try:
            model.clean('iter')
        except AttributeError:
            pass

        # collect output
        state_analysis_list.append(state_analysis)

    # model cleanup
    try:
        model.clean('time')
    except AttributeError:
        pass

    return state_analysis_list


def _convergence(misfit_list, obs_error, noise_factor, min_residual, option):
    """ Calculate convergence metrics.

    Also returns convergence decision and log message.
    """
    # Convergence: discrepancy principle
    noise_level = np.sqrt(np.trace(obs_error))
    if noise_factor is None:
        conv_discrepancy = False
    else:
        noise_criteria = noise_factor * noise_level
        conv_discrepancy = misfit_list[-1] < noise_criteria

    # Convergence: residual of misfit
    iteration = len(misfit_list) - 1
    if iteration > 0:
        residual = abs(misfit_list[iteration] - misfit_list[iteration-1])
        residual /= abs(misfit_list[0])
    else:
        residual = np.nan
    if min_residual is None:
        conv_residual = False
    else:
        conv_residual = residual < min_residual

    # Convergence
    if option == 'max':
        conv = False
    elif option == 'discrepancy':
        conv = conv_discrepancy
    elif option == 'residual':
        conv = conv_residual

    # log
    log_message = f"    Convergence (variance): {conv_discrepancy}"
    log_message += f"\n      Norm of misfit: {misfit_list[-1]}"
    log_message += f"\n      Noise level: {noise_criteria}"
    log_message += f"\n    Convergence (residual): {conv_residual}"
    log_message += f"\n      Relative iterative residual: {residual}"
    log_message += f"\n       Relative convergence criterion: {min_residual}"
    return conv, log_message, (residual, noise_criteria)


def _log_level(verbosity):
    """Return log level for specified verbosity level. """
    message = 'Verbosity should be between -2 and 9'
    if verbosity > 9:
        warnings.warn(message, RuntimeWarning)
        verbosity = 9
    elif verbosity < -1:
        warnings.warn(message, RuntimeWarning)
        verbosity = -1
    level = 29 - verbosity
    return level


def _perturb_vec(mean, cov, nsamps, perturb_diag=1e-10):
    """ Create samples of a random vector.

    Uses a multivariate Gaussian distribution.
    Returns matrix of samples where each column is a sample.
    """
    ndim = len(mean)
    # check symmetric
    if not np.allclose(cov, cov.T):
        raise ValueError('Covariance matrix is not symmetric.')
    # add small value to diagonal for numerical stability of Cholesky decomp.
    cov += np.eye(ndim)*perturb_diag
    # Cholesky decomposition
    l_mat = np.linalg.cholesky(cov)
    # create correlated perturbations
    x_mat = np.random.normal(loc=0.0, scale=1.0, size=(ndim, nsamps))
    perturb = np.matmul(l_mat, x_mat)
    return np.tile(mean, (nsamps, 1)).T + perturb, perturb


def _create_dir(dir,):
    """ Create directory if it does not exist. """
    if not os.path.exists(dir):
        os.makedirs(dir)
