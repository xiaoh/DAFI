"""
Collection of ensemble gradient algorithms.
"""

import numpy as np


def _preproc(x_samp, y_samp, x_bl, y_bl, y_data, baseline_as_mean=False):
    """ Pre-processing common to the different methods. """
    nsamples = x_samp.shape[1]
    _preproc_1darray = lambda x : np.expand_dims(np.squeeze(x), -1) 
    x_bl = _preproc_1darray(x_bl)
    y_bl = _preproc_1darray(y_bl)
    y_data = _preproc_1darray(y_data)
    nx = len(x_bl)
    ny = len(y_bl)
    if baseline_as_mean:
        delta_x = x_samp - x_bl
        delta_y = y_samp - y_bl
    else: 
        delta_x = x_samp - np.mean(x_samp, axis=1, keepdims=True) 
        delta_y = y_samp - np.mean(y_samp, axis=1, keepdims=True) 
    innovation = y_bl - y_data
    return nsamples, nx, ny, x_bl, y_bl, y_data, delta_x, delta_y, innovation


def direct(x_samp, y_samp, x_bl, y_bl, y_data, Rinv, baseline_as_mean=False, **kwargs):
    nsamples, nx, ny, x_bl, y_bl, y_data, delta_x, delta_y, innovation = _preproc(
        x_samp, y_samp, x_bl, y_bl, y_data, baseline_as_mean)
    dy = delta_y @ np.linalg.pinv(delta_x)
    return dy.T @ Rinv @ innovation


def precondition(x_samp, y_samp, x_bl, y_bl, y_data, Rinv, baseline_as_mean=False, **kwargs): 
    """ Premultiply gradient by state covariance (EnOpt). """
    nsamples, nx, ny, x_bl, y_bl, y_data, delta_x, delta_y, innovation = _preproc(
        x_samp, y_samp, x_bl, y_bl, y_data, baseline_as_mean)
    Cxy = (1.0/nsamples) * delta_x @ delta_y.T
    return Cxy @ Rinv @ innovation   


def precondition2(x_samp, y_samp, x_bl, y_bl, y_data, Rinv, baseline_as_mean=False, **kwargs):
    """ Premultiply gradient by state covariance twice. """
    nsamples, nx, ny, x_bl, y_bl, y_data, delta_x, delta_y, innovation = _preproc(
        x_samp, y_samp, x_bl, y_bl, y_data, baseline_as_mean)
    Cxy = (1.0/nsamples) * delta_x @ delta_y.T
    Cx = (1.0/nsamples) * delta_x @ delta_x.T
    return Cx @ Cxy @ Rinv @ innovation   


def projection(x_samp, y_samp, x_bl, y_bl, y_data, Rinv, baseline_as_mean=False, weights=None, eps=1e-12):
    """ Regularized projection method derived by us. """
    nsamples, nx, ny, x_bl, y_bl, y_data, delta_x, delta_y, innovation = _preproc(
        x_samp, y_samp, x_bl, y_bl, y_data, baseline_as_mean)
    #dJdbeta
    dJdbeta = delta_y.T @ Rinv @ innovation
    # dbetadx
    if weights is not None:
        W = np.diag(weights)
    else:
        W = np.eye(nx)
    xWx = delta_x.T @ W @ delta_x
    xWx_inv =  np.linalg.inv(xWx + eps * np.eye(nsamples))
    dbetadx = xWx_inv @ delta_x.T @ W
    return dbetadx.T @ dJdbeta
