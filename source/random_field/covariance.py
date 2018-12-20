# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Generate covariance matrices. """

# standard library imports
import time

# third party imports
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# generate covariance matrix
def generate_cov(kernel, corr_flag=False, stddev=None, stddev_constant=True,
                 sp_tol=0.001, perform_checks=False, tol=1e-05, verbose=0,
                 **kwargs):
    """ Create a sparse covariance matrix from specified kernel and tolerances.

    This is a wrapper for th specified kernel.
    Either the covariance or correlation kernel can be specified. Indicate
    which one with the ``corr_flag`` variable. Several checks are also
    performed on the covariance/correlation matrices. Performing checks is
    turned off by default since the check for positive-definite is very
    slow (computes an eigenvalue).
    """
    if verbose > 0:
        tic = time.time()
        print("Calculating covariance matrix.")
    if corr_flag:
        # check inputs
        if stddev is None:
            error_message = 'The standard deviation field needs to be ' + \
                'specified when using a correlation kernel.'
            raise ValueError(error_message)
        # calculate correlation
        corr = kernel(**kwargs)
        # check correlation
        if perform_checks:
            check = check_corr_diag(corr, tol)
            if not check:
                error_message = 'Diagonals of correlation matrix must be 1.0.'
                raise ValueError(error_message)
            check = check_corr_offdiag(corr, tol)
            if not check:
                error_message = 'Entries of correlation matrix must be ' + \
                    'between -1.0 and 1.0.'
                raise ValueError(error_message)
        # calculate covariance
        cov = corr_to_cov(corr, stddev, stddev_constant)
    else:
        cov = kernel(**kwargs)
    # convert covariance to sparse matrix
    cov = dense_to_sparse(cov, sp_tol)
    npoints = cov.shape[0]*cov.shape[1]
    if verbose > 0:
        print('  Sparse covariance has {:d} / {:d} values ({:.1%})'.format(
            cov.size, npoints, 1.0*cov.size/npoints))
    # check covariance
    if perform_checks:
        check = check_symmetric(cov, rtol=tol, atol=0.0)
        if not check:
            raise ValueError('Not symmetric.')
        if verbose > 1:
            tic2 = time.time()
            print("  Checking if covariance is positive-definite ...")
        check = check_positive_definite(cov, tol)
        if not check:
            raise ValueError("Not possitive-definitie.")
        if verbose > 1:
            toc2 = time.time()
            print("    Time to check positive-definite: {:.1f}s".format(
                toc2-tic2))
    # report time
    toc = time.time()
    if verbose > 0:
        print("  Time to calculate covariance matrix: {:.1f}s".format(toc-tic))
    return cov


def corr_to_cov(corr, stddev, stddev_constant=True):
    """ Convert a correlation matrix to a covariance matrix. """
    if stddev_constant:
        cov = stddev**2 * corr
    else:
        stddev = np.atleast_2d(stddev)
        cov = corr*np.dot(stddev.T, stddev)
    return cov


def dense_to_sparse(mat, tol):
    """ Convert matrix to sparse by setting small entries to zero.
    """
    indicator_mat = np.abs(mat) > tol
    indicator_mat = indicator_mat.astype(float)
    return sp.csc_matrix(mat*indicator_mat)


def check_corr_diag(corr, tol=1e-05):
    """ """
    return np.allclose(corr.diagonal(), 1.0, rtol=0.0, atol=tol)


def check_corr_offdiag(corr, tol=1e-05):
    tmp = np.abs(corr)
    tmp = tmp[tmp > 1.0]
    if tmp.size != 0:
        tmp = np.max(tmp)
    else:
        tmp = 1.0
    return np.isclose(tmp, 1.0, rtol=0.0, atol=tol)


def check_symmetric(mat, rtol=1e-05, atol=0.0):
    """ """
    mat_2 = mat.T
    tmp = np.abs(mat - mat_2) - rtol * np.abs(mat_2)
    return (tmp.max() - atol <= 0)


def check_positive_definite(mat, tol=1e-05):
    """ Checks if a symmetric matrix is positive-definite. """
    # TODO: square exponential fails if (l != 1). Why?
    min_eig = spla.eigsh(mat, k=1, which='SA', return_eigenvectors=False)[0]
    return (min_eig + tol >= 0)


# utilities
def sparse_to_nan(mat):
    """Convert a sparse matrix to a matrix with NaNs instead of zeros.
    """
    spnan = np.ones(mat.shape)*np.nan
    nonsparse = sp.find(mat)
    for (irow, icol, val) in zip(nonsparse[0], nonsparse[1], nonsparse[2]):
        spnan[irow, icol] = val
    return spnan


# kernels
def kernel_sqrexp(coords, length_scales, constant_length_scales=True):
    """
    """
    npoints = coords.shape[0]
    nphys_dims = coords.shape[1]
    # alpha = 2 specifying a Squre exponential kernel
    alpha = 2
    # calculate
    exp = np.zeros([npoints, npoints])

    def vec_to_mat(vec):
        vec = np.atleast_2d(vec)
        return np.sqrt(np.dot(vec.T, vec))

    for ipdim in range(nphys_dims):
        pos_1, pos_2 = np.meshgrid(coords[:, ipdim], coords[:, ipdim])
        lensc = length_scales[ipdim]
        if not constant_length_scales:
            lensc = vec_to_mat(lensc)
        exp += ((pos_1 - pos_2) / (lensc))**alpha
    return np.exp(-exp)
    # return np.exp(-0.5*exp)


def kernel_input_file(filename):
    """
    """
    # TODO: perform the calculations here.
    return np.loadtxt(filename)
