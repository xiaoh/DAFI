# Copyright 2020 Virginia Polytechnic Institute and State University.
""" Covariance matrices. """

# standard library imports
import time

# third party imports
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg


def generate_cov(kernel, stddev, sp_tol=1e-8, **kwargs):
    """ Generate a covariance matrix using the specified correlatiion
    kernel and standard deviation.

    Parameters
    ----------
    kernel : function
        Function that returns a correlation matrix. All additional
        arguments are passed to this function.
    stddev : ndarray
        Standard deviation of each state. Alternatively, provide a float
        for a constant standard deviation.
        *dtype=float*, *ndim=1*, *shape=(nstate)*
    sp_tol : float
        Tolerance for sparse matrix. Any entry with correlation less
        than ``sp_tol`` will be set to zero.

    Returns
    -------
    cov : scipy.sparse.csc_matrix
        Covariance matrix.
        *dtype=float*, *ndim=2*, *shape=(nstate, nstate)*
    """
    # correlation matrix
    corr = kernel(**kwargs)
    corr = array_to_sparse(corr, sp_tol)

    # constant stddev
    if len(np.atleast_1d(np.squeeze(np.array(stddev)))) == 1:
        stddev = np.ones(corr.shape[0]) * stddev

    # covariance matrix
    return corr_to_cov(corr, stddev)


def array_to_sparse(mat, tol):
    """ Convert numpy array to sparse by setting small entries to zero.

    Parameters
    ----------
    mat : ndarray
        Numpy array.
    tol : float
        Tolerance for sparse matrix. Any entry less than ``tol`` will
        be set to zero.

    Returns
    -------
    mat_sp : csc_matrix
        Matrix converted to sparse matrix.
    """
    indicator_mat = np.abs(mat) > tol
    indicator_mat = indicator_mat.astype(float)
    return sp.csc_matrix(mat*indicator_mat)


def check_mat(mat, type='cov', tol=1e-08):
    """ Perform checks for correlation or covariance matrices.

    Checks if matrix is symmetric and positive definite. For a
    correlation matrix also checks that the diagonal terms are close to
    1 and the off diagonal have magnitude less than or equal to 1.

    Parameters
    ----------
    mat : ndarray
        Correlation or covariance matrix. Can be ndarray, matrix, or
        scipy sparse matrix.
        *dtype=float*, *ndim=2*, *shape=(N, N)*
    type : str
        Matrix type: use 'corr' for a correlation matrix and 'cov' for a
        covariance matrix.
    tol : float
        Tolerance  used when checking if two values are close to  each
        other.

    Returns
    -------
    passed : bool
        Whether all tests passed succesfully.
    message : str
        Information on which tests did not pass.
    """

    def check_corr_diag(mat, tol):
        """ """
        return np.allclose(mat.diagonal(), 1.0, rtol=0.0, atol=tol)


    def check_corr_offdiag(mat, atol, rtol=0.0):
        tmp = np.abs(mat)
        tmp = tmp[tmp > 1.0]
        if tmp.size != 0:
            tmp = np.max(tmp)
        else:
            tmp = 1.0
        return np.isclose(tmp, 1.0, atol)


    def check_symmetric(mat, rtol, atol=0.0):
        """ """
        mat_2 = mat.T
        tmp = np.abs(mat - mat_2) - rtol * np.abs(mat_2)
        return (tmp.max() - atol <= 0)


    def check_positive_definite(mat, tol=1e-05, eps=1e-8):
        """ Checks if a symmetric matrix is positive-definite.

        Not very stable for very small eigenvalues.
        """
        min_eig = splinalg.eigsh(
            mat, k=1, which='SA', return_eigenvectors=False)
        min_eig = min_eig[0]
        return (min_eig + tol >= 0)

    # make sparse
    mat = sp.csc_matrix(mat)

    # perform checks
    passed = True
    message = ""
    if type == 'corr':
        diag = check_corr_diag(mat, tol)
        off_diag = check_corr_offdiag(mat, tol)
        if not diag:
            passed = False
            message += "\nNot all diagonal entries equal 1. "
        if not off_diag:
            passed = False
            message += "\nNot all off-diagonal entries between -1 and 1. "
    if type in ['cov', 'corr']:
        symmetric = check_symmetric(mat, tol)
        positive_definite = check_positive_definite(mat, tol)
        if not symmetric:
            passed = False
            message += "\nNot Symmetric. "
        if not positive_definite:
            passed = False
            message += "\nNot positive_definite. "
    return passed, message


def sparse_to_nan(mat):
    """Convert a sparse matrix to a matrix with NaNs instead of zeros.
    """
    spnan = np.ones(mat.shape)*np.nan
    nonsparse = sp.find(mat)
    for (irow, icol, val) in zip(nonsparse[0], nonsparse[1], nonsparse[2]):
        spnan[irow, icol] = val
    return spnan


def corr_to_cov(corr, stddev):
    """ Convert a correlation matrix to a covariance matrix.

    Parameters
    ----------
    corr : scipy.sparse.csc_matrix
        Correlation matrix.
        *dtype=float*, *ndim=2*, *shape=(nstate, nstate)*
    stddev : ndarray
        Standard deviation of each state. Alternatively, provide a float
        for a constant standard deviation.
        *dtype=float*, *ndim=1*, *shape=(nstate)*

    Returns
    -------
    cov : scipy.sparse.csc_matrix
        Covariance matrix.
        *dtype=float*, *ndim=2*, *shape=(nstate, nstate)*
    """
    if len(np.atleast_1d(np.squeeze(np.array(stddev)))) == 1:
        # constant stddev
        cov = stddev**2 * corr
    else:
        stddev = np.atleast_2d(stddev)
        cov = corr * np.dot(stddev.T, stddev)
    return cov


def cov_to_corr(cov):
    """ Convert a covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov : ndarray
        Covariance matrix.
        *dtype=float*, *ndim=2*, *shape=(nstate, nstate)*

    Returns
    -------
    corr : ndarray
        Correlation matrix.
        *dtype=float*, *ndim=2*, *shape=(nstate, nstate)*
    """
    stddev = np.sqrt(cov.diagonal())
    stddev = np.atleast_2d(stddev)
    corr = cov / np.dot(stddev.T, stddev)
    return corr


def source_cov_to_output_corr(cov, weights, mat):
    """ Convert the input field covariance to the output field
    correlation.

    The input and output field are related by a PDE described by the
    matrix ``mat``.
    This is used to create PDE-informed covariance matrices. See:

        Wu, Jin-Long, et al. *“Physics-Informed Covariance Kernel for
        Model-Form Uncertainty Quantification with Application to
        Turbulent Flows.”* Computers & Fluids, vol. 193, Oct. 2019,
        p. 104292. `doi:10.1016/j.compfluid.2019.104292`_.

    .. _doi:10.1016/j.compfluid.2019.104292:
       doi:10.1016/j.compfluid.2019.104292

    Parameters
    ----------
    cov : ndarray
        Covariance matrix.
        *dtype=float*, *ndim=2*, *shape=(nstate, nstate)*
    weights : ndarray
        Weight (e.g. cell volume) associated with each state.
        *dtype=float*, *ndim=1*, *shape=(nstate)*
    mat : ndarray
        Matrix corresponding to the PDE.
        *dtype=float*, *ndim=2*, *shape=(nstate, nstate)*

    Returns
    -------
    corr : ndarray
        Correlation structure of output field.
        *dtype=float*, *ndim=2*, *shape=(nstate, nstate)*
    """
    weights = np.squeeze(weights)
    weights = np.atleast_2d(weights)
    weight_mat = np.sqrt(np.dot(weights.T, weights))
    cov = cov * weight_mat
    inv_mat = np.linalg.solve(mat, np.identity(mat.shape[0]))
    # L*cov*L^T
    cov_out = np.dot(np.dot(inv_mat, cov), np.transpose(inv_mat))
    return cov_to_corr(cov_out)


# kernels
def kernel_sqrexp(coords, length_scales):
    """ Create a correlation matrix using the square exponential
    function.

    Parameters
    ----------
    coords : ndarray
        Array of coordinates. Each row correspond to a different point
        and the number of columns is the number of physical dimensions
        (e.g. 3 for (x,y,z)).
        *dtype=float*, *ndim=2*, *shape=(npoints, ndims)*
    length_scales : list
        Length scale for each physical dimensions. List length is ndims.
        Each entry is either a one dimensional ndarray of length nstate
        (length scale field) or a float (constant length scale).

    Returns
    -------
    corr : ndarray
        Correlation matrix.
        *dtype=float*, *ndim=2*, *shape=(nstate, nstate)*
    """
    npoints = coords.shape[0]
    nphys_dims = coords.shape[1]
    alpha = 2.0
    # calculate
    exp = np.zeros([npoints, npoints])

    def vec_to_mat(vec):
        vec = np.atleast_2d(vec)
        return np.sqrt(np.dot(vec.T, vec))

    for ipdim in range(nphys_dims):
        pos_1, pos_2 = np.meshgrid(coords[:, ipdim], coords[:, ipdim])
        lensc = length_scales[ipdim]
        constant_lscale = len(np.atleast_1d(np.squeeze(np.array(lensc)))) == 1
        if not constant_length_scale:
            lensc = vec_to_mat(lensc)
        exp += ((pos_1 - pos_2) / (lensc))**alpha
    return np.exp(-0.5*exp)


def kernel_input_file(filename, Type='corr'):
    """ Create a correlation matrix by reading it from a file.

    Parameters
    ----------
    filename : str
        Name (path) of file containing hte correlation or covariance
        matrix.
    Type : str
        The type of matrix contained in the file. Options are 'corr' for
        correlation matrix or 'cov' for covariance matrix.

    Returns
    -------
    corr : ndarray
        Correlation matrix.
        *dtype=float*, *ndim=2*, *shape=(nstate, nstate)*
    """
    corr = np.loadtxt(filename)
    if type == 'cov':
        corr = cov_to_corr(corr)
    return corr


def kernel_mixed_periodic_sqrexp(coords, length_scales, factor=6.0,
        period=None):
    """ Create a correlation matrix using the square exponential
    function in some directions and the periodic kernel in others.

    Parameters
    ----------
    coords : ndarray
        Array of coordinates. Each row correspond to a different point
        and the number of columns is the number of physical dimensions
        (e.g. 3 for (x,y,z)).
        *dtype=float*, *ndim=2*, *shape=(npoints, ndims)*
    length_scales : list
        Length scale for each physical dimensions. List length is ndims.
        Each entry is either a one dimensional ndarray of length nstate
        (length scale field) or a float (constant length scale).
        For periodic directions the ``factor`` argument is used (see
        ``factor``).
    factor : float
        Factor used in the physical interpretation of the periodic
        length scale. The provided lengthscale (:math:`l`) is modified as
        :math:`l = l * factor / p` where :math:`p` is the periodicity.
        A factor of about 6 results in similar physical interpretation
        of the provided length scale as for the non-periodic directions.
    period : list
        List of periodicity for each physical dimension (length ndims).
        Each entry is either a float (periodicity) or *'None'* for
        non-periodic directions.

    Returns
    -------
    corr : ndarray
        Correlation matrix.
        *dtype=float*, *ndim=2*, *shape=(nstate, nstate)*
    """
    npoints = coords.shape[0]
    nphys_dims = coords.shape[1]
    alpha = 2.0
    if period is None:
        period = [None]*nphys_dims

    def vec_to_mat(vec):
        vec = np.atleast_2d(vec)
        return np.sqrt(np.dot(vec.T, vec))

    # calculate
    exp = np.zeros([npoints, npoints])
    for ipdim in range(nphys_dims):
        pos_1, pos_2 = np.meshgrid(coords[:, ipdim], coords[:, ipdim])
        dpos = (pos_1 - pos_2)
        lensc = length_scales[ipdim]
        constant_lscale = len(np.atleast_1d(np.squeeze(np.array(lensc)))) == 1
        if not constant_length_scales:
            lensc = vec_to_mat(lensc)
        per = period[ipdim]
        if per is None:
            exp += -0.5*(dpos / (lensc))**alpha
        else:
            lensc = lensc * factor / per
            exp += -2.0*(np.sin(np.abs(dpos)*np.pi/per) / lensc)**2
    return np.exp(exp)
