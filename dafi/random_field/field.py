# Copyright 2020 Virginia Polytechnic Institute and State University.
""" Random fields representation and manipulation.

These functions can be called directly from ``dafi.random_field``, e.g.

.. code-block:: python

   >>> dafi.random_field.calc_kl_modes(*args)
"""

# standard library imports
import warnings

# third party imports
import numpy as np
from scipy import sparse as sp
from scipy.sparse import linalg as splinalg
from scipy import interpolate


# KL decomposition
def calc_kl_modes(cov, nmodes=None, weight_field=None, eps=1e-8,
                  normalize=True):
    """ Calculate the first N Karhunen-Loève modes for a covariance
    field.

    Converts the covariance to a sparse matrix if it is not one yet.

    Parameters
    ----------
    cov : ndarray
        Covariance matrix. Can be ndarray, matrix, or scipy sparse
        matrix. *dtype=float*, *ndim=2*, *shape=(nstate, nstate)*
    nmodes : int
        Number of KL modes to obtain.
    weight_field : ndarray
        Weight (e.g. cell volume) associated with each state.
        Default ones (1). *dtype=float*, *ndim=1*, *shape=(nstate)*
    eps : float
        Small quantity to add to the diagonal of the covariance matrix
        for numerical stability.
    normalize : bool
        Whether to normalize (norm = 1) the KL modes.

    Returns
    -------
    eig_vals : ndarray
        Eigenvalue associated with each mode.
        *dtype=float*, *ndim=1*, *shape=(nmodes)*
    kl_modes : ndarray
        KL modes (eigenvectors).
        *dtype=float*, *ndim=2*, *shape=(nstate, nmodes)*
    """
    # convert to sparse matrix
    cov = sp.csc_matrix(cov)

    # default values
    nstate = cov.shape[0]
    if nmodes == None:
        nmodes = nstate-1
    weight_field = _get_weight_field(weight_field, nstate)

    # add small value to diagonal
    cov = cov + sp.eye(cov.shape[0], format='csc')*eps

    weight_field = np.squeeze(weight_field)
    weight_vec = np.atleast_2d(weight_field)
    weight_mat = np.sqrt(np.dot(weight_vec.T, weight_vec))
    cov_weighted = cov.multiply(weight_mat)

    # perform the eig-decomposition
    eig_vals, eig_vecs = sp.linalg.eigsh(cov_weighted, k=nmodes)

    # sort the eig-value and eig-vectors in a descending order
    ascending_order = eig_vals.argsort()
    descending_order = ascending_order[::-1]
    eig_vals = eig_vals[descending_order]
    eig_vecs = eig_vecs[:, descending_order]

    # normalized KL modes
    weight_diag = np.diag(np.sqrt(weight_field))
    kl_modes = np.dot(np.linalg.inv(weight_diag), eig_vecs)  # normalized

    # check if negative eigenvalues
    for imode in np.arange(nmodes):
        neg_eigv = False
        if eig_vals[imode] < 0:
            neg_eigv = True
            warn_message = f'Negative eigenvalue for mode {imode}.'
            warnings.warn(warn_message)
            kl_modes[:, imode] *= 0.
    if neg_eigv:
        warn_message = 'Some modes have negative eigenvalues. ' + \
            'The number of KL modes might be too large. ' + \
            "Alternatively, use a larger value for 'eps'."

    # weight by appropriate variance
    if not normalize:
        kl_modes = scale_kl_modes(eig_vals, kl_modes)

    return eig_vals, kl_modes


def calc_kl_modes_coverage(cov, coverage, weight_field=None, eps=1e-8):
    """ Calculate all KL modes and return only those required to achieve
    a certain coverage of the variance.

    Parameters
    ----------
    cov : ndarray
        Covariance matrix. Can be ndarray, matrix, or scipy sparse
        matrix. *dtype=float*, *ndim=2*, *shape=(nstate, nstate)*
    coverage : float
        Desired percentage coverage of the variance. Value between 0-1.
    weight_field : ndarray
        Weight (e.g. cell volume) associated with each state.
        Default ones (1). *dtype=float*, *ndim=1*, *shape=(nstate)*
    eps : float
        Small quantity to add to the diagonal of the covariance matrix
        for numerical stability.

    Returns
    -------
    eig_vals : ndarray
        Eigenvalue associated with each mode. For the first N modes such
        that the desired coverage of the variance is achieved.
        *dtype=float*, *ndim=1*, *shape=(N)*
    kl_modes : ndarray
        first N  KL modes (eigenvectors)  such that the desired coverage
        of the variance is achieved.
        *dtype=float*, *ndim=2*, *shape=(nstate, N)*
    """
    # convert to sparse matrix
    cov = sp.csc_matrix(cov)

    # default values
    nstate = cov.shape[0]
    weight_field = _get_weight_field(weight_field, nstate)

    # get all KL modes
    nmodes = nstate - 1
    eig_vals, kl_modes = calc_kl_modes(cov, nmodes, weight_field, eps)

    # return only those KL modes required for desired coverage
    cummalative_variance = kl_coverage(cov, eig_vals, weight_field)
    coverage_index = np.argmax(cummalative_variance >= coverage)
    return eig_vals[:coverage_index], kl_modes[:, :coverage_index]


def scale_kl_modes(eig_vals, kl_modes_norm):
    """ Weight the KL modes by the appropriate variance.

    Parameters
    ----------
    eig_vals : ndarray
        Eigenvalue associated with each mode.
        *dtype=float*, *ndim=1*, *shape=(nmodes)*
    kl_modes_norm : ndarray
        Normalized (norm = 1) KL modes (eigenvectors).
        *dtype=float*, *ndim=2*, *shape=(nstate, nmodes)*

    Returns
    -------
    kl_modes_weighted : ndarray
        KL modes with correct magnitude.
        *dtype=float*, *ndim=2*, *shape=(nstate, nmodes)*
    """
    nmodes = len(eig_vals)
    kl_modes_weighted = kl_modes_norm.copy()
    for imode in np.arange(nmodes):
        kl_modes_weighted[:, imode] *= np.sqrt(eig_vals[imode])
    return kl_modes_weighted


def kl_coverage(cov, eig_vals, weight_field=None):
    """ Calculate the percentage of the covariance covered by the the
    first N KL modes for N from 1-nmodes.

    Parameters
    ----------
    cov : ndarray
        Covariance matrix. Can be ndarray, matrix, or scipy sparse
        matrix. *dtype=float*, *ndim=2*, *shape=(nstate, nstate)*
    eig_vals : ndarray
        Eigenvalues associated with each mode.
        *dtype=float*, *ndim=1*, *shape=(nmodes)*
    weight_field : ndarray
        Weight (e.g. cell volume) associated with each state.
        *dtype=float*, *ndim=1*, *shape=(nstate)*

    Returns
    -------
    coverage: ndarray
        Cumulative variance coverage of the first N modes. Each value
        is 0-1 and increasing.
        *dtype=float*, *ndim=1*, *shape=(nmodes)*
    """
    # make sparse if its not already
    cov = sp.csc_matrix(cov)

    # default values
    nstate = cov.shape[0]
    weight_field = _get_weight_field(weight_field, nstate)

    # calculate coverage
    weight_vec = np.atleast_2d(weight_field)
    weight_mat = np.sqrt(np.dot(weight_vec.T, weight_vec))
    cov_weighted = cov.multiply(weight_mat)
    cov_trace = np.sum(cov_weighted.diagonal())
    return np.cumsum(eig_vals) / cov_trace


def reconstruct_kl(modes, coeffs, mean=None):
    """ Reconstruct a field using KL modes and given coefficients.

    Can create multiple fields by providing two dimensional array of
    coefficients.

    Parameters
    ----------
    modes : ndarray
        KL modes. *dtype=float*, *ndim=2*, *shape=(nstate, nmodes)*
    coeffs : ndarray
        Array of coefficients.
        *dtype=float*, *ndim=2*, *shape=(nmodes, nsamples)*
    mean : ndarray
        Mean vector. *dtype=float*, *ndim=1*, *shape=(nstate)*

    Returns
    -------
    fields : ndarray
        Reconstructed fields.
        *dtype=float*, *ndim=2*, *shape=(nstate, nsamples)*
    """
    # number of modes, samples, and states
    if len(coeffs) == 1:
        coeffs = np.expand_dims(coeffs, 1)
    nmodes, nsamps = coeffs.shape
    nstate = modes.shape[0]

    # mean vector
    if mean == None:
        mean = np.zeros(nstate)
    mean = np.expand_dims(np.squeeze(mean), axis=1)

    # create samples
    fields = np.tile(mean, [nsamps])
    for imode in range(nmodes):
        vec1 = np.atleast_2d(coeffs[imode, :])
        vec2 = np.atleast_2d(modes[:, imode])
        fields += np.dot(vec1.T, vec2).T
    return fields


def project_kl(field, modes, weight_field=None):
    """ Project a field onto a set of modes.

    Parameters
    ----------
    field : ndarray
        Scalar field. *dtype=float*, *ndim=1*, *shape=(ncells)*
    modes : ndarray
        KL modes. *dtype=float*, *ndim=2*, *shape=(nstate, nmodes)*
    weight_field : ndarray
        Weight (e.g. cell volume) associated with each state.
        *dtype=float*, *ndim=1*, *shape=(nstate)*

    Returns
    -------
    coeffs : ndarray
        Projection magnitude.
        *dtype=float*, *ndim=1*, *shape=(nmodes)*
    """
    # default values
    nstate = len(field)
    weight_field = _get_weight_field(weight_field, nstate)

    nstate, nmode = modes.shape

    coeffs = []
    for imode in range(nmode):
        mode = modes[:, imode]
        coeffs.append(projection_magnitude(field, mode, weight_field))
    return np.array(coeffs)


def _get_weight_field(weight_field, nstate):
    """Pre-process provided weight field. """
    # default value
    if weight_field is None:
        weight_field = np.ones(nstate)
    # constant value
    if len(np.atleast_1d(np.squeeze(np.array(weight_field)))) == 1:
        weight_field = np.ones(nstate)*weight_field
    return weight_field


# linear algebra on scalar fields
def integral(field, weight_field):
    """ Calculate the integral of a field.

    Parameters
    ----------
    field : ndarray
        Scalar field. *dtype=float*, *ndim=1*, *shape=(ncells)*
    weight_field : ndarray
        Cell volumes. *dtype=float*, *ndim=1*, *shape=(ncells)*

    Returns
    -------
    field_integral : float
        The integral of the field over the domain.
    """
    nstate = len(field)
    weight_field = _get_weight_field(weight_field, nstate)
    return np.sum(field * weight_field)


def inner_product(field_1, field_2, weight_field):
    """ Calculate the inner product between two fields.

    The two fields share the same weights.

    Parameters
    ----------
    field_1 : ndarray
        One scalar field. *dtype=float*, *ndim=1*, *shape=(ncells)*
    field_2 : ndarray
        Another scalar field.
        *dtype=float*, *ndim=1*, *shape=(ncells)*
    weight_field : ndarray
        Cell volumes. *dtype=float*, *ndim=1*, *shape=(ncells)*

    Returns
    -------
    product : float
        The inner product between the two fields.
    """
    return integral(field_1 * field_2, weight_field)


def norm(field, weight_field):
    """ Calculate the L2-norm of a field.

    Parameters
    ----------
    field : ndarray
        Scalar field. *dtype=float*, *ndim=1*, *shape=(ncells)*
    weight_field : ndarray
        Cell volumes. *dtype=float*, *ndim=1*, *shape=(ncells)*

    Returns
    -------
    field_norm : float
        The norm of the field.
    """
    return np.sqrt(inner_product(field, field, weight_field))


def unit_field(field, weight_field):
    """ Calculate the unit field (norm = 1) in same direction.

    Parameters
    ----------
    field : ndarray
        Scalar field. *dtype=float*, *ndim=1*, *shape=(ncells)*
    weight_field : ndarray
        Cell volumes. *dtype=float*, *ndim=1*, *shape=(ncells)*

    Returns
    -------
    field_normed : ndarray
        Normalized (norm = 1) scalar field.
        *dtype=float*, *ndim=1*, *shape=(ncells)*
    """
    return field / norm(field, weight_field)


def projection_magnitude(field_1, field_2, weight_field):
    """ Get magnitude of projection of field_1 onto field_2.

    The two fields share the same weights.

    Parameters
    ----------
    field_1 : ndarray
        Scalar field being projected.
        *dtype=float*, *ndim=1*, *shape=(ncells)*
    field_2 : ndarray
        Scalar field used for projection direction.
        *dtype=float*, *ndim=1*, *shape=(ncells)*
    weight_field : ndarray
        Cell volumes.
        *dtype=float*, *ndim=1*, *shape=(ncells)*

    Returns
    -------
    magnitude : float
        magnitude of the projected field.
    """
    magnitude = inner_product(field_1, field_2, weight_field) \
        / norm(field_2, weight_field)
    return magnitude


def projection(field_1, field_2, weight_field):
    """ Project field_1 onto field_2.

    The two fields share the same weights.

    Parameters
    ----------
    field_1 : ndarray
        Scalar field being projected.
        *dtype=float*, *ndim=1*, *shape=(ncells)*
    field_2 : ndarray
        Scalar field used for projection direction.
        *dtype=float*, *ndim=1*, *shape=(ncells)*
    weight_field : ndarray
        Cell volumes.
        *dtype=float*, *ndim=1*, *shape=(ncells)*

    Returns
    -------
    projected_field : ndarray
        Projected field.
        *dtype=float*, *ndim=1*, *shape=(ncells)*
    """
    magnitude = projection_magnitude(field_1, field_2, weight_field)
    direction = unit_field(field_2, weight_field)
    return magnitude*direction


# interpolation
def interpolate_field_rbf(data, coords, kernel, length_scale):
    """ Interpolate data using a radial basis function (RBF) to create a
    field.

    Parameters
    ----------
    data : ndarray
        Sparse data to create interpolation from. For an NxM array, the
        number of data points is N, the number of dimensions
        (coordinates) is M-1, and the Mth column is the data value.
        *dtype=float*, *ndim=2*, *shape=(N, M)*
    coords : ndarray
        Coordinates of the cell centers of the full discretized field.
        The RBF will be evaluated at these points.
        *dtype=float*, *ndim=2*, *shape=(ncells, M-1)*
    kernel : str
        Kernel (function) of the RBF. See *'function'* input of
        `scipy.interpolate.Rbf`_ for list of options.
    length_scale : float
        Length scale parameter (epsilon in `scipy.interpolate.Rbf`_)
        in the RBF kernel.

    Returns
    -------
    field : ndarray
        Full field. *dtype=float*, *ndim=1*, *shape=(ncells)*
    """
    args1 = []
    args2 = []
    ncoord = coords.shape[1]
    for icoord in range(ncoord):
        args1.append(data[:, icoord])
        args2.append(coords[:, icoord])
    interp_func = interpolate.Rbf(
        *args1, function=kernel, epsilon=length_scale)
    return interp_func(*args2)


# Gaussian process: generate samples
def gp_samples_cholesky(cov, nsamples, mean=None, eps=1e-8):
    """ Generate samples of a Gaussian Process using Cholesky
    decomposition.

    Error
    -----
    NOT IMPLEMENTED: need to figure out how to account for weights.

    Parameters
    ----------
    cov : ndarray
        Covariance matrix. Can be ndarray, matrix, or scipy sparse
        matrix. *dtype=float*, *ndim=2*, *shape=(nstate, nstate)*
    nsamples : int
        Number of samples to generate.
    mean : ndarray
        Mean vector. *dtype=float*, *ndim=1*, *shape=(nstate)*
    eps : float
        Small quantity to add to the diagonal of the covariance matrix
        for numerical stability.

    Returns
    -------
    samples : ndarray
        Matrix of samples.
        *dtype=float*, *ndim=2*, *shape=(nstate, nsamples)*
    """
    # TODO: account for weight_field
    raise NotImplementedError
    # make sparse if its not already
    cov = sp.csc_matrix(cov)
    nstate = cov.shape[0]

    # add small value to diagonal
    cov = cov + sp.eye(nstate, format='csc')*eps

    # mean vector
    if mean == None:
        mean = np.zeros(cov.shape[0])
    mean = np.expand_dims(np.squeeze(mean), axis=1)

    # Create samples using Cholesky Decomposition
    L = sparse_cholesky(cov)
    a = np.random.normal(size=(nstate, nsamples))
    perturb = L.dot(a)

    return mean + perturb.toarray()


def sparse_cholesky(cov):
    """ Compute the Cholesky decomposition for a sparse (scipy) matrix.

    Adapted from `gist.github.com/omitakahiro`_.

    Error
    ----
    NOT IMPLEMENTED: need to figure out how to account for weights.

    Parameters
    ----------
    cov : ndarray
      Covariance matrix. Can be ndarray, matrix, or scipy sparse
      matrix. *dtype=float*, *ndim=2*, *shape=(nstate, nstate)*

    Returns
    -------
    lower: scipy.sparse.csc_matrix
        Lower triangular Cholesky factor of the covariance matrix.
    """
    # TODO: account for weight_field
    raise NotImplementedError
    # convert to sparse matrix
    cov = sp.csc_matrix(cov)

    # LU decomposition
    LU = sparse.linalg.splu(A, diag_pivot_thresh=0)

    # check the matrix A is positive definite.
    n = cov.shape[0]
    posd = (LU.perm_r == np.arange(n)).all() and (LU.U.diagonal() > 0).all()
    if not posd:
        raise ValueError('The matrix is not positive definite')

    return LU.L.dot(sparse.diags(LU.U.diagonal()**0.5))


def gp_samples_kl(cov, nsamples, nmodes=None, mean=None, eps=1e-8):
    """ Generate samples of a Gaussian Process using KL decomposition.

    Parameters
    ----------
    cov : ndarray
        Covariance matrix. Can be ndarray, matrix, or scipy sparse
        matrix. *dtype=float*, *ndim=2*, *shape=(nstate, nstate)*
    nsamples : int
        Number of samples to generate.
    nmodes : int
        Number of modes to use when generating samples. *'None'* to use
        all modes.
    mean : ndarray
        Mean vector. *dtype=float*, *ndim=1*, *shape=(nstate)*
    eps : float
        Small quantity to add to the diagonal of the covariance matrix
        for numerical stability.

    Returns
    -------
    samples : ndarray
        Matrix of samples.
        *dtype=float*, *ndim=2*, *shape=(nstate, nsamples)*
    """
    # KL decomposition
    eigv, klmodes = calc_kl_modes(cov, nmodes, weight_field, eps, False)
    if nmodes == None:
        nmodes = len(eigv)

    # create samples
    coeffs = np.random.normal(0, 1, [nmodes, nsamples])
    return reconstruct_kl(modes, coeffs, mean)


def gp_samples_kl_coverage(cov, nsamples, coverage=0.99, mean=None, eps=1e-8):
    """ Generate samples of a Gaussian Process using KL decomposition.

    Only the firs N modes required to get the desired variance coverage
    are used.

    Parameters
    ----------
    cov : ndarray
        Covariance matrix. Can be ndarray, matrix, or scipy sparse
        matrix. *dtype=float*, *ndim=2*, *shape=(nstate, nstate)*
    nsamples : int
        Number of samples to generate.
    coverage : float
        Desired percentage coverage of the variance. Value between 0-1.
    mean : ndarray
        Mean vector. *dtype=float*, *ndim=1*, *shape=(nstate)*
    eps : float
        Small quantity to add to the diagonal of the covariance matrix
        for numerical stability.

    Returns
    -------
    samples : ndarray
        Matrix of samples.
        *dtype=float*, *ndim=2*, *shape=(nstate, nsamples)*
    """
    # KL decomposition
    eigv, klmodes = calc_kl_modes_coverage(
        cov, nmodes, weight_field, eps, False)
    nmodes = len(eigv)

    # create samples
    coeffs = np.random.normal(0, 1, [nmodes, nsamples])
    return reconstruct_kl(modes, coeffs, mean)
