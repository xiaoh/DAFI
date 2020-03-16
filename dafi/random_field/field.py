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
    if nmodes is None:
        nmodes = nstate-1
    weight_field = _preprocess_field(weight_field, nstate, 1.0)

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


def calc_kl_modes_coverage(cov, coverage, weight_field=None, eps=1e-8,
                           max_modes=None, normalize=True):
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
    normalize : bool
        Whether to normalize (norm = 1) the KL modes.

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
    weight_field = _preprocess_field(weight_field, nstate, 1.0)
    if max_modes is None:
        max_modes = nstate - 1

    # get the first max_modes KL modes
    eig_vals, kl_modes = calc_kl_modes(
        cov, max_modes, weight_field, eps, normalize)

    # return only those KL modes required for desired coverage
    cummalative_variance = kl_coverage(cov, eig_vals, weight_field)
    coverage_index = np.argmax(cummalative_variance >= coverage)
    if coverage_index == 0:
        coverage_index = max_modes
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
    weight_field = _preprocess_field(weight_field, nstate, 1.0)

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
    if len(coeffs.shape) == 1:
        coeffs = np.expand_dims(coeffs, 1)
    nmodes, nsamps = coeffs.shape
    nstate = modes.shape[0]

    # mean vector
    mean = _preprocess_field(mean, nstate, 0.0)
    mean = np.expand_dims(np.squeeze(mean), axis=1)

    # create samples
    fields = np.tile(mean, [nsamps])
    for imode in range(nmodes):
        vec1 = np.atleast_2d(coeffs[imode, :])
        vec2 = np.atleast_2d(modes[:, imode])
        fields += np.dot(vec1.T, vec2).T
    return fields


def project_kl(field, modes, weight_field=None, mean=None):
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
    mean : ndarray
        Mean vector. *dtype=float*, *ndim=1*, *shape=(nstate)*

    Returns
    -------
    coeffs : ndarray
        Projection magnitude.
        *dtype=float*, *ndim=1*, *shape=(nmodes)*
    """
    nstate, nmode = modes.shape
    mean = _preprocess_field(mean, nstate, 0.0)

    coeffs = []
    for imode in range(nmode):
        mode = modes[:, imode]
        coeffs.append(projection_magnitude(field-mean, mode, weight_field))
    return np.array(coeffs)


def _preprocess_field(field, nstate, default):
    """Pre-process provided weight field. """
    # default value
    if field is None:
        field = np.ones(nstate)*default
    # constant value
    if len(np.atleast_1d(np.squeeze(np.array(field)))) == 1:
        field = np.ones(nstate)*field
    return field


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
    weight_field = _preprocess_field(weight_field, nstate, 1.0)
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
    magnitude = inner_product(field_1, field_2, weight_field) / \
        (norm(field_2, weight_field)**2)
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
    field from sparse specifications.

    This is used for instance to specify a variance field based on
    expert knowledge.

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
    # make sparse if its not already
    cov = sp.csc_matrix(cov)
    nstate = cov.shape[0]

    # add small value to diagonal
    cov = cov + sp.eye(nstate, format='csc')*eps

    # mean vector
    mean = _preprocess_field(mean, nstate, 0.0)
    mean = np.expand_dims(np.squeeze(mean), axis=1)

    # Create samples using Cholesky Decomposition
    L = sparse_cholesky(cov)
    a = np.random.normal(size=(nstate, nsamples))
    perturb = L.dot(a)
    return mean + perturb


def sparse_cholesky(cov):
    """ Compute the Cholesky decomposition for a sparse (scipy) matrix.

    Adapted from `gist.github.com/omitakahiro`_.

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
    # convert to sparse matrix
    cov = sp.csc_matrix(cov)

    # LU decomposition
    LU = splinalg.splu(cov, diag_pivot_thresh=0)

    # check the matrix is positive definite.
    n = cov.shape[0]
    posd = (LU.perm_r == np.arange(n)).all() and (LU.U.diagonal() > 0).all()
    if not posd:
        raise ValueError('The matrix is not positive definite')

    return LU.L.dot(sp.diags(LU.U.diagonal()**0.5))


def gp_samples_kl(cov, nsamples, weight_field, nmodes=None, mean=None,
                  eps=1e-8):
    """ Generate samples of a Gaussian Process using KL decomposition.

    Parameters
    ----------
    cov : ndarray
        Covariance matrix. Can be ndarray, matrix, or scipy sparse
        matrix. *dtype=float*, *ndim=2*, *shape=(nstate, nstate)*
    nsamples : int
        Number of samples to generate.
    weight_field : ndarray
        Weight (e.g. cell volume) associated with each state.
        *dtype=float*, *ndim=1*, *shape=(nstate)*
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
    if nmodes is None:
        nmodes = len(eigv)

    # create samples
    coeffs = np.random.normal(0, 1, [nmodes, nsamples])
    return reconstruct_kl(modes, coeffs, mean)


def gp_samples_klmodes(modes, nsamples, mean=None):
    """ Generate samples of a Gaussian Process using the given KL
    modes.

    Parameters
    ----------
    modes : ndarray
        KL modes. *dtype=float*, *ndim=2*, *shape=(nstate, nmodes)*
    nsamples : int
        Number of samples to generate.
    mean : ndarray
        Mean vector. *dtype=float*, *ndim=1*, *shape=(nstate)*

    Returns
    -------
    samples : ndarray
        Matrix of samples.
        *dtype=float*, *ndim=2*, *shape=(nstate, nsamples)*
    """
    # create samples
    nmodes = modes.shape[1]
    coeffs = np.random.normal(0, 1, [nmodes, nsamples])
    return reconstruct_kl(modes, coeffs, mean)


def gp_samples_kl_coverage(cov, nsamples, weight_field, coverage=0.99,
                           max_modes=None, mean=None, eps=1e-8):
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
    weight_field : ndarray
        Weight (e.g. cell volume) associated with each state.
        *dtype=float*, *ndim=1*, *shape=(nstate)*
    coverage : float
        Desired percentage coverage of the variance. Value between 0-1.
    max_modes : int
        Maximum number of modes used. This is the number of modes that
        is calculated. If less are needed to achieve the desired
        coverage the additional ones are discarded.
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
    nmodes : int
        Number of modes used to achieve the requested coverage.
    """
    # KL decomposition
    eigv, klmodes = calc_kl_modes_coverage(
        cov, coverage, weight_field, eps, max_modes, False)
    nmodes = len(eigv)

    # create samples
    coeffs = np.random.normal(0, 1, [nmodes, nsamples])
    return reconstruct_kl(klmodes, coeffs, mean), nmodes


def gp_sqrexp_samples(nsamples, coords, stddev, length_scales, mean=None,
                      weight_field=None, max_modes=None):
    """ Generate samples from a Gaussian Process with square exponential
    correlation kernel.

    This is a convinience function for new users or simple cases.
    It create the covariance matrix, does the KL decomposition, keeps
    the required modes for 99% coverage, and create the samples.

    Parameters
    ----------
    nsamples : int
        Number of samples to generate.
    coords : ndarray
        Array of coordinates. Each row correspond to a different point
        and the number of columns is the number of physical dimensions
        (e.g. 3 for (x,y,z)).
        *dtype=float*, *ndim=2*, *shape=(npoints, ndims)*
    stddev : ndarray
        Standard deviation of each state. Alternatively, provide a float
        for a constant standard deviation.
        *dtype=float*, *ndim=1*, *shape=(nstate)*
    length_scales : list
        Length scale for each physical dimensions. List length is ndims.
        Each entry is either a one dimensional ndarray of length nstate
        (length scale field) or a float (constant length scale).
    mean : ndarray
        Mean vector. *dtype=float*, *ndim=1*, *shape=(nstate)*
    weight_field : ndarray
        Weight (e.g. cell volume) associated with each state.
        *dtype=float*, *ndim=1*, *shape=(nstate)*
    max_modes : int
        Maximum number of modes used. This is the number of modes that
        is calculated. If less are needed to achieve 99% coverage the
        additional ones are discarded.
    """
    from dafi.random_field.covariance import generate_cov
    cov = generate_cov(
        'sqrexp', stddev, coords=coords, length_scales=length_scales)
    samples, _ = gp_samples_kl_coverage(
        cov, nsamples, weight_field, 0.99, max_modes, mean)
    return samples


# Random field class
class GaussianProcess(object):
    # TODO: Docstrings

    def __init__(self, klmodes, mean=None, weights=None, func=None,
                 funcinv=None):
        self.klmodes = klmodes
        self.ncell, self.nmodes = self.klmodes.shape
        self.mean = _preprocess_field(mean, self.ncell, 0.0)
        self.weights = _preprocess_field(mean, self.weights, 1.0)

        def func_identity(x):
            return x

        if func is None:
            func = func_identity
        if funcinv is None:
            funcinv = func_identity
        self.func = func
        self.funcinv = funcinv

    def sample_coeffs(self, nsamples):
        """ """
        return np.random.normal(0, 1, [self.nmodes, nsamples])

    def sample_gp(self, nsamples, mean=self.mean):
        """ """
        coeffs = self.sample_coeffs(nsamples)
        return reconstruct_kl(self.klmodes, coeffs, mean), coeffs

    def sample_func(self, nsamples, mean=self.mean):
        """ """
        coeffs = self.sample_coeffs(nsamples)
        samps_gp = reconstruct_kl(self.klmodes, coeffs, mean)
        return self.func(samps_gp), coeffs

    def reconstruct_gp(self, coeffs, mean=self.mean):
        return reconstruct_kl(self.klmodes, coeffs, mean)

    def reconstruct_func(self, coeffs, mean=self.mean):
        val_gp = reconstruct_kl(self.klmodes, coeffs, mean)
        return self.func(val_gp)

    def pdf(self, coeffs):
        return np.exp(logpdf(coeffs))

    def logpdf(self, coeffs):
        if len(coeffs.shape) == 1:
            coeffs = np.expand_dims(coeffs, 1)
        norm_coeff = np.linalg.norm(coeffs, axis=0)
        const = np.log((2*np.pi)**(-self.ncell/2))
        return const + -0.5*norm_coeff**2

    def project_gp_field(self, field, mean=None):
        return project_kl(field, self.klmodes, self.weights, mean)

    def project_func_field(self, field, mean=None):
        field = self.funcinv(field)
        mean = _preprocess_field(mean)
        mean = self.funcinv(mean)
        return project_kl(field, self.klmodes, self.weights, mean)


class LogNormal(object):
    # TODO: Docstrings

    def __init__(self, klmodes_gp, median=1.0, weights=None):
        """
        """
        median = _preprocess_field(median)
        self.median_func = np.expand_dims(np.squeeze(median), 1)

        def func(x):
            return self.median_func * np.exp(x)

        def funcinv(y):
            return np.log(y / self.median_func)

        super(self.__class__, self).__init__(
            klmodes_gp, mean=0.0, weights, func, funcinv)
