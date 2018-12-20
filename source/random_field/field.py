# -*- coding: utf-8 -*-
# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Random fields representation and manipulation. """

# standard library imports
import warnings

# third party imports
import numpy as np
import scipy.sparse as sp
import numpy.linalg as la


# KL decomposition functions
def calc_kl_modes(nmodes, cov, weight_field, normalize=False):
    """ Calculate the first N Karhunen-Loève modes for a covariance
    field.
    """
    weight_field = np.squeeze(weight_field)
    weight_vec = np.atleast_2d(weight_field)
    weight_mat = np.sqrt(np.dot(weight_vec.T, weight_vec))
    cov_weighted = cov.multiply(weight_mat)
    # perform the eig-decomposition
    eig_vals, eig_vecs = sp.linalg.eigsh(cov_weighted, k=nmodes, which='LA')
    # sort the eig-value and eig-vectors in a descending order
    ascending_order = eig_vals.argsort()
    descending_order = ascending_order[::-1]
    eig_vals = eig_vals[descending_order]
    eig_vecs = eig_vecs[:, descending_order]
    # weighted eigVec
    weight_diag = np.diag(np.sqrt(weight_field))
    eig_vecs_weighted = np.dot(la.inv(weight_diag), eig_vecs)
    # KLModes is eigVecWeighted * sqrt(eig_val)
    kl_modes = np.zeros([len(weight_field), nmodes])
    for imode in np.arange(nmodes):
        if eig_vals[imode] >= 0:
            kl_modes[:, imode] = eig_vecs_weighted[:, imode] \
                * np.sqrt(eig_vals[imode])
            # normalize modes
            if normalize:
                kl_modes[:, imode] = unit(kl_modes[:, imode], weight_field)
        else:
            warn_message = 'Negative eigenvalue detected at nmodes=' + \
                '{}: number of KL modes might be too large!'.format(imode)
            warnings.warn(warn_message)
            kl_modes[:, imode] = eig_vecs_weighted[:, imode] * 0
    return eig_vals, kl_modes


def kl_coverage(cov, eig_vals, weight_field):
    """ Calculate the percentage of the covariance covered by N KL
    modes.
    """
    weight_vec = np.atleast_2d(weight_field)
    weight_mat = np.sqrt(np.dot(weight_vec.T, weight_vec))
    cov_weighted = cov.multiply(weight_mat)
    cov_trace = sum(cov_weighted.diagonal())
    return sum(eig_vals) / cov_trace


# linear algebra functions
def inner_product(field_1, field_2, weight_field):
    """ Calculate the inner product between two fields. """
    return np.sum(field_1 * field_2 * weight_field)


def norm(field, weight_field):
    """ Calculate the L2-norm of a field. """
    return np.sqrt(inner_product(field, field, weight_field))


def unit(field, weight_field):
    """ Calculate the unit field in same direction. """
    return field / norm(field, weight_field)


def projection_magnitude(field_1, field_2, weight_field):
    """ Get magnitude of projection of field1 onto field2. """
    magnitude = inner_product(field_1, field_2, weight_field) \
        / norm(field_2, weight_field)
    return magnitude


def projection(field_1, field_2, weight_field):
    """ Project field_1 onto field_2. """
    magnitude = projection_magnitude(field_1, field_2, weight_field)
    direction = unit(field_2, weight_field)
    return magnitude*direction


# interpolation functions
def interpolate_field_rbf(data, coords, kernel, length_scale):
    """ Interpolate data using a radial basis function to create a
    field.
    """
    args1 = []
    args2 = []
    ncoord = data.shape[1]
    for icoord in range(ncoord):
        args1.append(data[:, icoord])
        args2.append(coords[:, icoord])
    interp_func = interpolate.Rbf(*args1, function=kernel,
                                  epsilon=length_scale)
    return interp_func(*args2)


# random field classes
class RandomField(object):
    """ Parent class (template). Need to overight some functions.
    """

    def __init__(self, zero_mean=False, **kwargs):
        """
        cov is scipy.sparse.csc
        """
        # name
        if 'name' in kwargs:
            self.name = kwargs['name']
        else:
            self.name = None
        # number of points
        if 'npoints' in kwargs:
            self.npoints = kwargs['npoints']
        else:
            self.npoints = None
        # number of spatial coordinates (1-3)
        if 'nspatial_dims' in kwargs:
            self.nspatial_dims = kwargs['nspatial_dims']
        else:
            self.nspatial_dims = None
        # coordinates
        if 'coords' in kwargs:
            self.coords = kwargs['coords']
            self._set_attribute('npoints', self.coords.shape[0])
            self._set_attribute('nspatial_dims', self.coords.shape[1])
        else:
            self.coords = None
        # weights (cell length, area, or volume)
        if 'weight_field' in kwargs:
            self.weight_field = kwargs['weight_field']
            self._set_attribute('npoints', self.weight_field.shape[0])
        else:
            self.weight_field = None
        # covariance
        if 'cov' in kwargs:
            self.cov = kwargs['cov']
            self._set_attribute('npoints', self.weight_field.shape[0])
        else:
            self.cov = None
        # kl modes (decomposition of covariance)
        if 'kl_modes' in kwargs:
            self.kl_modes = kwargs['kl_modes']
            self.nmodes = self.kl_modes.shape[1]
            self._set_attribute('npoints', self.kl_modes.shape[0])
        else:
            self.kl_modes = None
            self.nmodes = None
        # mean field
        if not zero_mean:
            self.mean = kwargs['mean']
        else:
            self.mean = np.zeros(self.npoints)
        self._set_attribute('npoints', self.mean.shape[0])

    def __str__(self):
        str_info = "Scalar random field."
        return str_info

    def rand_coeff(self, nsamp):
        """
        """
        raise NotImplementedError(
            "Needs to be implemented in the child class!")
        return np.random.normal(0, 1, nsamp)

    def calc_kl_modes(self, nmodes, normalize=False):
        """
        """
        self.nmodes = nmodes
        self.kl_eig_vals, self.kl_modes = calc_kl_modes(
            self.nmodes, self.cov, self.weight_field, normalize)

    def sample_full(self, nsamp):
        """
        """
        raise NotImplementedError(
            "Needs to be implemented in the child class!")
        samps = np.zeros([self.npoints, nsamps])
        return samps

    def sample_kl_reduced(self, nsamp, nmode, return_coeffs=False):
        """
        """
        coeffs = self.rand_coeff([nmode, nsamp])
        samples = self.reconstruct_kl_reduced(coeffs)
        if return_coeffs:
            out = (samples, coeffs)
        else:
            out = samples
        return out

    def reconstruct_kl_reduced(self, coeffs):
        """
        """
        if len(coeffs.shape) == 1:
            coeffs = np.expand_dims(coeffs, 1)
        nmodes, nsamps = coeffs.shape
        field = np.tile(self.mean, [nsamps, 1]).T
        for imode in range(nmodes):
            vec1 = np.atleast_2d(coeffs[imode, :])
            vec2 = np.atleast_2d(self.kl_modes[:, imode])
            field += np.dot(vec1.T, vec2).T
        return field

    def project_kl_reduced(self, nmode, field):
        """
        """
        # TODO: handle matrices like other functions
        coeffs = []
        for imode in range(nmode):
            mode = self.kl_modes[:, imode]
            # TODO: why was JX using eigenvalues for the coefficients?
            # eig = self.kl_eig_vals[imode]
            # coeffs.append(np.sum(self.weight_field * field * mode) / eig)
            coeffs.append(projection_magnitude(field, mode, self.weight_field))
        return np.array(coeffs)

    def _set_attribute(self, name, value):
        """ """
        current_val = getattr(self, name)
        if current_val is None:
            setattr(self, name, value)
        else:
            if current_val != value:
                raise ValueError('Dimensions do not match.')


class GaussianProcess(RandomField):
    """
    """

    def __init__(self, **kwargs):
        """
        """
        super(self.__class__, self).__init__(**kwargs)
        if 'cov_cholesky_l' in kwargs:
            self.cov_cholesky_l = kwargs['cov_cholesky_l']
            self._set_attribute('npoints', self.cov_cholesky_l.shape[0])
        else:
            self.cov_cholesky_l = None

    def __str__(self):
        str_info = "Gaussian process scalar random field."
        return str_info

    def sample_full(self, nsamps):
        """
        """
        # TODO: Not sure working properly. See tutorial (cannot match)
        if self.cov_cholesky_l is None:
            # TODO: cholesky decomposition for sparse matrix
            self.cov_cholesky_l = la.cholesky(self.cov.todense())
        x_mat = np.random.normal(loc=0.0, scale=1.0,
                                 size=(self.npoints, nsamps))
        perturb = np.matmul(self.cov_cholesky_l, x_mat)
        samples = np.tile(self.mean, (nsamps, 1)).T + perturb
        return samples.A

    def rand_coeff(self, nsamp):
        """ """
        return np.random.normal(0, 1, nsamp)
