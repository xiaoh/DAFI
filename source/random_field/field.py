# -*- coding: utf-8 -*-
# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Random fields representation and manipulation. """

# standard library imports
import warnings

# third party imports
import numpy as np
import scipy.sparse as sp
import numpy.linalg as la


# functions
def calc_kl_modes(nmodes, cov, weight_field, normalize=False):
    """ Calculate the first N Karhunen-Loève modes for a covariance
    field.
    """
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
    magnitude = inner_product(field_1, field_2, weight_field) \
        / norm(field_2, weight_field)
    return magnitude


def projection(field_1, field_2, weight_field):
    """ Project field_1 onto field_2. """
    magnitude = projection_magnitude(field_1, field_2, weight_field)
    direction = unit(field_2, weight_field)
    return magnitude*direction


# random field classes
class RandomField(object):
    """ Parent class (template). Need to overright some functions.
    """

    def __init__(self, mean, nspatial_dims=3, **kwargs):
        """
        cov is scipy.sparse.csc
        """
        self.mean = mean
        self.ndim = len(mean)
        self.nspatial_dims = nspatial_dims
        if 'kl_modes' in kwargs:
            self.kl_modes = kwargs['kl_modes']
            self.nmodes = len(self.kl_modes)
        else:
            self.kl_modes = None
        if 'cov' in kwargs:
            self.cov = kwargs['cov']
        else:
            self.cov = None
        if 'weight_field' in kwargs:
            self.weight_field = kwargs['weight_field']
        else:
            self.weight_field = None

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
        if self.cov is None or self.weight_field is None:
            raise RunTimeError(
                'self.cov and self.weight_field must be defined.')
        self.nmodes = nmodes
        self.kl_eig_vals, self.kl_modes = calc_kl_modes(
            self.nmodes, self.cov, self.weight_field, normalize)

    def sample_full(self, nsamp):
        """
        """
        raise NotImplementedError(
            "Needs to be implemented in the child class!")
        samps = np.zeros([self.ndim, nsamps])
        return samps

    def sample_kl_reduced(self, nsamp, nmode):
        """
        """
        samp_list = []
        for isamp in range(nsamp):
            samp = self.mean.copy()
            coeff = self.rand_coeff(nmode)
            for imode in range(nmode):
                samp += coeff[imode]*self.kl_modes[:, imode]
            samp_list.append(samp)
        return samp_list

    def reconstruct_kl_reduced(self, coeffs):
        """
        """
        nmode = len(coeffs)
        field = self.mean.copy()
        for imode in range(nmode):
            field += coeffs[imode] * self.kl_modes[:, imode]
        return field

    def project_kl_reduced(self, nmode, field):
        """
        """
        coeffs = []
        for imode in range(nmode):
            mode = self.kl_modes[:, imode]
            # TODO: why was JX using eigenvalues for the coefficients?
            # eig = self.kl_eig_vals[imode]
            # coeffs.append(np.sum(self.weight_field * field * mode) / eig)
            coeffs.append(projection_magnitude(field, mode, self.weight_field))
        return coeffs


class GaussianProcess(RandomField):
    """
    """

    def __init__(self, mean, nspatial_dims, **kwargs):
        """
        """
        super(self.__class__, self).__init__(mean, nspatial_dims, **kwargs)
        if 'cov_cholesky_l' in kwargs:
            self.cov_cholesky_l = kwargs['cov_cholesky_l']
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
        x_mat = np.random.normal(loc=0.0, scale=1.0, size=(self.ndim, nsamps))
        perturb = np.matmul(self.cov_cholesky_l, x_mat)
        samples = np.tile(self.mean, (nsamps, 1)).T + perturb
        return samples.A

    def rand_coeff(self, nsamp):
        """ """
        return np.random.normal(0, 1, nsamp)
