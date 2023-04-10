
"""
Collection of regularization algorithms.
"""

import numpy as np

import neuralnet


# L1
def l1(w, w_tf=None):
    w = neuralnet.flatten_weights(w)
    J = np.linalg.norm(w, ord=1)
    dJdw = np.sign(w)
    return J, dJdw

lasso = l1


# L2
def l2(w):
    w = neuralnet.flatten_weights(w)
    J = np.linalg.norm(w, ord=2)**2
    dJdw = 2*w
    return J, dJdw

ridge = tikhonov = l2


# group lasso
def group_lasso(w):
    J = 0.
    dJdw = np.array([])
    for i, iweight in enumerate(w):
        if not (i % 2 == 0):
            # bias vectors
            iweight = np.atleast_2d(iweight).T
        for group in iweight:
            dim = len(group)
            L2 = np.linalg.norm(group, ord=2)
            J += np.sqrt(dim) * L2
            if np.allclose(group, np.zeros(group.shape)):
                dJdg = np.zeros(group.shape)
            else:
                dJdg = np.sqrt(dim) / L2 * group
            dJdw = np.concatenate([dJdw, dJdg])
    return J, dJdw


def sparse_group_lasso(w):
    J1, dJdw1 = lasso(w)
    J2, dJdw2 = group_lasso(w)
    return J1+J2, dJdw1+dJdw2
