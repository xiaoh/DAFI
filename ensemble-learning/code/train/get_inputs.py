#!/usr/bin/env python3

"""
As module:
    provides the 'get_inputs' method
As executable:
    Create the scalar invariants and tensorial basis from an OpenFOAM
    run.
    First run the following OpenFOAM utilities in the OpenFOAM case
    directory:
        >> postProcess -func 'turbulenceFields(R, k, epsilon)'
        >> postProcess -func 'grad(U)'
"""

import os

import numpy as np

from dafi.random_field import foam_utilities as foam


TENSORSQRTDIM = 3
TENSORDIM = 9
DEVSYMTENSORDIM = 5
DEVSYMTENSOR_INDEX = [0,1,2,4,5]

NSCALARINVARIANTS = 5
NTENSORBASIS = 10


def _thirdtrace(x):
    return 1./3.*np.trace(x)*np.eye(TENSORSQRTDIM)


def get_inputs(gradU, time_scale):
    ncells = len(time_scale)
    assert len(gradU)==ncells
    theta = np.zeros([ncells, NSCALARINVARIANTS])
    T = np.zeros([ncells, DEVSYMTENSORDIM, NTENSORBASIS])
    for i, (igradU, it) in enumerate(zip(gradU, time_scale)):
        # velocity gradient: symmetric and anti-symmetric components
        igradU = igradU.reshape([TENSORSQRTDIM, TENSORSQRTDIM])
        S = 0.5*(igradU + igradU.T)
        R = 0.5*(igradU - igradU.T)
        # normalized: linear
        S *= it
        R *= it
        # combinations: quadratic
        SS = S @ S
        SR = S @ R
        RS = R @ S
        RR = R @ R
        # combinations: cubic
        RSS = RS @ S
        RRS = RR @ S
        SRR = SR @ R
        SSS = SS @ S
        SSR = SS @ R
        # combinations: quartic
        RSRR = RS @ RR
        RRSR = RR @ SR
        SRSS = SR @ SS
        SSRS = SS @ RS
        RRSS = RR @ SS
        SSRR = SS @ RR
        # combinations: quintic
        RSSRR = RSS @ RR
        RRSSR = RRS @ SR

        # tensorial basis
        T1 = S
        T2 = SR - RS
        T3 = SS - _thirdtrace(SS)
        T4 = RR - _thirdtrace(RR)
        T5 = RSS - SSR
        T6 = RRS + SRR - 2.*_thirdtrace(SRR)
        T7 = RSRR - RRSR
        T8 = SRSS - SSRS
        T9 = RRSS + SSRR - 2.*_thirdtrace(SSRR)
        T10 = RSSRR - RRSSR
        # make symmteric and combine
        T_list = [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]
        for j, iT in enumerate(T_list):
            iT = iT.reshape([TENSORDIM])
            symmetric = True
            symm_pairs = [(1, 3), (2, 6), (5, 7)]
            for (si, sj) in symm_pairs:
                symmetric = symmetric and np.isclose(iT[si], iT[sj])
            deviatoric = np.isclose(iT[0]+iT[4]+iT[8], 0.0)
            check = None
            if check=='assert':
                assert symmetric
                assert deviatoric
            elif check=='print':
                if not symmetric:
                    print(f'Warning: T_{j+1} at cell {i} not symmetric, symmetry pairs: ({iT[1]}, {iT[3]}), ({iT[2]}, {iT[6]}), ({iT[5]}, {iT[7]}), ')
                if not deviatoric:
                    print(f'Warning: T_{j+1} at cell {i} not deviatoric, trace(T_{j+1}) = {iT[0]+iT[4]+iT[8]}')
            T[i, :, j] = iT[DEVSYMTENSOR_INDEX]

        # scalar invariants
        theta[i, 0] = np.trace(SS)
        theta[i, 1] = np.trace(RR)
        theta[i, 2] = np.trace(SSS)
        theta[i, 3] = np.trace(RRS)
        theta[i, 4] = np.trace(RRSS)
    return theta, T

def get_inputs_loc_norm(gradU, time_scale):
    ncells = len(time_scale)
    assert len(gradU)==ncells
    theta = np.zeros([ncells, NSCALARINVARIANTS])
    # T = np.zeros([ncells, DEVSYMTENSORDIM, NTENSORBASIS])
    for i, (igradU, it) in enumerate(zip(gradU, time_scale)):
        # velocity gradient: symmetric and anti-symmetric components
        igradU = igradU.reshape([TENSORSQRTDIM, TENSORSQRTDIM])
        S = 0.5*(igradU + igradU.T)
        R = 0.5*(igradU - igradU.T)
        # normalized: linear
        S *= it / (1 + it * np.linalg.norm(S))
        R *= it / (1 + it * np.linalg.norm(R))
        # combinations: quadratic
        SS = S @ S
        SR = S @ R
        RS = R @ S
        RR = R @ R
        # combinations: cubic
        RSS = RS @ S
        RRS = RR @ S
        SRR = SR @ R
        SSS = SS @ S
        SSR = SS @ R
        # combinations: quartic
        RSRR = RS @ RR
        RRSR = RR @ SR
        SRSS = SR @ SS
        SSRS = SS @ RS
        RRSS = RR @ SS
        SSRR = SS @ RR
        # combinations: quintic
        RSSRR = RSS @ RR
        RRSSR = RRS @ SR

        # scalar invariants
        theta[i, 0] = np.trace(SS)
        theta[i, 1] = np.trace(RR)
        theta[i, 2] = np.trace(SSS)
        theta[i, 3] = np.trace(RRS)
        theta[i, 4] = np.trace(RRSS)
    return theta

if __name__ == "__main__":
    # case details
    foamcase = input('OpenFOAM case: ')
    foamtimedir = input('Time directory: ')
    savedir = input('Save directory: ')

    # file names
    file_scalar_k = os.path.join(foamcase, foamtimedir, 'turbulenceProperties:k')
    file_scalar_eps = os.path.join(
        foamcase, foamtimedir, 'turbulenceProperties:epsilon')
    file_tensor_gradU = os.path.join(foamcase, foamtimedir, 'grad(U)')

    # TKE
    k_file = foam.read_field_file(file_scalar_k)
    k = k_file['internal_field']['value']

    # epsilon
    eps_file = foam.read_field_file(file_scalar_eps)
    eps = eps_file['internal_field']['value']

    # turbulence time scale
    time_scale = k/eps

    # grad U
    gradU_file = foam.read_field_file(file_tensor_gradU)
    gradU = gradU_file['internal_field']['value']

    # initialize variables
    ncells = len(k)
    theta = np.zeros([ncells, NSCALARINVARIANTS])
    T = np.zeros([ncells, DEVSYMTENSORDIM, NTENSORBASIS])

    # calculate tensorial basis and scalar invariants
    theta, T = get_inputs(gradU, time_scale)

    # save inputs
    files = [('tke', k), ('epsilon', eps), ('time_scale', time_scale),
             ('scalar_invariants', theta), ('basis_tensors', T)]
    for file, val in files:
        file = os.path.join(savedir, file + '.npy')
        np.save(file, val)
        print(f'Wrote file: {file}')
