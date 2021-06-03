#!/usr/bin/env python3

import numpy as np

from dafi import random_field as rf


coverage = 0.99
foam_case = "baseline_foam"

foam_rc = None  # modify if needed to run in your system

kernel = "mixed_periodic_sqrexp"
stddev = 1.0
length_scales = [0.25, 0.25]
period = [9.0, None]

cell_coords = rf.foam.get_cell_centres(foam_case, foam_rc=foam_rc)
cell_coords = cell_coords[:, :2]
cell_volumes = rf.foam.get_cell_volumes(foam_case, foam_rc=foam_rc)

kernel_kwargs = {'coords': cell_coords,
                 'length_scales': length_scales, 'period': period}

b1 = rf.foam.get_cell_centres(foam_case, group="topWall", foam_rc=foam_rc)
b2 = rf.foam.get_cell_centres(foam_case, group="bottomWall", foam_rc=foam_rc)
dirichlet_coords = np.vstack([b1[:, :2], b2[:, :2]])

_, cov = rf.covariance.bc_cov(
    kernel=kernel, stddev=stddev, dirichlet_coords=dirichlet_coords,
    kernel_kwargs=kernel_kwargs)

_, modes = rf.calc_kl_modes_coverage(
    cov, coverage, weight_field=cell_volumes, normalize=False)

mode_out_file = "klmodes"
np.savetxt(mode_out_file, modes)
