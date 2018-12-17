#!/usr/bin/env python
# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Example of creating and manipulating random fields. """

# third party imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# local imports
import random_field as rf


# inputs - coordinates
nxpoints = 20
nypoints = 20
ncells = nxpoints*nypoints

# inputs - length scales
length_scales = (1, 1)

# inputs - Number of modes to plot
nmodes = ncells-1
nmodes_list = [200, nmodes]

# calculations
# coordinates
xpos, ypos = np.meshgrid(np.arange(nxpoints), np.arange(nypoints))
xpos = np.atleast_2d(xpos.ravel()).T
ypos = np.atleast_2d(ypos.ravel()).T
coords = np.hstack([xpos, ypos])

# square exponential kernel and inputs
kernel_func = rf.covariance.kernel_sqrexp
kernel_args = {'coords': coords, 'length_scales': length_scales}

# generate covariance
cov = rf.covariance.generate_cov(
    kernel_func, perform_checks=False, sp_tol=1e-10, **kernel_args)

# initiliaze a Gaussian process random field object
nspatial_dims = 2
mean = np.zeros(ncells)
weight_field = np.ones(ncells)
rfield = rf.GaussianProcess(mean=mean, cov=cov, weight_field=weight_field,
                            nspatial_dims=nspatial_dims)

# calculate KL modes
rfield.calc_kl_modes(nmodes)
for imode in nmodes_list:
    coverage = rf.kl_coverage(
        rfield.cov, rfield.kl_eig_vals[:imode], rfield.weight_field)
    print('{:d} modes ({:.1%}) can cover {:.1%} of the covariance.'.format(
        imode, 1.*imode/ncells, coverage))

# create a "full" sample
tfield = rfield.sample_full(1)[:, 0]
coeffs = rfield.project_kl_reduced(nmodes, tfield-rfield.mean)

# plot
fig, axarr = plt.subplots(1, len(nmodes_list)+1, sharey=True)

tfield = tfield.reshape(nypoints, nxpoints)
im = axarr[-1].matshow(tfield)
axarr[-1].xaxis.set_ticks_position('bottom')
axarr[-1].set_title('Truth')
axarr[-1].set_aspect('equal')
divider = make_axes_locatable(axarr[-1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
vmin = im.colorbar.vmin
vmax = im.colorbar.vmax

for ifig, imode in enumerate(nmodes_list):
    field = rfield.reconstruct_kl_reduced(coeffs[:imode])
    field = field.reshape(nypoints, nxpoints)
    axarr[ifig].matshow(field, vmin=vmin, vmax=vmax)
    axarr[ifig].set_title('{:d} Modes'.format(imode))
    axarr[ifig].xaxis.set_ticks_position('bottom')
    axarr[ifig].set_aspect('equal')

plt.show()
