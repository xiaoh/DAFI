#!/usr/bin/env python3
# Copyright 2018 Virginia Polytechnic Insampletitute and State University.
""" This module is for postprocessing the data for the Lorenz model. """

# standard library imports
import os
import importlib

# third party imports
import numpy as np
import matplotlib.pyplot as plt
import yaml

# local import
from lorenz import lorenz #TODO: import lorenz model with importlib


input_file = 'dafi.in'
with open(input_file, 'r') as f:
   input_dafi = yaml.load(f, yaml.SafeLoader)

# what to plot
plot_truth = True
plot_obs = True
plot_sampmean = True
plot_samps = True
plot_baseline = True

# constants
nstate = 4
nstateorg = 3
nparams = 1
observe_state = [True, False, True]

# create directory
savedir = 'results_postprocess'
if not os.path.exists(savedir):
    os.makedirs(savedir)

# directories where results are saved
da_dir = "results_dafi"
lorenz_dir = "results_lorenz"

# enable LaTex
plt.rc('text', usetex=True)

# get observations and truth
obs = np.loadtxt(lorenz_dir + os.sep + 'obs.dat')
truth = np.loadtxt(lorenz_dir + os.sep + 'truth.dat')
ndastep = obs.shape[0]
da_time = obs[:, 0]

# get initial states (Xi)
if plot_samps or plot_sampmean or plot_baseline:
    X0 = np.loadtxt(da_dir + os.sep + '/xf/xf_0')
    # X0m = np.loadtxt(da_dir + os.sep + 'X_0_mean')
    X0m = np.array([-8.5, -7, 27.0, 28.0]) #TODO
    nsamples = X0.shape[1]

if plot_samps or plot_sampmean:
    # get forecast and analysis states (Xf, Xa)

    def read_dasteps(name, nstep=ndastep):
        prefix = da_dir + os.sep + name + os.sep + name
        filenames = [prefix + '_' + str(istep) for istep in range(nstep)]
        return np.stack([np.loadtxt(f) for f in filenames])

    Xa = read_dasteps('xa')
    Xf = read_dasteps('xf')
    Xam = np.mean(Xa, axis=2)
    Xfm = np.mean(Xf, axis=2)
    t_matrix = np.tile(da_time, (nsamples, 1)).T

    # get in-between values for X

    prefix = lorenz_dir + os.sep + 'states' + os.sep + 'time'
    filenames = [prefix + '_' + str(istep+1) for istep in range(ndastep-1)]
    time_all = np.concatenate([np.loadtxt(f) for f in filenames])
    time_all = np.concatenate([time_all, [da_time[-1]]])

    prefix = lorenz_dir + os.sep + 'states' + os.sep + 'dastep_'
    Xall = []
    for isamp in range(nsamples):
        post = '_samp_{}'.format(isamp)
        filenames = [prefix + str(istep+1) + post for istep in range(ndastep-1)]
        Xall += [np.concatenate([np.loadtxt(f) for f in filenames], axis=0)]
    Xall = np.stack(Xall, axis=2)
    Xall = np.concatenate([Xall, Xa[-1:, :3, :]], axis=0)
    Xallm = np.mean(Xall, axis=2)
    t_all_mat = np.tile(time_all, (nsamples, 1)).T

# get baseline
if plot_baseline:
    t_base = truth[:, 0]
    params = np.loadtxt(lorenz_dir + os.sep + 'params.dat')
    Xbase = lorenz(t_base, X0m[:3], [X0m[3:], params[0], params[1]])

# get rho
if plot_samps or plot_sampmean:
    istate = 3
    time_rho = [0] + [val for pair in zip(da_time, da_time) for val in pair]
    val1 = Xf[:, istate, :]
    val2 = Xa[:, istate, :]
    Rho_all = [X0[istate, :]] + \
        [val for pair in zip(val1, val2) for val in pair]
    Rho_all = np.array(Rho_all)
    Rho_allm = np.mean(Rho_all, axis=1)
    t_rho_mat = np.tile(time_rho, (nsamples, 1)).T
rho_true = np.loadtxt(lorenz_dir + os.sep + 'rho.dat')

# plot states
fig1, axarr1 = plt.subplots(nstateorg, 1, sharex=True)
state_ind = 0
lines1 = []
for istate, ax in enumerate(axarr1):
    if plot_samps:
        plot = ax.plot(t_all_mat, Xall[:, istate, :], 'g--', alpha=0.25,
                       label='Samples')
        ax.plot([0]*nsamples, X0[istate, :], 'g*', alpha=0.25, fillstyle='full',
                markersize=5)
        ax.plot(t_matrix, Xa[:, istate, :], 'g.', alpha=0.25, fillstyle='full',
                markersize=5)
        ax.plot(t_matrix, Xf[:, istate, :], 'g.', alpha=0.25, fillstyle='none',
                markersize=5)
        if istate == 0:
            lines1.append(plot[0])
    if plot_truth:
        plot = ax.plot(truth[:, 0], truth[:, istate+1], 'k-',
                       alpha=1.0, lw=2.5, label='Truth')
        if istate == 0:
            lines1.append(plot[0])
    if plot_baseline:
        plot = ax.plot(t_base, Xbase[:, istate], 'c--', alpha=1.0,
                       label='Baseline')
        if istate == 0:
            lines1.append(plot[0])
    if plot_sampmean:
        plot = ax.plot(time_all, Xallm[:, istate], 'b--', label='Sample mean')
        ax.plot(0, X0m[istate], 'b*', fillstyle='full', markersize=5)
        ax.plot(da_time, Xam[:, istate], 'b.', fillstyle='full', markersize=5)
        ax.plot(da_time, Xfm[:, istate], 'b.', fillstyle='none', markersize=5)
        if istate == 0:
            lines1.append(plot[0])
    if plot_obs and observe_state[istate]:
        plot = ax.plot(obs[:, 0], obs[:, state_ind + 1], 'r.', markersize=5,
                       label='Observations')
        if state_ind == 0:
            lines1.append(plot[0])
        state_ind += 1

# Plot rho
lines2 = []
istate = 3
fig2, ax2 = plt.subplots()
if plot_samps:
    plot = ax2.plot(t_rho_mat, Rho_all, 'g--', alpha=0.25, label='Samples')
    ax2.plot([0]*nsamples, X0[istate, :], 'g*', markersize=5, alpha=0.25)
    ax2.plot(t_matrix, Xa[:, istate, :], 'g.', alpha=0.25, fillstyle='full',
             markersize=5)
    ax2.plot(t_matrix, Xf[:, istate, :], 'g.', alpha=0.25, fillstyle='none',
             markersize=5)
    lines2.append(plot[0])
if plot_baseline:
    plot = ax2.plot([0.0, da_time[-1]], [X0m[istate], X0m[istate]], 'c--',
                    alpha=1.0, label='Baseline')
    lines2.append(plot[0])
if plot_truth:
    plot = ax2.plot([0.0, da_time[-1]], [rho_true, rho_true], 'k-', alpha=1.0,
                    lw=2.5, label='Truth')
    lines2.append(plot[0])
    pass
if plot_sampmean:
    plot = ax2.plot(time_rho, Rho_allm, 'b--', label='Sample mean')
    ax2.plot(0, X0m[istate], 'b*', markersize=5)
    ax2.plot(da_time, Xam[:, istate], 'b.', fillstyle='full', markersize=5)
    ax2.plot(da_time, Xfm[:, istate], 'b.', fillstyle='none', markersize=5)
    lines2.append(plot[0])

# format, add legend, save
axarr1[-1].set_xlabel('time [s]')
orgstate_name = ['x', 'y', 'z']
for istate, ax in enumerate(axarr1):
    ax.set_ylabel(orgstate_name[istate])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
labels1 = [line.get_label() for line in lines1]
fig1.legend(lines1, labels1, loc='center right')

ax2.set_xlabel('time [s]')
ax2.set_ylabel(r'$\rho$')
box = ax2.get_position()
ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
labels2 = [line.get_label() for line in lines2]
fig2.legend(lines2, labels2, loc='center right')

fig1.savefig(savedir + os.sep + 'plot_state.pdf')
fig2.savefig(savedir + os.sep + 'plot_rho.pdf')

plt.show()
plt.close('all')
