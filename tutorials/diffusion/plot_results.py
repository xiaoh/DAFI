#!/usr/bin/env python3
# Copyright 2018 Virginia Polytechnic Insampletitute and State University.
""" This module is to postprocess the data for the heat diffusion model. """

# standard library imports
import os

# third party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
import yaml

mpl.rcParams.update({'text.usetex': True,
                     'text.latex.preamble': ['\\usepackage{gensymb}'], })


# create save directory
save_dir = 'results_figures'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# files and directories
dafi_input_file = 'dafi.in'
model_input_file = 'diffusion.in'
results_dir = 'results_diffusion'
dafi_results_dir = 'results'

# read required data
with open(os.path.join(dafi_results_dir, 't_0', 'iteration')) as file:
    final_step = int(file.read())
with open(model_input_file, 'r') as f:
    model_dict = yaml.load(f, yaml.SafeLoader)
with open(dafi_input_file, 'r') as f:
    dafi_dict = yaml.load(f, yaml.SafeLoader)['dafi']
mu_o = model_dict['prior_mean']
x_coor_obs = model_dict['obs_locations']
x_coor = np.loadtxt(os.path.join(results_dir, 'x_coor.dat'))
klmodes = np.loadtxt(os.path.join(results_dir, 'KLmodes.dat'))

# read truth
u_truth = np.loadtxt(os.path.join(results_dir, 'u_truth.dat'))
mu_truth = np.loadtxt(os.path.join(results_dir, 'mu_truth.dat'))
mu_truth /= mu_o
obs = np.loadtxt(os.path.join(results_dir, 'obs.dat'))
obs_error = np.loadtxt(os.path.join(results_dir, 'std_obs.dat'))
obs_error *= dafi_dict['convergence_factor']

# read baseline
u_base = np.loadtxt(os.path.join(results_dir, 'u_baseline.dat'))

# reconstruct diffusivity from KL modes
def reconstruct_inferred_mu(omega_mat, klmodes):
    """ reconstruct inferred diffusivity field. """
    nmodes, nsamps = omega_mat.shape
    ncells = klmodes.shape[0]
    mu = np.zeros([ncells, nsamps])
    for isamp in range(nsamps):
        for imode in range(nmodes):
            mu[:, isamp] += omega_mat[imode, isamp] * klmodes[:, imode]
        mu[:, isamp] = np.exp(mu[:, isamp])
    return mu

# plot results
prior = ('prior', 0, 'xf')
posterior = ('posterior', final_step, 'xa')
for case, step, fname in [prior, posterior]:
    # load results
    u_mat = np.loadtxt(f'./results_diffusion/U.{step+1}')
    u_mean = np.mean(u_mat, 1)
    omega_file = os.path.join(dafi_results_dir, fname, f'{fname}_0')
    omega_mat = np.loadtxt(omega_file)
    mu_mat = reconstruct_inferred_mu(omega_mat, klmodes)
    mu_mean = np.sum(mu_mat, 1) / mu_mat.shape[1]

    # plot results: output field
    fig1, ax1 = plt.subplots()
    u1 = ax1.plot(x_coor, u_mat, '-', color='0.7', lw=0.2)
    u2 = ax1.plot(x_coor, u_mean, 'b-.', label='sample mean')
    u3 = ax1.plot(x_coor, u_base, 'r--', label='baseline')
    u4 = ax1.plot(x_coor, u_truth, 'k-', label='truth')
    u5 = ax1.errorbar(x_coor_obs, obs, yerr=obs_error, fmt='kx')
    ax1.set_xlabel(r'position $\xi_1/L$')
    ax1.set_ylabel(r'output field $u$')
    # legend
    lines = [u1[0], u2[0], u3[0], u4[0], u5[0]]
    labels = ['samples', u2[0].get_label(), u3[0].get_label(),
              u4[0].get_label(), 'observations']
    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])
    ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))
    # save figure
    figure_name = os.path.join(save_dir, f'DA_u_{case}.pdf')
    plt.savefig(figure_name)
    if case == 'posterior':
        plt.show()
    plt.close()

    # plot results: diffusivity
    fig2, ax2 = plt.subplots()
    v1 = plt.plot(x_coor, mu_mat, '-', color='0.7', lw=0.2)
    v2 = plt.plot(x_coor, mu_mean, 'b-.', label='sample mean')
    v3 = plt.plot(x_coor, np.ones(x_coor.shape)*1., 'r--', label='baseline')
    v4 = plt.plot(x_coor[1:-1], mu_truth[:-1], 'k-', label='truth')
    ax2.set_xlabel(r'position $\xi_1/L$')
    ax2.set_ylabel(r'diffusivity $\mu/\mu_0$')
    # legend
    lines = [v1[0], v2[0], v3[0], v4[0]]
    labels = ['samples', v2[0].get_label(), v3[0].get_label(),
              v4[0].get_label()]
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])
    ax2.legend(lines, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))
    # save figure
    figure_name = os.path.join(save_dir, f'DA_mu_{case}.pdf')
    plt.savefig(figure_name)
    plt.close()
