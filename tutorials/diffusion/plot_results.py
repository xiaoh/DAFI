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


if not os.path.exists('results_figures/'):
    os.mkdir('results_figures/')


def plot_inferred_mu(step, nmodes):
    """Plot inferred diffusivity field"""
    x_coor = np.loadtxt('./results_diffusion/x_coor.dat')
    KLmodes = np.loadtxt('./results_diffusion/KLmodes.dat')
    omega_all = np.loadtxt('./results/t_0/xa/xa_'+str(step))
    vx = np.zeros((len(x_coor), len(omega_all[0, :])))
    for i in range(len(omega_all[0, :])):
        for j in range(len(omega_all[:, 0])):
            vx[:, i] += omega_all[j, i] * KLmodes[:, j]
        vx[:, i] = np.exp(vx[:, i])
    samp_mean = np.sum(vx, 1)/vx.shape[1]
    xcoor_matrix = np.tile(x_coor, (len(omega_all[0, :]), 1))
    v1 = plt.plot(x_coor, vx, '-', color='0.7', lw=0.2)
    v2 = plt.plot(x_coor, samp_mean, 'b-.', label='sample mean')
    v3 = plt.plot(x_coor, np.ones(x_coor.shape)*1., 'r--', label='baseline')
    return v1, v2, v3


def main():
    # read required parameters
    with open('dafi.in', 'r') as f:
        input_dict = yaml.load(f, yaml.SafeLoader)['dafi']
    path, dirs, files = os.walk('./results/t_0/xa').__next__()
    final_step = len(files) - 1
    with open('diffusion.in', 'r') as f:
        model_dict = yaml.load(f, yaml.SafeLoader)
    nmodes = int(model_dict['nmodes'])
    mu_o = float(model_dict['prior_mean'])
    x_coor_obs = model_dict['obs_locations']
    x_coor = np.loadtxt('./results_diffusion/x_coor.dat')

    # read truth
    u_truth = np.loadtxt('./results_diffusion/u_truth.dat')
    mu_truth = np.loadtxt('./results_diffusion/mu_truth.dat') / mu_o
    obs = np.mean(np.loadtxt('./results/y/y_0'), 1)

    # plot results
    for i in range(2):
        if i == 0:
            case = 'prior'
            step = 0
        if i == 1:
            case = 'posterior'
            step = final_step

        # plot results in velocity
        fig1, ax1 = plt.subplots()
        u_mat = np.loadtxt(f'./results_diffusion/U.{step+1}')
        u_mean = np.mean(u_mat, 1)
        u1 = ax1.plot(x_coor, u_mat, '-', color='0.7', lw=0.2)
        u2 = ax1.plot(x_coor, u_mean, 'b-.', label='sample mean')
        u3 = ax1.plot(x_coor, u_truth, 'k-', label='truth')
        u4 = ax1.plot(x_coor_obs, obs, 'kx', label='observations')
        plt.xlabel(r'position $\xi_1/L$')
        plt.ylabel(r'output field $u$')
        plt.legend(
            [u1[0], u2[0], u3[0], u4[0]],
            ['samples', u2[0].get_label(), u3[0].get_label(), u4[0].get_label()],
            loc='best')
        figure_name = './figures/DA_u_' + case + '.png'
        plt.savefig(figure_name)

        fig2, ax2 = plt.subplots()
        ax2 = plt.subplot(111)
        v1, v2, v3 = plot_inferred_mu(step, nmodes)
        v4 = plt.plot(x_coor[1:-1], mu_truth[:-1], 'k-', label='truth')
        plt.xlabel(r'position $\xi_1/L$')
        plt.ylabel(r'diffusivity $\mu/\mu_0$')
        plt.legend(
            [v1[0], v2[0], v3[0], v4[0]],
            ['samples', v2[0].get_label(), v3[0].get_label(), v4[0].get_label()],
            loc='best')
        # save figure
        figure_name = './figures/DA_mu_'+case+'.png'
        plt.savefig(figure_name)


if __name__ == "__main__":
    main()
