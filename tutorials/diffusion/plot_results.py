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

mpl.rcParams.update({'text.latex.preamble': ['\\usepackage{gensymb}'], })
plt.style.use('./style.mplstyle')
# import pylab
# fig = pylab.figure()
# figlegend = pylab.figure(figsize=(3,2))


if not os.path.exists('figures/'):
    os.mkdir('figures/')


def plot_KLmodes(nmodes):
    """ Plot KL modes. """
    x_coor = np.loadtxt('x_coor.dat')
    KLmodes = np.loadtxt('KLmodes.dat')
    m1 = plt.plot(x_coor, KLmodes[:, 0], 'g-', label='mode 1')
    m2 = plt.plot(x_coor, KLmodes[:, 1], 'r-', label='mode 2')
    m3 = plt.plot(x_coor, KLmodes[:, 2], 'b-', label='mode 3')
    if nmodes == 3:
        return m1, m2, m3
    if nmodes == 5:
        m4 = plt.plot(x_coor, KLmodes[:, 3], 'k-', label='mode 4')
        m5 = plt.plot(x_coor, KLmodes[:, 4], 'y-', label='mode 5')
        return m1, m2, m3, m4, m5


def plot_omega_samples(ind, step, nmodes):
    """ Plot omega samples and sample mean."""
    x_coor = get_obs_coor()
    omega = []
    omega_mean = []
    for i in range(step):
        omega_all = np.loadtxt('./results/t_0/xa/xa_'+str(i+1))[-nmodes:, :]
        if i == 0:
            omega = omega_all[ind, :]
        else:
            omega = np.column_stack((omega, omega_all[ind, :]))
        omega_mean.append(np.mean(omega_all[ind, :]))
    omega_mean = np.array(omega_mean)
    step = np.arange(step)
    # tile time series
    step_matrix = np.tile(step, (len(omega_all[0, :]), 1))
    # plot samples
    o = plt.plot(
        step_matrix, omega, '-', color='0.7', label='Samples')
    # plot sample mean
    om = plt.plot(step, omega_mean, 'b-.', lw=0.2, label='Sample mean')
    return o, om


def plot_omega_truth(ind, step):
    omega_truth = np.loadtxt('omega_truth.dat')
    truth = omega_truth[ind]
    x = np.arange(step)
    y = truth*np.ones(len(x))
    o_truth = plt.plot(x, y, 'k-', alpha=1.0, lw=2.5, label='Truth')
    return o_truth


def get_obs_coor():
    # x_coor_obs = np.zeros(10)
    x_coor = np.loadtxt('x_coor.dat')
    # for j in range(1, 11):
        # x_coor_obs[j-1] = x_coor[5*j]
    return x_coor[[25,50,75]]


def plot_samples(step):
    """ Plot samples and sample mean."""
    # read x coordinate
    x_coor = get_obs_coor()
    hx_mean = []
    model_obs = np.loadtxt('./results/t_0/Hx/Hx_'+str(step))
    hx = model_obs
    # hx = np.zeros((len(x_coor),len(model_obs[0,:])))
    # for i in range(len(model_obs[0, :])):
    #     for j in range(len(model_obs[:, 0])):
    #         hx[:, i] += model_obs[j, i]
    #     hx[:, i] = np.exp(hx[:, i])
    hx_mean = hx.mean(axis=1)
    # tile time series
    xcoor_matrix = np.tile(x_coor, (len(hx[0, :]), 1))
    # plot samples
    p1 = plt.plot(xcoor_matrix, hx.T, '-', color='0.7', label='Samples', lw=0.2, alpha=1)
    # plot samples
    # p1 = plt.plot(x_coor, hx, '-', color='0.7', label='Samples', lw=0.2, alpha=1)
    # plot sample mean
    p2 = plt.plot(x_coor, hx_mean, 'b-.', lw=1.5, label='Sample mean')
    return p1, p2


def plot_truth():
    """ Plot truth. """
    # read truth file
    x_coor = np.loadtxt('x_coor.dat')
    truth = np.loadtxt('u_truth.dat')
    # plot truth
    p3 = plt.plot(x_coor, truth, 'k-', label='Truth')
    return p3



def plot_obs():
    """ Plot observations. """
    # read time series
    x_coor = get_obs_coor()
    # get noised ensemble observation
    obs = np.loadtxt('./results/y/y_0')
    obs_mean = obs.mean(axis=1)
    # plot observation
    p4 = plt.plot(x_coor, obs_mean, 'kx', label='Observations')
    return p4


def plot_force():
    """ Plot force source"""
    x_coor = np.loadtxt('x_coor.dat')
    fx = plt.plot(x_coor, np.sin(2*np.pi*x_coor/5), 'r-', label='heat source')
    return fx


def plot_inferred_mu(step, nmodes, mu_o):
    """Plot inferred diffusivity field"""
    x_coor = np.loadtxt('x_coor.dat')
    KLmodes = np.loadtxt('KLmodes.dat')
    omega_all = np.loadtxt('./results/t_0/xa/xa_'+str(step))#[-nmodes:, :]
    vx = np.zeros((len(x_coor), len(omega_all[0, :])))
    for i in range(len(omega_all[0, :])):
        for j in range(len(omega_all[:, 0])):
            vx[:, i] += omega_all[j, i] * KLmodes[:, j]
        vx[:, i] = np.exp(vx[:, i]) / mu_o
    samp_mean = np.sum(vx, 1)/vx.shape[1]
    xcoor_matrix = np.tile(x_coor, (len(omega_all[0, :]), 1))
    v1 = plt.plot(x_coor, vx, '-', color='0.7', lw=0.2)
    v2 = plt.plot(x_coor, samp_mean, 'b-.', label='Sample Mean')
    v3 = plt.plot(x_coor, np.ones(x_coor.shape)*1., 'r--', label='baseline')


def plot_mu_truth(nmodes, mu_o):
    """ Plot observations. """
    # read x coordinate
    x_coor = np.loadtxt('x_coor.dat')
    # read truth
    truth = np.loadtxt('./mu_truth.dat') / mu_o
    # plot truth
    v2 = plt.plot(x_coor[1:-1], truth, 'k-', label='Truth')
    # read truth of KL expansion coefficient
    omega_truth = np.loadtxt('omega_truth.dat')
    # read modes
    KLmodes = np.loadtxt('KLmodes.dat')
    # obtain projected truth
    v_truth = np.zeros((len(x_coor)))
    for i in range(nmodes):
        v_truth += omega_truth[i] * KLmodes[:, i]
    # v3 = plt.plot(x_coor, np.exp(v_truth), 'r--', label='projected truth')
    return v2


def main():
    # read required parameters
    with open('dafi.in', 'r') as f:
       input_dict = yaml.load(f, yaml.SafeLoader)['dafi']
    final_step = int(input_dict['max_iterations']) -1
    with open('diffusion.in', 'r') as f:
       model_dict = yaml.load(f, yaml.SafeLoader)
    nmodes =  int(model_dict['nmodes'])

    # velocity
    mu_o = float(model_dict['mu_init'])
    x_coor = np.loadtxt('x_coor.dat')
    truth = np.loadtxt('u_truth.dat')
    x_coor_obs = get_obs_coor()
    obs = np.mean(np.loadtxt('./results/y/y_0'), 1)

    # prior
    prior_iteration = 0
    u_mat_prior = np.loadtxt(f'./results_diffusion/U.{prior_iteration}')
    u_mean_prior = np.mean(u_mat_prior, 1)

    fig, ax = plt.subplots()
    ax.plot(x_coor, u_mat_prior, '-', color='0.7', lw=0.2)
    ax.plot(x_coor, u_mean_prior, 'b-.')
    ax.plot(x_coor, truth, 'k-', label='Truth')
    ax.plot(x_coor_obs, obs, 'kx')
    plt.xlabel(r'position $\xi_1/L$')
    plt.ylabel(r'output field $u$')
    figure_name = './figures/DA_' + 'prior' + '.pdf'
    plt.savefig(figure_name)

    # posterior
    posterior_iteration = final_step
    u_mat_posterior = np.loadtxt(f'./results_diffusion/U.{posterior_iteration}')
    u_mean_posterior = np.mean(u_mat_posterior, 1)

    fig, ax = plt.subplots()
    ax.plot(x_coor, u_mat_posterior, '-', color='0.7', lw=0.2)
    ax.plot(x_coor, u_mean_posterior, 'b-.')
    ax.plot(x_coor, truth, 'k-', label='Truth')
    ax.plot(x_coor_obs, obs, 'kx')
    plt.xlabel(r'position $\xi_1/L$')
    plt.ylabel(r'output field $u$')
    figure_name = './figures/DA_' + 'posterior' + '.pdf'
    plt.savefig(figure_name)


    # make prior and posterior plot in observation space
    # for i in range(2):
    #     if i == 0:
    #         case = 'prior'
    #         step = 1
    #     if i == 1:
    #         case = 'posterior'
    #         step = final_step
    #     fig1, ax1 = plt.subplots()
    #     ax1 = plt.subplot(111)
    #     p1, p2 = plot_samples(step)
    #     p3 = plot_truth()
    #     p4 = plot_obs()
    #     plt.xlabel(r'$x/L$')
    #     plt.ylabel(r'$u$')
    #     line = Line2D([0], [0], linestyle='-', color='g', alpha=0.5)
    #     label_1 = 'samples'
    #     label_2 = p2[0].get_label()
    #     label_3 = p3[0].get_label()
    #     label_4 = p4[0].get_label()
    #     box = ax1.get_position()
    #     ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #     # plt.legend(
    #     #     [line, p2[0], p3[0], p4[0]],
    #     #     [label_1, label_2, label_3, label_4],
    #     #     loc='center left',
    #     #     bbox_to_anchor=(1, 0.5))
    #     # save figure
    #     figure_name = './figures/DA_'+case+'.pdf'
    #     plt.savefig(figure_name)

    # plot inferred mu
    for i in range(2):
        if i == 0:
            case = 'prior'
            step = 1
        if i == 1:
            case = 'posterior'
            step = final_step
        fig2, ax2 = plt.subplots()
        ax2 = plt.subplot(111)
        v1 = plot_inferred_mu(step, nmodes, mu_o)
        v2 = plot_mu_truth(nmodes, mu_o)
        plt.xlabel(r'position $\xi_1/L$')
        plt.ylabel(r'diffusivity $\mu/\mu_0$')
        # plt.legend(
        #     [line, v2[0], v3[0]],
        #     [label_1, v2[0].get_label(), v3[0].get_label()],
        #     loc='best')
        # save figure
        figure_name = './figures/DA_mu_'+case+'.pdf'
        plt.savefig(figure_name)

    # # plot modes
    # fig3, ax3 = plt.subplots()
    # ax3 = plt.subplot(111)
    # if nmodes == 5:
    #     m1, m2, m3, m4, m5 = plot_KLmodes(nmodes)
    # if nmodes == 3:
    #     m1, m2, m3 = plot_KLmodes(nmodes)
    # plt.xlabel('x coordinate')
    # plt.ylabel('mode')
    # # plt.legend(loc='best')
    # figure_name = './figures/DA_mode.pdf'
    # plt.savefig(figure_name)

    # # plot omega evolution
    # for ind in range(nmodes):
    #     fig3, ax3 = plt.subplots()
    #     ax3 = plt.subplot(111)
    #     o, om = plot_omega_samples(ind, step, nmodes)
    #     o_truth = plot_omega_truth(ind, step)
    #     plt.xlabel('iterations')
    #     plt.ylabel('omega')
    #     line = Line2D([0], [0], linestyle='-', color='g', alpha=0.5)
    #     label_1 = 'samples'
    #     label_2 = om[0].get_label()
    #     label_3 = o_truth[0].get_label()
    #     # plt.legend(
    #     #     [line, om[0], o_truth[0]],
    #     #     [label_1, label_2, label_3],
    #     #     loc='best')
    #     # save figure
    #     figure_name = './figures/DA_omega'+str(ind+1)+'.pdf'
    #     plt.savefig(figure_name)

    # # plot force term
    # fig4, ax4 = plt.subplots()
    # ax4 = plt.subplot(111)
    # fx = plot_force()
    # plt.xlabel('x coordinate')
    # plt.ylabel('heat flux')
    # # plt.legend(loc='best')
    # # save figure
    # figure_name = './figures/DA_force.pdf'
    # plt.savefig(figure_name)

if __name__ == "__main__":
    main()
    # plt.show()
