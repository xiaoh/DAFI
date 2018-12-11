#!/usr/bin/env python2
# Copyright 2018 Virginia Polytechnic Insampletitute and State University.
""" This module is to postprocess the data for the heat diffusion model. """

# standard library imports
import os

# third party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#local imports
from dainv.utilities import read_input_data

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
        omega_all = np.loadtxt('./results/Xa_'+str(i+1))[-nmodes:, :]
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
        step_matrix, omega, 'g-', lw=0.2, alpha=0.5, label='sample')
    # plot sample mean
    om = plt.plot(step, omega_mean, 'b-', lw=1, label='sample mean')
    return o, om


def plot_omega_truth(ind, step):
    omega_truth = np.loadtxt('omega_truth.dat')
    truth = omega_truth[ind]
    x = np.arange(step)
    y = truth*np.ones(len(x))
    o_truth = plt.plot(x, y, 'k-', label='truth')
    return o_truth


def get_obs_coor():
    x_coor_obs = np.zeros(10)
    x_coor = np.loadtxt('x_coor.dat')
    for j in range(1, 11):
        x_coor_obs[j-1] = x_coor[5*j]
    return x_coor_obs


def plot_samples(step):
    """ Plot samples and sample mean."""
    # read x coordinate
    x_coor = get_obs_coor()
    hx_mean = []
    model_obs = np.loadtxt('./results/HX_'+str(step))
    hx = model_obs
    hx_mean = hx.mean(axis=1)
    # tile time series
    xcoor_matrix = np.tile(x_coor, (len(hx[0, :]), 1))
    # plot samples
    p1 = plt.plot(
        xcoor_matrix, hx.T, 'g-', lw=0.2, alpha=0.5, label='sample')
    # plot sample mean
    p2 = plt.plot(x_coor, hx_mean, 'b-', lw=1, label='sample mean')
    return p1, p2


def plot_truth():
    """ Plot truth. """
    # read truth file
    x_coor = np.loadtxt('x_coor.dat')
    truth = np.loadtxt('u_truth.dat')
    # plot truth
    p3 = plt.plot(x_coor, truth, 'k-', label='truth')
    return p3


def plot_obs():
    """ Plot observations. """
    # read time series
    x_coor = get_obs_coor()
    # get noised ensemble observation
    obs = np.loadtxt('./results/obs_100')
    obs_mean = obs.mean(axis=1)
    # plot observation
    p4 = plt.plot(x_coor, obs_mean, 'r.', label='observation')
    return p4


def plot_force():
    """ Plot force source"""
    x_coor = np.loadtxt('x_coor.dat')
    fx = plt.plot(x_coor, np.sin(2*np.pi*x_coor/5), 'r-', label='heat source')
    return fx


def plot_inferred_mu(step, nmodes):
    """Plot inferred diffusivity field"""
    x_coor = np.loadtxt('x_coor.dat')
    KLmodes = np.loadtxt('KLmodes.dat')
    omega_all = np.loadtxt('./results/Xa_'+str(step))[-nmodes:, :]
    vx = np.zeros((len(x_coor), len(omega_all[0, :])))
    for i in range(len(omega_all[0, :])):
        for j in range(len(omega_all[:, 0])):
            vx[:, i] += omega_all[j, i] * KLmodes[:, j]
        vx[:, i] = np.exp(vx[:, i])
    xcoor_matrix = np.tile(x_coor, (len(omega_all[0, :]), 1))
    v1 = plt.plot(x_coor, vx, 'g-', lw=0.2, alpha=0.5, label='sample')


def plot_mu_truth(nmodes):
    """ Plot observations. """
    # read x coordinate
    x_coor = np.loadtxt('x_coor.dat')
    # read truth
    truth = np.loadtxt('./mu_truth.dat')
    # plot truth
    v2 = plt.plot(x_coor[:-1], truth, 'k-', label='truth')
    # read truth of KL expansion coefficient
    omega_truth = np.loadtxt('omega_truth.dat')
    # read modes
    KLmodes = np.loadtxt('KLmodes.dat')
    # obtain projected truth
    v_truth = np.zeros((len(x_coor)))
    for i in range(nmodes):
        v_truth += omega_truth[i] * KLmodes[:, i]
    v3 = plt.plot(x_coor, np.exp(v_truth), 'r-', label='projected truth')
    return v2, v3


def main():
    # read required parameters
    da_dict = read_input_data('dainv.in')
    final_step = int(da_dict['max_da_iteration'])
    param_dict = read_input_data('diffusion.in')
    nmodes = int(param_dict['nmodes'])
    # make prior and posterior plot in observation space
    for i in range(2):
        if i == 0:
            case = 'prior'
            step = 1
        if i == 1:
            case = 'posterior'
            step = final_step
        fig1, ax1 = plt.subplots()
        ax1 = plt.subplot(111)
        p1, p2 = plot_samples(step)
        p3 = plot_truth()
        p4 = plot_obs()
        plt.xlabel('x coordinate')
        plt.ylabel('u')
        line = Line2D([0], [0], linestyle='-', color='g', alpha=0.5)
        label_1 = 'samples'
        label_2 = p2[0].get_label()
        label_3 = p3[0].get_label()
        label_4 = p4[0].get_label()
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(
            [line, p2[0], p3[0], p4[0]],
            [label_1, label_2, label_3, label_4],
            loc='center left',
            bbox_to_anchor=(1, 0.5))
        # save figure
        figure_name = './figures/DA_'+case+'.png'
        plt.savefig(figure_name)

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
        v1 = plot_inferred_mu(step, nmodes)
        v2, v3 = plot_mu_truth(nmodes)
        plt.xlabel('x coordinate')
        plt.ylabel(r'$\mu$')
        plt.legend(
            [line, v2[0], v3[0]],
            [label_1, v2[0].get_label(), v3[0].get_label()],
            loc='best')
        # save figure
        figure_name = './figures/DA_mu_'+case+'.png'
        plt.savefig(figure_name)

    # plot modes
    fig3, ax3 = plt.subplots()
    ax3 = plt.subplot(111)
    if nmodes == 5:
        m1, m2, m3, m4, m5 = plot_KLmodes(nmodes)
    if nmodes == 3:
        m1, m2, m3 = plot_KLmodes(nmodes)
    plt.xlabel('x coordinate')
    plt.ylabel('mode')
    plt.legend(loc='best')
    figure_name = './figures/DA_mode.png'
    plt.savefig(figure_name)

    # plot omega evolution
    for ind in range(nmodes):
        fig3, ax3 = plt.subplots()
        ax3 = plt.subplot(111)
        o, om = plot_omega_samples(ind, step, nmodes)
        o_truth = plot_omega_truth(ind, step)
        plt.xlabel('iterations')
        plt.ylabel('omega')
        line = Line2D([0], [0], linestyle='-', color='g', alpha=0.5)
        label_1 = 'samples'
        label_2 = om[0].get_label()
        label_3 = o_truth[0].get_label()
        plt.legend(
            [line, om[0], o_truth[0]],
            [label_1, label_2, label_3],
            loc='best')
        # save figure
        figure_name = './figures/DA_omega'+str(ind+1)+'.png'
        plt.savefig(figure_name)

    # plot force term
    fig4, ax4 = plt.subplots()
    ax4 = plt.subplot(111)
    fx = plot_force()
    plt.xlabel('x coordinate')
    plt.ylabel('heat flux')
    plt.legend(loc='best')
    # save figure
    figure_name = './figures/DA_force.png'
    plt.savefig(figure_name)


if __name__ == "__main__":
    main()
