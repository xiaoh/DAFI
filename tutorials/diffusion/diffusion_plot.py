#!/usr/bin/env python2
# Copyright 2018 Virginia Polytechnic Insampletitute and State University.
""" This module is to postprocess the data for the Lorenz model. """

# standard library imports
import os

# third party imports
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pdb

if not os.path.exists('figures/'):
    os.mkdir('figures/')


def plot_KLmodes():
    
    x_coor = np.loadtxt('x_coor.dat')
    KLmodes = np.loadtxt('KLmodes.dat')
    m1 = plt.plot(x_coor,KLmodes[:,0],'g-', label = 'mode 1')
    m2 = plt.plot(x_coor,KLmodes[:,1],'r-', label = 'mode 2')
    m3 = plt.plot(x_coor,KLmodes[:,2],'b-', label = 'mode 3')
    return m1, m2, m3


def plot_omega_samples(ind):
    """ Plot omega samples and sample mean."""
    x_coor = get_obs_coor()
    omega = []
    omega_mean = []

    for i in range(100):

        omega_all = np.loadtxt('./results/Xa_'+str(i+1))[-3:,:]
        if i == 0:
            omega = omega_all[ind, :]
        else:
            omega = np.column_stack((omega, omega_all[ind, :]))
        #pdb.set_trace()
        omega_mean.append(np.mean(omega_all[ind, :]))
    omega_mean = np.array(omega_mean)
    step = np.arange(100)
    # tile time series
    step_matrix = np.tile(step, (len(omega_all[0, :]), 1))
    # plot samples
    o = plt.plot(
        step_matrix, omega, 'g-', lw=0.2, alpha=0.5, label='sample')
    # plot sample mean
    om = plt.plot(step, omega_mean, 'b-', lw=1, label='sample mean')
    return o, om

def plot_omega_truth(ind):
    if ind == 0:
        truth = 22
    if ind == 1:
        truth = 8
    if ind == 2:
        truth = 10
    x = np.arange(100)
    y = truth*np.ones(len(x))
    o_truth = plt.plot(x, y, 'k-', label='truth')
    return o_truth

def get_obs_coor():
    x_coor_obs = np.zeros(10)
    x_coor = np.loadtxt('x_coor.dat')
    for j in range(1,11):
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
    truth = np.loadtxt('truth.dat')
    # plot truth
    p4 = plt.plot(x_coor, truth, 'k-', label='truth')
    return p4


def plot_obs():
    """ Plot observations. """
    # read time series
    x_coor = get_obs_coor()
    # get noised ensemble observation
    obs = np.loadtxt('./results/obs_100')
    obs_mean = obs.mean(axis=1)
    # plot observation
    p5 = plt.plot(x_coor, obs_mean, 'r.', label='observation')
    return p5


def main():
    # plot observation space
    for i in range(2):
        if i == 0:
            case = 'prior'
            step = 1
        if i == 100:
            case = 'posterior'
            step = 100
        fig1, ax1 = plt.subplots()
        ax1 = plt.subplot(111)
        p1, p2 = plot_samples(step)
        p4 = plot_truth()
        p5 = plot_obs()
        plt.xlabel('x coordinate')
        plt.ylabel('u')
        line = Line2D([0], [0], linestyle='-', color='g', alpha=0.5)
        label_1 = 'samples'
        label_2 = p2[0].get_label()
        label_4 = p4[0].get_label()
        label_5 = p5[0].get_label()
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(
            [line, p2[0], p4[0], p5[0]],
            [label_1, label_2, label_4, label_5],
            loc='center left',
            bbox_to_anchor=(1, 0.5))
        # save figure
        figure_name = './figures/DA_'+case+'.png'
        plt.savefig(figure_name)

    # plot modes
    fig2, ax2 = plt.subplots()
    ax2 = plt.subplot(111)
    m1,m2,m3 = plot_KLmodes()
    plt.xlabel('x coordinate')
    plt.ylabel('mode')
    plt.legend(loc='best')
    figure_name = './figures/DA_mode.png'
    plt.savefig(figure_name)

    # plot omega evolution
    
    for ind in range(3):
        fig3, ax3 = plt.subplots()
        ax3 = plt.subplot(111)
        o, om = plot_omega_samples(ind)
        o_truth = plot_omega_truth(ind)
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


if __name__ == "__main__":
    main()
