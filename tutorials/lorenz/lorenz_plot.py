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


if not os.path.exists('figures/'):
    os.mkdir('figures/')


def plot_samples(para, da_interval):
    """ Plot samples and sample mean."""
    if para == 'x':
        state_ind = 0
    if para == 'y':
        state_ind = 1
    if para == 'z':
        state_ind = 2
    # read time series
    t = np.loadtxt('truth.dat')[:, 0]
    # get observed time point
    da_t = t[da_interval: -1: da_interval]
    da_step = len(da_t)

    # intialize sequential Xa and mean
    hx_seq = []
    hx_mean = []

    for i in range(da_step):
        model_obs = np.loadtxt('./results/Xa_'+str(i+1))[:3, :]
        if i == 0:
            hx_seq = model_obs[state_ind, :]
            hx_mean.append(np.mean(model_obs[state_ind, :]))
        else:
            hx_seq = np.column_stack((hx_seq, model_obs[state_ind, :]))
            hx_mean.append(np.mean(model_obs[state_ind, :]))
    hx_mean = np.array(hx_mean)
    # tile time series
    t_matrix = np.tile(da_t, (len(hx_seq[:, 0]), 1))
    # plot samples
    p1 = plt.plot(
        t_matrix.T, hx_seq.T, 'g-', lw=0.2, alpha=0.5, label='sample')
    # plot sample mean
    p2 = plt.plot(da_t, hx_mean, 'b-', lw=1, label='sample mean')
    return p1, p2


def plot_truth(para):
    """ Plot truth. """
    # read truth file
    obs = np.loadtxt('truth_plot.dat', skiprows=1)
    # set which state varible to plot
    if para == 'x':
        state_ind = 1
    if para == 'y':
        state_ind = 2
    if para == 'z':
        state_ind = 3
    # plot truth
    p4 = plt.plot(obs[:, 0], obs[:, state_ind], 'k-', label='truth')
    return p4


def plot_obs(para, da_interval):
    """ Plot observations. """
    # set which state varible to plot
    if para == 'x':
        state_ind = 0
    if para == 'y':
        state_ind = 1
    if para == 'z':
        state_ind = 2
    # read time series
    time = np.loadtxt('truth.dat')[:, 0]
    # get observed time point
    da_t = time[da_interval:-1:da_interval]
    da_step = len(da_t)
    # get sequential noised observation
    obs_seq = []
    for i in range(da_step):
        obs = np.loadtxt('./results/obs_'+str(i+1))
        obs_seq.append(obs[state_ind, 0])
    obs_seq = np.array(obs_seq)
    # plot observation
    p5 = plt.plot(da_t, obs_seq, 'r.', label='observation')
    return p5


def main():
    # plot
    for para in ['x', 'y', 'z']:
        fig, ax1 = plt.subplots()
        ax1 = plt.subplot(111)
        da_interval = 5
        p1, p2 = plot_samples(para, da_interval)
        p4 = plot_truth(para)
        p5 = plot_obs(para, da_interval)
        plt.xlabel('time')
        plt.ylabel(para)
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
        figure_name = './figures/timeSeries_DA_' + para + '.png'
        plt.savefig(figure_name)


if __name__ == "__main__":
    main()
