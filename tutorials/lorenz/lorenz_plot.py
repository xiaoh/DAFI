#!/usr/bin/env python2
# Copyright 2018 Virginia Polytechnic Insampletitute and State University.
""" This module is to postprocess the da_ta for the Lorenz model"""

# standard library imports
import os

# third party imports
import numpy as np
from scipy.integrate import ode
import matplotlib
import matplotlib.pyplot as plt

if not os.path.exists('figures/'):
    os.mkdir('figures/')

def plot_samples(para):
    """plot samples and sample mean"""

    if para=='x': state_ind=0
    if para=='y': state_ind=1
    if para=='z': state_ind=2
    # read time series
    t = np.loadtxt('obs.txt')[:,0]
    # get observed time point
    da_t = t[10:-1:10]
    da_step = len(da_t)
    
    #intialize sequential HX and mean
    HX_seq = []
    HX_mean = []
    
    for i in range(da_step):
        HX = np.loadtxt('./debugData/HX_'+str(i+1)+'.0')
        if i == 0:
            HX_seq = HX[state_ind,:]
            HX_mean.append(np.mean(HX[state_ind,:]))
        else:
            HX_seq = np.column_stack((HX_seq, HX[state_ind,:]))
            HX_mean.append(np.mean(HX[state_ind,:]))
    HX_mean = np.array(HX_mean)
    # tile time series
    t_matrix = np.tile(da_t, (len(HX_seq[:,0]), 1))
    # plot samples
    p1 = plt.plot(t_matrix.T,HX_seq.T,'g-', lw=0.2, label = 'sample')
    # plot sample mean
    p2 =  plt.plot(da_t,HX_mean, 'b-',lw=1, label = 'sample mean')
    return p1,p2

def plot_truth(para):
    """ plot synthetic truth """

    # read truth file
    obs = np.loadtxt('obs.txt')
    # set which state varible to plot
    if para == 'x': state_ind = 1
    if para == 'y': state_ind = 2
    if para == 'z': state_ind = 3
    # plot truth
    p4 = plt.plot(obs[:,0],obs[:,state_ind], 'k-', label= 'truth')

    return p4

def plot_obs(para):
    """ plot observation """

    # set which state varible to plot
    if para == 'x': state_ind = 0
    if para == 'y': state_ind = 1
    if para == 'z': state_ind = 2
    # read time series
    t = np.loadtxt('obs.txt')[:,0]
    # get observed time point
    da_t = t[10:-1:10]
    da_step = len(da_t)
    # get sequential noised observation
    obs_seq = []
    for i in range(da_step):
        obs = np.loadtxt('./debugData/obs_'+str(i+1)+'.0')
        if i == 0:
            obs_seq.append(obs[state_ind,0])
        else:
            obs_seq.append(obs[state_ind,0])
    obs_seq = np.array(obs_seq)
    # plot observation
    p5 = plt.plot(da_t,obs_seq,'r.', label = 'observation')

    return p5   

def main(iShow=False):

    # plot
    plt.figure()
    ax1=plt.subplot(111)
    p1,p2 = plot_samples(para)
    p4 = plot_truth(para)
    p5 = plot_obs(para)
    plt.xlabel('time')
    plt.ylabel(para)
    matplotlib.rcParams.update({'font.size':15})
    # save figure
    figureName = './figures/timeSeries_DA_'+ para +'.pdf'
    plt.savefig(figureName)

if __name__ == "__main__":
    
    para = 'x'
    main()
    para = 'y'
    main()
    para = 'z'
    main()
