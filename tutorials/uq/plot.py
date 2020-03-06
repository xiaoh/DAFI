#!/usr/bin/env python3

import os

import yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# plot options & style
xlim_state = (0., 1.2)
ylim_state = (0., 1.2)
xlim_obs = (0., 2.2)
ylim_obs = (0., 2.2)
xlim_state_post = (.6, .9)
ylim_state_post = (.95, 1.25)
xlim_obs_post = (0.65, .95)
ylim_obs_post = (1.8, 2.1)
num_bins = 5
h_marg = 75
density = False
sns.set(style="white")
sns.set_style("ticks")
mpl.rcParams.update({'text.usetex': True, 'text.latex.preamble': ['\\usepackage{gensymb}'],})
#plt.style.use('./style.mplstyle')

# read obs
uq_input_file = 'uq.in'
with open(uq_input_file, 'r') as f:
    uq_dict = yaml.load(f, yaml.SafeLoader)
obs = np.array(uq_dict['obs'])

# create save directory
fig_dir = './figures'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# PLOT
cases = ['enkf', 'rml', 'mda']

for name in cases:
    # Load data
    dir = 'results_' + name
    x0 = np.loadtxt(os.path.join(dir, 'xf', 'xf_0'))
    x1 = np.loadtxt(os.path.join(dir, 'xa', 'xa_0'))
    Hx0 = np.loadtxt(os.path.join(dir, 'Hx', 'Hx_0'))
    Hx1 = np.loadtxt(os.path.join(dir, 'Hxa', 'Hxa_0'))

    # make Pandas DataFrame
    prior = pd.DataFrame(np.append(x0.T, Hx0.T, axis=1),
                         columns=['x1', 'x2', 'z1', 'z2'])
    posterior = pd.DataFrame(np.append(x1.T, Hx1.T, axis=1),
                             columns=['x1', 'x2', 'z1', 'z2'])
    # SCATTER and HISTOGRAM
    # STATE SPACE
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.scatter(posterior.x1, posterior.x2, color='salmon')
    ax1.scatter(prior.x1, prior.x2, color='0.75')
    ax1.plot(prior.x1.mean(), prior.x2.mean(), 'ko')
    ax1.plot(posterior.x1.mean(), posterior.x2.mean(), 'ro')
    ax1.set_xlabel(r'state $\mathsf{x}_1$')
    ax1.set_ylabel(r'state $\mathsf{x}_2$')
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # plt.savefig(os.path.join(fig_dir, name + '_ss.pdf'))

    # OBS SPACE
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.scatter(posterior.z1, posterior.z2, color='salmon')
    ax1.scatter(prior.z1, prior.z2, color='0.75')
    ax1.plot(prior.z1.mean(), prior.z2.mean(), 'ko')
    ax1.plot(posterior.z1.mean(), posterior.z2.mean(), 'ro')
    ax1.plot(obs[0], obs[1], 'kx')
    ax1.set_xlabel(r'observation $\mathsf{y}_1$')
    ax1.set_ylabel(r'observation$\mathsf{y}_2$')
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # plt.savefig(os.path.join(fig_dir, name + '_obs.pdf'))

    # KDEPLOTS
    # STATE SPACE
    plt.clf()
    sns.kdeplot(posterior.x1, posterior.x2, cmap='Reds', shade=True, shade_lowest=False)
    sns.kdeplot(prior.x1, prior.x2, cmap='bone_r', shade=True, shade_lowest=False)
    plt.xlabel(r'state $\mathsf{x}_1$')
    plt.ylabel(r'state $\mathsf{x}_2$')
    # # Hide the right and top spines
    sns.despine(top=True, right=True)
    plt.savefig(os.path.join(fig_dir, name + '_ss_kde.pdf'))

    plt.figure()
    plt.xlim(xlim_state_post)
    plt.ylim(ylim_state_post)
    sns.kdeplot(posterior.x1, posterior.x2, cmap='Reds', shade=True, shade_lowest=False)
    plt.xlabel(r'state $\mathsf{x}_1$')
    plt.ylabel(r'state $\mathsf{x}_2$')
    # Hide the right and top spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # plt.savefig(os.path.join(fig_dir, name + '_ss_kde_post.pdf'))


    # # OBS SPACE
    plt.clf()
    sns.kdeplot(posterior.z1, posterior.z2, cmap='Reds', shade=True, shade_lowest=False)
    sns.kdeplot(prior.z1, prior.z2, cmap='bone_r', shade=True, shade_lowest=False)
    plt.xlabel(r'observation $\mathsf{y}_1$')
    plt.ylabel(r'observation $\mathsf{y}_2$')
    # # Hide the right and top spines
    sns.despine(top=True, right=True)
    plt.plot(obs[0], obs[1], 'kx')
    plt.savefig(os.path.join(fig_dir, name + '_obs_kde.pdf'))

    plt.figure()
    plt.xlim(xlim_obs_post)
    plt.ylim(ylim_obs_post)
    sns.kdeplot(posterior.z1, posterior.z2, cmap='Reds', shade=True, shade_lowest=False)
    plt.plot(obs[0], obs[1], 'kx')
    plt.xlabel('$y_1$')
    plt.ylabel('$y_2$')
    # Hide the right and top spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # plt.savefig(os.path.join(fig_dir, name + '_obs_kde_post.pdf'))
