#!/usr/bin/env python
# Copyright 2020 Virginia Polytechnic Institute and State University.
""" Plotting script for the periodic hill OpenFOAM cases. """

# standard library imports
import os
import argparse

# third party imports
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scipy.stats as stats
import yaml


# set plot properties
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'text.usetex': True,
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'figure.autolayout': True,
    'image.cmap': 'magma',
    'axes.grid': False,
    'savefig.dpi': 500,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'font.size': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.figsize': [3.2, 1.75],
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'lines.linewidth': 2,
    'lines.markersize': 6,
}
mpl.rcParams.update(params)


def plot_domain(ls):
    """ Plot the simulation domain. """
    ypos = np.arange(0, 9, 0.01)
    yext = np.array([9, 9, 0, 0])
    hill = _pehill_profile(ypos)
    hill_c = np.append(hill[ypos>=4.5], hill[ypos<4.5])
    hext = np.array([0.0, 3.036, 3.036, 0.0])
    plt.plot(ypos, hill_c, ls[0], color=ls[1], lw=ls[2], dashes=ls[3],
             alpha=ls[4])
    plt.plot([0,9], [3.036, 3.036], ls[0], color=ls[1], lw=ls[2],
             dashes=ls[3], alpha=ls[4])
    plt.plot([0,0], [0, 3.036], '--', color=ls[1], lw=ls[2], alpha=0.5*ls[4])
    plt.plot([9,9], [0, 3.036], '--', color=ls[1], lw=ls[2], alpha=0.5*ls[4])


def plot_profile_x(filename, xpos, scale, xcol, ycol, norm, ls, cl, lw, dash,
                   alpha):
    data = np.loadtxt(filename, comments='%')
    yval = data[:,ycol]
    xval = data[:,xcol]/norm * scale + float(xpos)
    plot, = plt.plot(xval, yval, linestyle=ls, color=cl, lw=lw, dashes=dash,
        alpha=alpha)
    return plot, xval, yval


def plot_profile_y(filename, ypos, scale, xcol, ycol, norm, cyclic, ls, cl,
                   lw, dash, alpha):
    data = np.loadtxt(filename, comments='%')
    yval = data[:,ycol]/norm * scale + float(ypos)
    xval = data[:,xcol]
    xval_c1 = xval[xval>=4.5] - 4.5
    xval_c2 = xval[xval<4.5] + 4.5
    yval_c1 = yval[xval>=4.5]
    yval_c2 = yval[xval<4.5]
    plot, = plt.plot(xval_c1, yval_c1, linestyle=ls, marker='', color=cl,
                     lw=lw, dashes=dash, alpha=alpha)
    plot, = plt.plot(xval_c2, yval_c2, linestyle=ls, marker='', color=cl,
                     lw=lw, dashes=dash, alpha=alpha)
    if cyclic:
        plot, = plt.plot([xval_c1[-1], xval_c2[0]], [yval_c1[-1], yval_c2[0]],
                         linestyle='--', marker='', color=cl, lw=lw,
                         alpha=alpha*0.5)
    return plot, xval, yval


def _pehill_profile(ypos):
    'Calculate the shape of the periodic hill'
    ypos = np.array(ypos)
    xpos = ypos * 28.0
    hill = np.zeros(len(xpos))
    for i in range(len(xpos)):
        if xpos[i] > 126.0 :
            xpos[i] = 252.0 - xpos[i]
        if (xpos[i]>=0) and (xpos[i]<9) :
            hill[i] = np.minimum(
                28., 2.8e+01 + 0.0e+00*xpos[i] + \
                6.775070969851e-03*xpos[i]**2 - 2.124527775800e-03*xpos[i]**3)
        elif (xpos[i]>=9) and (xpos[i]<14) :
            hill[i] = 2.507355893131E+01 + 9.754803562315E-01*xpos[i] - \
                1.016116352781E-01*xpos[i]**2 + 1.889794677828E-03*xpos[i]**3
        elif (xpos[i]>=14) and (xpos[i]<20) :
            hill[i] = 2.579601052357E+01 + 8.206693007457E-01*xpos[i] - \
                9.055370274339E-02*xpos[i]**2 + 1.626510569859E-03*xpos[i]**3
        elif (xpos[i]>=20) and (xpos[i]<30) :
            hill[i] = 4.046435022819E+01 - 1.379581654948E+00*xpos[i] + \
                1.945884504128E-02*xpos[i]**2 - 2.070318932190E-04*xpos[i]**3
        elif (xpos[i]>=30) and (xpos[i]<40) :
            hill[i] = 1.792461334664E+01 + 8.743920332081E-01*xpos[i] - \
                5.567361123058E-02*xpos[i]**2 + 6.277731764683E-04*xpos[i]**3
        elif (xpos[i]>=40) and (xpos[i]<=54) :
            hill[i] = np.maximum(
                0., 5.639011190988E+01 - 2.010520359035E+00*xpos[i] + \
                1.644919857549E-02*xpos[i]**2 + 2.674976141766E-05*xpos[i]**3)
        elif (xpos[i]>54) and (xpos[i]<=126) :
            hill[i] = 0;
    return hill/28.0


def main(input_file):
    # define color for different profiles
    # [line_style, color, line_width, dashes, opacity]
    line_profile = ['-', 'k', 2.5, (None, None), 1.0]
    line_truth = ['-', 'k', 2.0, (None, None), 1.0]
    line_baseline = ['-', 'red', 1.6, (12, 3), 1.0]
    line_sample_mean = ['-', 'blue', 1.5, (12, 3), 1.0]
    line_sample_median = ['-', 'cyan', 1.5, (12, 3), 1.0]
    line_sample_mode = ['-', 'magenta', 1.5, (12, 3), 1.0]
    line_samples = ['-', 'grey', 1.0, (None, None), 0.25]

    # marker: style, color, edgewidth, size
    markers_obs = ['x', 'black', 2, 10]

    # legend names
    truth_name = 'Truth'
    baseline_name = 'Prior'
    samples_name = 'Samples'
    sample_mean_name = 'Posterior'
    sample_median_name = 'Sample Median'
    sample_mode_name = 'Sample Mode'
    obs_name = 'Observations'

    # read input file
    with open(input_file, 'r') as f:
        inputs = yaml.load(f, yaml.SafeLoader)
    dir_truth = inputs['truth_dir']
    dir_base = inputs['base_dir']
    dir_samples = inputs['sample_dir']
    obs_file = inputs['obs_file']
    save_dir = inputs['save_dir']
    field = inputs['field']
    field_name = inputs['field_name']
    index = inputs['index']
    norm = inputs['norm']
    norm_name = inputs['norm_name']
    scale = inputs['scale']
    da_step = inputs['da_step']
    x_pos_list = [str(x) for x in inputs['x_pos_list']]
    x_name_list = [str(x) for x in inputs['x_name_list']]
    y_pos_list = [str(y) for y in inputs['y_pos_list']]
    y_name_list = [str(y) for y in inputs['y_name_list']]
    y_cyclic = inputs['y_cyclic']
    nsamps = inputs['nsamps']

    show_plot = inputs.get('show_plot', False)
    fix_aspect = inputs.get('fix_aspect', True)
    plot_truth = inputs.get('plot_truth', False)
    plot_baseline = inputs.get('plot_baseline', False)
    plot_obs = inputs.get('plot_obs', True)
    plot_samples = inputs.get('plot_samples', True)
    plot_sample_mean = inputs.get('plot_sample_mean', True)
    plot_sample_median = inputs.get('plot_sample_median', False)
    plot_sample_mode = inputs.get('plot_sample_mode', False)
    file_name = inputs.get('file_name', 'figure.pdf')
    show_legend = inputs.get('show_legend', True)
    legend_top = inputs.get('legend_top', False)

    # set defaults
    try:
        os.makedirs(save_dir)
    except:
        pass
    if field_name is None:
        field_name = field
    if norm_name is None:
        if norm != 1.0:
            norm_name = str(norm)
        else:
            norm_name = ' '
    if scale==1.0:
        scale_name = ''
    else:
        scale_name = str(scale)
    xcol = index
    ycol = 0

    # start figure and plot domain
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    # ax1 = fig.add_subplot(2,1,1)
    # ax2 = fig.add_subplot(2,1,2)
    fig.canvas.set_window_title(input_file)

    tmp = np.array([float(i) for i in x_pos_list])
    tmp2 = np.array([float(i) for i in x_pos_list])
    tmp2[tmp<4.5] += 4.5
    tmp2[tmp>=4.5] -= 4.5
    x_pos_list = [str(i) for i in tmp2]

    plt.sca(ax1)
    plot_domain(line_profile)

    # plot profiles
    for xpos,xname in zip(x_pos_list,x_name_list):
        # truth
        if plot_truth:
            ls = line_truth
            filename = os.path.join(
                dir_truth, 'line_x{}_{}.xy'.format(xname, field))
            _ = plot_profile_x(filename, xpos, scale, xcol, ycol, norm,
                            ls[0], ls[1], ls[2], ls[3], ls[4])
        # baseline
        if plot_baseline:
            ls = line_baseline
            filename = os.path.join(
                dir_base, 'line_x{}_{}.xy'.format(xname, field))
            _ = plot_profile_x(filename, xpos, scale, xcol, ycol, norm,
                            ls[0], ls[1], ls[2], ls[3], ls[4])
        # samples
        plot_sample_stats = plot_sample_mean or plot_sample_median or \
            plot_sample_mode
        if plot_samples or plot_sample_stats:
            all_samps = []
            ls = line_samples
            for isamp in range(nsamps):
                filename = os.path.join(
                    dir_samples, 'sample_{:d}'.format(isamp),
                    'postProcessing', 'sampleDict', '{:d}'.format(da_step),
                    'line_x{}_{}.xy'.format(xname, field))
                if plot_samples:
                    _, xval, yval = plot_profile_x(
                        filename, xpos, scale, xcol, ycol, norm,
                        ls[0], ls[1], ls[2], ls[3], ls[4])
                else:
                    data = np.loadtxt(filename, comments='%')
                    yval = data[:,ycol]
                    xval = data[:,xcol]/norm * scale + float(xpos)
                all_samps.append(xval)
            if plot_sample_mean:
                mean = np.mean(np.array(all_samps), axis=0)
                ls = line_sample_mean
                plt.plot(mean, yval, linestyle=ls[0], color=ls[1], lw=ls[2],
                         dashes=ls[3], alpha=ls[4])
            if plot_sample_median:
                median = np.median(np.array(all_samps), axis=0)
                ls = line_sample_median
                plt.plot(median, yval, linestyle=ls[0], color=ls[1], lw=ls[2],
                         dashes=ls[3], alpha=ls[4])
            if plot_sample_mode:
                mode = np.squeeze(stats.mode(np.array(all_samps), axis=0).mode)
                ls = line_sample_mode
                plt.plot(mode, yval, linestyle=ls[0], color=ls[1], lw=ls[2],
                         dashes=ls[3], alpha=ls[4])

    # plot observations
    if plot_obs:
        obs_coord = np.atleast_2d(np.loadtxt(obs_location_file))
        xcoor = obs_coord[:,0]
        tmp = xcoor.copy()
        xcoor[tmp<4.5] += 4.5
        xcoor[tmp>=4.5] -= 4.5
        ycoor = obs_coord[:,1]
        obs_values = np.loadtxt(obs_values_file).reshape([len(xcoor), 3]).T
        obs_values = obs_values[index-1, :]
        xval = obs_values/norm * scale + xcoor
        plot, = plt.plot(xval, ycoor, linestyle='', marker=markers_obs[0],
            color=markers_obs[1], mew=markers_obs[2], ms=markers_obs[3])
    # plt.plot([4.0+4.5+0.00040392314135692377/norm * scale], [2.5], linestyle='', marker=markers_obs[0], color=markers_obs[1], mew=markers_obs[2], ms=markers_obs[3])


    # set figure properties and labels
    plt.axis([-1.5, 10.5, -0.5, 3.5])
    plt.ylabel(r'$x_2/H$')
    if norm == 1.0:
        lab = r'$\frac{x_1}{H}$, \quad \quad  $' + scale_name + field_name + ' + \\frac{x_1}{H}$'
    else:
        # lab = r'$\frac{x_1}{H}$, \quad \quad  $' + scale_name + '\\frac{' + field_name + '}{' + norm_name + '} + \\frac{x_1}{H}$'
        lab = r'$x_1/H$, \quad \quad  $' + scale_name +  field_name + r'/' + norm_name + r' + x_1/H$'
    plt.xlabel(lab)
    if fix_aspect:
        ax1.set_aspect(1.0)
    # fig = plt.gcf()
    # fig.set_size_inches(8, 4.0)

    # add legend
    lines = []
    labels = []
    if plot_truth:
        ls = line_truth
        lines.append(Line2D([0], [0], color=ls[1], lw=ls[2],
            dashes=ls[3], alpha=ls[4]))
        labels.append(truth_name)
    if plot_baseline:
        ls = line_baseline
        lines.append(Line2D([0], [0], color=ls[1], lw=ls[2],
            dashes=ls[3], alpha=ls[4]))
        labels.append(baseline_name)
    if plot_sample_mean:
        ls = line_sample_mean
        lines.append(Line2D([0], [0], color=ls[1], lw=ls[2],
            dashes=ls[3], alpha=ls[4]))
        labels.append(sample_mean_name)
    if plot_sample_median:
        ls = line_sample_median
        lines.append(Line2D([0], [0], color=ls[1], lw=ls[2],
            dashes=ls[3], alpha=ls[4]))
        labels.append(sample_median_name)
    if plot_sample_mode:
        ls = line_sample_mode
        lines.append(Line2D([0], [0], color=ls[1], lw=ls[2],
            dashes=ls[3], alpha=ls[4]))
        labels.append(sample_mode_name)
    if plot_samples:
        ls = line_samples
        lines.append(Line2D([0], [0], color=ls[1], lw=ls[2],
            dashes=ls[3], alpha=ls[4]))
        labels.append(samples_name)
    if plot_obs:
        lines.append(Line2D([0], [0], lw=0, marker=markers_obs[0],
            color=markers_obs[1], mew=markers_obs[2], ms=markers_obs[3]))
        labels.append(obs_name)
    if show_legend:
        box1 = ax1.get_position()
        # box2 = ax2.get_position()
        if legend_top:
            ax1.set_position([box1.x0, box1.y0+box1.height*0.05, box1.width,
                            box1.height * 0.9])
            # ax2.set_position([box2.x0, box2.y0+box2.height*0.05, box2.width,
                            # box2.height * 0.9])
            plt.legend(lines, labels,
                       loc='lower center', bbox_to_anchor=(0.5, 1.05),
                       fancybox=False, shadow=False, ncol=5)
        else:
           ax1.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])
           # ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])
           plt.legend(lines, labels,
                      loc='center left', bbox_to_anchor=(1.0, 0.5),
                      fancybox=False, shadow=False)
    # save/show
    plt.savefig(os.path.join(save_dir, file_name))
    if show_plot:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Periodic Hills Results')
    parser.add_argument('input_file', help='Name (path) of input file')
    args = parser.parse_args()
    main(args.input_file)
