#!/usr/bin/env python3
""" Plotting script for the periodic hill OpenFOAM case. """

# standard library imports
import os

# third party imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# set plot properties
params = {
    'text.latex.preamble': '\\usepackage{gensymb}',
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
    'figure.figsize': [5, 3],
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'lines.linewidth': 2,
    'lines.markersize': 6,
}
mpl.rcParams.update(params)


def _pehill_profile(yy, a):
    'Calculate the shape of the parameterized periodic hill'

    import numpy as np

    y = np.array(yy)
    x = y * 28
    ya = y

    h = np.zeros(len(x))

    for i in range(len(x)):
        if (x[i] >= 0) and (x[i] < 54):
            ya[i] *= a
        elif (x[i] > 54) and (x[i] <= 126):
            ya[i] -= (54/28.0*(1-a))
        elif (x[i] > 126) and (x[i] <= 198):
            ya[i] -= (54/28.0*(1-a))
        elif (x[i] > 198) and (x[i] <= 252):
            ya[i] -= (54/28.0*(1-a))
            ya[i] -= (x[i]-198)*(1-a)/28.0

    for i in range(len(x)):
        if x[i] > 126.0:
            x[i] = 252.0 - x[i]

        if (x[i] >= 0) and (x[i] < 9):
            h[i] = np.minimum(28., 2.8e+01 + 0.0e+00*x[i] +
                              6.775070969851e-03*x[i]**2 - 2.124527775800e-03*x[i]**3)
        elif (x[i] >= 9) and (x[i] < 14):
            h[i] = 2.507355893131E+01 + 9.754803562315E-01*x[i] - \
                1.016116352781E-01*x[i]**2 + 1.889794677828E-03*x[i]**3
        elif (x[i] >= 14) and (x[i] < 20):
            h[i] = 2.579601052357E+01 + 8.206693007457E-01*x[i] - \
                9.055370274339E-02*x[i]**2 + 1.626510569859E-03*x[i]**3
        elif (x[i] >= 20) and (x[i] < 30):
            h[i] = 4.046435022819E+01 - 1.379581654948E+00*x[i] + \
                1.945884504128E-02*x[i]**2 - 2.070318932190E-04*x[i]**3
        elif (x[i] >= 30) and (x[i] < 40):
            h[i] = 1.792461334664E+01 + 8.743920332081E-01*x[i] - \
                5.567361123058E-02*x[i]**2 + 6.277731764683E-04*x[i]**3
        elif (x[i] >= 40) and (x[i] <= 54):
            h[i] = np.maximum(0., 5.639011190988E+01 - 2.010520359035E+00*x[i] +
                              1.644919857549E-02*x[i]**2 + 2.674976141766E-05*x[i]**3)
        elif (x[i] > 54) and (x[i] <= 126):
            h[i] = 0

    hout = h/28.0
    return ya, hout


def plot_domain(ls, ypos, alpha):
    """ Plot the simulation domain. """
    yh, hill = _pehill_profile(ypos, alpha)
    L = np.max(yh)
    plt.plot(yh, hill, ls[0], color=ls[1], lw=ls[2], dashes=ls[3],
             alpha=ls[4])
    plt.plot([0, L], [3.036, 3.036], ls[0], color=ls[1], lw=ls[2],
             dashes=ls[3], alpha=ls[4])
    plt.plot([0, 0], [1, 3.036], '--', color=ls[1], lw=ls[2], alpha=0.5*ls[4])
    plt.plot([L, L], [1, 3.036], '--', color=ls[1], lw=ls[2], alpha=0.5*ls[4])
    return L


def plot_domain_centered(ls, ypos, alpha):
    """ Plot the simulation domain. """
    yh, hill = _pehill_profile(ypos, alpha)
    L = np.max(yh)
    hill_c = np.append(hill[yh >= L/2], hill[yh < L/2])
    yh_c = np.append(yh[yh >= L/2] - L/2, yh[yh < L/2] + L/2)
    plt.plot(yh_c, hill_c, ls[0], color=ls[1], lw=ls[2], dashes=ls[3],
             alpha=ls[4])
    plt.plot([0, L], [3.036, 3.036], ls[0], color=ls[1], lw=ls[2],
             dashes=ls[3], alpha=ls[4])
    plt.plot([0, 0], [0, 3.036], '--', color=ls[1], lw=ls[2], alpha=0.5*ls[4])
    plt.plot([L, L], [0, 3.036], '--', color=ls[1], lw=ls[2], alpha=0.5*ls[4])
    return L


def plot_profile_x(filename, xpos, scale, xcol, ycol, norm, ls, cl, lw, dash,
                   alpha, L):
    data = np.loadtxt(filename, comments='%')
    yval = data[:, ycol]
    xval = data[:, xcol]/norm * scale + float(xpos)
    plot, = plt.plot(xval, yval, linestyle=ls, color=cl, lw=lw, dashes=dash,
                     alpha=alpha)
    return plot, xval, yval


def plot_profile_x_centered(filename, xpos, scale, xcol, ycol, norm, ls, cl,
                            lw, dash, alpha, L):
    xpos += L/2 if xpos < L/2 else -L/2
    plot, xval, yval = plot_profile_x(filename, xpos, scale, xcol, ycol, norm,
                                      ls, cl, lw, dash, alpha, L)
    return plot, xval, yval


def main():
    # from case setup
    nsamples = 50
    niter = 10
    foam_end_time = 10000

    # options
    savefig = True
    showfig = True
    legend_top = True
    centered = False
    x_pos_list = [0, 2, 4.5, 7]  # 2, 7 are observation locations.
    x_name_list = [str(i).replace('.', 'p') for i in x_pos_list]

    alpha = 1.0  # baseline periodic hills geometry
    ycol = 0  # y is in column 0
    xcol = 1  # Ux and nu_t are in column 1 of their respective files

    # define color for different profiles
    # [line_style, color, line_width, dashes, opacity]
    line_profile = ['-', 'k', 1.7, (None, None), 1.0]
    line_truth = ['-', 'k', 1, (None, None), 1.0]
    line_base = ['-', 'tab:red', 1, (9, 3), 1.0]
    line_inferred = ['-', 'tab:blue', 1, (5, 2), 1.0]
    line_samples = ['-', 'tab:grey', 0.5, (None, None), 0.5]

    # color, marker, linestyle, markersize, markeredgewidth
    line_obs = ['black', 'x', '', 6, 1.5]

    # legend names
    truth_name = 'truth'
    base_name = 'baseline'
    inferred_name = 'Sample Mean'
    samples_name = 'Samples'
    obs_name = "Observations"

    # observations
    obsfile = 'pre_processing/obs'
    obs = np.loadtxt(obsfile)
    obs = obs[obs[:, 3] == 0, :]

    # create figure
    def plot_figure(case, field):
        if case == 'prior':
            iter = 0
        elif case == 'posterior':
            iter = niter
        else:
            raise ValueError("'case' must be one of 'prior' or 'posterior'")

        if field == 'Ux':
            field_name = 'U_x'
            field_filename = 'U'
            norm = 1.0
            norm_name = 'U_b'
            scale = 2.0
        elif field == 'nut':
            field_name = r'\nu_t'
            field_filename = 'nut'
            norm = 0.00017857142857142857
            norm_name = r'\nu'
            scale = 0.01
        else:
            raise ValueError("'field_name' must be one of 'Ux' or 'nut'")

        # start figure and plot domain
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ypos = np.arange(0, 9, 0.01)
        plot_dom = plot_domain_centered if centered else plot_domain
        L = plot_dom(line_profile, ypos, alpha)

        # plot observations
        ls = line_obs
        if centered:
            for i in range(len(obs)):
                obs[i, 0] += L/2 if xpos < L/2 else -L/2
        obs_x = obs[:, 0] + obs[:, 4]*scale/norm
        obs_y = obs[:, 1]
        plt.plot(obs_x, obs_y, color=ls[0], marker=ls[1], ls=ls[2],
                 markersize=ls[3], markeredgewidth=ls[4])

        # plot profiles
        plot_prof = plot_profile_x_centered if centered else plot_profile_x
        for xpos, xname in zip(x_pos_list, x_name_list):
            # samples
            ls = line_samples
            sample_mean = 0
            for i in range(nsamples):
                filename = f"results_nutFoam/sample_{i:d}/postProcessing/" +\
                    f"sampleDict/{iter}/line_x{xname}_{field_filename}.xy"
                _, xval, yval = plot_prof(
                    filename, xpos, scale, xcol, ycol, norm, ls[0], ls[1],
                    ls[2], ls[3], ls[4], L)
                # update mean
                sample_mean += xval
            sample_mean /= nsamples
            ls = line_inferred
            _ = plt.plot(xval, yval, linestyle=ls[0], color=ls[1], lw=ls[2],
                         dashes=ls[3], alpha=ls[4])

            # truth
            ls = line_truth
            filename = 'pre_processing/truth_foam/postProcessing/sampleDict/'
            filename += f'{foam_end_time}/line_x{xname}_{field_filename}.xy'
            _ = plot_prof(filename, xpos, scale, xcol, ycol, norm,
                          ls[0], ls[1], ls[2], ls[3], ls[4], L)

            # baseline
            ls = line_base
            filename = 'pre_processing/baseline_foam/postProcessing/sampleDict/'
            filename += f'{foam_end_time}/line_x{xname}_{field_filename}.xy'
            _ = plot_prof(filename, xpos, scale, xcol, ycol, norm,
                          ls[0], ls[1], ls[2], ls[3], ls[4], L)

        # set figure properties and labels
        scale_name = str(scale)
        plt.axis([-1.5, L + 1.5, -0.5, 3.5])
        plt.ylabel(r'$y/H$')
        lab = r'$x/H$, \quad \quad  $' + scale_name + field_name + r'/' + \
            norm_name + r' + x/H$'
        plt.xlabel(lab)
        ax1.set_aspect(1.0)

        # legend
        lines = []
        labels = []

        ls = line_truth
        lines.append(Line2D(
            [0], [0], color=ls[1], lw=ls[2], dashes=ls[3], alpha=ls[4]))
        labels.append(truth_name)

        ls = line_base
        lines.append(Line2D(
            [0], [0], color=ls[1], lw=ls[2], dashes=ls[3], alpha=ls[4]))
        labels.append(base_name)

        ls = line_samples
        lines.append(Line2D(
            [0], [0], color=ls[1], lw=ls[2], dashes=ls[3], alpha=ls[4]))
        labels.append(samples_name)

        ls = line_inferred
        lines.append(Line2D(
            [0], [0], color=ls[1], lw=ls[2], dashes=ls[3], alpha=ls[4]))
        labels.append(inferred_name)

        ls = line_obs
        lines.append(Line2D(
            [0], [0], color=ls[0], marker=ls[1], ls=ls[2], markersize=ls[3],
            markeredgewidth=ls[4]))
        labels.append(obs_name)

        box1 = ax1.get_position()
        if legend_top:
            ax1.set_position([box1.x0, box1.y0+box1.height*0.05, box1.width,
                              box1.height * 0.9])
            plt.legend(lines, labels, handlelength=4,
                       loc='lower center', bbox_to_anchor=(0.5, 1.05),
                       fancybox=False, shadow=False, ncol=5)
        else:
            ax1.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])
            plt.legend(lines, labels, handlelength=4,
                       loc='center left', bbox_to_anchor=(1.0, 0.5),
                       fancybox=False, shadow=False)

        # save/show
        if savefig:
            figname = f"hills_profiles_{field}_{case}"
            plt.savefig(f"{figname}.pdf")

    plot_figure('prior', 'Ux')
    plot_figure('prior', 'nut')
    plot_figure('posterior', 'Ux')
    plot_figure('posterior', 'nut')

    if showfig:
        plt.show()


if __name__ == "__main__":
    main()
