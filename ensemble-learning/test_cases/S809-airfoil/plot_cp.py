
import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt
import pdb
# set plot properties
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    # 'image.cmap': 'viridis',
    'axes.grid': False,
    'savefig.dpi': 300,
    'axes.labelsize': 20, # 10
    'axes.titlesize': 20,
    'font.size': 20,
    'legend.fontsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'text.usetex': True,
    'figure.figsize': [5, 4],
    'font.family': 'serif',
}
mpl.rcParams.update(params)

cp_AoA8_exp = np.loadtxt('inputs/data/S809_Cp_AoA8.txt', delimiter=',')

cp_AoA14_exp = np.loadtxt('inputs/data/S809_Cp_AoA14.txt', delimiter=',')


cp_AoA8_baseline = np.loadtxt('inputs/S809_komega_AoA8/postProcessing/sampleDict/10000/p_walls_interpolated.raw', skiprows=2)

cp_AoA14_baseline = np.loadtxt('inputs/S809_komega_AoA14/postProcessing/surfaces/10000/p_walls_interpolated.raw', skiprows=2)


cp_AoA8_infer = np.loadtxt('results_ensemble/sample_0/foam_base_AoA8/postProcessing/sampleDict/20/p_walls_interpolated.raw', skiprows=2)

cp_AoA14_infer = np.loadtxt('results_ensemble/sample_0/foam_base_AoA14/postProcessing/sampleDict/20/p_walls_interpolated.raw', skiprows=2)

cp_infer = cp_AoA14_infer
cp_infer_b = []
cp_infer_t = []

for i in range(cp_infer.shape[0]):
    if cp_infer[i, 1] > 0:
        cp_infer_t.append(cp_infer[i, :])
    else:
        cp_infer_b.append(cp_infer[i, :])
cp_infer_b = np.array(cp_infer_b)
cp_infer_t = np.array(cp_infer_t)
cp_infer_b.sort(axis=0)
cp_infer_t.sort(axis=0)

fig, ax = plt.subplots()

plt.plot(cp_AoA8_exp[:, 0], -cp_AoA8_exp[:, 1], 'ko', markersize=5, fillstyle='none', label='exp')
plt.plot(cp_AoA8_infer[:, 0]/0.6, -cp_AoA8_infer[:, 3]/ 0.5/ 48/ 48, 'b-', lw=3, label='learned') #, markersize=2)
plt.plot([cp_AoA8_infer[0, 0]/0.6, cp_AoA8_infer[-1, 0]/0.6],
    [-cp_AoA8_infer[0, 3]/ 0.5/ 48/ 48, -cp_AoA8_infer[-1, 3]/ 0.5/ 48/ 48], 'b-', lw=3) #, markersize=2)

plt.plot(cp_AoA8_baseline[:, 0]/0.6, -cp_AoA8_baseline[:, 3]/ 0.5/ 48/ 48, 'r-', lw=3, dashes=[6,4,2,4], label=r'$k-\omega$')
plt.plot([cp_AoA8_baseline[0, 0]/0.6, cp_AoA8_baseline[-1, 0]/0.6],
    [-cp_AoA8_baseline[0, 3]/ 0.5/ 48/ 48, -cp_AoA8_baseline[-1, 3]/ 0.5/ 48/ 48], 'r-', lw=3, dashes=[6,4,2,4]) #, markersize=2)

ax.set_xlabel(f'$x/c$')
ax.set_ylabel(f'$-C_p$')
plt.legend()
plt.tight_layout()
plt.savefig('cp_AoA8.pdf')
plt.show()



fig, ax = plt.subplots()

plt.plot(cp_AoA14_exp[:, 0], -cp_AoA14_exp[:, 1], 'ko', markersize=5, fillstyle='none', label='exp')

plt.plot(cp_AoA14_infer[17:, 0]/0.6, -cp_AoA14_infer[17:, 3]/ 0.5/ 48/ 48, 'b-', lw=2, label='learned') #, markersize=2)
plt.plot([cp_AoA14_infer[17, 0]/0.6, cp_AoA14_infer[-1, 0]/0.6],
    [-cp_AoA14_infer[17, 3]/ 0.5/ 48/ 48, -cp_AoA14_infer[-1, 3]/ 0.5/ 48/ 48], 'b-', lw=3) #, markersize=2)

plt.plot(cp_AoA14_baseline[:, 0]/0.6, -cp_AoA14_baseline[:, 3]/ 0.5/ 48/ 48, 'r-', lw=3, dashes=[6,4,2,4], label=r'$k-\omega$')
plt.plot([cp_AoA14_baseline[0, 0]/0.6, cp_AoA14_baseline[-1, 0]/0.6],
    [-cp_AoA14_baseline[0, 3]/ 0.5/ 48/ 48, -cp_AoA14_baseline[-1, 3]/ 0.5/ 48/ 48], 'r-', lw=3, dashes=[6,4,2,4]) #, markersize=2)

ax.set_xlabel(f'$x/c$')
ax.set_ylabel(f'$-C_p$')
plt.tight_layout()
plt.savefig('cp_AoA14.pdf')
plt.show()
