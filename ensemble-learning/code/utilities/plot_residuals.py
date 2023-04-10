#!/usr/bin/env python3

"""
Executable. 
Plot the residuals from an OpenFOAM run. 
First run the following OpenFOAM utilities in the OpenFOAM case directory:
    >> foamLog <logname>
Where <logname> is the file where the log (screen output) was saved
(redirected, e.g. log.simpleFoam in "simpleFoam >> log.simpleFoam")
"""

import os

import numpy as np
import matplotlib.pyplot as plt


# dir = input('logs directory: ')
dir = 'logs'

files = [
    'Ux_0',
    'Uy_0',
    'Uz_0',
    'p_0',
    'k_0',
    'epsilon_0',
    'omega_0',
    'Uax_0',
    'Uay_0',
    'Uaz_0',
    'pa_0'
    ]

plt.figure()
for file in files:
    try:
        data = np.loadtxt(os.path.join(dir, file))
        plt.semilogy(data[:, 0], data[:, 1], label=file, alpha=0.5)
    except:
        pass
plt.legend()

plt.savefig('residuals.pdf')
plt.savefig('residuals.png')
plt.show()