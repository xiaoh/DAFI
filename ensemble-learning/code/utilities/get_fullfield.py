#!/usr/bin/env python3

""" 
Executable. 
Read and save the velocity components from an OpenFOAM run. 
Resulting files used as data for training with full-field velocity. 
"""

import os

import numpy as np

from dafi.random_field import foam_utilities as foam


foamcase = input('OpenFOAM case: ')
foamtimedir = input('Time directory: ')
savedir = input('Save directory: ')

file = os.path.join(foamcase, foamtimedir, 'U')
U = foam.read_vector_field(file)

names = ['x', 'y', 'z']

for i, iname in enumerate(names):
    file = os.path.join(savedir, f'U{iname}FullField')
    np.savetxt(file, U[:, i])