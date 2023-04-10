#!/usr/bin/env python3

""" 
Executable. 
Interpolate field at a point and save. 
Resulting file used as data for training with sparse velocity. 
"""

import numpy as np

import dafi.random_field as rf

foam_dir = input('OpenFOAM case: ')
field_file = input('Field file:')
px = float(input('Point x:'))
py = float(input('Point y:'))
pz = float(input('Point z:'))
save_file = input('Save file:')

field = np.loadtxt(field_file)
point = np.array([[px, py, pz]])
coordinates = rf.foam.get_cell_centres(foam_dir)
connectivity = rf.foam.get_neighbors(foam_dir)
Hmat = rf.inverse_distance_weights(coordinates, connectivity, point)

value = float(Hmat.dot(field))
np.savetxt(save_file, np.array([[px, py, pz, value]]))