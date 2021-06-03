#!/usr/bin/env python3

import os

import numpy as np

from dafi import random_field as rf

foam_rc = None  # modify if needed to run in your system
foam_rc = "/Users/cmichel/.OpenFOAMrc"

foam_case = 'truth_foam'
time_dir = '10000'
field_file = 'U'

coordinates = rf.foam.get_cell_centres(foam_case, foam_rc=foam_rc)
connectivity = rf.foam.get_neighbors(foam_case)
U_file = os.path.join(foam_case, time_dir, 'U')
vel = rf.foam.read_vector_field(U_file)
vel_x = vel[:, 0]
vel_y = vel[:, 1]

# List of points for observation. Both Ux and Uy will be observed
points = np.array([
    [2.0, 1.0, 0.05],
    [7.0, 2.5, 0.05],
])

# standard deviation of observation for [Ux_1, Uy_1, ..., Ux_N, Uy_N]
stddev = [0.01, 0.001, 0.01, 0.001]

# create observations using true value + random error
values = []
for i, point in enumerate(points):
    point = np.atleast_2d(point)
    Hmat = rf.inverse_distance_weights(coordinates, connectivity, point)
    values.append(float(Hmat.dot(vel_x)) + np.random.randn()*stddev[2*i])
    values.append(float(Hmat.dot(vel_y)) + np.random.randn()*stddev[2*i+1])

# organize data
DATA = np.ones([len(points)*2, 6])

for i in range(len(points)):
    for k in range(2):
        j = 2*i + k
        DATA[j, :3] = points[i, :]
        DATA[j, 3] = k
        DATA[j, 4] = values[j]
        DATA[j, 5] = stddev[j]

# write file
header = "# x, y, z, field, value, stddev\n"
header += r"# field {0:Ux, 1:Uy, 2:Uz}"
save_file = 'obs'
np.savetxt(save_file, DATA, header=header, fmt='%f %f %f %i %f %f ')
