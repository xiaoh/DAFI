# Copyright Jianxun Wang (jianxunwang@berkeley.edu) 2017
"""
	A simple dynamic system used in the demo codes of EnKF
"""
import numpy as np
from scipy.integrate import ode
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb

def Lorenz63(t, x):
	"""
	Define Lorenz 63 system
	"""
	# parse parameters
	sigma = 10.
	beta = 8./3.
	rho = x[3]

	# ODEs
	dx = np.zeros([4, 1])
	dx[0] = sigma * (-x[0] + x[1])
	dx[1] = rho * x[0] - x[1] - x[0]*x[2]
	dx[2] = x[0] * x[1] - beta * x[2]
	dx[3] = 0

	return dx

def main():
	"""
	Direct solve the forward system
	"""
	timeSeries = np.arange(0., tEnd+dt, dt)
	solver = ode(Lorenz63)
	solver.set_integrator('dopri5')
	solver.set_initial_value(x_init, tStart)
	# solve ode
	x = np.empty([len(timeSeries), 4])
	x[0] = x_init
	k = 1
	while solver.successful() and solver.t < tEnd:
	    solver.integrate(timeSeries[k])
	    x[k] = solver.y
	    k += 1
	plotSystem2D(timeSeries, x)
	plotSystem3D(x)

def plotSystem2D(t, x):
	"""
	Plot Lorenz63 system
	"""
	paraNameVec = ['x', 'y', 'z', 'rho']
	nDim = x.shape[1]
	for iDim in np.arange(nDim):
		plt.figure(1)
		plt.clf()
		a1, = plt.plot(t, x[:, iDim], lw=2, color='blue')
		plt.xlabel('time')
		plt.ylabel(paraNameVec[iDim])
		matplotlib.rcParams.update({'font.size':15})
		figureName = './figs/timeSeries_'+ paraNameVec[iDim] +'.pdf'
		plt.savefig(figureName)			

def plotSystem3D(x):
	"""
	Plot 3D Lorenz63 system
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot(x[:, 0], x[:, 1], x[:, 2], label='Lorenz63')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	matplotlib.rcParams.update({'font.size':15})
	plt.grid()
	figureName = './figs/Lorenz63.png'
	plt.savefig(figureName)	

if __name__ == "__main__":
	# specify control parameters
	tStart = 0.;
	tEnd = 40.;
	dt = 0.025;
	# specify initial condition
	x_init = [1, 1.2, 1, 28]
	main()
