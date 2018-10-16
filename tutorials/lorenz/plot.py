# Copyright Xinlei Zhang (xinlei.zhang@ensam.eu) 2018
"""
	A simple dynamic system used in the demo codes of EnKF
"""
import numpy as np
from scipy.integrate import ode
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb


def plotSamples(para):
	"""
		plot samples and sample mean
	"""
	if para=='x': ind=0
	if para=='y': ind=1
	if para=='z': ind=2
	t = np.loadtxt('obs.txt')[:,0]
	# pdb.set_trace()
	DAt = t[11:-1:10]
	DAstep = len(DAt) #(t-1)/10

	Allt = np.tile(DAt, (50, 1))
	#pdb.set_trace()
	AllHX = []
	meanHX = []
	meanHX = np.array(meanHX)
	for i in range(DAstep):
		HX = np.loadtxt('./debugData/HX_'+str(i+1)+'.0')
		if i == 0:
			AllHX = HX[ind,:]

			meanHX = np.append(meanHX, np.mean(HX[ind,:]))
		else:
			#pdb.set_trace()
			AllHX = np.column_stack((AllHX, HX[ind,:]))
			meanHX = np.append(meanHX, np.mean(HX[ind,:]))

	np.savetxt('AllHX.txt',AllHX)
	# pdb.set_trace()
	p1 = 0 #plt.plot(Allt.T,AllHX.T,'g-')

	p2 =  plt.plot(DAt,meanHX, 'g-')
	return p1,p2

def plotTruth(para):
	"""
	plot synthetic truth
	"""
	# parse parameters
	obs = np.loadtxt('obs.txt')
	if para == 'x': ind = 1
	if para == 'y': ind = 2
	if para == 'z': ind = 3
	p4 = plt.plot(obs[:,0],obs[:,ind],'k-')
	return p4

def plotObs(para):
	obs = np.loadtxt('obs.txt')
	if para == 'x': ind = 1
	if para == 'y': ind = 2
	if para == 'z': ind = 3
	p5 = plt.plot(obs[10:-1:10,0],obs[10:-1:10,ind],'r.')
	return p5

def main():
	"""
	Direct solve the forward system
	"""
	para='x'
	plt.figure()
	ax1=plt.subplot(111)
	p1,p2 = plotSamples(para)
	p4 = plotTruth(para)
	p5 = plotObs(para)
	plt.xlabel('time')
	plt.ylabel(para)
	matplotlib.rcParams.update({'font.size':15})
	figureName = './figs/timeSeries_DA_'+ para +'.pdf'
	plt.savefig(figureName)

if __name__ == "__main__":
	main()
