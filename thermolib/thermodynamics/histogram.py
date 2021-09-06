#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 - 2021 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
# Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium;
# all rights reserved unless otherwise stated.
#
# This file is part of a library developed by Louis Vanduyfhuys at
# the Center for Molecular Modeling under supervision of prof. Veronique
# Van Speybroeck. Usage of this package should be authorized by prof. Van
# Van Speybroeck.

from matplotlib.colors import Normalize
from molmod.units import *
from molmod.constants import *

import matplotlib.pyplot as pp
import matplotlib.cm as cm
import numpy as np

from thermolib.thermodynamics.fep import BaseFreeEnergyProfile
from thermolib.tools import integrate, integrate2d

__all__ = ['Histogram1D', 'Histogram2D', 'plot_histograms']

class Histogram1D(object):
	def __init__(self, cvs, ps, plower=None, pupper=None, cv_unit='au', cv_label='CV'):
		'''
			Class to implement the estimation of a probability histogram in terms of a collective variable from a trajectory series corresponding to that collective variable.

			:param cvs: array corresponding the collective variable grid points
			:type data: np.ndarray

			:param ps: array corresponding to the histogram probability values at the grid points
			:type bins: np.ndarray

			:param plower: lower bound of the error on the probability histogram values, defaults to None
			:type do_error: np.ndarray, optional

			:param pupper: upper bound of the error on the probability histogram values, defaults to None
			:type pupper: np.ndarray, optional

			:param cv_unit: the unit in which cv will be plotted/printed, defaults to 'au'
			:type cv_unit: str, optional

			:param cv_label: the label of the cv that will be used on plots, defaults to 'CV'
			:type cv_label: str, optional
		'''
		self.cvs = cvs.copy()
		self.ps = ps.copy()
		self.plower = plower
		self.pupper = pupper
		self.cv_unit = cv_unit
		self.cv_label = cv_label

	def copy(self):
		plower = None
		if self.plower is not None:
			plower = self.plower.copy()
		pupper = None
		if self.pupper is not None:
			pupper = self.pupper.copy()
		return Histogram1D(self.cvs.copy(), self.ps.copy(), plower=plower, pupper=pupper, cv_unit=self.cv_unit, cv_label=self.cv_label)

	@classmethod
	def from_average(cls, histograms, error_estimate=None, nsigma=2):
		'''
			Start from a set of histograms and compute and return the averaged histogram. If error_estimate is set to 'std', an error on the histogram will be computed from the standard deviation within the set of histograms.

			:param histograms: set of histrograms to be averaged
			:type histograms: list(Histogram1D)

			:param error_estimate: indicate if and how to perform error analysis. One of following options is available:
			
				*  **std** -- compute error from the standard deviation within the set of histograms.
				*  **None** -- do not estimate the error.
			
			Defaults to None, i.e. no error estimate.
			:type error_estimate: str, optional

			:param nsigma: only relevant when error estimation is turned on (i.e. when keyword ``error_estimate`` is not None), this option defines how large the error interval should be in terms of the standard deviation sigma. A ``nsigma=2`` implies a 2-sigma error bar (corresponding to 95% confidence interval) will be returned. Defaults to 2
			:type nsigma: int, optional

			:return: averages histogram
			:rtype: Histrogram1D
		'''
		#sanity checks
		cv_unit, cv_label = None, None
		cvs = None
		for hist in histograms:
			if cvs is None:
				cvs = hist.cvs.copy()
			else:
				assert (abs(cvs-hist.cvs)<1e-12*np.abs(cvs.mean())).all(), 'Cannot average the histograms as they do not have consistent CV grids'
			if cv_unit is None:
				cv_unit = hist.cv_unit
			else:
				assert cv_unit==hist.cv_unit, 'Inconsistent CV unit definition in histograms'
			if cv_label is None:
				cv_label = hist.cv_label
			else:
				assert cv_label==hist.cv_label, 'Inconsistent CV label definition in histograms'
		#collect probability distributions
		pss = np.array([hist.ps for hist in histograms])
		#average histograms
		ps = pss.mean(axis=0)
		#compute error if requested
		plower, pupper = None, None
		if error_estimate is not None and error_estimate.lower() in ['std']:
			perr = pss.std(axis=0,ddof=1)
			plower = ps - nsigma*perr
			plower[plower<0] = 0.0
			pupper = ps + nsigma*perr
		return cls(cvs, ps, plower=plower, pupper=pupper, cv_unit=cv_unit, cv_label=cv_label)

	@classmethod
	def from_single_trajectory(cls, data, bins, error_estimate=None, nsigma=2, cv_unit='au', cv_label='CV'):
		'''
			Routine to estimate of a probability histogram in terms of a collective variable from a trajectory series corresponding to that collective variable.

			:param data: array corresponding to the trajectory series of the collective variable
			:type data: np.ndarray

			:param bins: array representing the edges of the bins for which a histogram will be constructed. This argument is parsed to `the numpy.histogram routine <https://numpy.org/doc/stable/reference/generated/numpy.histogram.html>`_. Hence, more information on its meaning and allowed values can be found there.
			:type bins: np.ndarray

			:param error_estimate: indicate if and how to perform error analysis. One of following options is available:
			
				*  **mle_p** -- compute error through the asymptotic normality of the maximum likelihood estimator for the probability itself. WARNING: due to positivity constraint of the probability, this only works for low variance. Otherwise the standard error interval for the normal distribution (i.e. mle +- n*sigma) might give rise to negative lower error bars.
				*  **mle_f** -- compute error through the asymptotic normality of the maximum likelihood estimator for -log(p) (hence for the beta-scaled free energy). This estimation does not suffor from the same WARNING as for ``mle_p``. Furthermore, in case of low variance, the error estimation using ``mle_f`` and ``mle_p`` are consistent (i.e. one can be computed from the other using f=-log(p)).
				*  **None** -- do not estimate the error.
			
			Defaults to None, i.e. no error estimate.
			:type error_estimate: str, optional

			:param nsigma: only relevant when error estimation is turned on (i.e. when keyword ``error_estimate`` is not None), this option defines how large the error interval should be in terms of the standard deviation sigma. A ``nsigma=2`` implies a 2-sigma error bar (corresponding to 95% confidence interval) will be returned. Defaults to 2
			:type nsigma: int, optional

			:param cv_unit: the unit in which cv will be plotted/printed, defaults to 'au'
			:type cv_unit: str, optional

			:param cv_label: the label of the cv that will be used on plots, defaults to 'CV'
			:type cv_label: str, optional
		'''
		#initialization
		cvs = None
		ps = None
		Ntot = len(data)
		#generate histogram using numpy.histogram routine
		ns, bin_edges = np.histogram(data, bins=bins, density=False)
		cvs = 0.5*(bin_edges[:-1]+bin_edges[1:]) # bin centers
		ps = ns/Ntot
		#estimate of upper and lower boundary of n-sigma confidence interval
		plower = None
		pupper = None
		if error_estimate=='mle_p':
			perrors = np.sqrt(ps*(1-ps)/Ntot)
			plower = ps - nsigma*perrors
			plower[plower<0] = 0.0
			pupper = ps + nsigma*perrors
		elif error_estimate=='mle_f':
			#we first compute the error bar interval on f=-log(p) and then transform it to one on p itself.
			fs, ferrors = np.zeros(len(ps))*np.nan, np.zeros(len(ps))*np.nan
			fs[ps>0] = -np.log(ps[ps>0])
			ferrors[ps>0] = np.sqrt((np.exp(fs[ps>0])-1)/Ntot)
			fupper  = fs + nsigma*ferrors
			flower  = fs - nsigma*ferrors
			pupper, plower = np.zeros(len(ps))*np.nan, np.zeros(len(ps))*np.nan
			pupper[ps>0] = np.exp(-flower[ps>0])
			plower[ps>0] = np.exp(-fupper[ps>0])
			pupper[np.isinf(pupper)] = np.nan
		elif error_estimate is not None:
			raise ValueError('Invalid value for error_estimate argument, received %s. Check documentation for allowed values.' %error_estimate)
		return cls(cvs, ps, plower=plower, pupper=pupper, cv_unit=cv_unit, cv_label=cv_label)

	@classmethod
	def from_wham(cls, bins, trajectories, biasses, temp, error_estimate=None, nsigma=2, bias_subgrid_num=20, Nscf=1000, convergence=1e-6, verbose=False, cv_unit='au', cv_label='CV'):
		'''
			Routine that implements the Weighted Histogram Analysis Method (WHAM) for reconstructing the overall probability histogram from a series of biased molecular simulations.

			:param bins: number of bins for the CV grid or array representing the bin edges for th CV grid.
			:type bins: int or np.ndarray(float)

			:param trajectories: list or array of numpy arrays containing the CV trajectory data for each simulation. Alternatively, a list of PLUMED file names containing the trajectory data can be specified as well. The arguments trajectories and biasses should be of the same length.
			:type trajectories: list(np.ndarray)/np.ndarray(np.ndarray)

			:param biasses: list of callables, each representing a function to compute the bias at a given value of the collective variable. The arguments trajectories and biasses should be of the same length.
			:type biasses: list(callable)

			:param temp: the temperature at which all simulations were performed
			:type temp: float

			:param error_estimate: indicate if and how to perform error analysis. One of following options is available:
			
				*  **mle_p** -- compute error through the asymptotic normality of the maximum likelihood estimator for the probability itself. WARNING: due to positivity constraint of the probability, this only works for high probability and low variance. Otherwise the standard error interval for the normal distribution (i.e. mle +- n*sigma) might give rise to negative lower error bars.
				*  **mle_f** -- compute error through the asymptotic normality of the maximum likelihood estimator for -log(p) (hence for the beta-scaled free energy). This estimation does not suffor from the same WARNING as for ``mle_p``. Furthermore, in case of high probability and low variance, the error estimation using ``mle_f`` and ``mle_p`` are consistent (i.e. one can be computed from the other using f=-log(p)).
				*  **None** -- do not estimate the error.
			
			Defaults to None, i.e. no error estimate.
			:type error_estimate: str, optional

			:param nsigma: only relevant when error estimation is turned on (i.e. when keyword ``error_estimate`` is not None), this option defines how large the error interval should be in terms of the standard deviation sigma. A ``nsigma=2`` implies a 2-sigma error bar (corresponding to 95% confidence interval) will be returned. Defaults to 2
			:type nsigma: int, optional
			
			:param bias_subgrid_num: the number of grid points for the sub-grid used to compute the integrated boltzmann factor of the bias in each CV bin.
			:type bias_subgrid_num: optional, defaults to 20

			:param Nscf: the maximum number of steps in the self-consistent loop to solve the WHAM equations
			:type Nscf: int, defaults to 1000

			:param convergence: convergence criterium for the WHAM self consistent solver. The SCF loop will stop whenever the integrated absolute difference between consecutive probability densities is less then the specified value.
			:type convergence: float, defaults to 1e-6

			:param verbose: set to True to turn on more verbosity during the self consistent solution cycles of the WHAM equations, defaults to False.
			:type verbose: bool, optional

			:param cv_unit: the unit in which cv will be plotted/printed, defaults to 'au'
			:type cv_unit: str, optional

			:param cv_label: the label of the cv that will be used on plots, defaults to 'CV'
			:type cv_label: str, optional

			:return: probability histogram
			:rtype: Histogram1D
		'''
		#checks and initialization
		assert len(biasses)==len(trajectories), 'The arguments trajectories and biasses should be of the same length.'
		beta = 1/(boltzmann*temp)
		Nsims = len(biasses)
		#Prerocess trajectory argument: load files if file names are given instead of raw data, determine and store the number of simulation steps in each simulation:
		Nis = []
		data = []
		for i, trajectory in enumerate(trajectories):
			if isinstance(trajectory, str):
				trajectory = np.loadtxt(trajectory)[:,1] #first column is the time, second column is the CV
			Nis.append(len(trajectory))
			data.append(trajectory)
		if len(data)>0:
			trajectory = data
		Nis = np.array(Nis)
		#Preprocess the bins argument and redefine it to represent the bin_edges. We need to do this beforehand to make sure that when calling the numpy.histogram routine with this bins argument, each histogram will have a consistent bin_edges array and hence consistent histogram.
		if isinstance(bins, int):
			raise NotImplementedError
			cv_min = None #TODO
			cv_max = None #TODO
			cv_delta = (cv_max-cv_min)/bins #this 
			bins = np.arange(cv_min, cv_max+cv_delta, cv_delta)			
		bin_centers = 0.5*(bins[:-1]+bins[1:])
		deltas = bins[1:]-bins[:-1]
		grid_non_uniformity = deltas.std()/deltas.mean()
		assert grid_non_uniformity<1e-6, 'CV grid defined by bins argument should be of uniform spacing!'
		delta =deltas.mean()
		Ngrid = len(bin_centers)
		#generate the individual histograms using numpy.histogram
		Hs = np.zeros([Nsims, Ngrid])
		for i, data in enumerate(trajectories):
			ns, edges = np.histogram(data, bins, density=False)
			assert (bins==edges).all()
			Hs[i,:] = ns.copy()
			N = ns.sum()
			if Nis[i]!=N:
				print("WARNING: the CV range you specified for the histogram does not cover the entire simulation range in trajectory %i. Simulation samples outside the given CV range were cropped out." %(i))
				Nis[i] = N
		#compute the boltzmann factors of the biases in each grid interval
		#b_ik = 1/delta*int(exp(-beta*W_i(q)), q=Q_k-delta/2...Q_k+delta/2)
		bs = np.zeros([Nsims, Ngrid])
		for i, bias in enumerate(biasses):
			for k, center in enumerate(bin_centers):
				subdelta = delta/bias_subgrid_num
				subgrid = np.arange(center-delta/2, center+delta/2 + subdelta, subdelta)
				bs[i,k] = integrate(subgrid, np.exp(-beta*bias(subgrid)))/delta
		#some init printing
		if verbose:
			print('Initialization:')
			print('  Number of simulations = ', Nsims)
			for i, Ni in enumerate(Nis):
				print('    simulation %i has %i steps' %(i,Ni))
			print('  CV grid [%s]: start = %.3f    end = %.3f    delta = %.3f    N = %i' %(cv_unit, bins.min()/parse_unit(cv_unit), bins.max()/parse_unit(cv_unit), delta/parse_unit(cv_unit), Ngrid))
			print('')
			print('Starting WHAM SCF loop:')
		#self consistent loop to solve the WHAM equations
		as_old = np.ones(Ngrid)/Ngrid #initialize probability density array (which should sum to 1)
		nominator = Hs.sum(axis=0) #precomputable factor in WHAM update equations
		for iscf in range(Nscf):
			#compute new normalization factors
			fs = np.zeros(Nsims)
			for i in range(Nsims):
				fs[i] = 1.0/np.dot(bs[i,:],as_old)
			#compute new probabilities
			as_new = as_old.copy()
			for k in range(Ngrid):
				denominator = sum([Nis[i]*fs[i]*bs[i,k] for i in range(Nsims)])
				as_new[k] = nominator[k]/denominator			
			as_new /= as_new.sum() #enforce normalization
			#check convergence
			integrated_diff = np.abs(as_new-as_old).sum()
			if verbose:
				max = as_new.max()
				imax = np.where(as_new==max)[0]
				#min = as_new.min()
				#imin = np.where(as_new==min)[0]
				print('cycle %i/%i' %(iscf+1,Nscf))
				print('  norm prob. dens. [au] = %.3e' %as_new.sum())
				print('  max prob. dens.  [au] = %.3e' %max)
				print('  cv for max       [%s] = ' %cv_unit, bin_centers[imax]/parse_unit(cv_unit))
				#print('  min prob. dens.  [au] = %.3e' %min)
				#print('  cv for min       [%s] = ' %cv_unit, bin_centers[imin]/parse_unit(cv_unit))
				print('  Integr. Diff.    [au] = %.3e' %integrated_diff)
				print('')
			if integrated_diff<convergence:
				print('WHAM SCF Converged!')
				break
			else:
				if iscf==Nscf-1:
					print('WARNING: could not converge WHAM equations to convergence of %.3e in %i steps!' %(convergence, Nscf))
			as_old = as_new.copy()
		cvs = bin_centers.copy()
		ps = as_new.copy()
		#compute final normalization factors
		for i in range(Nsims):
			fs[i] = 1.0/np.dot(bs[i,:],as_new)
		plower = None
		pupper = None
		if error_estimate in ['mle_p']:
			#construct the extended Fisher information matrix corresponding to histogram counts directly as the weighted sum of the indivual Fisher matrices of each simulation separately.
			I = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])
			for i in range(Nsims):
				Ii = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])
				for k in range(Ngrid):
					#avoid singularity at ps[k]=0, this is an inherint isssue to mle_p not present in the mle_f error estimation
					if ps[k]>0:
						Ii[k,k] = fs[i]*bs[i,k]/ps[k]
					Ii[k,Ngrid+i] = bs[i,k]
					Ii[Ngrid+i,k] = bs[i,k]
					Ii[k,Ngrid+Nsims+i] = fs[i]*bs[i,k]
					Ii[Ngrid+Nsims+i,k] = fs[i]*bs[i,k]
					Ii[k,-1] = 1
					Ii[-1,k] = 1
				Ii[Ngrid+i,Ngrid+i] = 1/fs[i]**2
				Ii[Ngrid+i,Ngrid+Nsims+i] = 1/fs[i]
				Ii[Ngrid+Nsims+i,Ngrid+i] = 1/fs[i]
				I += Nis[i]*Ii
			#the covariance matrix is now simply the inverse of the total Fisher matrix. However, for histogram bins with count 0 (and hence probability 0), we cannot estimate the corresponding error. This can be seen above as ps[k]=0 gives an divergence. Therefore, we define a mask corresponding to only non-zero probabilities and define and inverted the masked information matrix
			mask = np.ones([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1], dtype=bool)
			for k in range(Ngrid):
				if ps[k]==0:
					mask[k,:] = np.zeros(Ngrid+2*Nsims+1, dtype=bool)
					mask[:,k] = np.zeros(Ngrid+2*Nsims+1, dtype=bool)
			Nmask2 = len(I[mask])
			assert abs(int(np.sqrt(Nmask2))-np.sqrt(Nmask2))==0
			Nmask = int(np.sqrt(Nmask2))
			Imask = I[mask].reshape([Nmask,Nmask])
			sigma = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])*np.nan
			sigma[mask] = np.linalg.inv(Imask).reshape([Nmask2])
			#the error bar on the probability of bin k is now simply the (k,k)-diagonal element of sigma
			perr = np.sqrt(np.diagonal(sigma)[:Ngrid])
			plower = ps - nsigma*perr
			plower[plower<0] = 0.0
			pupper = ps + nsigma*perr
		elif error_estimate in ['mle_f']:
			#construct the extended Fisher information matrix corresponding to minus the logarithm of the histogram counts (which are related to the free energy values) as the weighted sum of the indivual Fisher matrices of each simulation separately
			I = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])
			for i in range(Nsims):
				Ii = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])
				for k in range(Ngrid):
					#note that ps=exp(-gs) where gs are the degrees of freedom in the mle_f method (minus logarithm of the histogram counts)
					Ii[k,k] = ps[k]*fs[i]*bs[i,k]
					Ii[k,Ngrid+i] = -ps[k]*bs[i,k]
					Ii[Ngrid+i,k] = -ps[k]*bs[i,k]
					Ii[k,Ngrid+Nsims+i] = -fs[i]*ps[k]*bs[i,k]
					Ii[Ngrid+Nsims+i,k] = -fs[i]*ps[k]*bs[i,k]
					Ii[k,-1] = -ps[k]
					Ii[-1,k] = -ps[k]
				Ii[Ngrid+i,Ngrid+i] = 1/fs[i]**2
				Ii[Ngrid+i,Ngrid+Nsims+i] = 1/fs[i]
				Ii[Ngrid+Nsims+i,Ngrid+i] = 1/fs[i]
				I += Nis[i]*Ii
			#the covariance matrix is now simply the inverse of the total Fisher matrix. However, for histogram bins with count 0 (and hence probability 0), we cannot estimate the corresponding error. This can be seen above as ps[k]=0 gives an divergence. Therefore, we define a mask corresponding to only non-zero probabilities and define and inverted the masked information matrix
			mask = np.ones([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1], dtype=bool)
			for k in range(Ngrid):
				if ps[k]==0:
					mask[k,:] = np.zeros(Ngrid+2*Nsims+1, dtype=bool)
					mask[:,k] = np.zeros(Ngrid+2*Nsims+1, dtype=bool)
			Nmask2 = len(I[mask])
			assert abs(int(np.sqrt(Nmask2))-np.sqrt(Nmask2))==0
			Nmask = int(np.sqrt(Nmask2))
			Imask = I[mask].reshape([Nmask,Nmask])
			sigma = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])*np.nan
			sigma[mask] = np.linalg.inv(Imask).reshape([Nmask2])
			#the error bar on the g-value (i.e. minus logarithm of the probability) of bin k is now simply the (k,k)-diagonal element of sigma
			gerr = np.sqrt(np.diagonal(sigma)[:Ngrid])
			plower = ps*np.exp(-nsigma*gerr)
			pupper = ps*np.exp(nsigma*gerr)
		elif error_estimate is not None and error_estimate not in ["None"]:
			raise ValueError('Received invalid value for keyword argument error_estimate, got %s. See documentation for valid choices.' %error_estimate)
		return cls(cvs, ps, plower=plower, pupper=pupper)

	@classmethod
	def from_fep(cls, fep, temp):
		'''
			Use the given free energy profile to construct the corresponding probability histogram at the given temperature.

			:param fep: free energy profile from which the probability histogram is computed
			:type fep: fep.BaseFreeEnergyProfile/fep.SimpleFreeEnergyProfile

			:param temp: the temperature at which the histogram input data was simulated
			:type temp: float

			:return: probability histogram corresponding to the given free energy profile
			:rtype: Histogram1D
		'''
		pupper = None
		plower = None
		ps = np.zeros(len(fep.fs))
		beta = 1.0/(boltzmann*temp)
		ps = np.exp(-beta*fep.fs)
		ps /= ps[~np.isnan(ps)].sum()
		if fep.fupper is not None and fep.flower is not None:
			pupper = np.zeros(len(fep.fs))
			plower = np.zeros(len(fep.fs))
			pupper = np.exp(-beta*fep.flower)
			plower = np.exp(-beta*fep.fupper)
			pupper /= pupper[~np.isnan(pupper)].sum()
			plower /= plower[~np.isnan(plower)].sum()
		return cls(fep.cvs, ps, pupper=pupper, plower=plower, cv_unit=fep.cv_unit, cv_label=fep.cv_label)

	def plot(self, fn, temp=None, flim=None):
		'''
			Make a plot of the probability histogram and possible the corresponding free energy (if the argument ``temp`` is specified).

			:param fn: file name of the resulting plot
			:type fn: str

			:param temp: the temperature for conversion of histogram to free energy profile. Specifying this number will add a free energy plot. Defaults to None
			:type temp: float, optional

			:param flim: upper limit of the free energy axis in plots. Defaults to None
			:type flim: float, optional
		'''
		plot_histograms(fn, [self], temp=temp, flim=flim)


class Histogram2D(object):
	def __init__(self, cv1s, cv2s, ps, plower=None, pupper=None, cv1_unit='au', cv2_unit='au', cv1_label='CV1', cv2_label='CV2'):
		'''
			Class to implement the estimation of a probability histogram in terms of a collective variable from a trajectory series corresponding to that collective variable.

			:param cv1s: 1D array corresponding the grid points of the first collective variable
			:type data: np.ndarray

			:param cv2s: 1D array corresponding the grid points of the second collective variable
			:type data: np.ndarray

			:param ps: 2D array corresponding to the histogram probability values at the (CV1,CV2) grid points
			:type bins: np.ndarray

			:param plower: lower bound of the error on the probability histogram values, defaults to None
			:type do_error: np.ndarray, optional

			:param pupper: upper bound of the error on the probability histogram values, defaults to None
			:type pupper: np.ndarray, optional

			:param cv1_unit: the unit in which the first collective variable will be plotted/printed, defaults to 'au'
			:type cv_unit: str, optional

			:param cv2_unit: the unit in which the second collective variable will be plotted/printed, defaults to 'au'
			:type cv_unit: str, optional

			:param cv1_label: the label of the first collective variable that will be used on plots, defaults to 'CV1'
			:type cv1_label: str, optional

			:param cv2_label: the label of the first collective variable that will be used on plots, defaults to 'CV2'
			:type cv2_label: str, optional
		'''
		self.cv1s = cv1s.copy()
		self.cv2s = cv2s.copy()
		self.ps = ps.copy()
		self.plower = plower
		self.pupper = pupper
		self.cv1_unit = cv1_unit
		self.cv2_unit = cv2_unit
		self.cv1_label = cv1_label
		self.cv2_label = cv2_label

	def copy(self):
		plower = None
		if self.plower is not None:
			plower = self.plower.copy()
		pupper = None
		if self.pupper is not None:
			pupper = self.pupper.copy()
		return Histogram1D(
			self.cvs1.copy(), self.cvs2.copy(), self.ps.copy(), plower=plower, pupper=pupper, 
			cv1_unit=self.cv1_unit, cv2_unit=self.cv2_unit, cv1_label=self.cv1_label, cv2_label=self.cv2_label
		)

	@classmethod
	def from_average(cls, histograms, error_estimate=None, nsigma=2):
		'''
			Start from a set of histograms and compute and return the averaged histogram. If error_estimate is set to 'std', an error on the histogram will be computed from the standard deviation within the set of histograms.

			:param histograms: set of histrograms to be averaged
			:type histograms: list(Histogram1D)

			:param error_estimate: indicate if and how to perform error analysis. One of following options is available:
			
				*  **std** -- compute error from the standard deviation within the set of histograms.
				*  **None** -- do not estimate the error.
			
			Defaults to None, i.e. no error estimate.
			:type error_estimate: str, optional

			:param nsigma: only relevant when error estimation is turned on (i.e. when keyword ``error_estimate`` is not None), this option defines how large the error interval should be in terms of the standard deviation sigma. A ``nsigma=2`` implies a 2-sigma error bar (corresponding to 95% confidence interval) will be returned. Defaults to 2
			:type nsigma: int, optional

			:return: averages histogram
			:rtype: Histrogram1D
		'''
		#sanity checks
		cv1_unit, cv2_unit, cv1_label, cv2_label = None, None, None, None
		cv1s, cv2s = None, None
		for hist in histograms:
			if cv1s is None:
				cv1s = hist.cv1s.copy()
			else:
				assert (abs(cv1s-hist.cv1s)<1e-12*np.abs(cv1s.mean())).all(), 'Cannot average the histograms as they do not have consistent CV1 grids'
			if cv2s is None:
				cv2s = hist.cv2s.copy()
			else:
				assert (abs(cv2s-hist.cv2s)<1e-12*np.abs(cv2s.mean())).all(), 'Cannot average the histograms as they do not have consistent CV2 grids'
			if cv1_unit is None:
				cv1_unit = hist.cv1_unit
			else:
				assert cv1_unit==hist.cv1_unit, 'Inconsistent CV1 unit definition in histograms'
			if cv2_unit is None:
				cv2_unit = hist.cv2_unit
			else:
				assert cv2_unit==hist.cv2_unit, 'Inconsistent CV2 unit definition in histograms'
			if cv1_label is None:
				cv1_label = hist.cv1_label
			else:
				assert cv1_label==hist.cv1_label, 'Inconsistent CV1 label definition in histograms'
			if cv2_label is None:
				cv2_label = hist.cv2_label
			else:
				assert cv2_label==hist.cv2_label, 'Inconsistent CV2 label definition in histograms'
		#collect probability distributions
		pss = np.array([hist.ps for hist in histograms])
		#average histograms
		ps = pss.mean(axis=0)
		#compute error if requested
		plower, pupper = None, None
		if error_estimate is not None and error_estimate.lower() in ['std']:
			perr = pss.std(axis=0,ddof=1)
			plower = ps - nsigma*perr
			plower[plower<0] = 0.0
			pupper = ps + nsigma*perr
		return cls(cv1s, cv2s, ps, plower=plower, pupper=pupper, cv1_unit=cv1_unit, cv2_unit=cv2_unit, cv1_label=cv1_label, cv2_label=cv2_label)

	@classmethod
	def from_single_trajectory(cls, data, bins, error_estimate=None, nsigma=2, cv1_unit='au', cv2_unit='au', cv1_label='CV1', cv2_label='CV2'):
		'''
			Routine to estimate of a probability histogram in terms of two collective variables from a trajectory series corresponding to that collective variable.

			:param data: 2D array corresponding to the trajectory series of the two collective variables. The first column is assumed to correspond to the first collective variable, the second column to the second CV.
			:type data: np.ndarray([N,2])

			:param bins: array representing the edges of the bins for which a histogram will be constructed. This argument is parsed to `the numpy.histogram2d routine <https://numpy.org/doc/stable/reference/generated/numpy.histogram.html>`_. Hence, more information on its meaning and allowed values can be found there.
			:type bins: np.ndarray

			:param error_estimate: indicate if and how to perform error analysis. One of following options is available:
			
				*  **mle_p** -- compute error through the asymptotic normality of the maximum likelihood estimator for the probability itself. WARNING: due to positivity constraint of the probability, this only works for low variance. Otherwise the standard error interval for the normal distribution (i.e. mle +- n*sigma) might give rise to negative lower error bars.
				*  **mle_f** -- compute error through the asymptotic normality of the maximum likelihood estimator for -log(p) (hence for the beta-scaled free energy). This estimation does not suffor from the same WARNING as for ``mle_p``. Furthermore, in case of low variance, the error estimation using ``mle_f`` and ``mle_p`` are consistent (i.e. one can be computed from the other using f=-log(p)).
				*  **None** -- do not estimate the error.
			
			Defaults to None, i.e. no error estimate.
			:type error_estimate: str, optional

			:param nsigma: only relevant when error estimation is turned on (i.e. when keyword ``error_estimate`` is not None), this option defines how large the error interval should be in terms of the standard deviation sigma. A ``nsigma=2`` implies a 2-sigma error bar (corresponding to 95% confidence interval) will be returned. Defaults to 2
			:type nsigma: int, optional

			:param cv1_unit: the unit in which the first collective variable will be plotted/printed, defaults to 'au'
			:type cv_unit: str, optional

			:param cv2_unit: the unit in which the second collective variable will be plotted/printed, defaults to 'au'
			:type cv_unit: str, optional

			:param cv1_label: the label of the first collective variable that will be used on plots, defaults to 'CV1'
			:type cv1_label: str, optional

			:param cv2_label: the label of the first collective variable that will be used on plots, defaults to 'CV2'
			:type cv2_label: str, optional
		'''
		#initialization
		cv1s, cv2s = None, None
		ps = None
		Ntot = len(data)
		#generate histogram using numpy.histogram routine
		ns, cv1_edges, cv2_edges = np.histogram2d(data[0,:], data[1,:], bins=bins, density=False)
		cv1s = 0.5*(cv1_edges[:-1]+cv1_edges[1:]) # bin centers vor CV1
		cv2s = 0.5*(cv2_edges[:-1]+cv2_edges[1:]) # bin centers vor CV2
		ps = ns/Ntot
		#estimate of upper and lower boundary of n-sigma confidence interval
		plower = None
		pupper = None
		if error_estimate=='mle_p':
			perrors = np.sqrt(ps*(1-ps)/Ntot)
			plower = ps - nsigma*perrors
			plower[plower<0] = 0.0
			pupper = ps + nsigma*perrors
		elif error_estimate=='mle_f':
			#we first compute the error bar interval on f=-log(p) and then transform it to one on p itself.
			fs, ferrors = np.zeros(len(ps))*np.nan, np.zeros(len(ps))*np.nan
			fs[ps>0] = -np.log(ps[ps>0])
			ferrors[ps>0] = np.sqrt((np.exp(fs[ps>0])-1)/Ntot)
			fupper  = fs + nsigma*ferrors
			flower  = fs - nsigma*ferrors
			pupper, plower = np.zeros(len(ps))*np.nan, np.zeros(len(ps))*np.nan
			pupper[ps>0] = np.exp(-flower[ps>0])
			plower[ps>0] = np.exp(-fupper[ps>0])
			pupper[np.isinf(pupper)] = np.nan
		elif error_estimate is not None:
			raise ValueError('Invalid value for error_estimate argument, received %s. Check documentation for allowed values.' %error_estimate)
		return cls(cv1s, cv2s, ps, plower=plower, pupper=pupper, cv1_unit=cv1_unit, cv2_unit=cv2_unit, cv1_label=cv1_label, cv2_label=cv2_label)

	@classmethod
	def from_wham(cls, bins, trajectories, biasses, temp, pinit=None, error_estimate=None, nsigma=2, bias_subgrid_num=20, Nscf=1000, convergence=1e-6, verbose=False, cv1_unit='au', cv2_unit='au', cv1_label='CV1', cv2_label='CV2', plot_biases=False):
		'''
			Routine that implements the Weighted Histogram Analysis Method (WHAM) for reconstructing the overall 2D probability histogram from a series of biased molecular simulations in terms of two collective variables CV1 and CV2.

			:param data: 2D array corresponding to the trajectory series of the two collective variables. The first column is assumed to correspond to the first collective variable, the second column to the second CV.
			:type data: np.ndarray([N,2])

			:param bins: list of the form [bins1, bins2] where bins1 and bins2 are numpy arrays each representing the bin edges of their corresponding CV for which a histogram will be constructed. For example the following definition:

				[np.arange(-1,1.05,0.05), np.arange(0,5.1,0.1)]

			will result in a 2D histogram with bins of width 0.05 between -1 and 1 for CV1 and bins of width 0.1 between 0 and 5 for CV2.
			:type bins: np.ndarray

			:param trajectories: list or array of 2D numpy arrays containing the [CV1,CV2] trajectory data for each simulation. Alternatively, a list of PLUMED file names containing the trajectory data can be specified as well. The arguments trajectories and biasses should be of the same length.
			:type trajectories: list(np.ndarray([Nmd,2]))/np.ndarray([Ntraj,Nmd,2])

			:param biasses: list of callables, each representing a function to compute the bias at a given value of the collective variables CV1 and CV2. The arguments trajectories and biasses should be of the same length.
			:type biasses: list(callable)

			:param temp: the temperature at which all simulations were performed
			:type temp: float

			:param pinit: initial guess for the probability density. If None is given, a uniform distribution is used as initial guess.
			:type pinit: np.ndarray, optional, default=None
			
			:param error_estimate: indicate if and how to perform error analysis. One of following options is available:
			
				*  **mle_p** -- compute error through the asymptotic normality of the maximum likelihood estimator for the probability itself. WARNING: due to positivity constraint of the probability, this only works for high probability and low variance. Otherwise the standard error interval for the normal distribution (i.e. mle +- n*sigma) might give rise to negative lower error bars.
				*  **mle_f** -- compute error through the asymptotic normality of the maximum likelihood estimator for -log(p) (hence for the beta-scaled free energy). This estimation does not suffor from the same WARNING as for ``mle_p``. Furthermore, in case of high probability and low variance, the error estimation using ``mle_f`` and ``mle_p`` are consistent (i.e. one can be computed from the other using f=-log(p)).
				*  **None** -- do not estimate the error.
			
			Defaults to None, i.e. no error estimate.
			:type error_estimate: str, optional

			:param nsigma: only relevant when error estimation is turned on (i.e. when keyword ``error_estimate`` is not None), this option defines how large the error interval should be in terms of the standard deviation sigma. A ``nsigma=2`` implies a 2-sigma error bar (corresponding to 95% confidence interval) will be returned. Defaults to 2
			:type nsigma: int, optional
			
			:param bias_subgrid_num: the number of grid points along each CV for the sub-grid used to compute the integrated boltzmann factor of the bias in each CV1,CV2 bin. Either a single integer is given, corresponding to identical number of subgrid points for both CVs, or a list of two integers corresponding the number of grid points in the two CVs respectively.
			:type bias_subgrid_num: optional, defaults to [20,20]

			:param Nscf: the maximum number of steps in the self-consistent loop to solve the WHAM equations
			:type Nscf: int, defaults to 1000

			:param convergence: convergence criterium for the WHAM self consistent solver. The SCF loop will stop whenever the integrated absolute difference between consecutive probability densities is less then the specified value.
			:type convergence: float, defaults to 1e-6

			:param verbose: set to True to turn on more verbosity during the self consistent solution cycles of the WHAM equations, defaults to False.
			:type verbose: bool, optional

			:param cv1_unit: the unit in which the first collective variable will be plotted/printed, defaults to 'au'
			:type cv_unit: str, optional

			:param cv2_unit: the unit in which the second collective variable will be plotted/printed, defaults to 'au'
			:type cv_unit: str, optional

			:param cv1_label: the label of the first collective variable that will be used on plots, defaults to 'CV1'
			:type cv1_label: str, optional

			:param cv2_label: the label of the first collective variable that will be used on plots, defaults to 'CV2'
			:type cv2_label: str, optional

			:param plot_biases: if set to True, a 2D plot of the boltzmann factor of the bias integrated over each bin will be made. Defaults to False.
			:type plot_biases: bool, optional.

			:return: 2D probability histogram
			:rtype: Histogram2D
		'''
		if verbose:
			print('Initialization')
			print('--------------')
		
		#checks and initialization
		assert len(biasses)==len(trajectories), 'The arguments trajectories and biasses should be of the same length.'
		beta = 1/(boltzmann*temp)
		Nsims = len(biasses)
		if isinstance(bias_subgrid_num, int):
			bias_subgrid_num = [bias_subgrid_num, bias_subgrid_num]
		else:
			assert isinstance(bias_subgrid_num,list) or isinstance(bias_subgrid_num,np.ndarray), 'bias_subgrid_num argument should be an integer or a list of two integers'
			assert len(bias_subgrid_num)==2, 'bias_subgrid_num argument should be an integer or a list of two integers'
			assert isinstance(bias_subgrid_num[0],int), 'bias_subgrid_num argument should be an integer or a list of two integers'
			assert isinstance(bias_subgrid_num[1],int), 'bias_subgrid_num argument should be an integer or a list of two integers'
		
		#Preprocess trajectory argument: load files if file names are given instead of raw data, determine and store the number of simulation steps in each simulation:
		if verbose:
			print('processing trajectories ...')
		Nis = []
		data = []
		for i, trajectory in enumerate(trajectories):
			if isinstance(trajectory, str):
				trajectory = np.loadtxt(trajectory)[:,1:3] #first column is the time, second column is the CV1, third column is the CV2
			Nis.append(len(trajectory))
			data.append(trajectory)
		Nis = np.array(Nis)
		
		#Preprocess the bins argument and redefine it to represent the bin_edges. We need to do this beforehand to make sure that when calling the numpy.histogram routine with this bins argument, each histogram will have a consistent bin_edges array and hence consistent histogram.
		if verbose:
			print('processing bins ...')
		assert isinstance(bins,list), "Bins argument should be a list of 2 numpy arrays with the corresponding bin edges. Current definition is not a list, but an object of type %s." %bins.__class__.__name__
		assert len(bins)==2, "Bins argument should be a list of 2 numpy arrays with the corresponding bin edges. Current list definition does not have two elements, but %i." %(len(bins))
		assert isinstance(bins[0],np.ndarray), "Bins argument should be a list of 2 numpy arrays with the corresponding bin edges. The first element in the current definition is not a numpy array but an object of type %s" %bins[0].__class__.__name__
		assert isinstance(bins[1],np.ndarray), "Bins argument should be a list of 2 numpy arrays with the corresponding bin edges. The second element in the current definition is not a numpy array but an object of type %s" %bins[1].__class__.__name__
		bin_centers1, bin_centers2 = 0.5*(bins[0][:-1]+bins[0][1:]), 0.5*(bins[1][:-1]+bins[1][1:])
		deltas1, deltas2 = bins[0][1:]-bins[0][:-1], bins[1][1:]-bins[1][:-1]
		grid_non_uniformity1, grid_non_uniformity2 = deltas1.std()/deltas1.mean(), deltas2.std()/deltas2.mean()
		assert grid_non_uniformity1<1e-6, 'CV1 grid defined by bins argument should be of uniform spacing! Non-uniform grids not implemented.'
		assert grid_non_uniformity2<1e-6, 'CV2 grid defined by bins argument should be of uniform spacing! Non-uniform grids not implemented.'
		delta1, delta2 = deltas1.mean(), deltas2.mean()
		Ngrid1, Ngrid2 = len(bin_centers1), len(bin_centers2)
		Ngrid = Ngrid1*Ngrid2
		
		#generate the individual histograms using numpy.histogram
		if verbose:
			print('generating individual histograms for each biased simulation ...')
		Hs = np.zeros([Nsims, Ngrid1, Ngrid2])
		for i, traj in enumerate(data):
			ns, edges1, edges2 = np.histogram2d(traj[:,0], traj[:,1], bins, density=False)
			assert (bins[0]==edges1).all()
			assert (bins[1]==edges2).all()
			Hs[i,:,:] = ns.copy()
			N = ns.sum()
			if not Nis[i]==N:
				print('WARNING: Histogram of trajectory %i should have total count of %i (=number of simulation steps), but found %f (are you sure the CV range is sufficient?). Number of simulation steps adjusted to match total histogram count.' %(i,Nis[i],N))
				Nis[i] = N
		
		#compute the boltzmann factors of the biases in each grid interval
		#b_ikl = 1/(delta1*delta2)*int(exp(-beta*W_i(q1,q2)), q1=Q_k-delta1/2...Q_k+delta1/2, q2=Q_l-delta2/2...Q_l+delta2/2)
		if verbose:
			print('computing bias on grid ...')
		bs = np.zeros([Nsims, Ngrid1, Ngrid2])
		for i, bias in enumerate(biasses):
			for k, center1 in enumerate(bin_centers1):
				subdelta1 = delta1/bias_subgrid_num[0]
				subgrid1 = np.arange(center1-delta1/2, center1+delta1/2 + subdelta1, subdelta1)
				for l, center2 in enumerate(bin_centers2):
					subdelta2 = delta2/bias_subgrid_num[1]
					subgrid2 = np.arange(center2-delta2/2, center2+delta2/2 + subdelta2, subdelta2)
					CV1, CV2 = np.meshgrid(subgrid1, subgrid2, indexing='ij')
					Ws = np.exp(-beta*bias(CV1,CV2))
					bs[i,k,l] = integrate2d(Ws, dx=subdelta1, dy=subdelta2)/(delta1*delta2)
			if plot_biases:
				bias.plot('bias_%i.png' %i, bin_centers1, bin_centers2)
		
		#some init printing
		if verbose:
			print('')
			print('WHAM setup')
			print('----------')
			print('Number of simulations = ', Nsims)
			for i, Ni in enumerate(Nis):
				print('    simulation %i has %i steps' %(i,Ni))
			print('CV1 grid [%s]: start = %.3f    end = %.3f    delta = %.3f    N = %i' %(cv1_unit, bins[0].min()/parse_unit(cv1_unit), bins[0].max()/parse_unit(cv1_unit), delta1/parse_unit(cv1_unit), Ngrid1))
			print('CV2 grid [%s]: start = %.3f    end = %.3f    delta = %.3f    N = %i' %(cv2_unit, bins[1].min()/parse_unit(cv2_unit), bins[1].max()/parse_unit(cv2_unit), delta2/parse_unit(cv2_unit), Ngrid2))
			print('')
			print('Starting WHAM SCF loop:')
			print('-----------------------')
		
		#self consistent loop to solve the WHAM equations
		##initialize probability density array (which should sum to 1)
		if pinit is None:
			as_old = np.ones([Ngrid1,Ngrid2])/Ngrid
		else:
			assert pinit.shape[0]==Ngrid1, 'Specified initial guess should be of shape (%i,%i), got (%i,%i)' %(Ngrid1,Ngrid2,pinit.shape[0],pinit.shape[1])
			assert pinit.shape[1]==Ngrid2, 'Specified initial guess should be of shape (%i,%i), got (%i,%i)' %(Ngrid1,Ngrid2,pinit.shape[0],pinit.shape[1])
			as_old = pinit.copy()/pinit.sum()
		nominator = Hs.sum(axis=0) #precomputable factor in WHAM update equations
		for iscf in range(Nscf):
			#compute new normalization factors
			fs = 1.0/np.einsum('ikl,kl->i', bs, as_old)
			#compute new probabilities
			as_new = np.zeros(as_old.shape)
			for k in range(Ngrid1):
				for l in range(Ngrid2):
					denominator = sum([Nis[i]*fs[i]*bs[i,k,l] for i in range(Nsims)])
					as_new[k,l] = nominator[k,l]/denominator			
			as_new /= as_new.sum() #enforce normalization
			#check convergence
			integrated_diff = np.abs(as_new-as_old).sum()
			if verbose:
				max = as_new.max()
				kmax, lmax = [indices[0] for indices in np.where(as_new==as_new.max())]
				print('cycle %i/%i' %(iscf+1,Nscf))
				print('  norm prob. dens. = %.3e au' %as_new.sum())
				print('  max prob. dens.  = %.3e au' %max)
				print('  cv1,cv2 for max  = %.3e %s , %.3f %s' %(bin_centers1[kmax]/parse_unit(cv1_unit), cv1_unit, bin_centers2[lmax]/parse_unit(cv2_unit), cv2_unit))
				print('  Integr. Diff.    = %.3e au' %integrated_diff)
				print('')
			if integrated_diff<convergence:
				print('WHAM SCF Converged!')
				break
			else:
				if iscf==Nscf-1:
					print('WARNING: could not converge WHAM equations to convergence of %.3e in %i steps!' %(convergence, Nscf))
			as_old = as_new.copy()
		
		#finalization
		cv1s = bin_centers1.copy()
		cv2s = bin_centers2.copy()
		ps = as_new.copy()
		fs = 1.0/np.einsum('ikl,kl->i', bs, as_new)
		plower = None
		pupper = None
		
		#error estimation
		if error_estimate in ['mle_p']:
			#construct the extended Fisher information matrix corresponding to histogram counts directly as the weighted sum of the indivual Fisher matrices of each simulation separately. This is very similar as in the 1D case. However, we first have to flatten the CV1,CV2 2D grid to a 1D CV12 grid. This is achieved with the flatten function (which flattens a 2D index to a 1D index). Deflattening of the CV12 grid to a 2D CV1,CV2 grid is achieved with the deflatten function (which deflattens a 1D index to a 2D index). Using the flatten function, the ps array is flattened and a conventional Fisher matrix can be constructed and inverted. Afterwards the covariance matrix is deflattened using the deflatten function to arrive to a multidimensional matrix giving (co)variances on the 2D probability array.
			def flatten(k,l):
				return Ngrid2*k+l
			def deflatten(K):
				k = int(K/Ngrid2)
				l = K - Ngrid2*k
				return k,l
			I = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])
			for i in range(Nsims):
				Ii = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])
				for k in range(Ngrid1):
					for l in range(Ngrid2):
						K = flatten(k,l)
						if ps[k,l]>0:
							Ii[K,K] = fs[i]*bs[i,k,l]/ps[k,l]
						Ii[K, Ngrid+i] = bs[i,k,l]
						Ii[Ngrid+i, K] = bs[i,k,l]
						Ii[K, Ngrid+Nsims+i] = fs[i]*bs[i,k,l]
						Ii[Ngrid+Nsims+i, K] = fs[i]*bs[i,k,l]
						Ii[K,-1] = 1
						Ii[-1,K] = 1
				Ii[Ngrid+i, Ngrid+i] = 1/fs[i]**2
				Ii[Ngrid+i, Ngrid+Nsims+i] = 1/fs[i]
				Ii[Ngrid+Nsims+i, Ngrid+i] = 1/fs[i]
				I += Nis[i]*Ii
			#the covariance matrix is now simply the inverse of the total Fisher matrix. However, for histogram bins with count 0 (and hence probability 0), we cannot estimate the corresponding error. This can be seen above as ps[k]=0 introduces a divergence. Therefore, we define a mask corresponding to only non-zero probabilities and define and invert the masked information matrix
			mask = np.ones([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1], dtype=bool)
			for k in range(Ngrid1):
				for l in range(Ngrid2):
					K = flatten(k,l)
					if ps[k,l]==0:
						mask[K,:] = np.zeros(Ngrid+2*Nsims+1, dtype=bool)
						mask[:,K] = np.zeros(Ngrid+2*Nsims+1, dtype=bool)
			Nmask2 = len(I[mask])
			assert abs(int(np.sqrt(Nmask2))-np.sqrt(Nmask2))==0 #consistency check, sqrt(Nmask2) should be integer valued
			Nmask = int(np.sqrt(Nmask2))
			Imask = I[mask].reshape([Nmask,Nmask])
			sigma = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])*np.nan
			sigma[mask] = np.linalg.inv(Imask).reshape([Nmask2])
			#the error bar on the probability of bin (k,l) is now simply the [K,K]-diagonal element of sigma (with [K,K] corresponding to deflattend [(k,l),(k,l)])
			perr = np.zeros([Ngrid1,Ngrid2])
			for K in range(Ngrid):
				k, l = deflatten(K)
				perr[k,l] = np.sqrt(sigma[K,K])
			plower = ps - nsigma*perr
			plower[plower<0] = 0.0
			pupper = ps + nsigma*perr
		elif error_estimate in ['mle_f']:
			######################################################################
			#TODO: has to be adapted to 2D variant as already done above for mle_p
			######################################################################
			
			#construct the extended Fisher information matrix corresponding to minus the logarithm of the histogram counts (which are related to the free energy values) as the weighted sum of the indivual Fisher matrices of each simulation separately
			I = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])
			for i in range(Nsims):
				Ii = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])
				for k in range(Ngrid):
					#note that ps=exp(-gs) where gs are the degrees of freedom in the mle_f method (minus logarithm of the histogram counts)
					Ii[k,k] = ps[k]*fs[i]*bs[i,k]
					Ii[k,Ngrid+i] = -ps[k]*bs[i,k]
					Ii[Ngrid+i,k] = -ps[k]*bs[i,k]
					Ii[k,Ngrid+Nsims+i] = -fs[i]*ps[k]*bs[i,k]
					Ii[Ngrid+Nsims+i,k] = -fs[i]*ps[k]*bs[i,k]
					Ii[k,-1] = -ps[k]
					Ii[-1,k] = -ps[k]
				Ii[Ngrid+i,Ngrid+i] = 1/fs[i]**2
				Ii[Ngrid+i,Ngrid+Nsims+i] = 1/fs[i]
				Ii[Ngrid+Nsims+i,Ngrid+i] = 1/fs[i]
				I += Nis[i]*Ii
			#the covariance matrix is now simply the inverse of the total Fisher matrix. However, for histogram bins with count 0 (and hence probability 0), we cannot estimate the corresponding error. This can be seen above as ps[k]=0 introduces a row (and column) or zeros resulting in a singular matrix. Therefore, we define a mask corresponding to only non-zero probabilities and define and inverted the masked information matrix
			mask = np.ones([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1], dtype=bool)
			for k in range(Ngrid):
				if ps[k]==0:
					mask[k,:] = np.zeros(Ngrid+2*Nsims+1, dtype=bool)
					mask[:,k] = np.zeros(Ngrid+2*Nsims+1, dtype=bool)
			Nmask2 = len(I[mask])
			assert abs(int(np.sqrt(Nmask2))-np.sqrt(Nmask2))==0
			Nmask = int(np.sqrt(Nmask2))
			Imask = I[mask].reshape([Nmask,Nmask])
			sigma = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])*np.nan
			sigma[mask] = np.linalg.inv(Imask).reshape([Nmask2])
			#the error bar on the g-value (i.e. minus logarithm of the probability) of bin k is now simply the (k,k)-diagonal element of sigma
			gerr = np.sqrt(np.diagonal(sigma)[:Ngrid])
			plower = ps*np.exp(-nsigma*gerr)
			pupper = ps*np.exp(nsigma*gerr)
		elif error_estimate is not None and error_estimate not in ["None"]:
			raise ValueError('Received invalid value for keyword argument error_estimate, got %s. See documentation for valid choices.' %error_estimate)
		
		#For consistency with how FreeEnergySurface2D is implemented, ps (and plower, pupper) need to be transposed
		ps = ps.T
		if plower is not None: plower = plower.T
		if pupper is not None: pupper = pupper.T
		return cls(cv1s, cv2s, ps, plower=plower, pupper=pupper)

	@classmethod
	def from_fes(cls, fes, temp):
		'''
			Use the given 2D free energy surface to construct the corresponding 2D probability histogram at the given temperature.

			:param fes: free energy surfave from which the probability histogram is computed
			:type fes: fep.FreeEnergySurface2D

			:param temp: the temperature at which the histogram input data was simulated
			:type temp: float

			:return: probability histogram corresponding to the given free energy profile
			:rtype: Histogram2D
		'''
		assert isinstance(fes, FreeEnergySurface2D), 'Input argument fes should be type FreeEnergySurface2D'
		pupper = None
		plower = None
		ps = np.zeros(len(fes.fs))
		beta = 1.0/(boltzmann*temp)
		ps = np.exp(-beta*fes.fs)
		ps /= ps[~np.isnan(ps)].sum()
		if fes.fupper is not None and fes.flower is not None:
			pupper = np.zeros(len(fes.fs))
			plower = np.zeros(len(fes.fs))
			pupper = np.exp(-beta*fes.flower)
			plower = np.exp(-beta*fes.fupper)
			pupper /= pupper[~np.isnan(pupper)].sum()
			plower /= plower[~np.isnan(plower)].sum()
		return cls(fes.cv1s, fes.cv2s, ps, pupper=pupper, plower=plower, cv1_unit=fes.cv1_unit, cv2_unit=fes.cv2_unit, cv1_label=fes.cv1_label, cv2_label=fes.cv2_label)

	def plot(self, fn, temp=None, flim=None):
		'''
			Make a 2D contour plot of the probability histogram and possible the corresponding free energy (if the argument ``temp`` is specified).

			:param fn: file name of the resulting plot
			:type fn: str

			:param temp: the temperature for conversion of histogram to free energy profile. Specifying this number will add a free energy plot. Defaults to None
			:type temp: float, optional

			:param flim: upper limit of the free energy axis in plots. Defaults to None
			:type flim: float, optional
		'''
		raise NotImplementedError


def plot_histograms(fn, histograms, temp=None, labels=None, flims=None, colors=None, linestyles=None, linewidths=None, set_ref='min'):
	'''
		Make a plot to compare multiple probability histograms and possible the corresponding free energy (if the argument ``temp`` is specified).

		:param fn: file name to write the figure to, the extension determines the format (PNG or PDF).
		:type fn: str

		:param histograms: list of histrograms to plot
		:type histograms: list(Histogram1D)

		:param temp: if defined, the free energy profile corresponding to each histogram will be computed and plotted with its corresponding temperature. If a single float is given, all histograms are assumed to be at the same temperature, if a list or array of floats is given, each entry is assumed to be the temperature of the corresponding entry in the input list of histograms. Defaults to None
		:type temp: float/list(float)/np.ndarray, optional

		:param labels: list of labels for the legend, one for each histogram. Defaults to None
		:type labels: list(str), optional

		:param flims: [lower,upper] limit of the free energy axis in plots. Defaults to None
		:type flims: float, optional

		:param colors: List of matplotlib color definitions for each entry in histograms. If an entry is None, a color will be chosen internally. Defaults to None, implying all colors are chosen internally.
		:type colors: List(str), optional

		:param linestyles: List of matplotlib line style definitions for each entry in histograms. If an entry is None, the default line style of '-' will be chosen . Defaults to None, implying all line styles are set to the default of '-'.
		:type linestyles: List(str), optional

		:param linewidths: List of matplotlib line width definitions for each entry in histograms. If an entry is None, the default line width of 1 will be chosen. Defaults to None, implying all line widths are set to the default of 2.
		:type linewidths: List(str), optional

		:param set_ref: set the reference of each corresponding free energy profile, see documentation of the set_ref routine of the free energy profile class for more information on the allowed values. Defaults to min, implying each profile is shifted vertically untill the minimum value equals zero.
		:type set_ref: str, optional
	'''
	#initialize
	linewidth_default = 2
	linestyle_default = '-'
	nhist = len(histograms)
	pp.clf()
	if temp is None:
		fig, axs = pp.subplots(nrows=1, ncols=1, squeeze=False)
	else:
		fig, axs = pp.subplots(nrows=1, ncols=2, squeeze=False)
	cmap = cm.get_cmap('tab10')
	cv_unit = histograms[0].cv_unit
	cv_label = histograms[0].cv_label
	fmax = -np.inf
	#add probability histogram and possible free energy plots	
	for ihist, hist in enumerate(histograms):
		#set label
		label = 'Histogram %i' %ihist
		if labels is not None:
			label = labels[ihist]
		#set color, linestyles, ...
		color = None
		if colors is not None:
			color = colors[ihist]
		if color is None:
			color = cmap(ihist)
		linewidth = None
		if linewidths is not None:
			linewidth = linewidths[ihist]
		if linewidth is None:
			linewidth = linewidth_default
		linestyle = None
		if linestyles is not None:
			linestyle = linestyles[ihist]
		if linestyle is None:
			linestyle = linestyle_default
		#plot probability
		axs[0,0].plot(hist.cvs/parse_unit(cv_unit), hist.ps/max(hist.ps), linewidth=linewidth, color=color, linestyle=linestyle, label=label)
		if hist.plower is not None and hist.pupper is not None:
			axs[0,0].fill_between(hist.cvs/parse_unit(cv_unit), hist.plower/max(hist.ps), hist.pupper/max(hist.ps), color=color, alpha=0.33)
		#free energy if requested
		if temp is not None:
			fep = BaseFreeEnergyProfile.from_histogram(hist, temp)
			if set_ref is not None:
				fep.set_ref(ref=set_ref)
			fmax = max(fmax, np.ceil(max(fep.fs[~np.isnan(fep.fs)]/kjmol)/10)*10)
			funit = fep.f_unit
			axs[0,1].plot(fep.cvs/parse_unit(cv_unit), fep.fs/parse_unit(funit), linewidth=linewidth, color=color, linestyle=linestyle, label=label)
			if fep.flower is not None and fep.fupper is not None:
				axs[0,1].fill_between(fep.cvs/parse_unit(cv_unit), fep.flower/parse_unit(funit), fep.fupper/parse_unit(funit), color=color, alpha=0.33)
	#decorate
	cv_range = [min(histograms[0].cvs/parse_unit(cv_unit)), max(histograms[0].cvs/parse_unit(cv_unit))]
	axs[0,0].set_xlabel('%s [%s]' %(cv_label, cv_unit), fontsize=14)
	axs[0,0].set_ylabel('Relative probability [-]', fontsize=14)
	axs[0,0].set_title('Probability histogram', fontsize=14)
	axs[0,0].set_xlim(cv_range)
	axs[0,0].legend(loc='best')
	if temp is not None:
		axs[0,1].set_xlabel('%s [%s]' %(cv_label, cv_unit), fontsize=14)
		axs[0,1].set_ylabel('F [%s]' %funit, fontsize=14)
		axs[0,1].set_title('Free energy profile', fontsize=14)
		axs[0,1].set_xlim(cv_range)
		if flims is None:
			axs[0,1].set_ylim([-1,fmax])
		else:
			axs[0,1].set_ylim(flims)
		axs[0,1].legend(loc='best')
	#save
	if temp is not None:
		fig.set_size_inches([16,8])
	else:
		fig.set_size_inches([8,8])
	pp.savefig(fn)
