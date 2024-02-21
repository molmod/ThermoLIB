#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 - 2023 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
# Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium;
# all rights reserved unless otherwise stated.
#
# This file is part of a library developed by Louis Vanduyfhuys at
# the Center for Molecular Modeling under supervision of prof. Veronique
# Van Speybroeck. Usage of this package should be authorized by prof. Van
# Van Speybroeck.


from molmod.units import *
from molmod.constants import *

import matplotlib.pyplot as pp
import matplotlib.cm as cm

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import time

from thermolib.thermodynamics.fep import BaseProfile, BaseFreeEnergyProfile
from thermolib.ext import wham1d_hs, wham1d_bias, wham1d_scf, wham1d_error, wham2d_hs, wham2d_bias, wham2d_scf, wham2d_error
from thermolib.error import GaussianDistribution, LogGaussianDistribution, Propagator, MultiGaussianDistribution, MultiLogGaussianDistribution
from thermolib.tools import fisher_matrix_mle_probdens, invert_fisher_to_covariance

__all__ = [
	'Histogram1D', 'Histogram2D', 'plot_histograms'
]

class Histogram1D(object):
	def __init__(self, cvs, ps, error=None, cv_output_unit='au', cv_label='CV'):
		'''
			Class to implement the estimation of a probability histogram in terms of a collective variable from a trajectory series corresponding to that collective variable.

			:param cvs: array corresponding the collective variable grid points, which should be in atomic units.
			:type data: np.ndarray

			:param ps: array corresponding to the histogram probability values at the grid points, which should be in atomic units.
			:type bins: np.ndarray

			:param error: error distribution on the histogram, defaults to None
			:type error: Distribution child class, optional

			:param cv_output_unit: the unit in which cv will be plotted/printed
			:type cv_output_unit: str, optional, default='au'

			:param cv_label: the label of the cv that will be used on plots
			:type cv_label: str, optional, default='CV'
		'''
		self.cvs = cvs.copy()
		self.ps = ps.copy()
		self.error = None
		if error is not None:
			self.error = error.copy()
		self.cv_output_unit = cv_output_unit
		self.cv_label = cv_label

	def plower(self, nsigma=2):
		assert self.error is not None, 'Plower cannot be computed because no error distribution was defined in the error attribute'
		plower, pupper = self.error.nsigma_conf_int(nsigma)
		return plower

	def pupper(self, nsigma=2):
		assert self.error is not None, 'Pupper cannot be computed because no error distribution was defined in the error attribute'
		plower, pupper = self.error.nsigma_conf_int(nsigma)
		return pupper

	def copy(self):
		if self.error is not None:
			error = self.error.copy()
		else:
			error = None
		return Histogram1D(self.cvs.copy(), self.ps.copy(), error=error, cv_output_unit=self.cv_output_unit, cv_label=self.cv_label)

	@classmethod
	def from_average(cls, histograms, error_estimate=None, cv_output_unit=None, cv_label=None):
		'''
			Start from a set of histograms and compute and return the averaged histogram. If error_estimate is set to 'std', an error on the histogram will be computed from the standard deviation within the set of histograms.

			:param histograms: set of histrograms to be averaged
			:type histograms: list(Histogram1D)

			:param error_estimate: indicate if and how to perform error analysis. One of following options is available:

				*  **std** -- compute error from the standard deviation within the set of histograms.
				*  **None** -- do not estimate the error.

			:type error_estimate: str, optional, default=None

			:param nsigma: only relevant when error estimation is turned on (i.e. when keyword ``error_estimate`` is not None), this option defines how large the error interval should be in terms of the standard deviation sigma. A ``nsigma=2`` implies a 2-sigma error bar (corresponding to 95% confidence interval) will be returned.
			:type nsigma: int, optional, default=2

			:param cv_output_unit: the unit in which cv will be plotted/printed. Defaults to the cv_output_unit of the first histogram given.
			:type cv_output_unit: str, optional

			:param cv_label: label for the new CV. Defaults to the cv_label of the first given histogram.
			:type cv_label: str, optional

			:return: averages histogram
			:rtype: Histrogram1D
		'''
		#sanity checks
		cvs = None
		for hist in histograms:
			if cvs is None:
				cvs = hist.cvs.copy()
			else:
				assert (abs(cvs-hist.cvs)<1e-12*np.abs(cvs.mean())).all(), 'Cannot average the histograms as they do not have consistent CV grids'
			if cv_output_unit is None:
				cv_output_unit = hist.cv_output_unit
			if cv_label is None:
				cv_label = hist.cv_label
			else:
				assert cv_label==hist.cv_label, 'Inconsistent CV label definition in histograms'
		#collect probability distributions
		pss = np.array([hist.ps for hist in histograms])
		#average histograms
		ps = pss.mean(axis=0)
		#compute error if requested
		error = None
		if error_estimate is not None and error_estimate.lower() in ['std']:
			perr = pss.std(axis=0,ddof=1)/np.sqrt(len(histograms))
			error = GaussianDistribution(ps, perr)
		return cls(cvs, ps,error=error, cv_output_unit=cv_output_unit, cv_label=cv_label)

	@classmethod
	def from_single_trajectory(cls, data, bins, error_estimate=None, error_p_thresshold=0.0, cv_output_unit='au', cv_label='CV'):
		'''
			Routine to estimate of a probability histogram in terms of a collective variable from a trajectory series corresponding to that collective variable.

			:param data: array corresponding to the trajectory series of the collective variable in atomic units
			:type data: np.ndarray

			:param bins: array representing the edges of the bins for which a histogram will be constructed. This argument is parsed to `the numpy.histogram routine <https://numpy.org/doc/stable/reference/generated/numpy.histogram.html>`_. Hence, more information on its meaning and allowed values can be found there.
			:type bins: np.ndarray

			:param error_estimate: indicate if and how to perform error analysis. One of following options is available:

				*  **mle_p/mle_p_cov** -- compute error through the asymptotic normality of the maximum likelihood estimator for the probability itself. WARNING: due to positivity constraint of the probability, this only works for low variance. Otherwise the standard error interval for the normal distribution (i.e. mle +- n*sigma) might give rise to negative lower error bars. In case mle_p_cov is given, the full covariance matrix will be accounted for instead of only the (uncorrelated) variance at each grid point.
				*  **mle_f/mle_f_cov** -- compute error through the asymptotic normality of the maximum likelihood estimator for -log(p) (hence for the beta-scaled free energy). This estimation does not suffor from the same WARNING as for ``mle_p``. Furthermore, in case of low variance, the error estimation using ``mle_f`` and ``mle_p`` are consistent (i.e. one can be computed from the other using f=-log(p)). In case mle_f_cov is given, the full covariance matrix will be accounted for instead of only the (uncorrelated) variance at each grid point.
				*  **None** -- do not estimate the error.

			:type error_estimate: str, optional, default=None

			:param nsigma: only relevant when error estimation is turned on (i.e. when keyword ``error_estimate`` is not None), this option defines how large the error interval should be in terms of the standard deviation sigma. A ``nsigma=2`` implies a 2-sigma error bar (corresponding to 95% confidence interval) will be returned.
			:type nsigma: int, optional, default=2

			:param cv_output_unit: the unit in which cv will be plotted/printed (not the unit of the input array, that is assumed to be atomic units)
			:type cv_output_unit: str, optional, default='au'

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
		error = None
		if error_estimate=='mle_p':
			perr = np.sqrt(ps*(1-ps)/Ntot)
			error = GaussianDistribution(ps, perr)
		elif error_estimate=='mle_f':
			#we first compute the error bar interval on f=-log(p) and then transform it to one on p itself.
			fs = -np.log(ps)
			ferr = np.sqrt((np.exp(fs)-1)/Ntot)
			error = LogGaussianDistribution(-fs, ferr)
		elif error_estimate=='mle_p_cov':
			F = fisher_matrix_mle_probdens(ps, method='mle_p_cov')
			cov = invert_fisher_to_covariance(F, ps, threshold=error_p_thresshold)
			error = MultiGaussianDistribution(ps, cov)
		elif error_estimate=='mle_f_cov':
			fs = -np.log(ps)
			F = fisher_matrix_mle_probdens(ps, method='mle_f_cov')
			cov = invert_fisher_to_covariance(F, ps, threshold=error_p_thresshold)
			error = MultiLogGaussianDistribution(-fs, cov)
		elif error_estimate is not None:
			raise ValueError('Invalid value for error_estimate argument, received %s. Check documentation for allowed values.' %error_estimate)
		return cls(cvs, ps, error=error, cv_output_unit=cv_output_unit, cv_label=cv_label)

	@classmethod
	def from_wham(cls, bins, traj_input, biasses, temp, error_estimate=None, corrtimes=None, bias_subgrid_num=20, Nscf=1000, convergence=1e-6, bias_thress=1e-3, cv_output_unit='au', cv_label='CV', verbose=None, verbosity='low'):
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
				*  **mle_p** -- compute error through the asymptotic normality of the maximum likelihood estimator for the probability itself and IGNORE the correlation between histogram bins. WARNING: due to positivity constraint of the probability, this only works for high probability and low variance. Otherwise the standard error interval for the normal distribution (i.e. mle +- n*sigma) might give rise to negative lower error bars.
				*  **mle_p_cov** -- compute error through the asymptotic normality of the maximum likelihood estimator for the probability itself and ACCOUNT for the correlation between histogram bins. WARNING: due to positivity constraint of the probability, this only works for high probability and low variance. Otherwise the standard error interval for the normal distribution (i.e. mle +- n*sigma) might give rise to negative lower error bars.
				*  **mle_f** -- compute error through the asymptotic normality of the maximum likelihood estimator for -log(p) (hence for the beta-scaled free energy) and IGNORE the correlation between histogram bins. This estimation does not suffor from the same WARNING as for ``mle_p``. Furthermore, in case of high probability and low variance, the error estimation using ``mle_f`` and ``mle_p`` are consistent (i.e. one can be computed from the other using f=-log(p)).
				*  **mle_f_cov** -- compute error through the asymptotic normality of the maximum likelihood estimator for -log(p) (hence for the beta-scaled free energy) and ACCOUNT for the correlation between histogram bins. This estimation does not suffor from the same WARNING as for ``mle_p``. Furthermore, in case of high probability and low variance, the error estimation using ``mle_f`` and ``mle_p`` are consistent (i.e. one can be computed from the other using f=-log(p)).
				*  **None** -- do not estimate the error.
			:type error_estimate: str, optional, default=None

			:param bias_subgrid_num: the number of grid points for the sub-grid used to compute the integrated boltzmann factor of the bias in each CV bin.
			:type bias_subgrid_num: optional, default=20

			:param Nscf: the maximum number of steps in the self-consistent loop to solve the WHAM equations
			:type Nscf: int, default=1000

			:param convergence: convergence criterium for the WHAM self consistent solver. The SCF loop will stop whenever the integrated absolute difference between consecutive probability densities is less then the specified value.
			:type convergence: float, default=1e-6

			:param verbose: set to True to turn on more verbosity during the self consistent solution cycles of the WHAM equations.
			:type verbose: bool, optional, default=False

			:param cv_output_unit: the unit in which cv will be plotted/printed
			:type cv_output_unit: str, optional, default='au'

			:param cv_label: the label of the cv that will be used on plots
			:type cv_label: str, optional, default='CV'

			:return: probability histogram
			:rtype: Histogram1D
		'''
		timings = {}
		timings['start'] = time.time()
		#backward compatibility between verbose and verbosity.
		if verbose is not None:
			print('The keyword verbose is depricated and will be removed in the near future. Use the keyword verbosity instead.')
			if verbose:
				verbosity = 'high'
			else:
				verbosity = 'none'
		if verbosity.lower() in ['medium', 'high']:
			print('Initialization ...')

		#checks and initialization
		assert len(biasses)==len(traj_input), 'The arguments traj_input and biasses should be of the same length.'
		beta = 1/(boltzmann*temp)
		Nsims = len(biasses)

		#Preprocess trajectory argument: load files if file names are given instead of raw data, determine and store the number of simulation steps in each simulation:
		if verbosity.lower() in ['high']:
			print('  processing trajectories')
		Nis = np.zeros(len(traj_input), dtype=int)
		trajectories = np.empty(len(traj_input), dtype=object)
		for i, trajectory in enumerate(traj_input):
			if isinstance(trajectory, str):
				trajectory = np.loadtxt(trajectory)[:,1] #first column is the time, second column is the CV
			else:
				assert isinstance(trajectory, np.ndarray)
			Nis[i] = len(trajectory)
			trajectories[i] = trajectory

		#Preprocess the bins argument and redefine it to represent the bin_edges. We need to do this beforehand to make sure that when calling the numpy.histogram routine with this bins argument, each histogram will have a consistent bin_edges array and hence consistent histogram.
		if verbosity.lower() in ['high']:
			print('  processing bins')
		if isinstance(bins, int):
			raise NotImplementedError
		bin_centers = 0.5*(bins[:-1]+bins[1:])
		deltas = bins[1:]-bins[:-1]
		grid_non_uniformity = deltas.std()/deltas.mean()
		assert grid_non_uniformity<1e-6, 'CV grid defined by bins argument should be of uniform spacing!'
		delta = deltas.mean()
		Ngrid = len(bin_centers)
		timings['init'] = time.time()

		#generate the individual histograms using numpy.histogram
		if verbosity.lower() in ['medium', 'high']:
			print('Constructing individual histograms for each biased simulation ...')
		Hs = wham1d_hs(Nsims, Ngrid, trajectories, bins, Nis)
		timings['hist'] = time.time()

		#compute the integrated boltzmann factors of the biases in each grid interval
		if verbosity.lower() in ['medium', 'high']:
			print('Computing bias on grid ...')
		bs = wham1d_bias(Nsims, Ngrid, beta, biasses, delta, bias_subgrid_num, bin_centers, thresshold=bias_thress)
		timings['bias'] = time.time()

		#some init printing
		if verbosity.lower() in ['high']:
			print()
			print('---------------------------------------------------------------------')
			print('WHAM SETUP')
			print('  number of simulations = ', Nsims)
			for i, Ni in enumerate(Nis):
				print('    simulation %i has %i steps' %(i,Ni))
			print('  CV grid [%s]: start = %.3f    end = %.3f    delta = %.3f    N = %i' %(cv_output_unit, bins.min()/parse_unit(cv_output_unit), bins.max()/parse_unit(cv_output_unit), delta/parse_unit(cv_output_unit), Ngrid))
			print('---------------------------------------------------------------------')
			print()

		if verbosity.lower() in ['medium', 'high']:
			print('Solving WHAM equations (SCF loop) ...')
		ps, fs, converged = wham1d_scf(Nis, Hs, bs, Nscf=Nscf, convergence=convergence, verbose=verbosity.lower() in ['high'])
		if verbosity.lower() in ['low', 'medium', 'high']:
			if bool(converged):
				print('SCF Converged!')
			else:
				print('SCF did not converge!')
		timings['scf'] = time.time()

		error = None
		if error_estimate is not None and error_estimate in ['mle_p', 'mle_p_cov', 'mle_f', 'mle_f_cov']:
			if verbosity.lower() in ['medium', 'high']:
				print('Estimating error ...')
			if corrtimes is None: corrtimes = np.ones(len(traj_input), float)
			error = wham1d_error(Nsims, Ngrid, Nis, ps, fs, bs, corrtimes, method=error_estimate, verbosity=verbosity)
		elif error_estimate is not None and error_estimate not in ["None"]:
			raise ValueError('Received invalid value for keyword argument error_estimate, got %s. See documentation for valid choices.' %error_estimate)
		timings['error'] = time.time()

		if verbosity.lower() in ['low', 'medium', 'high']:
			print('---------------------------------------------------------------------')
			print('TIMING SUMMARY')
			t = timings['init'] - timings['start']
			print('  initializing: %s' %(time.strftime('%Hh %Mm %S.{:03d}s'.format(int((t%1)*1000)), time.gmtime(t))))
			t = timings['hist'] - timings['init']
			print('  histograms  : %s' %(time.strftime('%Hh %Mm %S.{:03d}s'.format(int((t%1)*1000)), time.gmtime(t))))
			t = timings['bias'] - timings['hist']
			print('  bias poten. : %s' %(time.strftime('%Hh %Mm %S.{:03d}s'.format(int((t%1)*1000)), time.gmtime(t))))
			t = timings['scf'] - timings['bias']
			print('  solve scf   : %s' %(time.strftime('%Hh %Mm %S.{:03d}s'.format(int((t%1)*1000)), time.gmtime(t))))
			t = timings['error'] - timings['scf']
			print('  error est.  : %s' %(time.strftime('%Hh %Mm %S.{:03d}s'.format(int((t%1)*1000)), time.gmtime(t))))
			t = timings['error'] - timings['init']
			print('  TOTAL       : %s' %(time.strftime('%Hh %Mm %S.{:03d}s'.format(int((t%1)*1000)), time.gmtime(t))))
			print('---------------------------------------------------------------------')

		return cls(bin_centers.copy(), ps, error=error, cv_output_unit=cv_output_unit, cv_label=cv_label)

	@classmethod
	def from_wham_c(cls, bins, trajectories, biasses, temp, error_estimate=None, bias_subgrid_num=20, Nscf=1000, convergence=1e-6, cv_output_unit='au', cv_label='CV', verbosity='low'):
		'''This routine sole purpose is backward compatibility and serves as an alias for from_wham. 
		
		There used to be a distinction between the from_wham and from_wham_c routine (former was full python implementation, latter used Cython for speed up). This distinction has been removed after deliberate testing confirmed that both routines gave identical results. As a result, only the former from_wham_c routine (which is faster) remains, but it has been renamed to from_wham, while the current from_wham_c routine remains in place for backward compatibility.
		'''
		return cls.from_wham(bins, trajectories, biasses, temp, error_estimate=error_estimate, bias_subgrid_num=bias_subgrid_num, Nscf=Nscf, convergence=convergence, cv_output_unit=cv_output_unit, cv_label=cv_label, verbosity=verbosity)

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
		ps = np.zeros(len(fep.fs))
		beta = 1.0/(boltzmann*temp)
		ps[~np.isnan(fep.fs)] = np.exp(-beta*fep.fs[~np.isnan(fep.fs)] )
		norm = ps.sum()
		ps /= norm
		error = None
		if fep.error is not None:
			means = -beta*fep.error.means-np.log(norm) #last term is to assure normalization consistent with that of ps attribute
			stds = beta*fep.error.stds
			error = LogGaussianDistribution(means, stds)
		return cls(fep.cvs, ps, error=error, cv_output_unit=fep.cv_output_unit, cv_label=fep.cv_label)

	def plot(self, fn=None, obss=['value'], linestyles=None, linewidths=None, colors=None, cvlims=None, plims=None, show_legend=False, **plot_kwargs):
		'''
			Make a plot of the probability histogram.

			:param fn: file name of the resulting plot
			:type fn: str

			:param flim: upper limit of the free energy axis in plots.
			:type flim: float, optional, default=None
		'''
		#preprocess
		cvunit = parse_unit(self.cv_output_unit)
		punit = 1.0/cvunit
		
		#read data 
		data = []
		labels = []
		for obs in obss:
			values = None
			if obs.lower() in ['value']:
				values = self.ps.copy()
			elif obs.lower() in ['error-mean', 'mean']:
				assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
				values = self.error.mean()
			elif obs.lower() in ['error-lower', 'lower']:
				assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
				values = self.error.nsigma_conf_int(2)[0]
			elif obs.lower() in ['error-upper', 'upper']:
				assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
				values = self.error.nsigma_conf_int(2)[1]
			elif obs.lower() in ['error-half-upper-lower', 'width']:
				assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
				lower = self.error.nsigma_conf_int(2)[0]
				upper = self.error.nsigma_conf_int(2)[1]
				values = 0.5*np.abs(upper - lower)
			elif obs.lower() in ['error-sample', 'sample']:
				assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
				values = self.error.sample()
			if values is None: raise ValueError('Could not interpret observable %s' %obs)
			data.append(values)
			labels.append(obs)
		
		if self.error is not None:
			lower, upper = self.error.nsigma_conf_int(2)
			lower, upper = lower, upper
		if linestyles is None:
			linestyles = [None,]*len(data)
		if linewidths is None:
			linewidths = [1.0,]*len(data)
		if colors is None:
			colors = [None,]*len(data)

		#make plot
		pp.clf()
		for (ys,label,linestyle,linewidth,color) in zip(data,labels,linestyles, linewidths,colors):
			pp.plot(self.cvs/cvunit, ys/punit, label=label, linestyle=linestyle, linewidth=linewidth, color=color, **plot_kwargs)
		if self.error is not None:
			pp.fill_between(self.cvs/cvunit, lower/punit, upper/punit, **plot_kwargs, alpha=0.33)
		pp.xlabel('%s [%s]' %(self.cv_label, self.cv_output_unit), fontsize=16)
		pp.ylabel('Probability density [1/%s]' %(self.cv_output_unit), fontsize=16)
		if cvlims is not None: pp.xlim(cvlims)
		if plims is not None: pp.ylim(plims)
		if show_legend:
			pp.legend(loc='best')
		fig = pp.gcf()
		fig.set_size_inches([8,8])
		fig.tight_layout()
		if fn is not None:
			pp.savefig(fn)
		else:
			pp.show()
		return



class Histogram2D(object):
	def __init__(self, cv1s, cv2s, ps, error=None, cv1_output_unit='au', cv2_output_unit='au', cv1_label='CV1', cv2_label='CV2'):
		'''
			Class to implement the estimation of a probability histogram in terms of a collective variable from a trajectory series corresponding to that collective variable.

			:param cv1s: 1D array corresponding the grid points of the first collective variable, which should be in atomic units.
			:type data: np.ndarray

			:param cv2s: 1D array corresponding the grid points of the second collective variable, which should be in atomic units.
			:type data: np.ndarray

			:param ps: 2D array corresponding to the histogram probability values at the (CV1,CV2) grid points, which should be in atomic units.
			:type bins: np.ndarray

			:param error: error distribution on the histogram, defaults to None
            :type error: Distribution child class, optional

			:param cv1_output_unit: the unit in which the first collective variable will be plotted/printed.
			:type cv_output_unit: str, optional, default='au'

			:param cv2_output_unit: the unit in which the second collective variable will be plotted/printed.
			:type cv_output_unit: str, optional, default='au'

			:param cv1_label: the label of the first collective variable that will be used on plots.
			:type cv1_label: str, optional, default='CV1'

			:param cv2_label: the label of the first collective variable that will be used on plots.
			:type cv2_label: str, optional, default='CV2
		'''
		self.cv1s = cv1s.copy()
		self.cv2s = cv2s.copy()
		self.ps = ps.copy()
		self.error = None
		if error is not None:
			self.error = error.copy()
		self.cv1_output_unit = cv1_output_unit
		self.cv2_output_unit = cv2_output_unit
		self.cv1_label = cv1_label
		self.cv2_label = cv2_label

	def plower(self, nsigma=2):
		assert self.error is not None, 'Plower cannot be computed because no error distribution was defined in the error attribute'
		plower, pupper = self.error.nsigma_conf_int(nsigma)
		plower /= plower.sum()
		return plower

	def pupper(self, nsigma=2):
		assert self.error is not None, 'Pupper cannot be computed because no error distribution was defined in the error attribute'
		plower, pupper = self.error.nsigma_conf_int(nsigma)
		pupper /= pupper.sum()
		return pupper

	def copy(self):
		'''
		Make and return a copy of the current instance.

		:return: copy of the current instance
		:rtype: Histogram2D
		'''
		error = None
		if self.error is not None:
			error = self.error.copy()
		return Histogram2D(
			self.cvs1.copy(), self.cvs2.copy(), self.ps.copy(), error=error,
			cv1_output_unit=self.cv1_output_unit, cv2_output_unit=self.cv2_output_unit,
			cv1_label=self.cv1_label, cv2_label=self.cv2_label
		)

	@classmethod
	def from_average(cls, histograms, error_estimate=None, cv1_output_unit=None, cv2_output_unit=None, cv1_label=None, cv2_label=None):
		'''
			Start from a set of histograms and compute and return the averaged histogram. If error_estimate is set to 'std', an error on the histogram will be computed from the standard deviation within the set of histograms.

			:param histograms: set of histrograms to be averaged
			:type histograms: list(Histogram1D)

			:param error_estimate: indicate if and how to perform error analysis. One of following options is available:

				*  **std** -- compute error from the standard deviation within the set of histograms.
				*  **None** -- do not estimate the error.

			:type error_estimate: str, optional, default=None

			:param nsigma: only relevant when error estimation is turned on (i.e. when keyword ``error_estimate`` is not None), this option defines how large the error interval should be in terms of the standard deviation sigma. A ``nsigma=2`` implies a 2-sigma error bar (corresponding to 95% confidence interval) will be returned.
			:type nsigma: int, optional, default=2

			:param cv1_output_unit: the units for printing and plotting of CV1 values. Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_. Defaults to the cv1_output_unit of the first histogram given.
            :type cv1_output_unit: str or float, optional

			:param cv2_output_unit: the units for printing and plotting of CV2 values. Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_. Defaults to the cv2_output_unit of the first histogram given.
            :type cv2_output_unit: str or float, optional

			:param cv1_label: label for the first collective variable in plots. Defaults to the cv1_label of the first given histogram.
            :type cv1_label: str, optional

			:param cv2_label: label for the second collective variable in plots. Defaults to the cv2_label of the first given histogram.
            :type cv2_label: str, optional

			:return: averages histogram
			:rtype: Histrogram1D
		'''
		#sanity checks
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
			if cv1_output_unit is None:
				cv1_output_unit = hist.cv1_output_unit
			if cv2_output_unit is None:
				cv2_output_unit = hist.cv2_output_unit
			if cv1_label is None:
				cv1_label = hist.cv1_label
			if cv2_label is None:
				cv2_label = hist.cv2_label
		#collect probability distributions
		pss = np.array([hist.ps for hist in histograms])
		#average histograms
		ps = pss.mean(axis=0)
		#compute error if requested
		error = None
		if error_estimate is not None and error_estimate.lower() in ['std']:
			perr = pss.std(axis=0,ddof=1)
			error = GaussianDistribution(ps, perr)
		return cls(cv1s, cv2s, ps, error=error, cv1_output_unit=cv1_output_unit, cv2_output_unit=cv2_output_unit, cv1_label=cv1_label, cv2_label=cv2_label)
	
	@classmethod
	def from_single_trajectory(cls, data, bins, error_estimate=None, nsigma=2, cv1_output_unit='au', cv2_output_unit='au', cv1_label='CV1', cv2_label='CV2'):
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

			:param cv1_output_unit: the unit in which the first collective variable will be plotted/printed
			:type cv1_output_unit: str, optional, default='au'

			:param cv2_output_unit: the unit in which the second collective variable will be plotted/printed
			:type cv2_output_unit: str, optional, default='au'

			:param cv1_label: the label of the first collective variable that will be used on plots
			:type cv1_label: str, optional, default='CV1'

			:param cv2_label: the label of the first collective variable that will be used on plots
			:type cv2_label: str, optional, default='CV2'
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
		error = None
		if error_estimate=='mle_p':
			perrs = np.sqrt(ps*(1-ps)/Ntot)
			error = GaussianDistribution(ps, perrs)
		elif error_estimate=='mle_f':
			#we first compute the error bar interval on f=-log(p) using the MLE-F method...
			fs, ferrors = np.zeros(len(ps))*np.nan, np.zeros(len(ps))*np.nan
			fs[ps>0] = -np.log(ps[ps>0])
			ferrors[ps>0] = np.sqrt((np.exp(fs[ps>0])-1)/Ntot)
			#...and then use it to define a Log-Normal distributed error on p as a Normal distributed error on log(p)=-f
			error = LogGaussianDistribution(-fs, ferrors)
		elif error_estimate is not None:
			raise ValueError('Invalid value for error_estimate argument, received %s. Check documentation for allowed values.' %error_estimate)
		return cls(cv1s, cv2s, ps, error=error, cv1_output_unit=cv1_output_unit, cv2_output_unit=cv2_output_unit, cv1_label=cv1_label, cv2_label=cv2_label)
	
	@classmethod
	def from_wham(cls, bins, traj_input, biasses, temp, pinit=None, error_estimate=None, error_p_threshold=0.0, corrtimes=None, bias_subgrid_num=20, Nscf=1000, convergence=1e-6, bias_thress=1e-3, overflow_threshold=1e-150, cv1_output_unit='au', cv2_output_unit='au', cv1_label='CV1', cv2_label='CV2', plot_biases=False, verbose=None, verbosity='low'):
		'''
			Routine that implements the Weighted Histogram Analysis Method (WHAM) for reconstructing the overall 2D probability histogram from a series of biased molecular simulations in terms of two collective variables CV1 and CV2.
			
			:param data: 2D array corresponding to the trajectory series of the two collective variables. The first column is assumed to correspond to the first collective variable, the second column to the second CV.
			:type data: np.ndarray([N,2])

			:param bins: list of the form [bins1, bins2] where bins1 and bins2 are numpy arrays each representing the bin edges of their corresponding CV for which a histogram will be constructed. For example the following definition: 	[np.arange(-1,1.05,0.05), np.arange(0,5.1,0.1)]
			will result in a 2D histogram with bins of width 0.05 between -1 and 1 for CV1 and bins of width 0.1 between 0 and 5 for CV2.
			:type bins: np.ndarray

			:param trajectories: list or array of 2D numpy arrays containing the [CV1,CV2] trajectory data for each simulation. Alternatively, a list of PLUMED file names containing the trajectory data can be specified as well. The arguments trajectories and biasses should be of the same length.
			:type trajectories: list(np.ndarray([Nmd,2]))/np.ndarray([Ntraj,Nmd,2])

			:param biasses: list of callables, each representing a function to compute the bias at a given value of the collective variables CV1 and CV2. The arguments trajectories and biasses should be of the same length.
			:type biasses: list(callable)

			:param temp: the temperature at which all simulations were performed
			:type temp: float

			:param pinit: initial guess for the probability density, which is assumed to be in the 'xy'-indexing convention (see the "indexing" argument and the corresponding "Notes" section in `the numpy online documentation of the meshgrid routine <https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html>`_). If None is given, a uniform distribution is used as initial guess.
			:type pinit: np.ndarray, optional, default=None

			:param error_estimate: indicate if and how to perform error analysis. One of following options is available:

				* `mle_p` -- compute error through the asymptotic normality of the maximum likelihood estimator for the probability itself. WARNING: due to positivity constraint of the probability, this only works for high probability and low variance. Otherwise the standard error interval for the normal distribution (i.e. mle +- n*sigma) might give rise to negative lower error bars.
				
				* `mle_f` -- compute error through the asymptotic normality of the maximum likelihood estimator for -log(p) (hence for the beta-scaled free energy). This estimation does not suffer from the same WARNING as for ``mle_p``. Furthermore, in case of high probability and low variance, the error estimation using ``mle_f`` and ``mle_p`` are consistent (i.e. one can be computed from the other using f=-log(p)).
				
				* `None` -- do not estimate the error.

			:type error_estimate: str, optional, default=None

			:param bias_subgrid_num: the number of grid points along each CV for the sub-grid used to compute the integrated boltzmann factor of the bias in each CV1,CV2 bin. Either a single integer is given, corresponding to identical number of subgrid points for both CVs, or a list of two integers corresponding the number of grid points in the two CVs respectively.
			:type bias_subgrid_num: optional, defaults to [20,20]

			:param Nscf: the maximum number of steps in the self-consistent loop to solve the WHAM equations
			:type Nscf: int, defaults to 1000

			:param convergence: convergence criterium for the WHAM self consistent solver. The SCF loop will stop whenever the integrated absolute difference between consecutive probability densities is less then the specified value.
			:type convergence: float, defaults to 1e-6

			:param verbose: set to True to turn on more verbosity during the self consistent solution cycles of the WHAM equations, defaults to False.
			:type verbose: bool, optional

			:param cv1_output_unit: the unit in which the first collective variable will be plotted/printed
			:type cv1_output_unit: str, optional, default='au'

			:param cv2_output_unit: the unit in which the second collective variable will be plotted/printed
			:type cv2_output_unit: str, optional, default='au'

			:param cv1_label: the label of the first collective variable that will be used on plots
			:type cv1_label: str, optional, default='CV1'

			:param cv2_label: the label of the first collective variable that will be used on plots
			:type cv2_label: str, optional, default='CV2'

			:param plot_biases: if set to True, a 2D plot of the boltzmann factor of the bias integrated over each bin will be made.
			:type plot_biases: bool, optional, default=False

			:param bias_thress: threshold for determining whether the exact bin integration of the bias boltzmann factors is required, or the boltzmann factor of the bin center suffices as an approximation. Setting the threshold to 0 implies exact integration for all bins, setting the threshold to 1 implies the bin center approximation for all cells and setting the threshold to x implies that only those bins with a center approximation higher than x will be integrated exactly.
			:type bias_thress: double, optional, default=1e-3

			:param overflow_threshold: threshold used in the scf procedure to avoid overflow errors, this determines which simulations and which grid points to ignore. Decreasing it results in a FES with a larger maximum free energy (lower unbiased probability). If it is too low, imaginary errors (sigma^2) arise, so increase if necessary.
			:type overflow_threshold: double, optional, default=1e-150

			:return: 2D probability histogram
			:rtype: Histogram2D
		'''
		timings = {}
		timings['start'] = time.time()
		#backward compatibility between verbose and verbosity.
		if verbose is not None:
			print('The keyword verbose is depricated and will be removed in the near future. Use the keyword verbosity instead.')
			if verbose:
				verbosity = 'high'
			else:
				verbosity = 'none'
		if verbosity.lower() in ['medium', 'high']:
			print('Initialization ...')

		#checks and initialization
		assert len(biasses)==len(traj_input), 'The arguments trajectories and biasses should be of the same length.'
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
		if verbosity.lower() in ['high']:
			print('  processing trajectories')
		Nis = np.zeros(len(traj_input), dtype=int)
		trajectories = np.empty(len(traj_input), dtype=object)
		for i, trajectory in enumerate(traj_input):
			if isinstance(trajectory, str):
				trajectory = np.loadtxt(trajectory)[:,1:3] #first column is the time, second column is the CV1, third column is the CV2
			else:
				assert isinstance(trajectory, np.ndarray)
				assert trajectory.shape[1]==2, 'Input trajectories parsed as numpy.ndarray should be have a second dimension of length 2, got %i' %(trajectory.shape[1])
			Nis[i] = len(trajectory)
			trajectories[i] = trajectory

		#Preprocess the bins argument and redefine it to represent the bin_edges. We need to do this beforehand to make sure that when calling the numpy.histogram routine with this bins argument, each histogram will have a consistent bin_edges array and hence consistent histogram.
		if verbosity.lower() in ['high']:
			print('  processing bins')
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
		timings['init'] = time.time()

		#generate the individual histograms using numpy.histogram
		if verbosity.lower() in ['medium', 'high']:
			print('Constructing individual histograms for each biased simulation ...')
		Hs = wham2d_hs(Nsims, Ngrid1, Ngrid2, trajectories, bins[0], bins[1], Nis)
		timings['hist'] = time.time()

		#compute the boltzmann factors of the biases in each grid interval
		if verbosity.lower() in ['medium', 'high']:
			print('Computing bias on grid ...')
		bs = wham2d_bias(Nsims, Ngrid1, Ngrid2, beta, biasses, delta1, delta2, bias_subgrid_num[0], bias_subgrid_num[1], bin_centers1, bin_centers2, thresshold=bias_thress)
		if plot_biases:
			for i, bias in enumerate(biasses):
				bias.plot('bias_%i.png' %i, bin_centers1, bin_centers2)
		timings['bias'] = time.time()

		#some init printing
		if verbosity.lower() in ['high']:
			print()
			print('---------------------------------------------------------------------')
			print('WHAM SETUP')
			print('  number of simulations = ', Nsims)
			for i, Ni in enumerate(Nis):
				print('    simulation %i has %i steps' %(i,Ni))
			print('  CV1 grid [%s]: start = %.3f    end = %.3f    delta = %.3f    N = %i' %(cv1_output_unit, bins[0].min()/parse_unit(cv1_output_unit), bins[0].max()/parse_unit(cv1_output_unit), delta1/parse_unit(cv1_output_unit), Ngrid1))
			print('  CV2 grid [%s]: start = %.3f    end = %.3f    delta = %.3f    N = %i' %(cv2_output_unit, bins[1].min()/parse_unit(cv2_output_unit), bins[1].max()/parse_unit(cv2_output_unit), delta2/parse_unit(cv2_output_unit), Ngrid2))
			print('---------------------------------------------------------------------')
			print()

		if verbosity.lower() in ['medium', 'high']:
			print('Solving WHAM equations (SCF loop) ...')

		#self consistent loop to solve the WHAM equations
		if pinit is None:
			pinit = np.ones([Ngrid1,Ngrid2])/Ngrid
		else:
			#it is assumed pinit is given in 'xy'-indexing convention (such as for example when using the ps attribute of another Histogram2D instance)
			assert pinit.shape[0]==Ngrid2, 'Specified initial guess should be of shape (%i,%i), got (%i,%i)' %(Ngrid2,Ngrid1,pinit.shape[0],pinit.shape[1])
			assert pinit.shape[1]==Ngrid1, 'Specified initial guess should be of shape (%i,%i), got (%i,%i)' %(Ngrid2,Ngrid1,pinit.shape[0],pinit.shape[1])
			#however, this routine is written in the 'ij'-indexing convention, therefore, we transpose pinit here.
			pinit = pinit.T
			pinit /= pinit.sum()
		ps, fs, converged = wham2d_scf(Nis, Hs, bs, pinit, Nscf=Nscf, convergence=convergence, verbose=verbosity.lower() in ['high'], overflow_threshold=overflow_threshold)
		if verbosity.lower() in ['low', 'medium', 'high']:
			if bool(converged):
				print('  SCF Converged!')
			else:
				print('  SCF did not converge!')
		timings['scf'] = time.time()

		#error estimation
		error = None
		if error_estimate is not None:
			if verbosity.lower() in ['medium', 'high']:
				print('Estimating error ...')
			if corrtimes is None: corrtimes = np.ones(len(traj_input), float)
			error = wham2d_error(ps, fs, bs, Nis, corrtimes, method=error_estimate, p_threshold=error_p_threshold, verbosity=verbosity)

		timings['error'] = time.time()

		if verbosity.lower() in ['low', 'medium', 'high']:
			print()
			print('---------------------------------------------------------------------')
			print('TIMING SUMMARY')
			t = timings['init'] - timings['start']
			print('  initializing: %s' %(time.strftime('%Hh %Mm %S.{:03d}s'.format(int((t%1)*1000)), time.gmtime(t))))
			t = timings['hist'] - timings['init']
			print('  histograms  : %s' %(time.strftime('%Hh %Mm %S.{:03d}s'.format(int((t%1)*1000)), time.gmtime(t))))
			t = timings['bias'] - timings['hist']
			print('  bias poten. : %s' %(time.strftime('%Hh %Mm %S.{:03d}s'.format(int((t%1)*1000)), time.gmtime(t))))
			t = timings['scf'] - timings['bias']
			print('  solve scf   : %s' %(time.strftime('%Hh %Mm %S.{:03d}s'.format(int((t%1)*1000)), time.gmtime(t))))
			t = timings['error'] - timings['scf']
			print('  error est.  : %s' %(time.strftime('%Hh %Mm %S.{:03d}s'.format(int((t%1)*1000)), time.gmtime(t))))
			t = timings['error'] - timings['init']
			print('  TOTAL       : %s' %(time.strftime('%Hh %Mm %S.{:03d}s'.format(int((t%1)*1000)), time.gmtime(t))))
			print('---------------------------------------------------------------------')

		return cls(bin_centers1.copy(), bin_centers2.copy(), ps, error=error, cv1_output_unit=cv1_output_unit, cv2_output_unit=cv2_output_unit, cv1_label=cv1_label, cv2_label=cv2_label)

	@classmethod
	def from_wham_c(cls, bins, trajectories, biasses, temp, pinit=None, error_estimate=None, bias_subgrid_num=20, Nscf=1000, convergence=1e-6, cv1_output_unit='au', cv2_output_unit='au', cv1_label='CV1', cv2_label='CV2', plot_biases=False, verbose=None, verbosity='low'):
		'''This routine sole purpose is backward compatibility and serves as an alias for from_wham. 
		
		There used to be a distinction between the from_wham and from_wham_c routine (former was full python implementation, latter used Cython for speed up). This distinction has been removed after deliberate testing confirmed that both routines gave identical results. As a result, only the former from_wham_c routine (which is faster) remains, but it has been renamed to from_wham, while the current from_wham_c routine remains in place for backward compatibility.
		'''
		return cls.from_wham(bins, trajectories, biasses, temp, pinit=pinit, error_estimate=error_estimate, bias_subgrid_num=bias_subgrid_num, Nscf=Nscf, convergence=convergence, cv1_output_unit=cv1_output_unit, cv2_output_unit=cv2_output_unit, cv1_label=cv1_label, cv2_label=cv2_label, plot_biases=plot_biases, verbose=verbose, verbosity=verbosity)

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
		ps = np.zeros(len(fes.fs))
		beta = 1.0/(boltzmann*temp)
		ps = np.exp(-beta*fes.fs)
		ps /= ps[~np.isnan(ps)].sum()
		error = None
		if fes.error is not None:
			raise NotImplementedError #TODO
		return cls(fes.cv1s, fes.cv2s, ps, error=error, cv1_output_unit=fes.cv1_output_unit, cv2_output_unit=fes.cv2_output_unit, cv1_label=fes.cv1_label, cv2_label=fes.cv2_label)
	
	def average_cv_constraint_other(self, index, target_distribution=MultiGaussianDistribution, verbose=False):
		'''S
			Routine to compute the average of one CV (denoted as y/Y below, y for integration values and Y for resulting averaged values) 
		   	as function of the other CV (denoted as x below), i.e. the other CV Y is contraint to one of its bin values) using the following formula:

			Y(x) = int(y*p(x,y),y)/int(p(x,y),dy)

			:param index: the index of the CV which will be averaged (the other is then contraint). If index=1, then y=CV1 and x=CV2, while if index=2, then y=CV2 and x=CV1.
			:type index: int (1 or 2)

			:param error_estimate: Specify the method of error propagation, either by propagating the FES distribution samples (propdist) or by propagating the FES 2 sigma confidence interval (prop2sigma), defaults to propdist
			:type error_estimate: str, optional

			:return: xs, ys (and yerrs) with xs the constraint CV values, ys the averaged CV values and yerrs the error bar on the averaged values assuming do_err was set to True
			:rtype: xs, ys | xs, ys, yerrs all np.ndarrays
		'''
		#set x,y and p arrays according to chosen index
		if index==1:
			ys = self.cv1s
			xs = self.cv2s
			cv_output_unit = self.cv2_output_unit
			f_output_unit = self.cv1_output_unit
			cv_label = self.cv2_label
			f_label = self.cv1_label
			def function(ps_inp):
				ps = ps_inp/ps_inp[~np.isnan(ps_inp)].sum()
				Ys = np.zeros(len(xs))
				for j in range(len(xs)):
					mask = ~np.isnan(ps[:,j])
					T = (ys[mask]*ps[mask,j]).sum()
					N = (ps[mask,j]).sum()
					try:
						Ys[j] = T/N
					except (FloatingPointError, ZeroDivisionError):
						Ys[j] = np.nan
				return Ys
		elif index==2:
			ys = self.cv2s
			xs = self.cv1s
			cv_output_unit = self.cv1_output_unit
			f_output_unit = self.cv2_output_unit
			cv_label = self.cv1_label
			f_label = self.cv2_label
			def function(ps_inp):
				ps = ps_inp/ps_inp[~np.isnan(ps_inp)].sum()
				Ys = np.zeros(len(xs))
				for j in range(len(xs)):
					mask = ~np.isnan(ps[j,:])
					T = (ys[mask]*ps[j,mask]).sum()
					N = (ps[j,mask]).sum()
					try:
						Ys[j] = T/N
					except (FloatingPointError,ZeroDivisionError):
						Ys[j] = np.nan
				return Ys
		else:
			raise ValueError('Index should be 1 or 2 for 2D Histogram')
		#compute average Y (and possibly its error) on grid of x
		if self.error is not None:
			if verbose: print('Error propagation is done numerically')
			propagator = Propagator(verbose=verbose)
			error = propagator(function, self.error, target_distribution=target_distribution, samples_are_flattened=True)
			Ys = error.mean()
		else:
			if verbose: print('No error detected in current histogram, so no error propagation possible!')
			Ys = function(self.ps)
			error = None
		return BaseProfile(xs, Ys, error=error, cv_output_unit=cv_output_unit, f_output_unit=f_output_unit, cv_label=cv_label, f_label=f_label)
	
	def plot(self, fn=None, slicer=[slice(None),slice(None)], obss=['value'], linestyles=None, linewidths=None, colors=None, cv1lims=None, cv2lims=None, plims=None, ncolors=8, ref_max=False, **plot_kwargs):
		'''
            Plot x[slicer], where (possibly multiple) x is/Are specified in the keyword argument obss and the argument slicer defines the subset of data that will be plotted. The resulting graph will be a regular 1D plot or 2D contourplot, depending on the dimensionality of the data x[slicer].

            :param fn: name of the file to store graph in, defaults to 'condprob.png'
            :type fn: str, optional

            :param cmap: color map to be used, only relevant in case of 2D contourplot, defaults to pp.get_cmap('rainbow')
            :type cmap: color map from matplotlib, optional
        '''
        #preprocess
		assert isinstance(slicer, list) or isinstance(slicer, np.ndarray), 'Slicer should be list or array, instead got %s' %(slicer.__class__.__name__)
		assert len(slicer)==2, 'Slicer should be list of length 2, instead got list of length %i' %( len(slicer))

		if isinstance(slicer[0], slice) and isinstance(slicer[1], slice):
			ndim = 2
			xs = self.cv1s[slicer[0]]/parse_unit(self.cv1_output_unit)
			xlabel = self.cv1_label
			xlims = cv1lims
			ys = self.cv2s[slicer[1]]/parse_unit(self.cv2_output_unit)
			ylabel = '%s [%s]' %(self.cv2_label, self.cv2_output_unit)
			ylims = cv2lims
			title = 'P(:,:)'
		elif isinstance(slicer[0], slice) or isinstance(slicer[1],slice):
			ndim = 1
			if isinstance(slicer[0], slice):
				xs = self.cv1s[slicer[0]]/parse_unit(self.cv1_output_unit)
				xlabel = '%s [%s]' %(self.cv1_label, self.cv1_output_unit)
				xlims = cv1_lims
				title = 'F(:,%s[%i]=%.3f %s)' %(self.cv2_label,slicer[1],self.cv2s[slicer[1]]/parse_unit(self.cv2_output_unit),self.cv2_output_unit)
			else:
				xs = self.cv2s[slicer[1]]/parse_unit(self.cv2_output_unit)
				xlabel = '%s [%s]' %(self.cv2_label, self.cv2_output_unit)
				xlims = cv2_lims
				title = 'P(%s[%i]=%.3f %s,:)' %(self.cv1_label,slicer[0],self.cv1s[slicer[0]]/parse_unit(self.cv1_output_unit),self.cv1_output_unit)
		else:
			raise ValueError('At least one of the two elements in slicer should be a slice instance!')
		punit = 1.0/(parse_unit(self.cv1_output_unit)*parse_unit(self.cv2_output_unit))

		#read data 
		data = []
		labels = []
		for obs in obss:
			values = None
			if obs.lower() in ['value']:
				values = self.ps.copy()
			elif obs.lower() in ['error-mean', 'mean']:
				assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
				values = self.error.mean()
			elif obs.lower() in ['error-lower', 'lower']:
				assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
				values = self.error.nsigma_conf_int(2)[0]
			elif obs.lower() in ['error-upper', 'upper']:
				assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
				values = self.error.nsigma_conf_int(2)[1]
			elif obs.lower() in ['error-half-upper-lower', 'width']:
				assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
				lower = self.error.nsigma_conf_int(2)[0]
				upper = self.error.nsigma_conf_int(2)[1]
				values = 0.5*np.abs(upper - lower)
			elif obs.lower() in ['error-sample', 'sample']:
				assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
				values = self.error.sample()
			if values is None: raise ValueError('Could not interpret observable %s' %obs)
			data.append(values)
			labels.append(obs)

			if ref_max:
				values /= max(values)
			data.append(values)
			labels.append(obs)

		if ndim==1 and self.error is not None:
			lower, upper = self.error.nsigma_conf_int(2)
			lower, upper = lower[slicer], upper[slicer]
		if linestyles is None:
			linestyles = [None,]*len(data)
		if linewidths is None:
			linewidths = [1.0,]*len(data)
		if colors is None:
			colors = [None,]*len(data)

		#make plot
		pp.clf()
		if ndim==1:
			for (zs,label,linestyle,linewidth,color) in zip(data,labels,linestyles, linewidths,colors):
				pp.plot(xs, zs/punit, label=label, linestyle=linestyle, linewidth=linewidth, color=color, **plot_kwargs)
			if self.error is not None:
				pp.fill_between(xs, lower/punit, upper/punit, **plot_kwargs, alpha=0.33)
			pp.xlabel(xlabel, fontsize=16)
			pp.ylabel('Energy [%s]' %(self.f_output_unit), fontsize=16)
			pp.title('Derived profiles from %s' %title, fontsize=16)
			if xlims is not None: pp.xlim(xlims)
			if plims is not None: pp.ylim(plims)
			pp.legend(loc='best')
			fig = pp.gcf()
			fig.set_size_inches([8,8])
		elif ndim==2:
			if len(data)<4:
				fig, axs = pp.subplots(1,len(data))
				size = [8*len(data),8]
			elif len(data)==4:
				fig, axs = pp.subplots(2,2)
				size = [16,16]
			else:
				nrows = int(np.ceil(len(data)/3))
				fig, axs = pp.subplots(nrows,3)
				size = [24,8*nrows]
			for i, (multi_index, ax) in enumerate(np.ndenumerate(axs)):
				if flims is not None:
					delta = (flims[1]-flims[0])/ncolors
					levels = np.arange(flims[0], flims[1]+delta, delta)
					contourf = ax.contourf(xs, ys, data[i].T/punit, levels=levels, **plot_kwargs) #transpose data to convert ij indexing (internal) to xy indexing (for plotting)
					contour = ax.contour(xs, ys, data[i].T/punit, levels=levels, colors='gray') #transpose data to convert ij indexing (internal) to xy indexing (for plotting)
				else:
					contourf = ax.contourf(xs, ys, data[i].T/punit, **plot_kwargs) #transpose data to convert ij indexing (internal) to xy indexing (for plotting)
					contour = ax.contour(xs, ys, data[i].T/punit, colors='gray') #transpose data to convert ij indexing (internal) to xy indexing (for plotting)
				ax.set_xlabel(xlabel, fontsize=16)
				ax.set_ylabel(ylabel, fontsize=16)
				ax.set_title('%s of %s' %(labels[i],title), fontsize=16)
				if xlims is not None: ax.set_xlim(xlims)
				if ylims is not None: ax.set_ylim(ylims)
				cbar = pp.colorbar(contourf, ax=ax, extend='both')
				cbar.set_label('Free energy [%s]' %self.f_output_unit, fontsize=16)
				pp.clabel(contour, inline=1, fontsize=10)
			fig.set_size_inches(size)
		else:
			raise ValueError('Can only plot 1D or 2D pcond data, but received %i-d data. Make sure that the combination of qslice and cvslice results in 1 or 2 dimensional data.' %(len(data.shape)))

		fig.tight_layout()
		if fn is not None:
			pp.savefig(fn)
		else:
			pp.show()
		return



def plot_histograms(histograms, fn=None, temp=None, labels=None, flims=None, colors=None, linestyles=None, linewidths=None):
	'''
		Make a plot to compare multiple probability histograms and possible the corresponding free energy (if the argument ``temp`` is specified).

		:param fn: file name to write the figure to, the extension determines the format (PNG or PDF).
		:type fn: str

		:param histograms: list of histrograms to plot
		:type histograms: list(Histogram1D)

		:param temp: if defined, the free energy profile corresponding to each histogram will be computed and plotted with its corresponding temperature. If a single float is given, all histograms are assumed to be at the same temperature, if a list or array of floats is given, each entry is assumed to be the temperature of the corresponding entry in the input list of histograms.
		:type temp: float/list(float)/np.ndarray, optional, default=None

		:param labels: list of labels for the legend, one for each histogram.
		:type labels: list(str), optional, default=None

		:param flims: [lower,upper] limit of the free energy axis in plots.
		:type flims: float, optional, default=None

		:param colors: List of matplotlib color definitions for each entry in histograms. If an entry is None, a color will be chosen internally. Defaults to None, implying all colors are chosen internally.
		:type colors: List(str), optional

		:param linestyles: List of matplotlib line style definitions for each entry in histograms. If an entry is None, the default line style of '-' will be chosen . Defaults to None, implying all line styles are set to the default of '-'.
		:type linestyles: List(str), optional

		:param linewidths: List of matplotlib line width definitions for each entry in histograms. If an entry is None, the default line width of 1 will be chosen. Defaults to None, implying all line widths are set to the default of 2.
		:type linewidths: List(str), optional
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
	cv_unit = histograms[0].cv_output_unit
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
		if hist.error is not None:
			axs[0,0].fill_between(hist.cvs/parse_unit(cv_unit), hist.plower()/max(hist.ps), hist.pupper()/max(hist.ps), color=color, alpha=0.33)
		#free energy if requested
		if temp is not None:
			fep = BaseFreeEnergyProfile.from_histogram(hist, temp)
			fmax = max(fmax, np.ceil(max(fep.fs[~np.isnan(fep.fs)]/kjmol)/10)*10)
			funit = fep.f_output_unit
			axs[0,1].plot(fep.cvs/parse_unit(cv_unit), fep.fs/parse_unit(funit), linewidth=linewidth, color=color, linestyle=linestyle, label=label)
			if fep.error is not None:
				axs[0,1].fill_between(fep.cvs/parse_unit(cv_unit), fep.flower()/parse_unit(funit), fep.fupper()/parse_unit(funit), color=color, alpha=0.33)
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
	if fn is not None:
		pp.savefig(fn)
	else:
		pp.show()
