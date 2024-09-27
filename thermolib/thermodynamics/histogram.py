#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 - 2024 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
# Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium;
# all rights reserved unless otherwise stated.
#
# This file is part of a library developed by Louis Vanduyfhuys at
# the Center for Molecular Modeling under supervision of prof. Veronique
# Van Speybroeck. Usage of this package should be authorized by prof. Van
# Vanduyfhuys or prof. Van Speybroeck.


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
	'''
		Class representing a 1D probability histogram :math:`p(CV)` in terms of a single collective variable :math:`CV`.
	'''
	def __init__(self, cvs, ps, error=None, cv_output_unit='au', cv_label='CV'):
		'''
			:param cvs: the bin values of the collective variable CV, which should be in atomic units! If you need help properly converting to atomic units, we refer to the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
			:type cvs: np.ndarray

			:param ps: the histogram probability values at the given CV values. The probabilities should be in atomic units! If you need help properly converting to atomic units, we refer to the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
			:type ps: np.ndarray

			:param error: error distribution on the free energy profile
			:type error: child of :py:class:`Distribution <thermolib.error.Distribution>`, optional, default=None
			
			:param cv_output_unit: the units for printing and plotting of CV values (not the unit of the input array, that is assumed to be in atomic units). Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
			:type cv_output_unit: str, optional, default='au'
			
			:param cv_label: label for the CV for printing and plotting
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
		'''
            Return the lower limit of an n-sigma error bar on the histogram probability, i.e. :math:`\\mu - n\\sigma` with :math:`\\mu` the mean and :math:`\\sigma` the standard deviation.

            :param nsigma: defines the n-sigma error bar
            :type nsigma: int, optional, default=2

            :return: the lower limit of the n-sigma error bar
            :rtype: np.ndarray with dimensions determined by self.error

            :raises AssertionError: if self.error is not defined.
        '''
		assert self.error is not None, 'Plower cannot be computed because no error distribution was defined in the error attribute'
		plower, pupper = self.error.nsigma_conf_int(nsigma)
		return plower

	def pupper(self, nsigma=2):
		'''
            Return the upper limit of an n-sigma error bar on the histogram probability, i.e. :math:`\\mu + n\\sigma` with :math:`\\mu` the mean and :math:`\\sigma` the standard deviation.

            :param nsigma: defines the n-sigma error bar
            :type nsigma: int, optional, default=2

            :return: the upper limit of the n-sigma error bar
            :rtype: np.ndarray with dimensions determined by self.error

            :raises AssertionError: if self.error is not defined.
        '''
		assert self.error is not None, 'Pupper cannot be computed because no error distribution was defined in the error attribute'
		plower, pupper = self.error.nsigma_conf_int(nsigma)
		return pupper

	def copy(self):
		'''
            Make and return a copy of the current Histogram1D instance.
        '''
		if self.error is not None:
			error = self.error.copy()
		else:
			error = None
		return Histogram1D(self.cvs.copy(), self.ps.copy(), error=error, cv_output_unit=self.cv_output_unit, cv_label=self.cv_label)

	@classmethod
	def from_average(cls, histograms, error_estimate=None, cv_output_unit=None, cv_label=None):
		'''
			Start from a set of 1D histograms and compute and return the averaged histogram (and optionally the error bar).

			:param histograms: set of histrograms to be averaged
			:type histograms: list of :py:class:`Histogram1D <thermolib.thermodynamics.histogram.Histogram1D>` instances

			:param error_estimate: indicate how to perform error analysis. Currently, only one method is supported, 'std', which will compute the error bar from the standard deviation within the set of histograms.
			:type error_estimate: str, optional, default=None

			:param cv_output_unit: the unit in which cv will be plotted/printed. Defaults to the cv_output_unit of the first histogram given.
			:type cv_output_unit: str, optional

			:param cv_label: label for the new CV. Defaults to the cv_label of the first given histogram.
			:type cv_label: str, optional

			:raises AssertionError: if the histograms do not have a consistent CV grid
			:raises AssertionError: if the histograms do not have a consistent CV label
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
			Routine to estimate a 1D probability histogram in terms of a single collective variable from a series of samples of that collective variable.

			:param data: series of samples of the collective variable. Should be in atomic units! If you need help properly converting to atomic units, we refer to the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
			:type data: np.ndarray

			:param bins: the edges of the bins for which a histogram will be constructed. This argument is parsed to `the numpy.histogram routine <https://numpy.org/doc/stable/reference/generated/numpy.histogram.html>`_. Hence, more information on its meaning and allowed values can be found there.
			:type bins: np.ndarray

			:param error_estimate: indicate if and how to perform error analysis. One of following options is available:

				- **mle_p** - Estimating the error directly for the probability of each bin in the histogram. This method does not explicitly impose the positivity of the probability.

				- **mle_p_cov** - Estimate the full covariance matrix for the probability of all bins in the histogram. In other words, appart from the error on the probability/free energy of a bin itself, we now also account for the covariance between the probabilty/free energy of the bins. This method does not explicitly impose the positivity of the probability.

				- **mle_f** - Estimating the error for minus the logarithm of the probability, which is proportional to the free energy (hence f in mle_f). As the probability is expressed as :math:`\propto e^{-f}`, its positivity is explicitly accounted for.

				- **mle_f_cov** - Estimate the full covariance matrix for minus the logarithm of the probability of all bins in the histogram. In other words, appart from the error on the probabilty/free energy of a bin itself (including explicit positivity constraint), we now also account for the covariance between the probability/free energy of the bins.

			:type error_estimate: str or None, optional, default=None

			:param error_p_thresshold: only relevant when error estimation is enabled (see parameter ``error_estimate``). When ``error_p_thresshold`` is set to x, bins in the histogram for which the probability resulting from the trajectory is smaller than x will be disabled for error estimation (i.e. its error will be set to np.nan). This is similar as the error_p_thresshold keyword for the from_wham routine, for which the use is illustrated in :doc:`one of the tutorial notebooks <tut/advanced_projection>`.
			:type error_p_thresshold: float, optional, default=0.0

			:param cv_output_unit: the unit in which cv will be plotted/printed (not the unit of the input array, that is assumed to be atomic units). If you need help properly converting to atomic units, we refer to the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
			:type cv_output_unit: str, optional, default='au'

			:param cv_label: label of the cv that will be used for plotting and printing
			:type cv_label: str, optional, default='CV'

			:raises ValueError: if no valid error_estimate value is given
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
	def from_wham(cls, bins, traj_input, biasses, temp, error_estimate=None, corrtimes=None, error_p_thresshold=0.0, bias_subgrid_num=20, Nscf=1000, convergence=1e-6, bias_thress=1e-3, cv_output_unit='au', cv_label='CV', verbosity='low'):
		'''
			Routine that implements the Weighted Histogram Analysis Method (WHAM) for reconstructing the overall 1D probability histogram in terms of collective variable CV from a series of molecular simulations that are (possibly) biased in terms of CV.

			:param bins: number of bins for the CV grid or array representing the bin edges for th CV grid.
			:type bins: int or np.ndarray(float)

			:param traj_input: list of CV trajectories, one for each simulation. This input can be generated using the :py:meth:`read_wham_input <thermolib.tools.read_wham_input>` routine. The arguments trajectories and biasses should be of the same length.
			:type trajectories: list of np.ndarrays

			:param biasses: list of bias potentials, one for each simulation allowing to compute the bias at a given value of the collective variable in that simulation. This input can be generated using the :py:meth:`read_wham_input <thermolib.tools.read_wham_input>` routine. The arguments traj_input and biasses should be of the same length.
			:type biasses: list(callable)

			:param temp: the temperature at which all simulations were performed
			:type temp: float

			:param error_estimate: indicate if and how to perform error analysis. One of following options is available:

				- **mle_p** - Estimating the error directly for the probability of each bin in the histogram. This method does not explicitly impose the positivity of the probability.

				- **mle_p_cov** - Estimate the full covariance matrix for the probability of all bins in the histogram. In other words, appart from the error on the probability/free energy of a bin itself, we now also account for the covariance between the probabilty/free energy of the bins. This method does not explicitly impose the positivity of the probability.

				- **mle_f** - Estimating the error for minus the logarithm of the probability, which is proportional to the free energy (hence f in mle_f). As the probability is expressed as :math:`\propto e^{-f}`, its positivity is explicitly accounted for.

				- **mle_f_cov** - Estimate the full covariance matrix for minus the logarithm of the probability of all bins in the histogram. In other words, appart from the error on the probabilty/free energy of a bin itself (including explicit positivity constraint), we now also account for the covariance between the probability/free energy of the bins.

			:type error_estimate: str or None, optional, default=None

			:param corrtimes: list of (integrated) correlation times of the CV, one for each simulation. Such correlation times will be taken into account during the error estimation and hence make it more reliable. If set to None, the CV trajectories will be assumed to contain fully uncorrelated samples (which is not true when using trajectories representing each subsequent step from a molecular dynamics simulation). More information can be found in :ref:`the user guide <seclab_ug_errorestimation>`. This input can be generated using the :py:meth:`decorrelate <thermolib.tools.decorrelate>` routine. This argument needs to have the same length as the ``traj_input`` and ``biasses`` arguments.
			:type corrtimes: list or np.ndarray, optional, default=None
			
			:param error_p_thresshold: only relevant when error estimation is enabled (see parameter ``error_estimate``). When ``error_p_thresshold`` is set to x, bins in the histogram for which the probability resulting from the trajectory is smaller than x will be disabled for error estimation (i.e. its error will be set to np.nan). It is mainly usefull in the case of 2D histograms,as illustrated in :doc:`one of the tutorial notebooks <tut/advanced_projection>`.
			:type error_p_thresshold: float, optional, default=0.0

			:param bias_subgrid_num: see documentation for this argument in the :py:meth:`wham1d_bias <thermolib.ext.wham1d_bias>` routine
			:type bias_subgrid_num: int, optional, default=20

			:param Nscf: the maximum number of steps in the self-consistent loop to solve the WHAM equations
			:type Nscf: int, optional, default=1000

			:param convergence: convergence criterium for the WHAM self consistent solver. The SCF loop will stop whenever the integrated absolute difference between consecutive probability densities is less then the specified value.
			:type convergence: float, optional, default=1e-6

			:param bias_thress: see documentation for the thresshold argument in the :py:meth:`wham1d_bias <thermolib.ext.wham1d_bias>` routine
			:type bias_thress:

			:param verbosity: controls the level of verbosity for logging during the WHAM algorithm. 
			:type verbose: str, optional, allowed='none'|'low'|'medium'|'high', default='low'

			:param cv_output_unit: the unit in which cv will be plotted/printed. If you need help properly converting to atomic units, we refer to the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
			:type cv_output_unit: str, optional, default='au'

			:param cv_label: the label of the cv that will be used on plots
			:type cv_label: str, optional, default='CV'

			:raises AssertionError: if traj_input and biasses are not of equal length
			:raises AssertionError: if traj_input has an element that is not a numpy.ndarray
			:raises AssertionError: if the CV grid defined by bins argument does not have a uniform spacing.
			:raises ValueError: if an invalid definition for error_estimate is provided
		'''
		timings = {}
		timings['start'] = time.time()
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
				assert isinstance(trajectory, np.ndarray), 'All trajectories should be numpy.ndarrays'
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
			error = wham1d_error(Nsims, Ngrid, Nis, ps, fs, bs, corrtimes, method=error_estimate, verbosity=verbosity, p_thresshold=error_p_thresshold)
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
		'''
			.. deprecated:: v1.7

				This routine sole purpose is backward compatibility and serves as an alias for from_wham. Please start using the from_wham routine as this routine will be removed in the near future.
			
				There used to be a distinction between the from_wham and from_wham_c routine (former was full python implementation, latter used Cython for speed up). This distinction has been removed after deliberate testing confirmed that both routines gave identical results. As a result, only the former from_wham_c routine (which is faster) remains, but it has been renamed to from_wham, while the current from_wham_c routine remains in place for backward compatibility.
		'''
		return cls.from_wham(bins, trajectories, biasses, temp, error_estimate=error_estimate, bias_subgrid_num=bias_subgrid_num, Nscf=Nscf, convergence=convergence, cv_output_unit=cv_output_unit, cv_label=cv_label, verbosity=verbosity)

	@classmethod
	def from_fep(cls, fep, temp):
		'''
			Compute a probability histogram from the given free energy profile at the given temperature.

			:param fep: free energy profile from which the probability histogram is computed
			:type fep: fep.BaseFreeEnergyProfile/fep.SimpleFreeEnergyProfile

			:param temp: the temperature at which the histogram input data was simulated
			:type temp: float
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
			Plot the probability histogram as function of the CV. If the error distribution is stored in self.error, various statistical quantities besides the estimated mean probabilty ietsel (such as the error width, lower/upper limit on the error bar, random sample) can be plotted using the ``obss`` keyword. You can specify additional matplotlib keyword arguments that will be parsed to the matplotlib plotter (`plot` and/or `fill_between`) at the end of the argument list of this routine.

            :param fn: name of a file to save plot to. If None, the plot will not be saved to a file.
            :type fn: str, optional, default=None

            :param obss: Specify which statistical property/properties to plot. Multiple values are allowed, which will be plotted on the same figure. Following options are supported:

                - **value** - the values stored in self.ps
                - **mean** - the mean according to the error distribution, i.e. self.error.mean()
                - **lower** - the lower limit of the 2-sigma error bar (which corresponds to a 95% confidence interval in case of a normal distribution), i.e. self.error.nsigma_conf_int(2)[0]
                - **upper** - the upper limit of the 2-sigma error bar (which corresponds to a 95% confidence interval in case of a normal distribution), i.e. self.error.nsigma_conf_int(2)[1]
                - **error** - half the width of the 2-sigma error bar (which corresponds to a 95% confidence interval in case of a normal distribution), i.e. abs(upper-lower)/2
                - **sample** - a random sample taken from the error distribution, i.e. self.error.sample()
            :type obss: list, optional, default=['value']

            :param linestyles: Specify the line style (using matplotlib definitions) for each quantity requested in ``obss``. If None, all linestyles are set to '-'
            :type linestyles: list or None, optional, default=None

            :param linewidths: Specify the line width (using matplotlib definitions) for each quantity requested in ``obss``. If None, al linewidths are set to 1
            :type linewidths: list of strings or None, optional, default=None

            :param colors: Specify the color (using matplotlib definitions) for each quantity requested in ``obss``. If None, matplotlib will choose.
            :type colors: list of strings or None, optional, default=None

            :param cvlims: limits to the plotting range of the cv. If None, no limits are enforced
            :type cvlims: list of strings or None, optional, default=None

            :param plims: limits to the plotting range of the probability. If None, no limits are enforced
            :type plims: list of strings or None, optional, default=None

            :param show_legend: If True, the legend is shown in the plot
            :type show_legend: bool, optional, default=None
            
            :raises ValueError: if the ``obs`` argument contains an invalid specification
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
	'''
		Class representing a 2D probability histogram :math:`p(CV1,CV2)` in terms of two collective variable :math:`CV1` and :math:`CV2`.
	'''
	def __init__(self, cv1s, cv2s, ps, error=None, cv1_output_unit='au', cv2_output_unit='au', cv1_label='CV1', cv2_label='CV2'):
		'''
			:param cv1s: the bin values of the first collective variable CV1, which should be in atomic units! If you need help properly converting to atomic units, we refer to the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
			:type data: np.ndarray

			:param cv2s: the bin values of the second collective variable CV2, which should be in atomic units! If you need help properly converting to atomic units, we refer to the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
			:type data: np.ndarray

			:param ps: 2D array corresponding to  the histogram probability values at the given CV1,CV2 grid. The probabilities should be in atomic units! If you need help properly converting to atomic units, we refer to the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
			:type bins: np.ndarray

			:param error: error distribution on the free energy profile
			:type error: child of :py:class:`Distribution <thermolib.error.Distribution>`, optional, default=None

			:param cv1_output_unit: the units for printing and plotting of CV1 values (not the unit of the input array, that is assumed to be in atomic units). Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
			:type cv1_output_unit: str, optional, default='au'

			:param cv2_output_unit: the units for printing and plotting of CV2 values (not the unit of the input array, that is assumed to be in atomic units). Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
			:type cv2_output_unit: str, optional, default='au'

			:param cv1_label: label for CV1 for printing and plotting
			:type cv1_label: str, optional, default='CV1'

			:param cv2_label: label for CV2 for printing and plotting
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
		'''
            Return the lower limit of an n-sigma error bar on the histogram probability, i.e. :math:`\\mu - n\\sigma` with :math:`\\mu` the mean and :math:`\\sigma` the standard deviation.

            :param nsigma: defines the n-sigma error bar
            :type nsigma: int, optional, default=2

            :return: the lower limit of the n-sigma error bar
            :rtype: np.ndarray with dimensions determined by self.error

            :raises AssertionError: if self.error is not defined.
        '''
		assert self.error is not None, 'Plower cannot be computed because no error distribution was defined in the error attribute'
		plower, pupper = self.error.nsigma_conf_int(nsigma)
		plower /= plower.sum()
		return plower

	def pupper(self, nsigma=2):
		'''
            Return the upper limit of an n-sigma error bar on the histogram probability, i.e. :math:`\\mu + n\\sigma` with :math:`\\mu` the mean and :math:`\\sigma` the standard deviation.

            :param nsigma: defines the n-sigma error bar
            :type nsigma: int, optional, default=2

            :return: the upper limit of the n-sigma error bar
            :rtype: np.ndarray with dimensions determined by self.error

            :raises AssertionError: if self.error is not defined.
        '''
		assert self.error is not None, 'Pupper cannot be computed because no error distribution was defined in the error attribute'
		plower, pupper = self.error.nsigma_conf_int(nsigma)
		pupper /= pupper.sum()
		return pupper

	def copy(self):
		'''
			Make and return a copy of the current 2D probability histogram.
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
			Start from a set of 2D histograms and compute and return the averaged histogram (and optionally the error bar).

			:param histograms: set of histrograms to be averaged
			:type histograms: list of :py:class:`Histogram1D <thermolib.thermodynamics.histogram.Histogram1D>` instances

			:param error_estimate: indicate how to perform error analysis. Currently, only one method is supported, 'std', which will compute the error bar from the standard deviation within the set of histograms.
			:type error_estimate: str, optional, default=None

			:param cv1_output_unit: the unit in which the new CV1 will be plotted/printed. Defaults to the cv1_output_unit of the first histogram given.
			:type cv1_output_unit: str, optional

			:param cv2_output_unit: the unit in which the new CV2 will be plotted/printed. Defaults to the cv2_output_unit of the first histogram given.
			:type cv2_output_unit: str, optional

			:param cv1_label: label for the new CV1. Defaults to the cv1_label of the first given histogram.
			:type cv1_label: str, optional

			:param cv2_label: label for the new CV2. Defaults to the cv2_label of the first given histogram.
			:type cv2_label: str, optional

			:raises AssertionError: if the histograms do not have a consistent CV1 grid
			:raises AssertionError: if the histograms do not have a consistent CV2 grid

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
	def from_single_trajectory(cls, data, bins, error_estimate=None, cv1_output_unit='au', cv2_output_unit='au', cv1_label='CV1', cv2_label='CV2'):
		'''
			Routine to estimate a 2D probability histogram in terms of two collective variable from a series of samples of those two collective variables.

			:param data: array representing series of samples of the two collective variables. The first column is assumed to correspond to the first collective variable, the second column to the second CV.
			:type data: np.ndarray([Nsamples,2])

			:param bins: array representing the edges of the bins for which a histogram will be constructed. This argument is parsed to `the numpy.histogram2d routine <https://numpy.org/doc/stable/reference/generated/numpy.histogram.html>`_. Hence, more information on its meaning and allowed values can be found there.
			:type bins: np.ndarray

			param error_estimate: indicate if and how to perform error analysis. One of following options is available:

				- **mle_p** - Estimating the error directly for the probability of each bin in the histogram. This method does not explicitly impose the positivity of the probability.

				- **mle_f** - Estimating the error for minus the logarithm of the probability, which is proportional to the free energy (hence f in mle_f). As the probability is expressed as :math:`\propto e^{-f}`, its positivity is explicitly accounted for.

			:type error_estimate: str or None, optional, default=None

			:param cv1_output_unit: the unit in which CV1 will be plotted/printed (not the unit of the input array, that is assumed to be atomic units). If you need help properly converting to atomic units, we refer to the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
			:type cv1_output_unit: str, optional, default='au'

			:param cv2_output_unit: the unit in which CV2 will be plotted/printed (not the unit of the input array, that is assumed to be atomic units). If you need help properly converting to atomic units, we refer to the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
			:type cv2_output_unit: str, optional, default='au'

			:param cv1_label: label of CV1 that will be used for plotting and printing
			:type cv1_label: str, optional, default='CV1'

			:param cv2_label: label of CV2 that will be used for plotting and printing
			:type cv2_label: str, optional, default='CV2'

			:raises ValueError: if no valid error_estimate value is given
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
	def from_wham(cls, bins, traj_input, biasses, temp, pinit=None, error_estimate=None, error_p_threshold=0.0, corrtimes=None, bias_subgrid_num=20, Nscf=1000, convergence=1e-6, bias_thress=1e-3, overflow_threshold=1e-150, cv1_output_unit='au', cv2_output_unit='au', cv1_label='CV1', cv2_label='CV2', plot_biases=False, verbosity='low'):
		'''
			Routine that implements the Weighted Histogram Analysis Method (WHAM) for reconstructing the overall 2D probability histogram in terms of two collective variables CV1 and CV2 from a series of molecular simulations that are (possibly) biased in terms of CV1 and/or CV2.

			:param bins: list of the form [bins1, bins2] where bins1 and bins2 are numpy arrays each representing the bin edges of their corresponding CV for which a histogram will be constructed. For example the following definition: 	[np.arange(-1,1.05,0.05), np.arange(0,5.1,0.1)] will result in a 2D histogram with bins of width 0.05 between -1 and 1 for CV1 and bins of width 0.1 between 0 and 5 for CV2.
			:type bins: np.ndarray

			:param traj_input: list of [CV1,CV2] trajectories, one for each simulation. This input can be generated using the :py:meth:`read_wham_input <thermolib.tools.read_wham_input>` routine. The arguments trajectories and biasses should be of the same length.
			:type trajectories: list of np.ndarrays

			:param biasses: list of bias potentials, one for each simulation allowing to compute the bias at given values of the collective variables CV1 and CV2 in that simulation. This input can be generated using the :py:meth:`read_wham_input <thermolib.tools.read_wham_input>` routine. The arguments traj_input and biasses should be of the same length.
			:type biasses: list(callable)

			:param temp: the temperature at which all simulations were performed
			:type temp: float

			:param pinit: initial guess for the probability density, which is assumed to be in the 'xy'-indexing convention (see the "indexing" argument and the corresponding "Notes" section in `the numpy online documentation of the meshgrid routine <https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html>`_). If None is given, a uniform distribution is used as initial guess.
			:type pinit: np.ndarray, optional, default=None

			:param error_estimate: indicate if and how to perform error analysis. One of following options is available:

				- **mle_p** - Estimating the error directly for the probability of each bin in the histogram. This method does not explicitly impose the positivity of the probability.

				- **mle_p_cov** - Estimate the full covariance matrix for the probability of all bins in the histogram. In other words, appart from the error on the probability/free energy of a bin itself, we now also account for the covariance between the probabilty/free energy of the bins. This method does not explicitly impose the positivity of the probability.

				- **mle_f** - Estimating the error for minus the logarithm of the probability, which is proportional to the free energy (hence f in mle_f). As the probability is expressed as :math:`\propto e^{-f}`, its positivity is explicitly accounted for.

				- **mle_f_cov** - Estimate the full covariance matrix for minus the logarithm of the probability of all bins in the histogram. In other words, appart from the error on the probabilty/free energy of a bin itself (including explicit positivity constraint), we now also account for the covariance between the probability/free energy of the bins.

			:type error_estimate: str or None, optional, default=None

			:param error_p_thresshold: only relevant when error estimation is enabled (see parameter ``error_estimate``). When ``error_p_thresshold`` is set to x, bins in the histogram for which the probability resulting from the trajectory is smaller than x will be disabled for error estimation (i.e. its error will be set to np.nan). Its use is illustrated in :doc:`one of the tutorial notebooks <tut/advanced_projection>`.
			:type error_p_thresshold: float, optional, default=0.0

			:param corrtimes: list of (integrated) correlation times of the CVs, one for each simulation. Such correlation times will be taken into account during the error estimation and hence make it more reliable. If set to None, the CV trajectories will be assumed to contain fully uncorrelated samples (which is not true when using trajectories representing each subsequent step from a molecular dynamics simulation). More information can be found in :ref:`the user guide <seclab_ug_errorestimation>`. This input can be generated using the :py:meth:`decorrelate <thermolib.tools.decorrelate>` routine. This argument needs to have the same length as the ``traj_input`` and ``biasses`` arguments.
			:type corrtimes: list or np.ndarray, optional, default=None

			:param bias_subgrid_num: see documentation for this argument in the :py:meth:`wham2d_bias <thermolib.ext.wham2d_bias>` routine
			:type bias_subgrid_num: optional, defaults to [20,20]

			:param Nscf: the maximum number of steps in the self-consistent loop to solve the WHAM equations
			:type Nscf: int, defaults to 1000

			:param convergence: convergence criterium for the WHAM self consistent solver. The SCF loop will stop whenever the integrated absolute difference between consecutive probability densities is less then the specified value.
			:type convergence: float, defaults to 1e-6

			:param bias_thress: see documentation for this argument in the :py:meth:`wham1d_bias <thermolib.ext.wham1d_bias>` routine
			:type bias_thress: double, optional, default=1e-3

			:param overflow_threshold: see documentation for this argument in the :py:meth:`wham2d_scf <thermolib.ext.wham2d_scf>` routine
			:type overflow_threshold: double, optional, default=1e-150

			:param cv1_output_unit: the unit in which CV1 will be plotted/printed. If you need help properly converting to atomic units, we refer to the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
			:type cv1_output_unit: str, optional, default='au'

			:param cv2_output_unit: the unit in which CV2 will be plotted/printed. If you need help properly converting to atomic units, we refer to the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
			:type cv2_output_unit: str, optional, default='au'

			:param cv1_label: label of CV1 used for plotting and printing
			:type cv1_label: str, optional, default='CV1'

			:param cv2_label: label of CV2 used for plotting and printing
			:type cv2_label: str, optional, default='CV2'

			:param plot_biases: if set to True, a 2D plot of the boltzmann factor of the bias integrated over each bin will be made. This routine (mainly) exists for debugging purposes.
			:type plot_biases: bool, optional, default=False

			:param verbosity: controls the level of verbosity for logging during the WHAM algorithm. 
			:type verbose: str, optional, allowed='none'|'low'|'medium'|'high', default='low'
		'''
		timings = {}
		timings['start'] = time.time()
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
		'''
			.. deprecated:: v1.7

				This routine sole purpose is backward compatibility and serves as an alias for from_wham. Please start using the from_wham routine as this routine will be removed in the near future.
		
				There used to be a distinction between the from_wham and from_wham_c routine (former was full python implementation, latter used Cython for speed up). This distinction has been removed after deliberate testing confirmed that both routines gave identical results. As a result, only the former from_wham_c routine (which is faster) remains, but it has been renamed to from_wham, while the current from_wham_c routine remains in place for backward compatibility.
		'''
		return cls.from_wham(bins, trajectories, biasses, temp, pinit=pinit, error_estimate=error_estimate, bias_subgrid_num=bias_subgrid_num, Nscf=Nscf, convergence=convergence, cv1_output_unit=cv1_output_unit, cv2_output_unit=cv2_output_unit, cv1_label=cv1_label, cv2_label=cv2_label, plot_biases=plot_biases, verbose=verbose, verbosity=verbosity)

	@classmethod
	def from_fes(cls, fes, temp):
		'''
			Compute a probability histogram from the given free energy surface at the given temperature.

			:param fes: free energy surfave from which the probability histogram is computed
			:type fes: fep.FreeEnergySurface2D

			:param temp: the temperature at which the histogram input data was simulated
			:type temp: float
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
	
	def average_cv_constraint_other(self, index, target_distribution=MultiGaussianDistribution, propagator=Propagator()):
		'''
			Routine to compute a profile representing the average of one CV (denoted as y/Y below, y for integration values and Y for resulting averaged values) as function of the other CV (denoted as x below), i.e. the other CV is contraint to one of its bin values x. This is done using the following formula:

			.. math::
				
				Y(x) = \\frac{\\int y\\cdot p(x,y) dy}{\\int p(x,y)dy}
			
			:param index: the index of the CV which will be averaged (the other is then contraint). If index=1, then y=CV1 and x=CV2, while if index=2, then y=CV2 and x=CV1.
			:type index: int (1 or 2)
			
			:param target_distribution: model for the error distribution of the resulting profile.
			:type target_distribution: child instance of :py:class:`Distribution <thermolib.error.Distribution>`, optional, default=:py:class:`MultiGaussianDistribution <thermolib.error.MultiGaussianDistribution>`
			
			:param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken)
			:type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator()
			
			:return: Profile of the average
			:rtype: :py:class:`BaseProfile <thermolib.thermodynamics.fep.BaseProfile>`
			
			:raises ValueError: if index is not 1 or 2
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
			error = propagator(function, self.error, target_distribution=target_distribution, samples_are_flattened=True)
			Ys = error.mean()
		else:
			Ys = function(self.ps)
			error = None
		return BaseProfile(xs, Ys, error=error, cv_output_unit=cv_output_unit, f_output_unit=f_output_unit, cv_label=cv_label, f_label=f_label)
	
	def plot(self, fn=None, slicer=[slice(None),slice(None)], obss=['value'], linestyles=None, linewidths=None, colors=None, cv1_lims=None, cv2_lims=None, plims=None, ncolors=8, ref_max=False, **plot_kwargs):
		'''
            Make either a 2D contour plot of p(CV1,CV2) or a 1D sliced plot of the probability along a slice in the direction specified by the slicer argument. Appart from the value of the probability itself, other (statistical) related properties can be plotted as defined in the obbs argument. At the end of the argument list, you can also specify any matplotlib keyword arguments you wish to parse to the matplotlib plotter. E.g. if you want to specify the colormap, you can just add at the end of the arguments ``cmap='rainbow'``.

            :param fn: name of a file to save plot to. If None, the plot will not be saved to a file.
            :type fn: str, optional, default=None

            :param slicer: determines which degrees of freedom (CV1/CV2) vary/stay fixed in the plot. If slice(none) is specified, the probability will be plotted as function of the corresponding CV. If an integer `i` is specified, that corresponding CV will be kept fixed at its `i-th` value. Some examples:

                - [slice(None),slice(Nonne)] -- a 2D contour plot will be made of the probability as function of both CVs
                - [slice(None),10] -- a 1D plot will be made of the probability as function of CV1 with CV2 fixed at self.cv2s[10]
                - [23,slice(None)] -- a 1D plot will be made of the probability as function of CV2 with CV1 fixed at self.cv1s[23]
            :type slicer: list of `slices <https://www.w3schools.com/python/ref_func_slice.asp>`_ or integers, optional, default=[slice(None),slice(None)]

            :param obss: Specify which statistical property/properties to plot. Multiple values are allowed, which will be plotted on the same figure. Following options are supported:

                - **value** - the values stored in self.ps
                - **mean** - the mean according to the error distribution, i.e. self.error.mean()
                - **lower** - the lower limit of the 2-sigma error bar (which corresponds to a 95% confidence interval in case of a normal distribution), i.e. self.error.nsigma_conf_int(2)[0]
                - **upper** - the upper limit of the 2-sigma error bar (which corresponds to a 95% confidence interval in case of a normal distribution), i.e. self.error.nsigma_conf_int(2)[1]
                - **error** - half the width of the 2-sigma error bar (which corresponds to a 95% confidence interval in case of a normal distribution), i.e. abs(upper-lower)/2
                - **sample** - a random sample taken from the error distribution, i.e. self.error.sample()
            :type obss: list, optional, default=['value']

            :param linestyles: Specify the line style (using matplotlib definitions) for each quantity requested in ``obss``. If None, matplotlib will choose.
            :type linestyles: list or None, optional, default=None

            :param linewidths: Specify the line width (using matplotlib definitions) for each quantity requested in ``obss``. If None, matplotlib will choose.
            :type linewidths: list of strings or None, optional, default=None

            :param colors: Specify the color (using matplotlib definitions) for each quantity requested in ``obss``. If None, matplotlib will choose.
            :type colors: list of strings or None, optional, default=None

            :param cv1_lims: limits to the plotting range of CV1. If None, no limits are enforced
            :type cv1_lims: list of strings or None, optional, default=None

            :param cv2_lims: limits to the plotting range of CV2. If None, no limits are enforced
            :type cv2_lims: list of strings or None, optional, default=None

            :param plims: limits to the plotting range of the probability. If None, no limits are enforced
            :type plims: list of strings or None, optional, default=None

            :param ncolors: only relevant for 2D contour plot, represents the number of contours (and hence colors) to be used in plot.
            :type ncolors: int, optional, default=8

			:param ref_max: if True, the probability histogram(s) will be rescaled such that the maximum probability is 1.0
			:type ref_max: bool, optional, default=False 
        '''
        #preprocess
		assert isinstance(slicer, list) or isinstance(slicer, np.ndarray), 'Slicer should be list or array, instead got %s' %(slicer.__class__.__name__)
		assert len(slicer)==2, 'Slicer should be list of length 2, instead got list of length %i' %( len(slicer))

		if isinstance(slicer[0], slice) and isinstance(slicer[1], slice):
			ndim = 2
			xs = self.cv1s[slicer[0]]/parse_unit(self.cv1_output_unit)
			xlabel = self.cv1_label
			xlims = cv1_lims
			ys = self.cv2s[slicer[1]]/parse_unit(self.cv2_output_unit)
			ylabel = '%s [%s]' %(self.cv2_label, self.cv2_output_unit)
			ylims = cv2_lims
			title = 'p(:,:)'
		elif isinstance(slicer[0], slice) or isinstance(slicer[1],slice):
			ndim = 1
			if isinstance(slicer[0], slice):
				xs = self.cv1s[slicer[0]]/parse_unit(self.cv1_output_unit)
				xlabel = '%s [%s]' %(self.cv1_label, self.cv1_output_unit)
				xlims = cv1_lims
				title = 'p(:,%s[%i]=%.3f %s)' %(self.cv2_label,slicer[1],self.cv2s[slicer[1]]/parse_unit(self.cv2_output_unit),self.cv2_output_unit)
			else:
				xs = self.cv2s[slicer[1]]/parse_unit(self.cv2_output_unit)
				xlabel = '%s [%s]' %(self.cv2_label, self.cv2_output_unit)
				xlims = cv2_lims
				title = 'p(%s[%i]=%.3f %s,:)' %(self.cv1_label,slicer[0],self.cv1s[slicer[0]]/parse_unit(self.cv1_output_unit),self.cv1_output_unit)
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
				if plims is not None:
					delta = (plims[1]-plims[0])/ncolors
					levels = np.arange(plims[0], plims[1]+delta, delta)
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
		Make a plot to compare multiple 1D probability histograms (and possibly teh corresponding free energy profiles)
		
		:param histograms: list of histrograms to plot
		:type histograms: list of :py:class:`Histogram1D <thermolib.thermodynamics.histogram.Histogram1D>` or child classes
		
		:param fn: file name to save the figure to. If None, the plot will not be saved.
		:type fn: str, optional, default=None
		
		:param temp: if defined, the free energy profile corresponding to each histogram will be computed and plotted with its corresponding temperature. If a single float is given, all histograms are assumed to be at the same temperature, if a list or array of floats is given, each entry is assumed to be the temperature of the corresponding entry in the input list of histograms.
		:type temp: float/list(float)/np.ndarray, optional, default=None
		
		:param labels: list of labels for the legend, one for each histogram. Order is assumed to be consistent with profiles.
		:type labels: list(str), optional, default=None
		
		:param flims: [lower,upper] limit of the free energy axis in plots. Ignored if temp argument is not given.
		:type flims: float, optional, default=None
		
		:param colors: List of matplotlib color definitions for each entry in histograms. If an entry is None, a color will be chosen internally. If colors=None, all colors are chosen internally.
		:type colors: List(str), optional, default=None
		
		:param linestyles: List of matplotlib line style definitions for each entry in histograms. If an entry is None, the default line style of '-' will be chosen internally. If linestyles=None, all line styles are set to '-'.
		:type linestyles: List(str), optional, default=None
		
		:param linewidths: List of matplotlib line width definitions for each entry in histograms. If an entry is None, the default line width of 1 will be chosen. If linewidths=None, all line widths are set to 2.
		:type linewidths: List(str), optional, default=None
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
