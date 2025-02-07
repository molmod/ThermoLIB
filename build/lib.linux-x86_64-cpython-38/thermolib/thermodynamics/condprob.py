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
from molmod.io.xyz import XYZReader

from ase.io import read

from ..tools import integrate2d, invert_fisher_to_covariance, fisher_matrix_mle_probdens
from .fep import BaseProfile, BaseFreeEnergyProfile, FreeEnergySurface2D
from ..error import ncycles_default, Propagator, GaussianDistribution, LogGaussianDistribution, MultiGaussianDistribution, MultiLogGaussianDistribution, ErrorArray
from .trajectory import CVComputer, ColVarReader
from ..flatten import Flattener
#from thermolib.ext import fisher_matrix_mle_probdens, invert_fisher_to_covariance

import matplotlib.pyplot as pp
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

__all__ = ['ConditionalProbability1D1D', 'ConditionalProbability1D2D']


class ConditionalProbability(object):
    '''
        Class to compute conditional probabilities of the form :math:`p([qs]|[cvs])`, i.e. probability of finding states characterized with collective variables qs, on the condition that the states are also characterized by collective variables cvs. Such conditional probabiliy allows to convert a free energy surface/profile in terms of the collective variables :math:`cvs` to a free energy surface/profile in terms of the collective variables :math:`qs`.
    '''
    def __init__(self, nq, ncv, q_labels=None, cv_labels=None, q_units=None, cv_units=None, verbose=False):
        '''
            :param nq: dimension of q-space
            :type nq: int

            :param ncv: dimension of cv-space
            :type ncv: int

            :param q_labels: list of labels (one for each q) to be used in plots or prints
            :type q_labels: list of strings, optional, default=['Q1', ['Q2', [...]]]

            :param cv_labels: list of labels (one for each cv) to be used in plots or prints
            :type cv_labels: list of strings, optional, default=['CV1', ['CV2', [...]]]

            :param q_units: list of units (one for each q) to be used in plots or prints
            :type q_units: list of strings, optional, default=['au', ['au', [...]]]

            :param cv_labels: list of units (one for each cv) to be used in plots or prints
            :type cv_labels: list of strings, optional, default=['au', ['au', [...]]]

            :param verbose: make all class routines more verbose in their logging
            :type verbose: bool, optional, default=False
        '''
        self.nq = nq
        self.ncv = ncv
        self._finished = False
        self.verbose = verbose

        if q_labels is None:
            self.q_labels = ['Q%i' for i in range(1,nq+1)]
        else:
            self.q_labels = q_labels
        if cv_labels is None:
            self.cv_labels = ['CV%i' for i in range(1,ncv+1)]
        else:
            self.cv_labels = cv_labels
        if q_units is None:
            self.q_units = ['au',]*nq
        else:
            self.q_units = q_units
        if cv_units is None:
            self.cv_units = ['au',]*ncv
        else:
            self.cv_units = cv_units

        self.q_samples_persim = []
        self.cv_samples_persim = []
        self.q_corr_time_persim = [] #correlation time in q samples in each simulation
        self.num_samples_persim = [] #number of samples in each simulation
        self.num_sims = 0 # number of simulations

    def init_bins_grids(self, q_bins, cv_bins):
        '''
            Routine to initialize the bins used for constructing the conditional probability histograms

            :param q_bins: list of numpy arrays, each array corresponding to the bin edges for one of the Qs. Alternatively, a single numpy array can be given which is then interpreted as the bin edges for all of the Qs.
            :type q_bins: list(np.ndarray) or np.ndarray

            :param cv_bins: list of numpy arrays, each array corresponding to the bin edges for one of the CVs. Alternatively, a single numpy array can be given which is then interpreted as the bin edges for all of the CVs.
            :type cv_bins: list(np.ndarray) or np.ndarray
        '''
        if self.verbose: print('Initializing grids for CV and Q ...')
        #type checking and storing of input arguments
        if isinstance(q_bins, list):
            assert len(q_bins)==self.nq, 'When q_bins is a list, it should contain a number of arrays equal to the number of Qs (=%i) corresponding to the edges for each Q, but instead got a list of length %i' %(self.nq, len(q_bins))
            assert np.array([isinstance(x, np.ndarray) for x in q_bins]).all(), 'When q_bins is a list, all its elements should be a numpy array containing the bin edges of the corresponding Q.'
        else:
            assert isinstance(q_bins, np.ndarray), 'Argument q_bins should be a list of numpy arrays, one array for each Q, or a single numpy array valid for each Q'
            q_bins = [q_bins,]*self.nq
        if isinstance(cv_bins, list):
            assert len(cv_bins)==self.ncv, 'When cv_bins is a list, it should contain a number of arrays equal to the number of CVs (=%i) corresponding to the edges for each CV, but instead got a list of length %i' %(self.ncv, len(cv_bins))
            assert np.array([isinstance(x, np.ndarray) for x in cv_bins]).all(), 'When cv_bins is a list, all its elements should be a numpy array containing the bin edges of the corresponding CV.'
        else:
            assert isinstance(cv_bins, np.ndarray), 'Argument cv_bins should be a list of numpy arrays, one array for each CV, or a single numpy array valid for each CV'
            cv_bins = [cv_bins,]*self.ncv
        self.q_bins = q_bins
        self.cv_bins = cv_bins
        #construct grids for q, cv and q+cv, useful for iterating over later
        #self.qs is just a list of the bin centers for each q
        #self.q_grid is a meshgrid of the bin centers
        #same for cv
        if self.nq==1:
            self.qs = [0.5*(self.q_bins[0][:-1]+self.q_bins[0][1:])]
            #self.q_grid = self.qs[0]
        else:
            self.qs = [ 0.5*b[:-1]+0.5*b[1:] for b in self.q_bins ]
        self.q_grid = np.meshgrid(*self.qs, indexing='ij')
        if self.ncv==1:
            self.cvs = [0.5*(self.cv_bins[0][:-1]+self.cv_bins[0][1:])]
            #self.cv_grid = self.cvs[0]
        else:
            self.cvs = [ 0.5*b[:-1]+0.5*b[1:] for b in self.cv_bins ]
        self.cv_grid = np.meshgrid(*self.cvs, indexing='ij')
        self.cv_q_grid = np.meshgrid(*(self.qs+self.cvs), indexing='ij')

    def process_simulation(self, input_q, input_cv, corr_time=1.0):
        '''
            Routine to extract data samples for the CVs and Qs from given files. In a later stage, these samples will be used to histogram in terms of Q for each conditional CV.

            :param input_q: list of tuples of the form (fn, reader), one for each Q to be included, with fn the file name of the trajectory from which reader will extract the q-values
            :type input_q: list

            :param input_cv: list of tuples of the form (fn, reader), one for each CV to be included, with fn the file name of the trajectory from which reader will extract the cv-values
            :type input_cv: list
            
            :param corr_time: only relevant if error estimation is also desired, in which case corr_time represent the correlation time within the Q samples which will be used for more accurate error estimation.
            :type corr_time: float, optional, default=1.0

            :raises RuntimeError: if attempted to add additional simulations after the ``finish`` routine has been called already.
            :raises AssertionError: if any 2 Q-readers detect different number of samples in any simulation
            :raises AssertionError: if any 2 CV-readers detect different number of samples in any simulation
            :raises AssertionError: if any combination of one Q-reader and one CV-reader detect different number of samples in any simulation
        '''
        #init
        if self._finished:
            raise RuntimeError("Cannot read additional XYZ trajectory because current conditional probability has already been finished.")
        if self.verbose: print('Processing simulation %i  ' %(self.num_sims))
        num_samples = None
        q_samples = None
        cv_samples = None
        
        #extracting Q samples
        for fn, reader in input_q:
            if self.verbose: print('  reading samples for %s from file %s' %(reader.name, fn))
            data = reader(fn)
            if q_samples is None:
                q_samples = data.reshape(-1,1)
            else:
                q_samples = np.append(q_samples, data.reshape(-1,1), axis=1)
            if num_samples is None:
                num_samples = len(data)
            else:
                assert len(data)==num_samples, 'Each Q reader should detect equal number of samples for the same simulation!'
            #estimating number of independent Q-samples

        #extracting CV samples
        for fn, reader in input_cv:
            if self.verbose: print('  reading samples for %s from file %s' %(reader.name, fn))
            data = reader(fn)
            assert len(data)==num_samples, 'Each CV reader should detect equal number of samples as all other Qs and CVs for the same simulation!'
            if cv_samples is None:
                cv_samples = data.reshape(-1,1)
            else:
                cv_samples = np.append(cv_samples, data.reshape(-1,1), axis=1)

        #some final bookkeeping
        self.q_samples_persim.append(q_samples)
        self.cv_samples_persim.append(cv_samples)
        self.num_sims += 1
        self.num_samples_persim.append(num_samples)
        self.q_corr_time_persim.append(corr_time)
        if self.verbose: print('  detected %i samples, from which %i independent (correlation time = %i)' %(num_samples, num_samples/corr_time, corr_time))

    def set_ref(self, q_index=None, q_ref=None, cv_index=None, cv_ref=None):
        '''
            Routine that allows to re-reference all samples of a given Q/CV collective variable. This is usefull, e.g. if a certain collective variable represents an energy and you want to put the zero for that energy at the minimum across all samples.

            :param q_index: specify the index of the Q collective variable you want to re-ref with the given q_ref. If an integer is given, it should be larger than (or equal to) 0 and smaller then the number of Qs. If None, no Q re-reffing will be done. If set to 'all', all Qs will be re-reffed.
            :type q_index: int|'all'|None, optional, default=None

            :param q_ref: specify how to re-reference. If a float is given, substract the given float from all samples of all processed simulations of the corresponding Q. If the string 'global_minimum' is given, first compute the global min of all samples of all processed simulations of the corresponding Q and then substract that value from all those samples. If the string 'minimum_average' is given, first compute the minima of the samples of the corresponding Q for each simulation separately, then compute the average of all those simulation minima. Finally, substract that average from all the samples of all simulatios of the corresponding Q. Value None only allowed if q_index is also None, in which case no re-reffing is done for any Q.
            :type q_ref: float|str|None, optional, default=None

            :param cv_index: specify the index of the CVcollective variable you want to re-ref with the given cv_ref. If an integer is given, it should be larger than (or equal to) 0 and smaller then the number of CVs. If None, no CV re-reffing will be done. If set to 'all', all Qs will be re-reffed.
            :type cv_index: int|'all'|None, optional, default=None

            :param cv_ref: specify how to re-reference. If a float is given, substract the given float from all samples of all processed simulations of the corresponding CV. If the string 'global_minimum' is given, first compute the global min of all samples of all processed simulations of the corresponding CV and then substract that value from all those samples. If the string 'minimum_average' is given, first compute the minima of the samples of the corresponding CV for each simulation separately, then compute the average of all those simulation minima. Finally, substract that average from all the samples of all simulatios of the corresponding CV. Value None only allowed if cv_index is also None, in which case no re-reffing is done for any CV.
            :type cv_ref: float|str|None, optional, default=None
        '''
        #set q references
        if q_index is not None:
            if self.verbose: print('Setting Q reference')
            #check and init q_indices
            if q_index=='all':
                q_index = range(self.nq)
            elif isinstance(q_index, int):
                q_index = [q_index]
            elif isinstance(q_index, list):
                assert np.array([isinstance(index,int) for index in q_index]).all(), "q_index should be integer, list of integers or 'all'. Received %s"%(str(q_index))
            else:
                raise ValueError("q_index should be integer, list of integers or 'all'. Received %s"%(str(q_index)))
            for iq in q_index:
                if self.verbose: print('  of Q%i to %s ' %(iq,q_ref), end='')
                #check and init reference value
                if q_ref in ['global_minimum']:
                    q_ref = np.inf
                    for isim in range(self.num_sims):
                        q_ref = min([q_ref, np.nanmin(self.q_samples_persim[isim][:,iq])])
                    if self.verbose: print('which is %.6f' %(q_ref))
                elif q_ref in ['minimum_average']:
                    q_ref = np.inf
                    for isim in range(self.num_sims):
                        q_ref = min([q_ref, np.nanmean(self.q_samples_persim[isim][:,iq])])
                    if self.verbose: print('which is %.6f' %(q_ref))
                elif not isinstance(q_ref, float):
                    raise ValueError("q_ref should be either 'global_minimum', 'minimum_average' or a float value")
                if self.verbose: print('')
                #set ref
                for isim in range(self.num_sims):
                    self.q_samples_persim[isim][:,iq] -= q_ref
        #set q references
        if cv_index is not None:
            if self.verbose: print('Setting CV reference')
            #check and init cv_indices
            if cv_index=='all':
                cv_index = range(self.ncv)
            elif isinstance(cv_index, int):
                cv_index = [cv_index]
            elif isinstance(cv_index, list):
                assert np.array([isinstance(index,int) for index in cv_index]).all(), "cv_index should be integer, list of integers or 'all'. Received %s"%(str(cv_index))
            else:
                raise ValueError("cv_index should be integer, list of integers or 'all'. Received %s"%(str(cv_index)))
            for icv in cv_index:
                if self.verbose: print('  of CV%i to %s ' %(icv,cv_ref), end='')
                #check and init reference value
                if cv_ref in ['global_minimum']:
                    cv_ref = np.inf
                    for isim in range(self.num_sims):
                        cv_ref = min([cv_ref, self.cv_samples_persim[isim][:,icv].min()])
                    if self.verbose: print('which is %.6f' %(cv_ref))
                elif cv_ref in ['minimum_average']:
                    cv_ref = np.inf
                    for isim in range(self.num_sims):
                        cv_ref = min([cv_ref, self.cv_samples_persim[isim][:,icv].mean()])
                    if self.verbose: print('which is %.6f' %(cv_ref))
                if self.verbose: print('')
                elif not isinstance(cv_ref, float):
                    raise ValueError("cv_ref should be either 'global_minimum', 'minimum_average' or a float value")
                #set ref
                for isim in range(self.num_sims):
                    self.cv_samples_persim[isim][:,icv] -= cv_ref
    
    def finish(self, q_bins, cv_bins, error_estimate=None, error_p_threshold=0.0):
        '''
            Flag the process of adding simulation data as finished and process all samples to compute the conditional probability histograms.

            :param q_bins: see init_bins_grids
            :type q_bins: see init_bins_grids

            :param cv_bins: see init_bins_grids
            :type cv_bins: see init_bins_grids

            :param error_estimate: If not None, this will turn on error estimation an call the ``finish_error`` routine at then end. See ``finish_error`` routine for more information.

            :param error_p_threshold: Only relevant if error_estimate is not None. See ``finish_error`` routine for more information.

            :raises AssertionError: if attempting to finish the conditional probability that is already finished.
        '''
        assert not self._finished, "Can't finish current conditional probability as it has already been finished."
        if self.verbose: print('Finishing conditional probability')
        #init
        self.pconds_persim = []
        self.histcounts_persim = []
        self.histnorms_persim = []
        self.fishers_persim = []
        self.pconds = None
        self.fisher = None
        self.error = None
        #set grids
        self.init_bins_grids(q_bins, cv_bins)

        #loop over each simulation
        self.q_corr_time_persim = np.array(self.q_corr_time_persim)
        Htot = np.zeros(self.cv_q_grid[0].shape)
        Ntot = np.zeros(self.cv_grid[0].shape)
        if self.verbose: print('  constructing histograms for all simulations')
        for isim in range(self.num_sims):
            #compute and store histograms from current simulations
            data = np.append(self.q_samples_persim[isim], self.cv_samples_persim[isim], axis=1)
            H, edges = np.histogramdd(data, bins=self.q_bins+self.cv_bins)
            N, cv_edges = np.histogramdd(self.cv_samples_persim[isim], bins=self.cv_bins)
            self.histcounts_persim.append(H)
            self.histnorms_persim.append(N)
            #add histograms to global histogram
            Ntot += N
            Htot += H
            #compute and store corresponding conditional probability from current simulation
            ps = np.zeros(H.shape)*np.nan
            for cv_index, cv in np.ndenumerate(self.cv_grid):
                if self.ncv==1:
                    cv_index = (cv_index[1],)
                    Ncurrent = N[cv_index]
                    if Ncurrent>0: invNcurrent = 1.0/Ncurrent
                    else:          invNcurrent = np.nan
                else:
                    Ncurrent = N[cv_index]
                    invNcurrent = np.zeros(Ncurrent.shape)*np.nan
                    invNcurrent[Ncurrent>0] = 1.0/Ncurrent[Ncurrent>0]
                ps[(slice(None),)*self.nq + cv_index ] = H[(slice(None),)*self.nq + cv_index ]*invNcurrent
            self.pconds_persim.append(ps)

        if self.verbose: print('  constructing global conditional probability')
        self.pconds = np.zeros(Htot.shape)*np.nan
        for cv_index, cv in np.ndenumerate(self.cv_grid):
            if self.ncv==1:
                cv_index = (cv_index[1],)
                Ncurrent = Ntot[cv_index]
                if Ncurrent>0: invNcurrent = 1.0/Ncurrent
                else:          invNcurrent = np.nan
            else:
                Ncurrent = Ntot[cv_index]
                mask = Ncurrent>0
                invNcurrent = np.zeros(Ncurrent.shape)*np.nan
                invNcurrent[mask] = 1.0/Ncurrent[mask]
            self.pconds[(slice(None),)*self.nq + cv_index ] = Htot[(slice(None),)*self.nq + cv_index ]*invNcurrent

        #define error bars based on Fisher matrix is implemented in inheriting classes
        self.error = None
        if error_estimate is not None:
            self.finish_error(error_estimate, error_p_threshold=error_p_threshold)
        self._finished = True
        
    def finish_error(self, error_estimate, error_p_threshold=0.0):
        '''
            This routine is not (yet) implemented in the most general case. Instead, it is (will be) implemented in the child classes.
        '''
        raise NotImplementedError('Error estimation not yet implemented for %s' %self.__class__.__name__)

    def plot(self, slicer, fn=None, obss=['value'], linestyles=None, linewidths=None, colors=None, logscale=False, ylims=None, croplims=None, cmap=pp.get_cmap('rainbow'), **plot_kwargs):
        '''
            Plot self.pconds[slicer], where slicer needs to be chosen such that self.pconds[slice] is 1D or 2D. The resulting graph will respectively by a regular 1D plot or 2D contourplot.

            :param fn: name of the file to store graph in, defaults to 'condprob.png'
            :type fn: str, optional

            :param obss: Specify which statistical property/properties to plot. Multiple values are allowed, which will be plotted on the same figure. Following options are supported:

                - **value** - the values stored in self.pconds
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

            :param logscale: applying log scale to y-axis (in case of 1D plot) or color axis (in case of 2D plot), defaults to False
            :type logscale: bool, optional

            :param croplims: limits to the plotting range of the cv(s) in case of a 1D plot. If None, no cropping is done
            :type croplims: list, optional, default=None

            :param ylims: limits to the plotting range of the probability in case of a 1D plot. If None, no limits are enforced
            :type ylims: list, optional, default=None

            :param cmap: color map to be used, only relevant in case of 2D contourplot
            :type cmap: color map from matplotlib, optional, default=pp.get_cmap('rainbow')
        '''
        #preprocess
        assert self._finished, "Conditional probability needs to be finished before plotting it."
        assert isinstance(slicer, list) or isinstance(slicer, np.ndarray), 'slicer should be list or array, instead got %s' %(slicer.__class__.__name__)
        assert len(slicer)==self.nq+self.ncv, 'slicer should be list of length equal to sum of number of Qs (whic is %i) and number of CVs (which is %i), instead got list of length %i' %(self.nq, self.ncv, len(slicer))
        #read q slicer
        iqs = []
        qstring = []
        q_label = None
        for iq in range(self.nq):
            qslicer = slicer[iq]
            if isinstance(qslicer, slice):
                iqs.append(iq)
                qstring.append(self.q_labels[iq])
                q_label = '%s [%s]' %(self.q_labels[iq],self.q_units[iq])
            elif isinstance(qslicer, int):
                qstring.append('q=%.3f' %(self.qs[iq][qslicer]))
            else:
                raise ValueError('Slicer list elements should be of type Slice or integer, instead got %s' %(qslicer.__class__.__name__))
        qstring = ','.join(qstring)
        #read q slicer
        icvs = []
        cvstring = []
        cv_label = None
        for icv in range(self.ncv):
            cvslicer = slicer[self.nq+icv]
            if isinstance(cvslicer, slice):
                icvs.append(icv)
                cvstring.append(self.cv_labels[icv])
                cv_label = '%s [%s]' %(self.cv_labels[icv],self.cv_units[icv])
            elif isinstance(cvslicer, int):
                cvstring.append('cv=%.3f' %(self.cvs[icv][cvslicer]))
            else:
                raise ValueError('Slicer list elements should be of type Slice or integer, instead got %s' %(cvslicer.__class__.__name__))
        cvstring = ','.join(cvstring)
        #print('Q-slicer: ', qstring)
        #print('CV-slicer: ', cvstring)

        #read data 
        data = []
        labels = []
        ndim = None
        for obs in obss:
            xs = None
            if obs.lower() in ['value']:
                xs = self.pconds[tuple(slicer)].copy()
            elif obs.lower() in ['mean']:
                assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
                xs = self.error.mean()[tuple(slicer)]
            elif obs.lower() in ['lower']:
                assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
                xs = self.error.nsigma_conf_int(2)[0][tuple(slicer)]
            elif obs.lower() in ['upper']:
                assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
                xs = self.error.nsigma_conf_int(2)[1][tuple(slicer)]
            elif obs.lower() in ['sample']:
                assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
                xs = self.error.sample()[tuple(slicer)]
            elif obs.lower() in ['error']:
                assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
                lower = self.error.nsigma_conf_int(2)[0][tuple(slicer)]
                upper = self.error.nsigma_conf_int(2)[1][tuple(slicer)]
                xs = 0.5*np.abs(upper - lower)
            if xs is None: raise ValueError('Could not interpret observable %s' %obs)
            if ndim is None:
                ndim = len(xs.shape)
            else:
                assert ndim==len(xs.shape), 'Various observable data has inconsistent dimensions!'
            assert len(xs.shape)==len(icvs)+len(iqs), 'Inconsistency in sliced observable data for %s and detected number of Qs and CVs' %obs
            #crop if required
            if croplims is not None:
                xs[xs<croplims[0]] = np.nan
                xs[xs>croplims[1]] = np.nan
            #transform to logscale if required
            if logscale:
                xs = np.log(xs)
            #print('Retrieved data for %s with dimensions: %s' %(obs, xs.shape))
            data.append(xs)
            labels.append(obs)
        
        
        if ndim==1 and self.error is not None:
            lower, upper = self.error.nsigma_conf_int(2)
            lower, upper = lower[tuple(slicer)], upper[tuple(slicer)]
            if croplims is not None:
                lower[lower<croplims[0]] = np.nan
                lower[lower>croplims[1]] = np.nan
                upper[upper<croplims[0]] = np.nan
                upper[upper>croplims[1]] = np.nan
            if logscale:
                lower = np.log(lower)
                upper = np.log(upper)
        if linestyles is None:
            linestyles = [None,]*len(data)
        if linewidths is None:
            linewidths = [1.0,]*len(data)
        if colors is None:
            colors = [None,]*len(data)

        #make plot
        #print('Making %i-dimensional plot (icvs=%i,iqs=%i)' %(ndim,len(icvs),len(iqs)))
        pp.clf()
        if ndim==1:
            if len(iqs)==0 and len(icvs)==1:
                xs = self.cvs[icvs[0]]/parse_unit(self.cv_units[icvs[0]])
            elif len(icvs)==0 and len(iqs)==1:
                xs = self.qs[iqs[0]]/parse_unit(self.q_units[iqs[0]])
            else:
                raise RuntimeError('Inconsistency in shape of sliced q and cv data!')
            for (ys,label,linestyle,linewidth,color) in zip(data,labels,linestyles, linewidths,colors):
                pp.plot(xs, ys, label=label, linestyle=linestyle, linewidth=linewidth, color=color, **plot_kwargs)
            if self.error is not None:
                pp.fill_between(xs, lower, upper, **plot_kwargs, alpha=0.33)
            if q_label is not None and cv_label is None:
                pp.xlabel(q_label, fontsize=16)
            elif q_label is None and cv_label is not None:
                pp.xlabel(cv_label, fontsize=16)
            else:
                raise ValueError('Something went wrong in trying to determine the correct plot labels.')
            pp.ylabel('Probability', fontsize=16)
            pp.title('Conditional probability p(%s;%s)' %(qstring,cvstring), fontsize=16)
            if ylims is not None:
                pp.ylim(ylims)
            pp.legend(loc='best')
            fig = pp.gcf()
            fig.set_size_inches([8,8])
        elif ndim==2:
            if len(icvs)==2:
                xs = self.cvs[icvs[0]]/parse_unit(self.cv_units[icvs[0]])
                ys = self.cvs[icvs[1]]/parse_unit(self.cv_units[icvs[1]])
                do_ij_to_xz_indexing = True
            elif len(iqs)==2:
                xs = self.qs[iqs[0]]/parse_unit(self.q_units[iqs[0]])
                ys = self.qs[iqs[1]]/parse_unit(self.q_units[iqs[1]])
                do_ij_to_xz_indexing = True
            elif len(iqs)==1 and len(icvs)==1:
                xs = self.cvs[icvs[0]]/parse_unit(self.cv_units[icvs[0]])
                ys = self.qs[iqs[0]]/parse_unit(self.q_units[iqs[0]])
                do_ij_to_xz_indexing = False
            else:
                raise RuntimeError('Inconsistency in shape of sliced q and cv data!')
            
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
                if do_ij_to_xz_indexing:
                    contourf = ax.contourf(xs, ys, data[i].T, cmap=cmap, **plot_kwargs) #transpose data to convert ij indexing (internal) to xy indexing (for plotting)
                else:
                    contourf = ax.contourf(xs, ys, data[i], cmap=cmap, **plot_kwargs)
                ax.set_xlabel(cv_label, fontsize=16)
                ax.set_ylabel(q_label, fontsize=16)
                ax.set_title('%s p(%s;%s)' %(labels[i],qstring,cvstring), fontsize=16)
                pp.colorbar(contourf, ax=ax)
            fig.set_size_inches(size)
        else:
            raise ValueError('Can only plot 1D or 2D pcond data, but received %i-d data. Make sure that the combination of qslice and cvslice results in 1 or 2 dimensional data.' %(len(data.shape)))

        fig.tight_layout()
        if fn is not None:
            pp.savefig(fn)
        else:
            pp.show()
        return




class ConditionalProbability1D1D(ConditionalProbability):
    '''
        Class to compute and store conditional probabilities of the form :math:`p(q|cv)` which allow to convert a free energy profile in terms of the collective variable :math:`cv` to a free energy profile in terms of the collective variable :math:`q`.

    '''
    def __init__(self, q_label='Q', cv_label='CV', q_output_unit='au', cv_output_unit='au', verbose=False):
        '''
            :param q_bins: np.histogram argument for defining the bins of Q samples
            :type q_bins: see np.histogram and np.histogram2d, optional

            :param cv_bins: np.histogram argument for defining the bins of CV samples
            :type cv_bins: see np.histogram and np.histogram2d, optional

            :param q_label: label for Q used for plotting/logging, defaults to 'Q'
            :type q_label: str, optional
            
            :param cv_label: label for Q used for plotting/logging, defaults to 'CV'
            :type cv_label: str, optional

            :param verbose: make all class routines more verbose in their logging
            :type verbose: bool, optional, default=False
        '''
        ConditionalProbability.__init__(self, 1, 1, q_labels=[q_label], cv_labels=[cv_label], q_units=[q_output_unit], cv_units=[cv_output_unit], verbose=verbose)

    def process_trajectory_xyz(self, fns, Q, CV, sub=slice(None,None,None), verbose=False):
        '''
            Included for backwards compatibility , this routine will call the more general process_trajectory routine of the parent class.

            .. warning:: 
            
                It is no longer possible to finish automatically after processing a trajectory. Finishing must alwasy be done manually by calling the finish routine!

            Extract Q and CV samples from the given XYZ trajectories. These samples will later be utilized by the finish routine to construct the conditional probability. The trajectory files may also contain data from simulations that are biased in CV space (not Q space!!).

            :param fns: file name (or list of file names) which contain trajectories that are used to compute the conditional probability.
            :type fns: str or list(str)

            :param Q: collective variable used to compute the CV value from a trajectory file
            :type Q: CollectiveVariable

            :param CV: collective variable used to compute the CV value from a trajectory file
            :type CV: CollectiveVariable

            :param sub: python slice instance to subsample the trajectory
            :type sub: slice, optional, default=slice(None, None, None)

            :param verbose: set to True to increase verbosity of the CVComputer to compute CV values from the trajectories
            :type verbose: bool, optional, default=False
        '''
        cv_reader = CVComputer([CV], name=self.cv_labels[0], start=sub.start, stride=sub.step, end=sub.stop, verbose=verbose)
        q_reader = CVComputer([Q], name=self.q_labels[0], start=sub.start, stride=sub.step, end=sub.stop, verbose=verbose)
        for fn in fns:
            self.process_simulation([(fn, q_reader )], [(fn, cv_reader)])

    def process_trajectory_cvs(self, fns, col_q=1, col_cv=2, sub=slice(None,None,None), unit_q='au', unit_cv='au', verbose=False):
        '''
            Included for backwards compatibility, this routine will call the more general process_trajectory routine of the parent class. 
            
            .. warning:: 
            
                It is no longer possible to finish automatically after processing a trajectory. Finishing must alwasy be done manually by calling the finish routine!

            Extract Q and CV samples from the given COLVAR trajectories. These samples will later be utilized by the finish routine to construct the conditional probability. The trajectory files may also contain data from simulations that are biased in CV space (not Q space!!). Each CV trajectory file contains rows of the form

                time q cv

            If the trajectory file contains this data in a different order, it can be accounted for using the col_xx keyword arguments.

            :param fns: file name (or list of file names) of colvar files with the above formatting containing the trajectory data.
            :type fns: str or list(str)

            :param col_q: column index of the collective variable Q in the given input file
            :type col_q: int, optional, default=1

            :param col_cv: column index of the collective variable CV in the given input file
            :type col_cv: int, optional, default=2

            :param unit_q: unit in which the q values are stored in the file
            :type unit_q: str, optional, default='au'

            :param unit_cv: unit in which the cv values are stored in the file
            :type unit_cv: str, optional, default='au'
            
            :param sub: python slice instance to subsample the trajectory
            :type sub: slice, optional, default=slice(None, None, None)

            :param verbose: set to True to increase verbosity of the ColVarReader when reading samples from the trajectory files.
            :type verbose: bool, optional, default=False
        '''
        q_reader  = ColVarReader([col_q] , units=[unit_q] , name='Q' , start=sub.start, stride=sub.step, end=sub.stop, verbose=verbose)
        cv_reader = ColVarReader([col_cv], units=[unit_cv], name='CV', start=sub.start, stride=sub.step, end=sub.stop, verbose=verbose)
        for fn in fns:
            self.process_simulation([(fn, q_reader )], [(fn, cv_reader)])

    def finish_error(self, error_estimate, error_p_threshold=0.0):
        '''
            Estimate error bars on the conditional probability. This routine will be called by the ``finish`` routine if the error_estimate keyword parsed there is not None.

            :param error_estimate: indicate if and how to perform error analysis. One of following options is available:

				- **mle_p** - Estimating the error directly for the probability of each bin in the histogram. This method does not explicitly impose the positivity of the probability.

				- **mle_p_cov** - Estimate the full covariance matrix for the probability of all bins in the histogram. In other words, appart from the error on the probability/free energy of a bin itself, we now also account for the covariance between the probabilty/free energy of the bins. This method does not explicitly impose the positivity of the probability.

				- **mle_f** - Estimating the error for minus the logarithm of the probability, which is proportional to the free energy (hence f in mle_f). As the probability is expressed as :math:`\propto e^{-f}`, its positivity is explicitly accounted for.

				- **mle_f_cov** - Estimate the full covariance matrix for minus the logarithm of the probability of all bins in the histogram. In other words, appart from the error on the probabilty/free energy of a bin itself (including explicit positivity constraint), we now also account for the covariance between the probability/free energy of the bins.

			:type error_estimate: str

            :param error_p_threshold: When ``error_p_threshold`` is set to x, bins in the histogram for which the probability resulting from the trajectory is smaller than x will be disabled for error estimation (i.e. its error will be set to np.nan).
			:type error_p_threshold: float, optional, default=0.0

            :raises ValueError: if an invalid definition for error_estimate is provided
        '''
        if self.verbose: print('  computing Fisher matrices for each simulation')
        Ngrid_q = len(self.q_grid[0])  #extra [0] because only 1 Q and then self.q_grid=[np.ndarray(...)]
        #loop over each simulation
        self.fisher = np.zeros([len(self.q_grid[0])+1, len(self.q_grid[0])+1, len(self.cv_grid[0])])
        for isim in range(self.num_sims):
            #compute and store contribution to fisher matrix from current simulation as well as add contribution to global fisher matrix
            Is = []
            for cv_index, cv in enumerate(self.cv_grid[0]):
                #if self.verbose: print('    for simulation %i cv[%s]=%.3f' %(isim,str(cv_index),cv))
                I = fisher_matrix_mle_probdens(self.pconds_persim[isim][:,cv_index], method=error_estimate)
                Is.append(I)
                n = self.histnorms_persim[isim][cv_index]/self.q_corr_time_persim[isim]
                self.fisher[:,:,cv_index] += n*I
            self.fishers_persim.append(np.array(Is))
        
        if self.verbose: print('  extracting error for conditional probability')
        errors = []
        for cv_index, cv in enumerate(self.cv_grid[0]):
            ps = self.pconds[:,cv_index].copy()
            cov = invert_fisher_to_covariance(self.fisher[:,:,cv_index], ps, threshold=error_p_threshold)[:Ngrid_q,:Ngrid_q]
            if error_estimate in ['mle_p']:
                stds = np.sqrt(np.diagonal(cov))
                err = GaussianDistribution(ps, stds)
            elif error_estimate in ['mle_p_cov']:
                err = MultiGaussianDistribution(ps, cov)
            elif error_estimate in ['mle_f']:
                fs = -np.log(ps)
                stds = np.sqrt(np.diagonal(cov))
                err = LogGaussianDistribution(-fs, stds)
            elif error_estimate in ['mle_f_cov']:
                fs = -np.log(ps)
                err = MultiLogGaussianDistribution(-fs, cov)
            else:
                raise ValueError('Error estimation method %s not supported' %error_estimate)
            errors.append(err)
        self.error = ErrorArray(errors, axis=-1)

    def average(self, propagator=Propagator(target_distribution=MultiGaussianDistribution)):
        '''
            Compute the average of Q as function of CV

            :param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken, or the desired distribution of the targeted error). See documentation on the :py:class:`Propagator <thermolig.error.Propagator>` class for more info.
            :type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator(target_distribution=MultiGaussianDistribution)

            :returns: Q profile as function of CV
            :rtype: :py:class:`BaseProfile <thermolib.thermodynamics.fep.BaseProfile>`
        '''
        def function(pconds):
            qs = np.zeros(len(self.cvs[0]), dtype=float)
            norm = np.zeros(len(self.cvs[0]), dtype=float)
            for iq, q in enumerate(self.qs[0]):
                mask = ~np.isnan(pconds[iq,:])
                qs[mask]   += q*pconds[iq,mask]
                norm[mask] +=   pconds[iq,mask]
            qs[norm==0] = np.nan
            #normalization in case pconds would not be normalized along second axis
            qs[norm>0] /= norm[norm>0]
            return qs
        if self.error is None:
            xs = function(self.pconds)
            error = None
        else:
            error = propagator(function, self.error, samples_are_flattened=True)
            xs = error.mean()
        return BaseProfile(self.cvs[0], xs, error=error, cv_output_unit=self.cv_units[0], f_output_unit=self.q_units[0], cv_label=self.cv_labels[0], f_label=self.q_labels[0])

    def transform(self, fep, f_output_unit=None, f_label=None, f_output_class=BaseFreeEnergyProfile, propagator=Propagator(target_distribution=MultiGaussianDistribution)):
        '''
            Transform the provided 1D FES to a different 1D FES using the current conditional probability according to the formula. 

            .. math:: F(q) &= -kT \\ln\\left(\\int p(q|v)\\cdot e^{-\\beta F(v)} dv\\right)
            
            :param fep: the input free energy profile F(cv) which will be transformed towards F(q)
            :type fep: :py:class:`BaseFreeEnergyProfile <thermolib.thermodynamics.fep.BaseFreeEnergyProfile>`

            :param f_output_unit: the unit of the transformed free energy profile to be used in plotting and printing of energies. If None, the f_output_unit of the original free energy profile will be used.
            :type f_output_unit: str, optional, default=None

            :param f_label: the label of the transformed free energy profile to be used in plotting and printing. If None, the f_label of the original free energy profile will be used.
            :type f_output_unit: str, optional, default=None

            :param f_output_class: the class of the output free energy profile. If you want to use specific features of the :py:class:`SimpleFreeEnergyProfile <thermolib.thermodynamics.fep.SimpleFreeEnergyProfile>` class (such as e.g. automatic detection of reactant, transition state and product state micro/macrostates) in the transformed fep, define this argument as :py:class:`SimpleFreeEnergyProfile <thermolib.thermodynamics.fep.SimpleFreeEnergyProfile>`.
            :type f_output_class: class, optional, default= :py:class:`BaseFreeEnergyProfile <thermolib.thermodynamics.fep.BaseFreeEnergyProfile>`

            :param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken, or the desired distribution of the targeted error). See documentation on the :py:class:`Propagator <thermolig.error.Propagator>` class for more info.
            :type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator(target_distribution=MultiGaussianDistribution)

            :raises AssertionError: if the conditional probability has not been finished yetµ
            :raises AssertionError: if the fep argument is not a :py:class:`BaseFreeEnergyProfile <thermolib.thermodynamics.fep.BaseFreeEnergyProfile>`
            :raises AssertionError: if the cv grids in self.cvs[0] and fep.cvs are not consistent
        '''
        #consistency checks and initalization
        qs = self.qs[0]
        cvs = self.cvs[0]
        assert self._finished, "Conditional probability needs to be finished before applying at in transformations."
        assert isinstance(fep, BaseFreeEnergyProfile), 'Input argument should be instance of (child of) BaseFreeEnergyProfile, instead received %s' %fep.__class__.__name__
        assert len(fep.cvs)==len(cvs), 'Dimension of 1D CV in conditional probability inconsistent with 1D FEP'
        assert (abs(fep.cvs-cvs)<1e-6).all(), 'Values of 1D CV in conditional probability not identical to those of 1D FEP'
        if f_output_unit is None: f_output_unit = fep.f_output_unit
        if f_label is None: f_label = fep.f_label
        # Construct 1D FEP
        def transform(fs, pconds):
            mask = ~np.isnan(fs)
            ps = np.trapz(pconds[:,mask]*np.exp(-fep.beta*fs[mask]), x=cvs[mask])
            ps /= np.trapz(ps, x=qs)
            fs_new = np.zeros([len(qs)], float)*np.nan
            fs_new[ps>0] = -np.log(ps[ps>0])/fep.beta
            return fs_new
        if fep.error is not None:
            if self.error is not None:
                error = propagator(transform, fep.error, self.error)
            else:
                transf1 = lambda fs: transform(fs, self.pconds)
                error = propagator(transf1, fep.error)
            fs = error.mean()
        elif self.error is not None:
            transf2 = lambda pconds: transform(fep.fs, pconds)
            error = propagator(transf2, self.error)
            fs = error.mean()
        else:
            fs = transform(fep.fs, self.pconds)
            error = None
        return f_output_class(self.qs[0], fs, fep.T, error=error, cv_output_unit=self.q_units[0], f_output_unit=f_output_unit, cv_label=self.q_labels[0], f_label=f_label)

    def deproject(self, fep, f_output_unit=None, f_label=None, f_output_class=FreeEnergySurface2D, propagator=Propagator(target_distribution=MultiGaussianDistribution)):
        '''
            Deproject the provided 1D FEP F(q) to a 2D FES F(q,v) using the current conditional probability according to the formula

            .. math:: F(q_1,q_2) &= F(q_2)-kT \\ln\\left(p(q_1|q_2)\\right)

            :param fep: the free energy profile F(q_2) which will be transformed
            :type fep: (child of) BaseFreeEnergyProfile

            :param f_output_unit: the unit of the deprojected free energy profile to be used in plotting and printing of energies. If None, the f_output_unit of the original 1D free energy profile will be used.
            :type f_output_unit: str, optional, default=None

            :param f_label: the label of the deprojected free energy profile to be used in plotting and printing. If None, the f_label of the original 1D free energy profile will be used.
            :type f_label: str, optional, default=None

            :param f_output_class: the class of the output free energy profile, defaults to FreeEnergySurface2D
            :type f_output_class: class, optional

            :param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken, or the desired distribution of the targeted error). See documentation on the :py:class:`Propagator <thermolig.error.Propagator>` class for more info.
            :type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator(target_distribution=MultiGaussianDistribution)

            :raises AssertionError: if the conditional probability has not been finished yetµ
            :raises AssertionError: if the fep argument is not a :py:class:`BaseFreeEnergyProfile <thermolib.thermodynamics.fep.BaseFreeEnergyProfile>`
            :raises AssertionError: if the cv grids in self.cvs[0] and fep.cvs are not consistent
        '''
        #consistency checks and initalization
        qs, cvs = self.qs[0], self.cvs[0]
        assert self._finished, "Conditional probability needs to be finished before applying at in transformations."
        assert isinstance(fep, BaseFreeEnergyProfile), 'Input argument should be instance of (child of) BaseFreeEnergyProfile, instead received %s' %fep.__class__.__name__
        assert len(fep.cvs)==len(cvs), 'Dimension of collective variable v in conditional probability p(q|v) inconsistent with collective variable in 1D FEP'
        assert (abs(fep.cvs-cvs)<1e-6).all(), 'Values of collective variable v in conditional probability p(q|v) not identical to those of collective variable in 1D FEP'
        if f_label is None: f_label = fep.f_label
        if f_output_unit is None: f_output_unit = fep.f_output_unit
        # Construct 2D FES
        kT = boltzmann*fep.T
        def deproject(fs, pconds):
            fs_new = np.zeros([len(cvs),len(qs)], float)*np.nan
            for icv in range(len(cvs)):
                mask = ~np.isnan(pconds[:,icv])
                pconds[:,icv] /= pconds[mask,icv].sum() #assert pcond is proparly normalized (in case it was sampled from error distribution)
                for iq in range(len(qs)):
                    if pconds[iq,icv]>0:
                        fs_new[icv,iq] = fs[icv] - kT*np.log(pconds[iq,icv])
            return fs_new
        if fep.error is not None or self.error is not None:
            propagator.flattener = Flattener(len(cvs), len(qs))
            if fep.error is not None:
                if self.error is not None:
                    error = propagator(deproject, fep.error, self.error)
                else:
                    deproj1 = lambda fs: deproject(fs, self.pconds)
                    error = propagator(deproj1, fep.error)
            elif self.error is not None:
                deproj2 = lambda pconds: deproject(fep.fs, pconds)
                error = propagator(deproj2, self.error)
            fs = error.mean()
        else:
            fs = deproject(fep.fs, self.pconds)
            error = None
        return f_output_class(cvs, qs, fs, fep.T, error=error, cv1_output_unit=fep.cv_output_unit, cv2_output_unit=self.q_units[0], f_output_unit=f_output_unit, cv1_label=fep.cv_label, cv2_label=self.q_labels[0], f_label=f_label)




class ConditionalProbability1D2D(ConditionalProbability):
    '''
        Class to store and compute conditional probabilities of the form :math:`p(q_1,q_2|cv)` which can be used to transform a 1D free energy profile in terms of the collective variable :math:`cv` towards a 2D free energy surface in terms of the collective variables :math:`q_1` and :math:`q_2`.
    '''
    def __init__(self, q1_label='Q1', q2_label='Q2', cv_label='CV', q1_output_unit='au', q2_output_unit='au', cv_output_unit='au', verbose=False):
        '''
            :param q1_label: label for Q1 used for plotting/logging
            :type q1_label: str, optional, default='Q1'

            :param q2_label: label for Q2 used for plotting/logging
            :type q2_label: str, optional, default='Q2'
            
            :param cv_label: label for Q used for plotting/logging
            :type cv_label: str, optional, default='CV'

            :param verbose: set to True to increase verbosity of TrajectoryReaders used to compute the various CV samples along an XYZ trajectory
            :type verbose: bool, optional, default=False
        '''
        ConditionalProbability.__init__(self, 2, 1, q_labels=[q1_label,q2_label], cv_labels=[cv_label], q_units=[q1_output_unit,q2_output_unit], cv_units=[cv_output_unit], verbose=verbose)

    def process_trajectory_xyz(self, fns, Q1, Q2, CV, sub=slice(None,None,None)):
        '''
            Included for backwards compatibility, this routine will call the more general process_trajectory routine of the parent class.

            Extract Q1, Q2 and CV samples from the given XYZ trajectories. These samples will later be utilized by the finish routine to construct the conditional probability. The trajectory files may also contain data from simulations that are biased in CV space (not Q1/Q2 space!!).

            :param fns: file name (or list of file names) which contain trajectories that are used to compute the conditional probability.
            :type fns: str or list(str)

            :param Q1: collective variable used to compute the Q1 value from a trajectory file
            :type Q1: :py:class:`CollectiveVariable <thermolib.thermodynamics.cv.CollectiveVariable>`

            :param Q2: collective variable used to compute the Q2 value from a trajectory file
            :type Q2: :py:class:`CollectiveVariable <thermolib.thermodynamics.cv.CollectiveVariable>`

            :param CV: collective variable used to compute the CV value from a trajectory file
            :type CV: :py:class:`CollectiveVariable <thermolib.thermodynamics.cv.CollectiveVariable>`

            :param sub: python slice instance to subsample the trajectory
            :type sub: slice, optional, default=slice(None, None, None)
        '''
        cv_reader = CVComputer([CV], name=self.cv_labels[0], start=sub.start, stride=sub.step, end=sub.stop, verbose=self.verbose)
        q1_reader = CVComputer([Q1], name=self.q_labels[0], start=sub.start, stride=sub.step, end=sub.stop, verbose=self.verbose)
        q2_reader = CVComputer([Q2], name=self.q_labels[1], start=sub.start, stride=sub.step, end=sub.stop, verbose=self.verbose)
        for fn in fns:
            self.process_simulation([(fn,q1_reader), (fn,q2_reader)], [(fn,cv_reader)])

    def process_trajectory_cvs(self, fns, col_q1=1, col_q2=2, col_cv=3, unit_q1='au', unit_q2='au', unit_cv='au', sub=slice(None,None,None)):
        '''
            Included for backwards compatibility, this routine will call the more general process_trajectory routine of the parent class.

            Extract Q and CV samples from the given COLVAR trajectories. These samples will later be utilized by the finish routine to construct the conditional probability. The trajectory files may also contain data from simulations that are biased in CV space (not Q space!!). Each CV trajectory file contains rows of the form

                time q1 q2 cv

            If the trajectory file contains this data in a different order, it can be accounted for using the col_xx keyword arguments.

            :param fns: file name (or list of file names) of colvar files with the above formatting containing the trajectory data.
            :type fns: str or list(str)

            :param col_q1: column index of the collective variable Q1 in the given input file
            :type col_q1: int, optional, default=1

            :param col_q2: column index of the collective variable Q2 in the given input file
            :type col_q2: int, optional, defaults=2

            :param col_cv: column index of the collective variable CV in the given input file
            :type col_cv: int, optional, default=3

            :param unit_q1: unit in which the q1 values are stored in the file
            :type unit_q1: str, optional, default='au'

            :param unit_q2: unit in which the q2 values are stored in the file
            :type unit_q2: str, optional, default='au'

            :param unit_cv: unit in which the cv values are stored in the file
            :type unit_cv: str, optional, default='au'
            
            :param sub: python slice instance to subsample the trajectory
            :type sub: slice, optional, default=slice(None, None, None)
        '''
        q1_reader  = ColVarReader([col_q1] , units=[unit_q1] , name='Q1' , start=sub.start, stride=sub.step, end=sub.stop, verbose=self.verbose)
        q2_reader  = ColVarReader([col_q2] , units=[unit_q2] , name='Q2' , start=sub.start, stride=sub.step, end=sub.stop, verbose=self.verbose)
        cv_reader = ColVarReader([col_cv], units=[unit_cv], name='CV', start=sub.start, stride=sub.step, end=sub.stop, verbose=self.verbose)
        for fn in fns:
            self.process_simulation([(fn,q1_reader), (fn,q2_reader)], [(fn,cv_reader)])

    def deproject(self, fep, f_output_unit=None, f_label=None, f_output_class=FreeEnergySurface2D, propagator=Propagator(target_distribution=MultiGaussianDistribution)):
        '''
            Transform the provided 1D FEP to a 2D FES using the current conditional probability according to the formula

            .. math:: F(q_1,q_2) &= -kT\cdot\\ln\\left(\\int p(q_1,q_2|v)\\cdot e^{-\\beta F(v)}dv\\right)

            :param fep: the input free energy profile F(cv) which will be transformed towards F(q)
            :type fep: BaseFreeEnergyProfile or child classes

            :param f_output_unit: the unit of the deprojected free energy profile to be used in plotting and printing of energies. If None, the f_output_unit of the original 1D free energy profile will be used.
            :type f_output_unit: str, optional, default=None

            :param f_label: the label of the deprojected free energy profile to be used in plotting and printing. If None, the f_label of the original 1D free energy profile will be used.
            :type f_label: str, optional, default=None

            :param f_output_class: the class of the output free energy profile, defaults to FreeEnergySurface2D
            :type f_output_class: class, optional

            :param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken, or the desired distribution of the targeted error). See documentation on the :py:class:`Propagator <thermolig.error.Propagator>` class for more info.
            :type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator(target_distribution=MultiGaussianDistribution)

            :raises AssertionError: if the conditional probability has not been finished yetµ
            :raises AssertionError: if the fep argument is not a :py:class:`BaseFreeEnergyProfile <thermolib.thermodynamics.fep.BaseFreeEnergyProfile>`
            :raises AssertionError: if the cv grids in self.cvs[0] and fep.cvs are not consistent
        '''
        #consistency checks and initalization
        assert self._finished, "Conditional probability needs to be finished before applying at in transformations."
        assert isinstance(fep, BaseFreeEnergyProfile), 'Input argument should be instance of (child of) BaseFreeEnergyProfile, instead received %s' %fep.__class__.__name__
        assert len(fep.cvs)==len(self.cvs[0]), 'Dimension of 1D CV in conditional probability inconsistent with 1D FEP'
        assert (abs(fep.cvs-self.cvs[0])<1e-6).all(), 'Values of 1D CV in conditional probability not identical to those of 1D FEP'
        if f_label is None: f_label = fep.f_label
        if f_output_unit is None: f_output_unit = fep.f_output_unit
        #construct 2D FES
        def transform(fs, pconds):
            mask = ~np.isnan(fs)
            ps = np.trapz(pconds[...,mask]*np.exp(-fep.beta*fs[mask]), x=fep.cvs[mask])
            ps /= integrate2d(ps, x=self.qs[0], y=self.qs[1])
            fs_new = np.zeros(ps.shape, float)*np.nan
            fs_new[ps>0] = -np.log(ps[ps>0])/fep.beta
            return fs_new

        fs = transform(fep.fs, self.pconds)
        error = None
        if fep.error is not None:
            propagator.flattener = Flattener(len(self.qs[0]), len(self.qs[1]))
            if self.error is not None:
                error = propagator(transform, fep.error, self.error)
            else:
                transf1 = lambda fs: transform(fs, self.pconds)
                error = propagator(transf1, fep.error)
        elif self.error is not None:
            transf2 = lambda pconds: transform(fep.fs, pconds)
            error = propagator(transf2, self.error)
        return f_output_class(self.qs[0], self.qs[1], fs, fep.T, error=error, cv1_output_unit=self.q_units[0], cv2_output_unit=self.q_units[1], f_output_unit=f_output_unit, cv1_label=self.q_labels[0], cv2_label=self.q_labels[1], f_label=f_label)
