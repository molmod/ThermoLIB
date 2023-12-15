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
from molmod.io.xyz import XYZReader

from ase.io import read

from ..tools import integrate2d
from .fep import BaseProfile, BaseFreeEnergyProfile, FreeEnergySurface2D
from ..error import Propagator, GaussianDistribution, LogGaussianDistribution
from .trajectory import CVComputer, ColVarReader

import matplotlib.pyplot as pp
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

import numpy as np

__all__ = ['ConditionalProbability1D1D', 'ConditionalProbability1D2D']


class ConditionalProbability(object):
    '''
        Class to compute conditional probabilities of the form :math:`p([qs]|[cvs])`, i.e. probability of finding states characterized with collective variables qs, on the condition that the states are also characterized by collective variables cvs. Such conditional probabiliy allows to convert a free energy surface (or profile if only one cv is given) in terms of the collective variables :math:`cvs` to a free energy surface (or profile if only one q is given) in terms of the collective variables :math:`qs`.

    '''
    def __init__(self, nq, ncv, q_bins=20, cv_bins=20, q_labels=None, cv_labels=None, q_units=None, cv_units=None, verbose=False):
        '''
            :param nq: dimension of q-space
            :type nq: int

            :param ncv: dimension of cv-space
            :type ncv: int

            :param q_bins: bins for qs (see numpy.histogram, numpy.histogram2d and numpy.histogramnd for more information), defaults to 20 bins for each Q
            :type q_bins: see numpy.histogram, numpy.histogram2d and numpy.histogramnd, optional

            :param cv_bins: bins for cvs (see numpy.histogram, numpy.histogram2d and numpy.histogramnd for more information), defaults to 20 bins for each CV
            :type cv_bins: see numpy.histogram, numpy.histogram2d and numpy.histogramnd, optional

            :param q_labels: list of labels (one for each q) to be used in plots or prints, defaults to ['Q1', ['Q2', [...]]]
            :type q_labels: list of strings, optional

            :param cv_labels: list of labels (one for each cv) to be used in plots or prints, defaults to ['CV1', ['CV2', [...]]]
            :type cv_labels: list of strings, optional

            :param q_units: list of units (one for each q) to be used in plots or prints, defaults to ['au', ['au', [...]]]
            :type q_units: list of strings, optional

            :param cv_labels: list of units (one for each cv) to be used in plots or prints, defaults to ['au', ['au', [...]]]
            :type cv_labels: list of strings, optional
        '''
        self.nq = nq
        self.ncv = ncv
        self.q_bins = q_bins
        self.cv_bins = cv_bins
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
        self.qsamples = None
        self.cvsamples = None
        self.nsamples = 0
        self.qs = None
        self.cvs = None
        self.pconds = None
        self.error = None
        self._finished = False
        self.verbose = verbose
    
    def process_trajectory(self, fns, q_readers, cv_readers, finish=True, error_estimate=None, corr_times=None):
        '''
            extracting data samples of CVs and QS from given files (in fns) that later will be histogrammed per value of CVs.

            :param fns: file name (or list of file names) which contain trajectories that are used to compute the conditional probability.
            :type fns: str or list(str)

            :param q_readers: List of Trajectory readers (one for each Q) for extracting Q values from trajectory files using the process_trajectory routine
            :type q_readers: list of TrajectoryReaders

            :param cv_readers: List of Trajectory readers (one for each CV) for extracting CV values from trajectory files using the process_trajectory routine
            :type cv_readers: list of TrajectoryReaders

            :param finish: set this to True if the given file name(s) are the only relevant trajectories and hence the conditional probability should be computed from only these trajectories. Setting it to True will therefore trigger propper normalisation of the conditional probability. Set this to False if you intend to call the routine *process_trajectory_xyz* again later on with additional trajectory files.
            :type finish: bool, optional, default=True

            :param verbose: set to True to increase verbosity, defaults to False
            :type verbose: bool, optional
        '''
        if self._finished:
            raise RuntimeError("Cannot read additional XYZ trajectory because current conditional probability has already been finished.")
        if corr_times is not None:
            assert len(corr_times)==len(fns), 'Argument corr_times should be of same length of fns'
        else:
            corr_times = [None,]*len(fns)
        #Read Q values
        for q_reader in q_readers:
            qs = None
            for fn, ctime in zip(fns, corr_times):
                if self.verbose: print('Reading Q %s from file %s' %(q_reader.name, fn))
                data = q_reader(fn)
                if ctime is not None:
                    step = int(np.ceil(ctime))
                    if step==0: step = 1
                    data = data[::step]
                if qs is None:
                    qs = data
                else:
                    qs = np.append(qs, data, axis=0)
            if self.qsamples is None:
                self.qsamples = qs
            else:
                self.qsamples = np.append(self.qsamples, qs, axis=1)
        #Read CV values
        for cv_reader in cv_readers:
            cvs = None
            for fn, ctime in zip(fns, corr_times):
                if self.verbose: print('Reading CV %s from file %s' %(cv_reader.name, fn))
                data = cv_reader(fn)
                if ctime is not None:
                    step = int(np.ceil(ctime))
                    if step==0: step = 1
                    data = data[::step]
                if cvs is None:
                    cvs = data
                else:
                    cvs = np.append(cvs, data, axis=0)
            if self.cvsamples is None:
                self.cvsamples = cvs
            else:
                self.cvsamples = np.append(self.cvsamples, cvs, axis=1)
        #finish if required
        if finish:
            self.finish()
    
    def finish(self, q_bins=None, cv_bins=None, error_estimate=None):
        assert not self._finished, "Can't finish current conditional probability as it has already been finished."
        #overwrite self.q_bins and self.cv_bins if given
        if q_bins is not None: self.q_bins = q_bins
        if cv_bins is not None: self.cv_bins = cv_bins
        #consistency check
        if len(self.qsamples.shape)==1:
            assert self.nq==1, 'Conditional probability was initialized for %i Qs, but only read 1 Q from files up till now' %(self.nq)
            self.qsamples = self.qsamples.reshape([-1,1])
        else:
            assert self.qsamples.shape[1] == self.nq, 'Conditional probability was initialized for %i Qs, but read %i Qs from files up till now' %(self.nq, self.qsamples.shape[1])
        if len(self.cvsamples.shape)==1:
            assert self.ncv==1, 'Conditional probability was initialized for %i CVs, but only read 1 CV from files up till now' %(self.ncv)
            self.cvsamples = self.cvsamples.reshape([-1,1])
        else:
            assert self.cvsamples.shape[1] == self.ncv, 'Conditional probability was initialized for %i CVs, but read %i CVs from files up till now' %(self.ncv, self.cvsamples.shape[1])
        assert self.qsamples.shape[0] == self.cvsamples.shape[0], 'Inconsitent number of Q (%i) and CV (%i) samples up till now!' %(self.qsamples.shape[0], self.cvsamples.shape[0])
        self.nsamples = self.qsamples.shape[0]
        if self.verbose:
            print('Finishing condporb')
            print('  detected %i samples' %(self.nsamples))
        #construct ND histogram n(qs,cvs) from samples
        data = np.append(self.qsamples, self.cvsamples, axis=1)
        if self.verbose:
            print('  sample data has shape: ',data.shape)
        if isinstance(self.q_bins, int) and isinstance(self.cv_bins, int):
            if self.q_bins==self.cv_bins:
                bins = self.q_bins
            else:
                bins = [self.q_bins,]*self.nq + [self.cv_bins,]*self.cv_bins
        elif isinstance(self.q_bins, list) and isinstance(self.cv_bins, list):
            bins = self.q_bins + self.cv_bins
        else:
            raise NotImplementedError('Combination of q_bins and cv_bins not supported! Should be either both integers (giving the number of bins for each of its qs/cvs), or a list with an integer (number of bins)/nparray (bin edges) for each of its qs/cvs.')
        self.pconds, edges = np.histogramdd(data, bins=bins)
        self.qs = []
        for iq in range(self.nq):
            self.qs.append(0.5*(edges[iq][:-1]+edges[iq][1:]))
        self.cvs = [] 
        for icv in range(self.ncv):
            self.cvs.append(0.5*(edges[self.nq+icv][:-1]+edges[self.nq+icv][1:]))
        # Normalize conditional probability as well as some additional probability densities
        for cv_index, cv in np.ndenumerate(self.cvs):
            if len(self.cvs)==1:
                slist = (slice(None),)*self.nq + (cv_index[1],)
            else:
                slist = (slice(None),)*self.nq + cv_index
            #print('slist=',slist)
            #print('self.pconds[slist].shape=',self.pconds[slist].shape)
            norm = self.pconds[slist].sum()
            if norm>0:
                self.pconds[slist] /= norm
        #do error estimation if requested
        self.error = None
        if error_estimate=='mle_p':
            perr = np.sqrt(self.pconds*(1-self.pconds)/self.nsamples)
            self.error = GaussianDistribution(self.pconds, perr)
        elif error_estimate=='mle_f':
			#we first compute the error bar interval on f=-log(p) and then transform it to one on p itself.
            fs, ferr = np.zeros(self.pconds.shape)*np.nan, np.zeros(self.pconds.shape)*np.nan
            mask = self.pconds>0
            fs[mask] = -np.log(self.pconds[mask])
            ferr[mask] = np.sqrt((np.exp(fs[mask])-1)/self.nsamples)
            self.error = LogGaussianDistribution(-fs, ferr)
        elif error_estimate is not None:
            raise ValueError('Invalid value for error_estimate argument, received %s. Check documentation for allowed values.' %error_estimate)
        self._finished = True

    def plot(self, slicer, fn='condprob.png', title=None, logscale=False, croplims=None, cmap=pp.get_cmap('rainbow'), **plot_kwargs):
        '''
            Plot self.pconds[slice], where slice needs to be chosen such that self.pconds[slice] is 1D or 2D. The resulting graph will respectively by a regular 1D plot or 2D contourplot.

            :param fn: name of the file to store graph in, defaults to 'condprob.png'
            :type fn: str, optional

            :param x_unit: unit to be used for the cv/q on the x-axis, defaults to 'au'
            :type x_unit: str, optional

            :param y_unit: unit to be used for the cv/q on the y-axis, only relevant when 2D contourplot is made, defaults to 'au'
            :type y_unit: str, optional

            :param x_label: label to be used for the cv/q on the x-axis, defaults to 'Q'/'Q1' for q-data or 'CV'/'CV1' for cv-data
            :type x_label: str, optional

            :param y_label: label to be used for the cv/q on the x-axis, defaults to 'Q'/'Q2' for q-data or 'CV'/'CV2' for cv-data
            :type y_label: str, optional

            :param logscale: applying log scale to y-axis (in case of 1D plot) or color axis (in case of 2D plot), defaults to False
            :type logscale: bool, optional

            :param cmap: color map to be used, only relevant in case of 2D contourplot, defaults to pp.get_cmap('rainbow')
            :type cmap: color map from matplotlib, optional
        '''
        assert self._finished, "Conditional probability needs to be finished before plotting it."
        #preprocess
        assert isinstance(slicer, list) or isinstance(slicer, np.ndarray), 'slicer should be list or array, instead got %s' %(slicer.__class__.__name__)
        assert len(slicer)==self.nq+self.ncv, 'slicer should be list of length equal to sum of number of Qs (whic is %i) and number of CVs (which is %i), instead got list of length %i' %(self.nq, self.ncv, len(slicer))
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
        data = self.pconds[slicer].copy()
        if croplims is not None:
            data[data<croplims[0]] = np.nan
            data[data>croplims[1]] = np.nan
        if self.error is not None:
            lower, upper = self.error.nsigma_conf_int(2)
            lower, upper = lower[slicer], upper[slicer]
        if logscale:
            mask = data>0
            data[mask] = np.log(data[mask])
            data[~mask] = np.nan
            mask = lower>0
            lower[mask] = np.log(lower[mask])
            lower[~mask] = np.nan
            mask = upper>0
            upper[mask] = np.log(upper[mask])
            upper[~mask] = np.nan
        assert len(data.shape)==len(iqs)+len(icvs), 'Inconsistency in sliced pcond data and detected number of Qs and CVs'
        #make plot
        pp.clf()
        if len(data.shape)==1:
            if len(iqs)==0 and len(icvs)==1:
                xs = self.cvs[icvs[0]]/parse_unit(self.cv_units[icvs[0]])
            elif len(icvs)==0 and len(iqs)==1:
                xs = self.qs[iqs[0]]/parse_unit(self.q_units[iqs[0]])
            else:
                raise RuntimeError('Inconsistency in shape of sliced q and cv data!')
            pp.plot(xs, data, **plot_kwargs)
            if self.error is not None:
                pp.fill_between(xs, lower, upper, **plot_kwargs, alpha=0.33)
            if q_label is not None and cv_label is None:
                pp.xlabel(q_label, fontsize=16)
            elif q_label is None and cv_labe is not None:
                pp.xlabel(cv_label, fontsize=16)
            else:
                raise ValueError('Something went wrong in trying to determine the correct plot labels.')
            pp.ylabel('Probability', fontsize=16)
        elif len(data.shape)==2:
            if len(icvs)==2:
                xs = self.cvs[icvs[0]]/parse_unit(self.cv_units[icvs[0]])
                ys = self.cvs[icvs[1]]/parse_unit(self.cv_units[icvs[1]])
            elif len(iqs)==2:
                xs = self.qs[iqs[0]]/parse_unit(self.q_units[iqs[0]])
                ys = self.qs[iqs[1]]/parse_unit(self.q_units[iqs[1]])
            elif len(iqs)==1 and len(icvs)==1:
                xs = self.cvs[icvs[0]]/parse_unit(self.cv_units[icvs[0]])
                ys = self.qs[iqs[0]]/parse_unit(self.q_units[iqs[0]])
            else:
                raise RuntimeError('Inconsistency in shape of sliced q and cv data!')
            contourf = pp.contourf(xs, ys, data, cmap=cmap, **plot_kwargs)
            pp.xlabel(cv_label, fontsize=16)
            pp.ylabel(q_label, fontsize=16)
            pp.colorbar(contourf)
        else:
            raise ValueError('Can only plot 1D or 2D pcond data, but received %i-d data. Make sure that the combination of qslice and cvslice results in 1 or 2 dimensional data.' %(len(data.shape)))
        if title is None:
            title = 'Conditional probability p(%s;%s)' %(qstring,cvstring)
        pp.title(title)
        fig = pp.gcf()
        fig.set_size_inches([8,8])
        fig.tight_layout()
        pp.savefig(fn)
        return




class ConditionalProbability1D1D(ConditionalProbability):
    '''
        Routine to store and compute conditional probabilities of the form :math:`p(q_1|cv)` which allow to convert a free energy profile in terms of the collective variable :math:`cv` to a free energy profile in terms of the collective variable :math:`q_1`.

    '''
    def __init__(self, q_bins=None, cv_bins=None, q_label='Q', cv_label='CV', q_output_unit='au', cv_output_unit='au', verbose=False):
        '''
            :param q_bins: np.histogram argument for defining the bins of Q samples
            :type q_bins: see np.histogram and np.histogram2d, optional

            :param cv_bins: np.histogram argument for defining the bins of CV samples
            :type cv_bins: see np.histogram and np.histogram2d, optional

            :param q_label: label for Q used for plotting/logging, defaults to 'Q'
            :type q_label: str, optional
            
            :param cv_label: label for Q used for plotting/logging, defaults to 'CV'
            :type cv_label: str, optional
        '''
        ConditionalProbability.__init__(self, 1, 1, q_bins=q_bins, cv_bins=cv_bins, q_labels=[q_label], cv_labels=[cv_label], q_units=[q_output_unit], cv_units=[cv_output_unit], verbose=verbose)

    def process_trajectory_xyz(self, fns, Q, CV, sub=slice(None,None,None), finish=True, verbose=False):
        '''
            Included for backwards compatibility, this routine will call the more general process_trajectory routine of the parent class.

            Extract Q and CV samples from the given XYZ trajectories. These samples will later be utilized by the finish routine to construct the conditional probability. The trajectory files may also contain data from simulations that are biased in CV space (not Q space!!).

            :param fns: file name (or list of file names) which contain trajectories that are used to compute the conditional probability.
            :type fns: str or list(str)

            :param Q: collective variable used to compute the CV value from a trajectory file
            :type Q: CollectiveVariable

            :param CV: collective variable used to compute the CV value from a trajectory file
            :type CV: CollectiveVariable

            :param sub: python slice instance to subsample the trajectory, defaults to slice(None, None, None)
            :type sub: slice, optional

            :param finish: set to True if the given file name(s) are the only relevant trajectories and hence the conditional probability should be computed from only these trajectories. Setting it to True will trigger histogram construction and normalization. Set to False if you would like to call a process_trajectory routine again afterwards. Defaults to True
            :type finish: bool, optional

            :param verbose: set to True to increase verbosity, defaults to False
            :type verbose: bool, optional
        '''
        cv_reader = CVComputer([CV], name=self.cv_labels[0], start=sub.start, stride=sub.step, end=sub.stop, verbose=verbose)
        q_reader = CVComputer([Q], name=self.q_labels[0], start=sub.start, stride=sub.step, end=sub.stop, verbose=verbose)
        ConditionalProbability.process_trajectory(self, fns, [q_reader], [cv_reader], finish=finish, verbose=verbose)

    def process_trajectory_cvs(self, fns, col_q=1, col_cv=2, sub=slice(None,None,None), unit_q='au', unit_cv='au', finish=True, verbose=False):
        '''
            Included for backwards compatibility, this routine will call the more general process_trajectory routine of the parent class.

            Extract Q and CV samples from the given COLVAR trajectories. These samples will later be utilized by the finish routine to construct the conditional probability. The trajectory files may also contain data from simulations that are biased in CV space (not Q space!!). Each CV trajectory file contains rows of the form

                time q cv

            If the trajectory file contains this data in a different order, it can be accounted for using the col_xx keyword arguments.

            :param fns: file name (or list of file names) of colvar files with the above formatting containing the trajectory data.
            :type fns: str or list(str)

            :param col_q: column index of the collective variable Q in the given input file, defaults to 1
            :type col_q: int, optional

            :param col_cv: column index of the collective variable CV in the given input file, defaults to 2
            :type col_cv: int, optional

            :param unit_q: unit in which the q values are stored in the file, defaults to 'au'
            :type unit_q: str, optional

            :param unit_cv: unit in which the cv values are stored in the file, defaults to 'au'
            :type unit_cv: str, optional
            
            :param sub: python slice instance to subsample the trajectory, defaults to slice(None, None, None)
            :type sub: slice, optional

            :param finish: set to True if the given file name(s) are the only relevant trajectories and hence the conditional probability should be computed from only these trajectories. Setting it to True will trigger histogram construction and normalization. Set to False if you would like to call a process_trajectory routine again afterwards. Defaults to True
            :type finish: bool, optional

            :param verbose: set to True to increase verbosity, defaults to False
            :type verbose: bool, optional
        '''
        q_reader  = ColVarReader([col_q] , units=[unit_q] , name='Q' , start=sub.start, stride=sub.step, end=sub.stop, verbose=verbose)
        cv_reader = ColVarReader([col_cv], units=[unit_cv], name='CV', start=sub.start, stride=sub.step, end=sub.stop, verbose=verbose)
        ConditionalProbability.process_trajectory(self, fns, [q_reader], [cv_reader], finish=finish, verbose=verbose)

    def average(self, error_estimate='propdist'):
        '''
            Compute the average of CV as function of Q

            :param q_output_unit: unit in which the q values are plotted, defaults to 'au'
            :type q_output_unit: str, optional

            :param q_label: the label of the collective variable Q to be used in the plot, defaults to 'Q'
            :type q_label: str, optional

            :param cv_output_unit: unit in which the cv values are plotted, defaults to 'au'
            :type cv_output_unit: str, optional

            :param cv_label: the label of the collective variable CV to be used in the plot, defaults to 'CV'
            :type cv_label: str, optional

            :param error_estimate: Specify the method of error propagation, either by propagating the FES distribution samples (propdist) or by propagating the FES 2 sigma confidence interval (prop2sigma), defaults to propdist
            :type error_estimate: str, optional
        '''
        def function(ps):
            qs = np.zeros(len(self.cvs[0]), dtype=float)
            norm = np.zeros(len(self.cvs[0]), dtype=float)
            for iq, q in enumerate(self.qs[0]):
                mask = ~np.isnan(ps[iq,:])
                qs[mask]   += q*ps[iq,mask]
                norm[mask] +=   ps[iq,mask]
            qs[norm==0] = np.nan
            #normalization in case ps would not be normalized
            qs[norm>0] /= norm[norm>0]
            return qs
        xs = function(self.pconds)
        error = None
        if self.error is not None:
            if error_estimate.lower()=='propdist':
                propagator = Propagator(ncycles=self.error.ncycles, target_distribution=GaussianDistribution)
                error = propagator(function, self.error)
            elif error_estimate.lower()=='prop2sigma':
                plower, pupper = self.error.nsigma_conf_int(nsigma=2)
                xlower = function(plower)
                xupper = function(pupper)
                xerr = abs(xupper-xlower)/4
                error = GaussianDistribution(xs, xerr)
            else:
                raise NotImplementedError('Unsupported method for error estimation, received %s but should be propdist or prop2sigma.')
        return BaseProfile(self.cvs[0], xs, error=error, cv_output_unit=self.cv_units[0], f_output_unit=self.q_units[0], cv_label=self.cv_labels[0], f_label=self.q_labels[0])

    def transform(self, fep, q_output_unit='au', f_output_unit=None, q_label=None, f_output_class=BaseFreeEnergyProfile):
        '''
            Transform the provided 1D FES to a different 1D FES using the current conditional probability according to the formula. 

            .. math:: F(q) &= -kT \\ln\\left(\\int p(q|v)\\cdot e^{-\\beta F(v)} dv\\right)
            
            :param fep: the input free energy profile F(cv) which will be transformed towards F(q)
            :type fep: BaseFreeEnergyProfile or child classes

            :param q_output_unit: the unit of the new collective variable Q to be used in plotting and printing, defaults to 'au'
            :type q_output_unit: str, optional

            :param f_output_unit: the unit of the the transformed free energy profile to be used in plotting and printing of energies. By default, the f_output_unit of the given free energy profile will be used.
            :type f_output_unit: str, optional

            :param f_output_class: the class of the output free energy profile, defaults to BaseFreeEnergyProfile
            :type f_output_class: class, optional

            :param q_label: the label of the new collective variable Q to be used in plot labels. This argument is deprecated, the q_labels[0] as defined upon initializing the class instance is used by default.
            :type cv_label: str, optional
        '''
        #consistency checks and initalization
        qs = self.qs[0]
        cvs = self.cvs[0]
        assert self._finished, "Conditional probability needs to be finished before applying at in transformations."
        assert isinstance(fep, BaseFreeEnergyProfile), 'Input argument should be instance of (child of) BaseFreeEnergyProfile, instead received %s' %fep.__class__.__name__
        assert len(fep.cvs)==len(cvs), 'Dimension of 1D CV in conditional probability inconsistent with 1D FEP'
        assert (abs(fep.cvs-cvs)<1e-6).all(), 'Values of 1D CV in conditional probability not identical to those of 1D FEP'
        if q_label is None: q_label = self.q_labels[0]
        if f_output_unit is None: f_output_unit = fep.f_output_unit
        # Construct 1D FEP
        def transform(fs, pconds):
            mask = ~np.isnan(fs)
            ps = np.trapz(pconds[:,mask]*np.exp(-fep.beta*fs[mask]), x=cvs[mask])
            ps /= np.trapz(ps, x=qs)
            fs_new = np.zeros([len(qs)], float)*np.nan
            fs_new[ps>0] = -np.log(ps[ps>0])/fep.beta
            return fs_new
        fs = transform(fep.fs, self.pconds)
        error = None
        if fep.error is not None:
            propagator = Propagator(ncycles=fep.error.ncycles, target_distribution=GaussianDistribution)
            if self.error is not None:
                error = propagator(transform, fep.error, self.error)
            else:
                transf1 = lambda fs: transform(fs, self.pconds)
                error = propagator(transf1, fep.error)
        elif self.error is not None:
            propagator = Propagator(ncycles=self.error.ncycles, target_distribution=LogGaussianDistribution)
            transf2 = lambda pconds: transform(fep.fs, pconds)
            error = propagator(transf2, self.error)
        return f_output_class(self.qs[0], fs, fep.T, error=error, cv_output_unit=q_output_unit, f_output_unit=f_output_unit, cv_label=q_label)

    def deproject(self, fep, cv_output_unit='au', q_output_unit='au', f_output_unit=None, f_output_class=FreeEnergySurface2D, cv_label='CV', q_label='Q'):
        '''
            Deproject the provided 1D FEP F(q) to a 2D FES F(q,v) using the current conditional probability according to the formula

            .. math:: F(q_1,q_2) &= F(q_2)-kT \\ln\\left(p(q_1|q_2)\\right)

            :param fep: the free energy profile F(q_2) which will be transformed
            :type fep: (child of) BaseFreeEnergyProfile

            :param q_output_unit: the unit of the new additional collective variable Q to be used in plotting and printing, defaults to 'au'
            :type q_output_unit: str, optional

            :param cv_output_unit: the unit of the original collective variable CV to be used in plotting and printing, defaults to 'au'
            :type q2_output_unit: str, optional

            :param f_output_unit: the unit of the the transformed free energy profile to be used in plotting and printing of energies. If set to None, the f_output_unit of the given free energy profile will be use. Defaults to None
            :type f_output_unit: str, optional

            :param f_output_class: the class of the output free energy profile, defaults to FreeEnergySurface2D
            :type f_output_class: class, optional

            :param q_label: the label of the new additional collective variable Q to be used in plot labels. This argument is deprecated, the q_labels[0] as defined upon initializing the class instance is used by default.
            :type q_label: str, optional

            :param cv_label: the label of the oribinal collective variable CV to be used in plot labels. This argument is deprecated, the cv_labels[0] as defined upon initializing the class instance is used by default.
            :type cv_label: str, optional
        '''
        #consistency checks and initalization
        qs, cvs = self.qs[0], self.cvs[0]
        assert self._finished, "Conditional probability needs to be finished before applying at in transformations."
        assert isinstance(fep, BaseFreeEnergyProfile), 'Input argument should be instance of (child of) BaseFreeEnergyProfile, instead received %s' %fep.__class__.__name__
        assert len(fep.cvs)==len(cvs), 'Dimension of collective variable v in conditional probability p(q|v) inconsistent with collective variable in 1D FEP'
        assert (abs(fep.cvs-cvs)<1e-6).all(), 'Values of collective variable v in conditional probability p(q|v) not identical to those of collective variable in 1D FEP'
        if q_label is None: q_label = self.q_labels[0]
        if cv_label is None: cv_label = self.cv_labels[0]
        if f_output_unit is None: f_output_unit = fep.f_output_unit
        # Construct 2D FES
        kT = boltzmann*fep.T
        def deproject(fs, pconds):
            fs_new = np.zeros([len(qs),len(cvs)], float)*np.nan
            for iq in range(len(qs)):
                for icv in range(len(cvs)):
                    if self.pconds[iq,icv]>0:
                        fs_new[iq,icv] = fs[icv] - kT*np.log(pconds[iq,icv])
            return fs_new
        fs = deproject(fep.fs, self.pconds)
        error = None
        if fep.error is not None:
            propagator = Propagator(ncycles=fep.error.ncycles, target_distribution=GaussianDistribution)
            if self.error is not None:
                error = propagator(deproject, fep.error, self.error)
            else:
                deproj1 = lambda fs: deproject(fs, self.pconds)
                error = propagator(deproj1, fep.error)
        elif self.error is not None:
            propagator = Propagator(ncycles=self.error.ncycles, target_distribution=LogGaussianDistribution)
            deproj2 = lambda pconds: deproject(fep.fs, pconds)
            error = propagator(deproj2, self.error)
        return f_output_class(cvs, qs, fs, fep.T, error=error, cv1_output_unit=cv_output_unit, cv2_output_unit=q_output_unit, f_output_unit=f_output_unit, cv1_label=cv_label, cv2_label=q_label)




class ConditionalProbability1D2D(ConditionalProbability):
    '''
        Class to store and compute conditional probabilities of the form :math:`p(q1,q2|cv)` which can be used to transform a 1D free energy profile in terms of the collective variable *cv* towards a 2D free energy surface in terms of the collective variables :math:`q_{1}` and :math:`q_{2}`.

    '''
    def __init__(self, q1_bins=None, q2_bins=None, cv_bins=None, q1_label='Q1', q2_label='Q2', cv_label='CV', verbose=False):
        '''
            :param q1_bins: np.histogram argument for defining the bins of Q1 samples
            :type q1_bins: see np.histogram and np.histogram2d, optional
            .
            :param q2_bins: np.histogram argument for defining the bins of Q2 samples
            :type q2_bins: see np.histogram and np.histogram2d, optional

            :param cv_bins: np.histogram argument for defining the bins of CV samples
            :type cv_bins: see np.histogram and np.histogram2d, optional

            :param q1_label: label for Q1 used for plotting/logging, defaults to 'Q1'
            :type q1_label: str, optional

            :param q2_label: label for Q2 used for plotting/logging, defaults to 'Q2'
            :type q2_label: str, optional
            
            :param cv_label: label for Q used for plotting/logging, defaults to 'CV'
            :type cv_label: str, optional
        '''
        ConditionalProbability.__init__(self, 2, 1, q_bins=[q1_bins,q2_bins], cv_bins=cv_bins, q_labels=[q1_label,q2_label], cv_labels=[cv_label], verbose=verbose)

    def process_trajectory_xyz(self, fns, Q1, Q2, CV, sub=slice(None,None,None), finish=True):
        '''
            Included for backwards compatibility, this routine will call the more general process_trajectory routine of the parent class.

            Extract Q1, Q2 and CV samples from the given XYZ trajectories. These samples will later be utilized by the finish routine to construct the conditional probability. The trajectory files may also contain data from simulations that are biased in CV space (not Q1/Q2 space!!).

            :param fns: file name (or list of file names) which contain trajectories that are used to compute the conditional probability.
            :type fns: str or list(str)

            :param Q1: collective variable used to compute the Q1 value from a trajectory file
            :type Q1: CollectiveVariable

            :param Q2: collective variable used to compute the Q2 value from a trajectory file
            :type Q2: CollectiveVariable

            :param CV: collective variable used to compute the CV value from a trajectory file
            :type CV: CollectiveVariable

            :param sub: python slice instance to subsample the trajectory, defaults to slice(None, None, None)
            :type sub: slice, optional

            :param finish: set to True if the given file name(s) are the only relevant trajectories and hence the conditional probability should be computed from only these trajectories. Setting it to True will trigger histogram construction and normalization. Set to False if you would like to call a process_trajectory routine again afterwards. Defaults to True
            :type finish: bool, optional

            :param verbose: set to True to increase verbosity, defaults to False
            :type verbose: bool, optional
        '''
        cv_reader = CVComputer([CV], name=self.cv_labels[0], start=sub.start, stride=sub.step, end=sub.stop, verbose=verbose)
        q1_reader = CVComputer([Q1], name=self.q_labels[1], start=sub.start, stride=sub.step, end=sub.stop, verbose=verbose)
        q2_reader = CVComputer([Q2], name=self.q_labels[2], start=sub.start, stride=sub.step, end=sub.stop, verbose=verbose)
        ConditionalProbability.process_trajectory(self, fns, [q1_reader,q2_reader], [cv_reader], finish=finish, verbose=verbose)

    def process_trajectory_cvs(self, fns, col_q1=1, col_q2=2, col_cv=3, unit_q1='au', unit_q2='au', unit_cv='au', sub=slice(None,None,None), finish=True, verbose=False):
        '''
            Included for backwards compatibility, this routine will call the more general process_trajectory routine of the parent class.

            Extract Q and CV samples from the given COLVAR trajectories. These samples will later be utilized by the finish routine to construct the conditional probability. The trajectory files may also contain data from simulations that are biased in CV space (not Q space!!). Each CV trajectory file contains rows of the form

                time q1 q2 cv

            If the trajectory file contains this data in a different order, it can be accounted for using the col_xx keyword arguments.

            :param fns: file name (or list of file names) of colvar files with the above formatting containing the trajectory data.
            :type fns: str or list(str)

            :param col_q1: column index of the collective variable Q1 in the given input file, defaults to 1
            :type col_q1: int, optional

            :param col_q2: column index of the collective variable Q2 in the given input file, defaults to 2
            :type col_q2: int, optional

            :param col_cv: column index of the collective variable CV in the given input file, defaults to 3
            :type col_cv: int, optional

            :param unit_q1: unit in which the q1 values are stored in the file, defaults to 'au'
            :type unit_q1: str, optional

            :param unit_q2: unit in which the q2 values are stored in the file, defaults to 'au'
            :type unit_q2: str, optional

            :param unit_cv: unit in which the cv values are stored in the file, defaults to 'au'
            :type unit_cv: str, optional
            
            :param sub: python slice instance to subsample the trajectory, defaults to slice(None, None, None)
            :type sub: slice, optional

            :param finish: set to True if the given file name(s) are the only relevant trajectories and hence the conditional probability should be computed from only these trajectories. Setting it to True will trigger histogram construction and normalization. Set to False if you would like to call a process_trajectory routine again afterwards. Defaults to True
            :type finish: bool, optional

            :param verbose: set to True to increase verbosity, defaults to False
            :type verbose: bool, optional
        '''
        q1_reader  = ColVarReader([col_q1] , units=[unit_q1] , name='Q1' , start=sub.start, stride=sub.step, end=sub.stop, verbose=verbose)
        q2_reader  = ColVarReader([col_q2] , units=[unit_q2] , name='Q2' , start=sub.start, stride=sub.step, end=sub.stop, verbose=verbose)
        cv_reader = ColVarReader([col_cv], units=[unit_cv], name='CV', start=sub.start, stride=sub.step, end=sub.stop, verbose=verbose)
        ConditionalProbability.process_trajectory(self, fns, [q1_reader,q2_reader], [cv_reader], finish=finish, verbose=verbose)

    def transform(self, fep, q1_output_unit='au', q2_output_unit='au', q1_label=None, q2_label=None, f_output_unit='kjmol'):
        '''
            Transform the provided 1D FEP to a 2D FES using the current conditional probability according to the formula

            .. math:: F(q_1,q_2) &= -kT\cdot\\ln\\left(\\int p(q_1,q_2|v)\\cdot e^{-\\beta F(v)}dv\\right)

            :param fep: the input free energy profile F(cv) which will be transformed towards F(q)
            :type fep: BaseFreeEnergyProfile or child classes

            :param q1_output_unit: the unit of the new collective variable Q1 to be used in plotting and printing, defaults to 'au'
            :type q1_output_unit: str, optional

            :param q2_output_unit: the unit of the new collective variable Q2 to be used in plotting and printing, defaults to 'au'
            :type q2_output_unit: str, optional

            :param f_output_unit: the unit of the the transformed free energy profile to be used in plotting and printing of energies. By default, the f_output_unit of the given free energy profile will be used.
            :type f_output_unit: str, optional

            :param f_output_class: the class of the output free energy profile, defaults to BaseFreeEnergyProfile
            :type f_output_class: class, optional

            :param q1_label: the label of the new collective variable Q1 to be used in plot labels. This argument is deprecated, the q_labels[0] as defined upon initializing the class instance is used by default.
            :type q1_label: str, optional

            :param q2_label: the label of the new collective variable Q2 to be used in plot labels. This argument is deprecated, the q_labels[1] as defined upon initializing the class instance is used by default.
            :type q2_label: str, optional
        '''
        #consistency checks and initalization
        assert self._finished, "Conditional probability needs to be finished before applying at in transformations."
        assert isinstance(fep, BaseFreeEnergyProfile), 'Input argument should be instance of (child of) BaseFreeEnergyProfile, instead received %s' %fep.__class__.__name__
        assert len(fep.cvs)==len(self.cvs[0]), 'Dimension of 1D CV in conditional probability inconsistent with 1D FEP'
        assert (abs(fep.cvs-self.cvs[0])<1e-6).all(), 'Values of 1D CV in conditional probability not identical to those of 1D FEP'
        if q1_label is None: q1_label = self.q_labels[0]
        if q2_label is None: q2_label = self.q_labels[1]
        if f_output_unit is None: f_output_unit = fep.f_output_unit
        #construct 2D FES
        def transform(fs, pconds):
            ps = np.zeros([self.q2num, self.q1num], float)
            for icv, fcv in enumerate(fs):
                if not np.isnan(fcv):
                    ps += pconds[:,:,icv]*np.exp(-fep.beta*fcv)
            ps /= integrate2d(ps, x=self.qs[0], y=self.qs[1])
            fs_new = np.zeros([self.q2num, self.q1num], float)*np.nan
            fs_new[ps>0] = -np.log(ps[ps>0])/fep.beta
            return fs_new
        fs = transform(fep.fs, self.pconds)
        error = None
        if fep.error is not None:
            propagator = Propagator(ncycles=fep.error.ncycles, target_distribution=GaussianDistribution)
            if self.error is not None:
                error = propagator(transform, fep.error, self.error)
            else:
                transf1 = lambda fs: transform(fs, self.pconds)
                error = propagator(transf1, fep.error)
        elif self.error is not None:
            propagator = Propagator(ncycles=self.error.ncycles, target_distribution=LogGaussianDistribution)
            transf2 = lambda pconds: transform(fep.fs, pconds)
            error = propagator(transf2, self.error)
        return FreeEnergySurface2D(self.qs[0], self.qs[1], fs, fep.T, error=error, cv1_output_unit=q1_output_unit, cv2_output_unit=q2_output_unit, f_output_unit=f_output_unit, cv1_label=q1_label, cv2_label=q2_label)
