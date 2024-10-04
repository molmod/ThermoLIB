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

from __future__ import annotations

from molmod.units import *
from molmod.constants import *

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from thermolib.tools import integrate2d, format_scientific
from thermolib.thermodynamics.state import *
from thermolib.error import *
from thermolib.flatten import DummyFlattener, Flattener

import matplotlib.pyplot as pp
import matplotlib.cm as cm
from matplotlib import gridspec, rc


from sklearn.cluster import DBSCAN #clustering algorithm


__all__ = ['BaseProfile', 'BaseFreeEnergyProfile', 'SimpleFreeEnergyProfile', 'FreeEnergySurface2D', 'plot_profiles']


class BaseProfile(object):
    '''
        Base parent class to define a 1D profile of a property X as function of a certain collective variable (CV). This class will be used as the basis for (free) energy profiles.
    '''
    def __init__(self, cvs, fs, error=None, cv_output_unit='au', f_output_unit='au', cv_label='CV', f_label='X'):
        """
            :param cvs: the collective variable values, which should be in atomic units! If you need help properly converting to atomic units, we refer to the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type cvs: np.ndarray

            :param fs: the values of the property X, which should be in atomic units! If you need help properly converting to atomic units, we refer to the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type fs: np.ndarray

            :param error: error distribution on the profile, defaults to None
            :type error: child class of :py:class:`Distribution <thermolib.error.Distribution>` class, optional

            :param cv_output_unit: the units for printing and plotting of CV values (not the unit of the input array, that is assumed to be in atomic units). Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type cv_output_unit: str, default='au'
                
            :param f_output_unit: the units for printing and plotting of free energy values (not the unit of the input array, that is assumed to be in atomic units). Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type f_output_unit: str, default='kjmol'

            :param cv_label: label for the CV for printing and plotting
            :type cv_label: str, optional, default='CV'

            :param f_label: label for the observable X for printing and plotting
            :type f_label: str, optional, default='X'
        """
        assert len(cvs)==len(fs), "cvs and fs array should be of same length"
        self.cvs = cvs.copy()
        self.fs = fs.copy()
        self.error = error
        self.cv_output_unit = cv_output_unit
        self.f_output_unit = f_output_unit
        self.cv_label = cv_label
        self.f_label = f_label

    def flower(self, nsigma=2):
        '''
            Return the lower limit of an n-sigma error bar on the profile property, i.e. :math:`\\mu - n\\sigma` with :math:`\\mu` the mean and :math:`\\sigma` the standard deviation.

            :param nsigma: defines the n-sigma error bar
            :type nsigma: int, optional, default=2

            :return: the lower limit of the n-sigma error bar
            :rtype: np.ndarray with dimensions determined by self.error

            :raises AssertionError: if self.error is not defined.
        '''
        assert self.error is not None, 'Flower cannot be computed because no error distribution was defined in the error attribute'
        flower, fupper = self.error.nsigma_conf_int(nsigma)
        return flower

    def fupper(self, nsigma=2):
        '''
            Return the upper limit of an n-sigma error bar on the profile property, i.e. :math:`\\mu + n\\sigma` with :math:`\\mu` the mean and :math:`\\sigma` the standard deviation.

            :param nsigma: defines the n-sigma error bar
            :type nsigma: int, optional, default=2
            
            :return: the upper limit of the n-sigma error bar
            :rtype: np.ndarray with dimensions determined by self.error

            :raises AssertionError: if self.error is not defined.
        '''
        assert self.error is not None, 'Fupper cannot be computed because no error distribution was defined in the error attribute'
        flower, fupper = self.error.nsigma_conf_int(nsigma)
        return fupper

    @classmethod
    def from_txt(cls, fn, cvcol=0, fcol=1, fstdcol=None, cv_input_unit='au', f_input_unit='kjmol', cv_output_unit='au', f_output_unit='kjmol', cv_label='CV', f_label='X', cvrange=None, delimiter=None, reverse=False, cut_constant=False):
        '''
            Read the a property profile (and optionally its error bar) as function of a collective variable from a txt file.

            :param fn: the name of the txt file (assumed to be readable by numpy.loadtxt) containing the data
            :type fn: str

            :param cvcol: index of the column in which the collective variable is stored
            :type cvcol: int, default=0

            :param fcol: index of the column in which the observable X is stored.
            :type fcol: int, default=1

            :param fstdcol: index of the column in which the standard deviation of observable X is stored, which is used to construct an error distribution. If None, no standard deviation will be read and no error bar will be computed.
            :type fstdcol: int, default=None

            :param cv_input_unit: the units in which the CV values are stored in the file. Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type cv_input_unit: str or float, default='au'

            :param f_input_unit: the units in which the observable X values are stored in the file. Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type f_input_unit: str or float, default='kjmol'

            :param cv_output_unit: the units for printing and plotting of CV values (not the unit of the input array, that is defined by cv_input_unit). Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type cv_output_unit: str or float, default='au'

            :param f_output_unit: the units for printing and plotting of observable X values (not the unit of the input array, that is defined by x_input_unit). Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type f_output_unit: str or float, default='kjmol'

            :param cv_label: label for the CV for printing and plotting
            :type cv_label: str, optional, default='CV'

            :param f_label: label for the property X for printing and plotting
            :type f_label: str, optional, default='X'

            :param cvrange: only read the property X for CVs in the given range. If None, all data is read
            :type cvrange: tuple or list, default=None
            
            :param delimiter: The delimiter used in the txt input file to separate columns. If None, use the default of `the numpy.loadtxt routine <https://numpy.org/doc/1.20/reference/generated/numpy.loadtxt.html>`_ (i.e. whitespace).
            :type delimiter: str, optional, default=None
            
            :param reverse: if set to True, reverse the CV and X values (usefull to make sure reactant is on the left)
            :type reverse: bool, optional, default=False

            :param cut_constant: if set to True, the data points at the start and end of the data array that are constant will be cut. Usefull to cut out unsampled areas for large and small CV values. 
            :type cut_constant: bool, optional, default=False
        '''
        #TODO: deal with inf values in input file
        data = np.loadtxt(fn, delimiter=delimiter, dtype=float)
        cvs = data[:,cvcol]*parse_unit(cv_input_unit)
        fs = data[:,fcol]*parse_unit(f_input_unit)
        error = None
        if fstdcol is not None:
            ferr = data[:,fstdcol]*parse_unit(f_input_unit)
            error = GaussianDistribution(fs, ferr)
        if reverse:
            cvs = cvs[::-1]
            fs = fs[::-1]
            if error is not None:
                error.means = error.means[::-1]
                error.stds = error.stds[::-1]
        if cvrange is not None:
            indexes = []
            for i, cv in enumerate(cvs):
                if cvrange[0]<=cv and cv<=cvrange[1]:
                    indexes.append(i)
            cvs = cvs[np.array(indexes)]
            fs = fs[np.array(indexes)]
            if error is not None:
                error.means = error.means[np.array(indexes)]
                error.stds = error.stds[np.array(indexes)]
        if cut_constant:
            mask = np.ones(len(fs), bool)
            for i in range(len(fs)):
                if fs[i]==fs[0]:
                    mask[i] = False
                else:
                    break
            for j in range(len(fs))[::-1]:
                if fs[j]==fs[-1]:
                    mask[j] = False
                else:
                    break
            cvs = cvs[mask]
            fs = fs[mask]
            if error is not None:
                error.means = error.means[mask]
                error.stds = error.stds[mask]
        return cls(cvs, fs, error=error, cv_output_unit=cv_output_unit, cv_label=cv_label, f_output_unit=f_output_unit, f_label=f_label)

    def savetxt(self, fn_txt):
        '''
            Save the current profile as txt file. The values of CV and X are written in units specified by the cv_output_unit and f_output_unit attributes of the self instance. If the error attribute of self is not None, the corresponding std values will be written to the file as well in the third column.

            :param fn_txt: name for the output file
            :type fn_txt: str
        '''
        
        if self.error is None:
            header = '%s [%s]\t%s [%s]' %(self.cv_label, self.cv_output_unit, self.f_label, self.f_output_unit)
            data = np.vstack((self.cvs/parse_unit(self.cv_output_unit), self.fs/parse_unit(self.f_output_unit))).T
        else:
            header = '%s [%s]\t%s [%s]\t error [%s]' %(self.cv_label, self.cv_output_unit, self.f_label, self.f_output_unit, self.f_output_unit)
            data = np.vstack((self.cvs/parse_unit(self.cv_output_unit), self.fs/parse_unit(self.f_output_unit), self.error.std()/parse_unit(self.f_output_unit))).T
        np.savetxt(fn_txt, data, header=header)

    @classmethod
    def from_average(cls, profiles, cv_output_unit=None, cv_label=None, f_output_unit=None, f_label=None, error_estimate=None):
        '''
            Construct a profile as the average of a set of given profiles. 

            :param profiles: set of profiles to be averaged
            :type profiles: list of instances of :py:class:`BaseProfile <thermolib.thermodynamics.fep.BaseProfile>` (or one of its child classes such as :py:class:`BaseFreeEnergyProfile <thermolib.thermodynamics.fep.BaseFreeEnergyProfile>` or :py:class:`SimpleFreeEnergyProfile <thermolib.thermodynamics.fep.SimpleFreeEnergyProfile>`) 

            :param cv_output_unit: the units for printing and plotting of CV values. If None, use the value of the ``cv_output_unit`` attribute of the first profile. Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type cv_output_unit: str or float, optional, default=None

            :param cv_label: label for the collective variable in plots. If None, use the value of the ``cv_label`` attribute of the first profile.
            :type cv_label: str, optional, default=None

            :param f_output_unit: the units for printing and plotting of X values (not the unit of the input array, that is defined by x_input_unit). If None, use the value of the ``f_output_unit`` attribute of the first profile. Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type f_output_unit: str or float, optional, default=None.

            :param f_label: label for the X observable in plots. If None, use the value of the ``f_label`` attribute of the first profile.
            :type f_label: str, optional, default=None

            :param error_estimate: method of estimating the error. Currently, only *std* is supported, which computes the error from the standard deviation within the set of profiles.            
            :type error_estimate: str, optional, default=None
        '''
        #sanity checks
        cv_label = None
        cvs = None
        for prof in profiles:
            if cvs is None:
                cvs = prof.cvs.copy()
            else:
                assert (abs(cvs-prof.cvs)<1e-12*np.abs(cvs.mean())).all(), 'Cannot average the histograms as they do not have consistent CV grids'
            if cv_output_unit is None:
                cv_output_unit = prof.cv_output_unit
            if f_output_unit is None:
                f_output_unit = prof.f_output_unit
            if cv_label is None:
                cv_label = prof.cv_label
            else:
                assert cv_label==prof.cv_label, 'Inconsistent CV label definition in profiles (%s!=%s)' %(cv_label,prof.cv_label)
            if f_label is None:
                f_label = prof.f_label
        #collect free energy value
        fss = np.array([prof.fs for prof in profiles])
        #average histograms
        fs = fss.mean(axis=0)
        #compute error if requested
        error = None
        if error_estimate is not None and error_estimate.lower() in ['std']:
            ferr = fss.std(axis=0,ddof=1)/np.sqrt(len(profiles))
            error = GaussianDistribution(fs, ferr)
        if cls==BaseProfile:
            return cls(cvs, fs, error=error, cv_output_unit=cv_output_unit, cv_label=cv_label, f_output_unit=f_output_unit, f_label=f_label)
        elif cls in [BaseFreeEnergyProfile, SimpleFreeEnergyProfile]:
            temps = np.array([prof.T for prof in profiles])
            assert temps.std()/temps.mean()<1e-6, 'Cannot average free energy profiles because they do not have the same temperature!'
            return cls(cvs, fs, temps.mean(), error=error, cv_output_unit=cv_output_unit, cv_label=cv_label, f_output_unit=f_output_unit, f_label=f_label)
        else:
            raise NotImplementedError('routine from_average not implemented for class %s' %str(cls))

    def set_ref(self, ref='min'):
        '''
            Set the zero energy reference of the profile.

            :param ref: the choice for the zero reference of X. Currently only 'min', 'max' or an integer is implemented, resulting in setting the value of the minimum, maximum or X[i] to zero respectively.
            :type ref: str or int, optional, default='min'
        
            :raises IOError: invalid value for keyword argument ref is given. See doc above for choices.
        '''
        #Find index of reference state
        if isinstance(ref, int):
            ref = self.fs[ref]
        elif isinstance(ref, float):
            pass #in this case, ref itself is the profile y value that has to shifted to zero
        elif ref.lower() in ['min']:
            global_minimum = Minimum('glob_min' , cv_unit=self.cv_output_unit, f_unit=self.f_output_unit)
            global_minimum.compute(self, dist_prop='stationary')
            ref = global_minimum.get_F()
        elif ref.lower() in ['max']:
            global_maximum = Maximum('glob_min', cv_unit=self.cv_output_unit, f_unit=self.f_output_unit)
            global_maximum.compute(self, dist_prop='stationary')
            ref = global_maximum.get_F()
        else:
            raise IOError('Invalid REF specification, recieved %s. Check documentation for allowed options.' %ref)
        #Set ref
        self.fs -= ref
        if self.error is not None:
            self.error.set_ref(value=ref)

    def crop(self, cvrange):
        '''
            Crop the profile to the given cvrange and throw away cropped data. This routine will alter the data in the current profile.

            :param cvrange: the range of the collective variable defining the new range to which the FEP will be cropped.
            :type cvrange: tuple
        '''
        #cut off some unwanted regions
        cvs = self.cvs.copy()
        fs = self.fs.copy()
        if cvrange is not None:
            indexes = []
            for i, cv in enumerate(cvs):
                if cvrange[0]<=cv<=cvrange[1]:
                    indexes.append(i)
            cvs = cvs[np.array(indexes)]
            fs = fs[:,np.array(indexes)]

        self.cvs = cvs.copy()
        self.fs = fs.copy()

    def plot(self, fn: str|None=None, obss: list=['value'], linestyles: list|None=None, linewidths: list|None=None, colors: list|None=None, cvlims: list|None=None, flims: list|None=None, show_legend: bool=False, **plot_kwargs):
        '''
            Plot the property stored in the current profile as function of the CV. If the error distribution is stored in self.error, various statistical quantities besides the estimated mean property (such as the error width, lower/upper limit on the error bar, random sample) can be plotted using the ``obss`` keyword. You can specify additional matplotlib keyword arguments that will be parsed to the matplotlib plotter (`plot` and/or `fill_between`) at the end of the argument list of this routine.

            :param fn: name of a file to save plot to. If None, the plot will not be saved to a file.
            :type fn: str, optional, default=None

            :param obss: Specify which statistical property/properties to plot. Multiple values are allowed, which will be plotted on the same figure. Following options are supported:

                - **value** - the values stored in self.fs
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

            :param flims: limits to the plotting range of the X property. If None, no limits are enforced
            :type flims: list of strings or None, optional, default=None

            :param show_legend: If True, the legend is shown in the plot
            :type show_legend: bool, optional, default=None
            
            :raises ValueError: if the ``obs`` argument contains an invalid specification
        '''
        #preprocess
        cvunit = parse_unit(self.cv_output_unit)
        funit = parse_unit(self.f_output_unit)
        
        #read data 
        data = []
        labels = []
        for obs in obss:
            values = None
            if obs.lower() in ['value']:
                values = self.fs.copy()
            elif obs.lower() in ['mean']:
                assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
                values = self.error.mean()
            elif obs.lower() in ['lower']:
                assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
                values = self.error.nsigma_conf_int(2)[0]
            elif obs.lower() in ['upper']:
                assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
                values = self.error.nsigma_conf_int(2)[1]
            elif obs.lower() in ['error']:
                assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
                lower = self.error.nsigma_conf_int(2)[0]
                upper = self.error.nsigma_conf_int(2)[1]
                values = 0.5*np.abs(upper - lower)
            elif obs.lower() in ['sample']:
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
            pp.plot(self.cvs/cvunit, ys/funit, label=label, linestyle=linestyle, linewidth=linewidth, color=color, **plot_kwargs)
        if self.error is not None:
            pp.fill_between(self.cvs/cvunit, lower/funit, upper/funit, **plot_kwargs, alpha=0.33)
        pp.xlabel('%s [%s]' %(self.cv_label, self.cv_output_unit), fontsize=16)
        pp.ylabel('%s [%s]' %(self.f_label, self.f_output_unit), fontsize=16)
        if cvlims is not None: pp.xlim(cvlims)
        if flims is not None: pp.ylim(flims)
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

    def plot_corr_matrix(self, fn: str|None=None, flims: list|None=None, cmap: str='bwr', **plot_kwargs):
        '''
            Make a 2D filled contour plot of the correlation matrix (i.e. the normalized covariance matrix) containing the correlation between every pair of points on the X profile. This correlation ranges from +1 (red when cmap='bwr') over 0 (white when cmap='bwr') to -1 (blue when cmap='bwr'). For easy interpretation, the plot of the X profile itself is included above the correlation plot (top pane).

            :param fn: name of file to save the plot to. If None, the plot is not saved
            :type fn: str or None, optional, default=None

            :param flims: Limit the plotting range of the X property in the original profile plot included in the top pane.
            :type flims: list or None, optional, default=None

            :param cmap: Specification of the matplotlib color map for the 2D plot of the correlation matrix
            :type cmap: str, optional, default='bwr'

            :raises ValueError: when self.error is not specified (as this is needed to compute the correlation matrix)
        '''
        if self.error is None:
            raise ValueError('Can only plot correlation matrix if error estimation has been performed!')
        cvunit = parse_unit(self.cv_output_unit)
        funit = parse_unit(self.f_output_unit)
        pp.clf()
        fig, axs = pp.subplots(nrows=2, ncols=1, sharex=True)
        #plot free energy profile
        axs[0].plot(self.cvs/cvunit, self.fs/funit, **plot_kwargs)
        lower, upper = self.error.nsigma_conf_int(2)
        axs[0].fill_between(self.cvs/cvunit, lower/funit, upper/funit, **plot_kwargs, alpha=0.33)
        axs[0].set_ylabel('%s [%s]' %(self.f_label, self.f_output_unit), fontsize=16)
        if flims is not None: axs[0].set_ylim(flims)
        #plot correlation matrix
        data = self.error.corr(unflatten=False)
        axs[1].imshow(data, cmap=cmap, extent=[self.cvs[0]/cvunit, self.cvs[-1]/cvunit, self.cvs[-1]/cvunit, self.cvs[0]/cvunit], aspect='auto')
        axs[1].set_xlabel('%s [%s]' %(self.cv_label, self.cv_output_unit), fontsize=16)
        fig = pp.gcf()
        fig.tight_layout(h_pad=0.1)
        fig.set_size_inches([8,16])
        if fn is not None:
            pp.savefig(fn)
        else:
            pp.show()



class BaseFreeEnergyProfile(BaseProfile):
    '''
        Child class of :py:class:`BaseProfile <thermolib.thermodynamics.fep.BaseProfile>` to define a (free) energy profile :math:`F(CV)` (stored in self.fs) as function of a certain collective variable :math:`CV` (stored in self.cvs). This class will set some defaults choices as well as define  additional attributes and routines specific for manipulation of free energy profiles.
    '''
    def __init__(self, cvs, fs, temp, error=None, cv_output_unit='au', f_output_unit='kjmol', cv_label='CV', f_label='F'):
        """
            See documentation of :py:class:`BaseProfile <thermolib.thermodynamics.fep.BaseProfile>` for meaning of meaning of arguments not documented below..

            :param temp: the temperature at which the free energy is constructed, which should be in atomic units!
            :type temp: float
        """
        BaseProfile.__init__(self, cvs, fs, error=error, cv_output_unit=cv_output_unit, f_output_unit=f_output_unit, cv_label=cv_label, f_label=f_label)
        self.T = temp
        self.microstates = []
        self.macrostates = []

    @property
    def beta(self):
        return 1.0/(boltzmann*self.T)
        
    @classmethod
    def from_txt(cls, fn, temp, cvcol=0, fcol=1, fstdcol=None, cv_input_unit='au', f_input_unit='kjmol', cv_output_unit='au', f_output_unit='kjmol', cv_label='CV', f_label='F', cvrange=None, delimiter=None, reverse=False, cut_constant=False):
        '''
            See documentation off parent class routine :py:class:`BaseProfile.from_txt <thermolib.thermodynamics.fep.BaseProfile.from_txt>` for meaning of arguments not documented below.
        
            :param temp: temperature corresponding to the free (energy) profile (in atomic units, hence, in kelvin).
            :type temp: float
        '''
        profile = BaseProfile.from_txt(fn, cvcol=cvcol, fcol=fcol, fstdcol=fstdcol, cv_input_unit=cv_input_unit, f_input_unit=f_input_unit, cv_output_unit=cv_output_unit, f_output_unit=f_output_unit, cv_label=cv_label, f_label=f_label, cvrange=cvrange, delimiter=delimiter, reverse=reverse, cut_constant=cut_constant)
        error = None
        if profile.error is not None:
            error = profile.error.copy()
        return cls(profile.cvs, profile.fs, temp, error=error, cv_label=profile.cv_label, cv_output_unit=profile.cv_output_unit, f_output_unit=profile.f_output_unit)

    @classmethod
    def from_histogram(cls, histogram, temp, cv_output_unit=None, cv_label=None, f_label='F', f_output_unit='kjmol', propagator=Propagator()):
        '''
            Use the probability histogram :math:`p(CV)` to construct the corresponding free energy profile :math:`F(CV)` at the given temperature using the following formula

            .. math:: 

                F(CV) = -k_BT\\log\\left(p(CV)\\right)
        
            :param histogram: histogram from which the free energy profile is computed
            :type histogram: :py:class:`Histogram1D <thermolib.thermodynamics.histogram.Histogram1D>`

            :param temp: the temperature at which the histogram input data was simulated, in atomic units.
            :type temp: float

            :param cv_output_unit: the units for printing and plotting of CV values (not the unit of the input array, that is assumed to be in atomic units). Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module. If None is given, the cv_output_unit attribute of the histogram instance is used.
            :type cv_output_unit: str, default=None
                
            :param f_output_unit: the units for printing and plotting of free energy values (not the unit of the input array, that is assumed to be in atomic units). Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type f_output_unit: str, default='kjmol'

            :param cv_label: label for the CV for printing and plotting. If None is given, the cv_label attribute of the histogram instance is used.
            :type cv_label: str, optional, default=None

            :param f_label: label for the free energy for printing and plotting
            :type f_label: str, optional, default='F'

            :param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken)
            :type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator()
        '''
        fs = np.zeros(len(histogram.ps))*np.nan
        fs[histogram.ps>0] = -boltzmann*temp*np.log(histogram.ps[histogram.ps>0])
        error = None
        if histogram.error is not None:
            kT = boltzmann*temp
            if isinstance(histogram.error, LogGaussianDistribution):
                error = GaussianDistribution.log_from_loggaussian(histogram.error, scale=-kT)
            elif isinstance(histogram.error, MultiLogGaussianDistribution):
                error = MultiGaussianDistribution.log_from_loggaussian(histogram.error, scale=-kT)
            else:
                def function(ps):
                    result = np.zeros(ps.shape)*np.nan
                    result[ps>0] = -kT*np.log(ps[ps>0])
                    return result
                error =  propagator(function, histogram.error, target_distribution=type(histogram.error))
        if cv_output_unit is None:
            cv_output_unit = histogram.cv_output_unit
        if cv_label is None:
            cv_label = histogram.cv_label
        return cls(histogram.cvs, fs, temp, error=error, cv_output_unit=cv_output_unit, f_output_unit=f_output_unit, cv_label=cv_label, f_label=f_label)

    def process_states(self, *args, **kwargs):
        '''
            This routine is not implemented in the current class, but only in its child classes (see e.g. :py:meth:`SimpleFreeEnergyProfile.process_states <thermolib.thermodynamics.fep.SimpleFreeEnergyProfile.process_states>`)
        '''
        raise NotImplementedError('Cannot process states of a BaseFreeEnergyProfile. First convert to a SimpleFreeEnergyProfile.')

    def recollect(self, new_cvs, fn_plot=None, return_new_fes=False):
        '''
            Redefine the CV array to the new given array. For each interval of new CV values, collect all old free energy values for which the corresponding CV value falls in this new interval and average out. As such, this routine can be used to filter out noise on a given free energy profile by means of averaging.

            :param new_cvs:  Array of new CV values
            :type new_cvs: np.ndarray
            
            :param fn_plot: File name for comparison plot of old and new profile. If None, no such plot will be made
            :type fn_plot: str, optional, default=None
            
            :param return_new_fes: If set to False, the recollected data will be written to the current instance (overwritting the original data). If set to True, a new instance will be initialized with the recollected data and returned.
            :type return_new_fes: bool, optional, default=False
            
            :return: Returns an instance of the current class if the keyword argument `return_new_fes` is set to True. Otherwise, None is returned.
            :rtype: None or instance of same class as self
        '''
        assert new_cvs[0]<=self.cvs[0], 'First value of new cvs should be lower or equal to first value of original cvs, otherwise data will be lost. If you really want to delete data, use crop first.'
        assert self.cvs[-1]<=new_cvs[-1], 'Last value of new cvs should be greater or equal to last value of original cvs, otherwise data will be lost. If you really want to delete data, use crop first.'
        new_fs = np.zeros(len(new_cvs), float)*np.nan
        iold = 0
        for inew, cvnew in enumerate(new_cvs):
            #print('Processing new_cvs[%i]=%.3f'%(inew,cvnew))
            data = []
            if inew==0:
                lower = -np.inf
                upper = 0.5*(new_cvs[inew]+new_cvs[inew+1])
            elif inew==len(new_cvs)-1:
                lower = 0.5*(new_cvs[inew-1]+new_cvs[inew])
                upper = np.inf
            else:
                lower = 0.5*(new_cvs[inew-1]+new_cvs[inew])
                upper = 0.5*(new_cvs[inew]+new_cvs[inew+1])
            while iold<len(self.cvs) and lower<=self.cvs[iold]<upper:
                #print('  cvs_old[%i]=%.3f added to current data'%(iold,self.cvs[iold]))
                if not np.isnan(self.fs[iold]):
                    data.append(self.fs[iold])
                iold += 1
            #print('  no further old cvs found')
            if len(data)>0:
                new_fs[inew] = -np.log(sum(np.exp(-self.beta*np.array(data))))/self.beta #sum(data)/len(data)
                #print('  ==> averaged previous data=',data,' to f=%.3f' %(new_fs[inew]/kjmol))
            if iold>=len(self.cvs):
                #print('reached end of old cvs, breaking')
                break
        if fn_plot is not None:
            pp.clf()
            fig, ax = pp.subplots(nrows=1, ncols=1)
            #make free energy plot
            ax.plot(self.cvs/parse_unit(self.cv_output_unit), self.fs/parse_unit(self.f_output_unit), linewidth=1, color='0.3', label='Original')
            ax.plot(new_cvs/parse_unit(self.cv_output_unit), new_fs/parse_unit(self.f_output_unit), linewidth=3, color='sandybrown', label='Recollected')
            #decorate
            ax.set_xlabel('%s [%s]' %(self.cv_label, self.cv_output_unit))
            ax.set_ylabel('F [%s]' %self.f_output_unit)
            ax.set_title('Free energy profile')
            ax.set_xlim([min(new_cvs), max(new_cvs)])
            ax.legend(loc='best')
            #save
            fig.set_size_inches([8,8])
            pp.savefig(fn_plot)
        if return_new_fes:
            return self.__class__(new_cvs, new_fs, self.T, cv_output_unit=self.cv_output_unit, f_output_unit=self.f_output_unit, cv_label=self.cv_label)
        else:
            self.cvs = new_cvs[~np.isnan(new_fs)].copy()
            self.fs = new_fs[~np.isnan(new_fs)].copy()

    def transform_function(self, function, derivative=None, cv_label='Q', cv_output_unit='au'):
        '''
            Routine to transform the current free energy profile in terms of the original :math:`CV` towards a free energy profile in terms of a new collective variable :math:`Q=f(CV)` according to the formula:

            .. math:: 

                F_2(Q) &= F_1(f^{-1}(Q)) + k_B T \\log\\left[\\frac{df}{dCV}(f^{-1}(Q))\\right]

            :param function: The transformation function relating the old CV to the new Q, i.e. :math:`Q=f(CV)`
            :type function: callable

            :param derivative: The analytical derivative of the transformation function :math:`f`. If set to None, the derivative will be estimated through numerical differentiation.
            :type derivative: callable, optional, default=None

            :param cv_label: The label of the new collective variable used in plotting etc
            :type cv_label: str, optional, default='Q'

            :param cv_output_unit: The unit of the new collective varaible used in plotting and printing. Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type cv_output_unit: str, optional, default='au'

            :return: transformed free energy profile
            :rtype: the same class as the instance this routine is called upon

            :raise ValueError: if self.error is a distribution that is neither an instance of :py:class:`GaussianDistribution <thermolib.error.GaussianDistribution>` nor of :py:class:`MultiGaussianDistribution <thermolib.error.MultiGaussianDistribution>`
        '''
        qs = function(self.cvs)
        if derivative is None:
            eps = min(self.cvs[1:]-self.cvs[:-1])*0.001
            def derivative(q):
                return (function(q+eps/2)-function(q-eps/2))/eps
        dfs = derivative(self.cvs)
        fs = self.fs + np.log(abs(dfs))/self.beta
        error = None
        if self.error is not None:
            #as the transformation is (1) exactly known, (2) it transforms CV values and (3) doesn't change the error on the f
            if isinstance(self.error, GaussianDistribution):
                error = GaussianDistribution(fs, self.error.stds.copy())
            elif isinstance(self.error, MultiGaussianDistribution):
                error = MultiGaussianDistribution(fs, self.error.covariance.copy())
            else:
                raise ValueError('Error distribution %s is not supported in transform_function' %(type(self.error)))
        return self.__class__(qs, fs, self.T, error=error, cv_output_unit=cv_output_unit, f_output_unit=self.f_output_unit, cv_label=cv_label)



class SimpleFreeEnergyProfile(BaseFreeEnergyProfile):
    '''
        Class implementing a 'simple' 1D FEP representing a bi-stable profile with 2 minima corresponding to the reactant and process states and 1 local maximum corresponding to the transition state. As such, this class offers all features of the parent class :py:class:`BaseFreeEnergyProfile <thermolib.thermodynamics.fep.BaseFreeEnergyProfile>` as well as the functionality implemented in :py:meth:`process_states <thermolib.thermodynamics.fep.SimpleFreeEnergyProfile.process_states>` to automatically define the macrostates corresponding to reactant state and product state as well as the microstate corresponding to the transition state. 
                '''
    def __init__(self, cvs, fs, temp, error=None, cv_output_unit='au', f_output_unit='kjmol', cv_label='CV', f_label='F'):
        '''
            See documentation of :py:class:`BaseFreeEnergyProfile <thermolib.thermodynamics.fep.BaseFreeEnergyProfile>`.
        '''
        BaseFreeEnergyProfile.__init__(self, cvs, fs, temp, error=error, cv_output_unit=cv_output_unit, f_output_unit=f_output_unit, cv_label=cv_label, f_label=f_label)
        self.r  = None
        self.ts = None
        self.p  = None
        self.R = None
        self.P = None

    #@classmethod
    #def from_txt(cls, fn, temp, cvcol=0, fcol=1, fstdcol=None, cv_input_unit='au', f_input_unit='kjmol', cv_output_unit='au', f_output_unit='kjmol', cv_label='CV', f_label='F', cvrange=None, delimiter=None, reverse=False, cut_constant=False):
    #    '''
    #        Construct instance of SimpleFreeEnergyProfile from a txt file. For more information and argument documentation, see :py:class:`BaseFreeEnergyProfile <thermolib.thermodynamics.fep.BaseFreeEnergyProfile>`.
    #    '''
    #    profile = BaseProfile.from_txt(fn, cvcol=cvcol, fcol=fcol, fstdcol=fstdcol, cv_input_unit=cv_input_unit, f_input_unit=f_input_unit, cv_output_unit=cv_output_unit, f_output_unit=f_output_unit, cv_label=cv_label, f_label=f_label, cvrange=cvrange, delimiter=delimiter, reverse=reverse, cut_constant=cut_constant)
    #    error = None
    #    if profile.error is not None:
    #        error = profile.error.copy()
    #    return cls(profile.cvs, profile.fs, temp, error=error, cv_label=profile.cv_label, cv_output_unit=profile.cv_output_unit, f_output_unit=profile.f_output_unit)

    @classmethod
    def from_base(cls, base):
        '''
            Simple class method to transform a given instance of BaseFreeEnergyProfile into an instance of SimpleFreeEnergyProfile

            :param base: instance of :py:class:`BaseFreeEnergyProfile <thermolib.thermodynamics.fep.BaseBaseFreeEnergyProfileProfile>` to convert into an instance of :py:class:`SimpleFreeEnergyProfile <thermolib.thermodynamics.fep.SimpleFreeEnergyProfile>`
            :type base: :py:class:`BaseFreeEnergyProfile <thermolib.thermodynamics.fep.BaseBaseFreeEnergyProfileProfile>`
        '''
        error = None
        if base.error is not None:
            error = base.error.copy()
        return SimpleFreeEnergyProfile(base.cvs, base.fs, base.T, error=error, cv_label=base.cv_label, cv_output_unit=base.cv_output_unit, f_output_unit=base.f_output_unit)

    def process_states(self, lims=[-np.inf, None, None, np.inf], verbose=False, propagator=Propagator()):
        '''
            Routine to define:
            
                - a microstate representig the transition state (ts) as the local maximum within the given ts_range
                - a microstate representing the reactant (r) as local minimum left of ts
                - a microstate representing the product (p) as local minimum right of ts
                - a macrostate representing the reactant (R) as an integrated sum of microstates left of the ts
                - a macrostate representing the product (P) as an integrated sum of microstates right of the ts

            :param lims: list of 4 values [a,b,c,d] such that the reactant state minimum (r) should be within interval [a,b], the transition state maximum (ts) should be within interval [b,c] and the the product state minimum (p) should be within interval [c,d]. If b and c are both None, the transition state maximum is looked for in the entire range defined by [a,b] (which will fail if the transition state is only a local maximum but not the global maximum in that range). a can be specified as -np.inf and/or b can be specified as np.inf indicating no limits.
            :type lims: list[float], optional, default=[-np.inf,None, None, np.inf]

            :param verbose: If True, increase verbosity and print thermodynamic state properties
            :type verbose: bool, optional, default=False

            :param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken)
            :type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator()

            :raises AssertionError: if one of b,c is None, but not both
            :raises AssertionError: by a :py:class:`State <thermolib.thermodynamics.state.State>` child classes if it could not determine its micro/macrostate
        '''
        #define and find the transition state:
        if lims[1] is None or lims[2] is None:
            assert lims[1] is None and lims[2] is None, 'In argument lims of the form [a,b,c,d] if one of b,c is None, both should be None!'
            ts_range = [-np.inf, np.inf]
        else:
            ts_range = [lims[1], lims[2]]
        self.ts = Maximum('ts', cv_range=ts_range, cv_unit=self.cv_output_unit, f_unit=self.f_output_unit, propagator=propagator)
        self.ts.compute(self)
        if verbose: self.ts.print()
        cv_ts = self.cvs[self.ts.index]
    
        #define and find reactant microstate
        if lims[1] is None:
            r_range = [lims[0], cv_ts]
        else:
            r_range = [lims[0], lims[1]]
        self.r = Minimum('r', cv_range=r_range, cv_unit=self.cv_output_unit, f_unit=self.f_output_unit, propagator=propagator)
        self.r.compute(self)
        if verbose: self.r.print()
        
        #define and find reactant microstate
        if lims[2] is None:
            p_range = [cv_ts, lims[3]]
        else:
            p_range = [lims[2], lims[3]]
        self.p = Minimum('p', cv_range=p_range, cv_unit=self.cv_output_unit, f_unit=self.f_output_unit, propagator=propagator)
        self.p.compute(self)
        if verbose: self.p.print()
            
        #define macrostates
        self.R = Integrate('R', cv_range=[lims[0], self.ts] , beta=self.beta, cv_unit=self.cv_output_unit, f_unit=self.f_output_unit, propagator=propagator)
        self.R.compute(self)
        if verbose: self.R.print()

        self.P = Integrate('P', cv_range=[self.ts, lims[3]], beta=self.beta, cv_unit=self.cv_output_unit, f_unit=self.f_output_unit, propagator=propagator)
        self.P.compute(self)
        if verbose: self.P.print()

        #collect micro and macrostates
        self.microstates = [self.r, self.ts, self.p]
        self.macrostates = [self.R, self.P]

    def update_states(self):
        '''
            Routine to update the state definition. Usefull if the free energy profile has changed somehow.
        '''
        #process free energy profile to calculate microstates
        for state in self.microstates:
            state.compute(self)
        #process free energy profile to calculate macrostates
        for state in self.macrostates:
            state.compute(self)

    def set_ref(self, ref='min'):
        ''' 
            Set the energy reference of the free energy profile.

            :param ref: the choice for the energy reference, should be one of:

                - *m* or *min* for the global minimum
                - *r* or *reactant* for the reactant minimum       
                - *ts*, *trans_state* or *transition* for the transition state maximum
                - *p* or *product* for the product minimum

                The options r, ts and p are only available if the reactant, transition state and product have already been found by the routine process_states.
            :type ref: str, optional, default=min
            
            :raises IOError: invalid value for keyword argument ref is given. See doc above for choices.
            :raises AssertionError: if a microstate is not defined while the ref choice requires it
            :raises AssertionError: if the ref choice was set to min or max, but the global minimum/maximum could not be found
        '''
        #Find index of reference state
        if isinstance(ref, int):
            ref = self.fs[ref]
        elif isinstance(ref, float):
            pass #in this case, ref itself is the profile y value that has to shifted to zero
        elif ref.lower() in ['r', 'reactant']:
            assert self.r is not None, 'Reactant state not defined yet, did you already apply the process_states routine?'
            self.r.compute(self, dist_prop='stationary')
            ref = self.r.get_F()
        elif ref.lower() in ['p', 'product']:
            assert self.p is not None, 'Product state not defined yet, did you already apply the process_states routine?'
            self.p.compute(self, dist_prop='stationary')
            ref = self.p.get_F()
        elif ref.lower() in ['ts', 'trans_state', 'transition']:
            assert self.ts is not None, 'Transition state not defined yet, did you already apply the process_states routine?'
            self.ts.compute(self, dist_prop='stationary')
            ref = self.ts.get_F()
        elif ref.lower() in ['min']:
            global_minimum = Minimum('glob_min', cv_unit=self.cv_output_unit, f_unit=self.f_output_unit)
            global_minimum.compute(self, dist_prop='stationary')
            ref = global_minimum.get_F()
        elif ref.lower() in ['max']:
            global_maximum = Maximum('glob_min', cv_unit=self.cv_output_unit, f_unit=self.f_output_unit)
            global_maximum.compute(self, dist_prop='stationary')
            ref = global_maximum.get_F()
        else:
            raise IOError('Invalid REF specification, recieved %s and should be min, max, r, ts, p or an integer' %ref)
        #Set ref
        self.fs -= ref
        if self.error is not None:
            self.error.set_ref(value=ref)
        #Micro and macrostates need to be updated
        self.update_states()

    def print_states(self):
        '''
            Print information on the micro- and macrostates currently defined
        '''
        for microstate in self.microstates:
            microstate.print()
        for macrostate in self.macrostates:
            macrostate.print()

    def plot(self, fn: str|None=None, obss: list|str='thermo_kinetic', rate: object|None=None, linestyles: list|None=None, linewidths: list|None=None, colors: list|None=None, cvlims: list|None=None, flims: list|None=None, micro_marker: str='s', micro_color: str='r', micro_size: int=4, micro_linestyle: str='--', macro_linestyle: str='-', macro_color: str='b', do_latex: bool=False, show_legend: bool=False, fig_size: list|None=None):
        '''
            Plot the property stored in the current profile as function of the CV. If the error distribution is stored in self.error, various statistical quantities besides the mean (such as the error width, lower/upper limit on the error bar, random sample) can be plotted using the ``obss`` keyword. Alternatively, one can also plot the thermodynamic micro/macrostates (defined by :py:meth:`process_states`) and optionally the kinetic properties such as the rate constant and phenomenological barrier (see obss and rate keywords in the documentation below).

            :param fn: name of a file to save plot to. If None, the plot will not be saved to a file.
            :type fn: str, optional, default=None

            :param obss: Specify which statistical property/properties to plot. Multiple values from the list given below are allowed, which will be plotted on the same figure. Alternatively, one can also specify `obss=thermo_kinetic` which will plot the free energy profile, its error as well as highlight all micro and macrostates defined in the current instance.

                - **value** - the values stored in self.fs
                - **mean** - the mean according to the error distribution, i.e. self.error.mean()
                - **lower** - the lower limit of the 2-sigma error bar (which corresponds to a 95% confidence interval in case of a normal distribution), i.e. self.error.nsigma_conf_int(2)[0]
                - **upper** - the upper limit of the 2-sigma error bar (which corresponds to a 95% confidence interval in case of a normal distribution), i.e. self.error.nsigma_conf_int(2)[1]
                - **error** - half the width of the 2-sigma error bar (which corresponds to a 95% confidence interval in case of a normal distribution), i.e. abs(upper-lower)/2
                - **sample** - a random sample taken from the error distribution
            :type obss: list, optional, default='thermo_kinetic'

            :param rate: only relevent when ``obss=thermo_kinetic``, rate factor that allows to include inclusion of reaction rate and phenomenological free energy barriers to plot
            :type rate: :py:class:`RateFactorEquilibrium <thermolib.kinetics.rate.RateFactorEquilibrium>` or None, optional, default=None

            :param micro_marker: matplotlib marker style for indicating microstates
            :type micro_marker: str, optional, default='s'

            :param micro_color: matplotlib marker color for indicating microstates
            :type micro_color: str, optional, default='r'

            :param micro_size: matplotlib marker size for indicating microstates
            :type micro_size: str, optional, default=4

            :param macro_linestyle: matplotlib line style for indicating macrostates
            :type macro_linestyle: str, optional, default='-'

            :param macro_color: matplotlib line color for indicating macrostates
            :type macro_color: str, optional, default='b'

            :param linestyles: Specify the line style (using matplotlib definitions) for each quantity requested in ``obss``. If None, matplotlib will choose.
            :type linestyles: list or None, optional, default=None

            :param linewidths: Specify the line width (using matplotlib definitions) for each quantity requested in ``obss``. If None, matplotlib will choose.
            :type linewidths: list of strings or None, optional, default=None

            :param colors: Specify the color (using matplotlib definitions) for each quantity requested in ``obss``. If None, matplotlib will choose.
            :type colors: list of strings or None, optional, default=None

            :param cvlims: limits to the plotting range of the cv. If None, no limits are enforced
            :type cvlims: list of strings or None, optional, default=None

            :param flims: limits to the plotting range of the X property. If None, no limits are enforced
            :type flims: list of strings or None, optional, default=None

            :param do_latex: only relevent when ``obss=thermo_kinetic``, will format the numerical values on the side of the plot in LaTeX.
            :type do_latex: bool, optional, default=False

            :param show_legend: If True, the legend is shown in the plot
            :type show_legend: bool, optional, default=None

            :param fig_size: specify the matplotlib figure siz
            :type fig_size: list of two floats, optional, default=None
            
            :raises ValueError: if the ``obs`` argument contains an invalid specification
        '''
        if isinstance(obss, list):
            if fig_size is None: fig_size = [8,8]
            #preprocess
            cvunit = parse_unit(self.cv_output_unit)
            funit = parse_unit(self.f_output_unit)
            #read data 
            data = []
            labels = []
            for obs in obss:
                values = None
                if obs.lower() in ['value']:
                    values = self.fs.copy()
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
            #add error if present
            if self.error is not None:
                lower, upper = self.error.nsigma_conf_int(2)
                lower, upper = lower, upper
            #post processing default arguments
            if linestyles is None:
                linestyles = [None,]*len(data)
            if linewidths is None:
                linewidths = [1.0,]*len(data)
            if colors is None:
                colors = [None,]*len(data)
            #make plot
            pp.clf()
            for (ys,label,linestyle,linewidth,color) in zip(data,labels,linestyles, linewidths,colors):
                pp.plot(self.cvs/cvunit, ys/funit, label=label, linestyle=linestyle, linewidth=linewidth, color=color)
            if self.error is not None:
                pp.fill_between(self.cvs/cvunit, lower/funit, upper/funit, alpha=0.33)
            pp.xlabel('%s [%s]' %(self.cv_label, self.cv_output_unit), fontsize=16)
            pp.ylabel('%s [%s]' %(self.f_label, self.f_output_unit), fontsize=16)
            if cvlims is not None: pp.xlim(cvlims)
            if flims is not None: pp.ylim(flims)
            if show_legend:
                pp.legend(loc='best')
            fig = pp.gcf()
            fig.set_size_inches([8,8])
            fig.tight_layout()
        elif obss=='thermo_kinetic':
            if fig_size is None: fig_size = [16,8]
            rc('text', usetex=do_latex)
            pp.clf()
            fig = pp.gcf()
            gs  = gridspec.GridSpec(1,2, width_ratios=[2,1])
            ax = fig.add_subplot(gs[0])
            axs = [ax]
            #make free energy plot
            axs[0].plot(self.cvs/parse_unit(self.cv_output_unit), self.fs/parse_unit(self.f_output_unit), linewidth=1, color='0.2')
            if self.error is not None:
                axs[0].fill_between(self.cvs/parse_unit(self.cv_output_unit), self.flower()/parse_unit(self.f_output_unit), self.fupper()/parse_unit(self.f_output_unit), alpha=0.33)
            #plot vline for transition state if defined
            cv_width = max(self.cvs)-min(self.cvs)
            if flims is None:
                ylower = -1+min([state.get_F()/kjmol for state in self.macrostates]+[0])
                yupper = max(self.fs[~np.isnan(self.fs)])/kjmol+10
            else:
                ylower, yupper = flims
            #plot microstates
            for microstate in self.microstates:
                axs[0].plot(microstate.get_cv()/parse_unit(self.cv_output_unit), microstate.get_F()/parse_unit(self.f_output_unit), linestyle='none', marker=micro_marker, color=micro_color, markersize=micro_size)
                axs[0].text((microstate.get_cv()+0.01*cv_width)/parse_unit(self.cv_output_unit), microstate.get_F()/parse_unit(self.f_output_unit), microstate.text('f'), color=micro_color, fontsize=12)
                axs[0].axvline(x=microstate.get_cv()/parse_unit(self.cv_output_unit), ymin=0, ymax=(microstate.get_F()/parse_unit(self.f_output_unit)-ylower)/(yupper-ylower), linestyle=micro_linestyle, color=micro_color, linewidth=1)
                if rate is not None:
                    axs[0].fill_betweenx([ylower, max(self.fs)/kjmol], x1=rate.CV_TS_lims[0]/parse_unit(self.cv_output_unit), x2=rate.CV_TS_lims[1]/parse_unit(self.cv_output_unit), alpha=0.33, color='k')
            #plot macrostates
            for macrostate in self.macrostates:
                xcen = (macrostate.get_cv()-min(self.cvs))/cv_width
                xwidth = macrostate.cvstd/cv_width
                ycen = macrostate.get_F()/parse_unit(self.f_output_unit)
                axs[0].axhline(y=ycen, xmin=xcen-xwidth, xmax=xcen+xwidth, linestyle=macro_linestyle, color=macro_color, linewidth=2)
                axs[0].text((macrostate.get_cv())/parse_unit(self.cv_output_unit), ycen+0.5, macrostate.text('f'), color=macro_color, fontsize=12)
            #cv_output_unit
            axs[0].set_xlabel('%s [%s]' %(self.cv_label, self.cv_output_unit), fontsize=14)
            axs[0].set_ylabel('Energy [%s]' %self.f_output_unit, fontsize=14)
            axs[0].set_title('Free energy profile F(CV)', fontsize=16)
            if cvlims is None:
                axs[0].set_xlim([min(self.cvs/parse_unit(self.cv_output_unit)), max(self.cvs/parse_unit(self.cv_output_unit))])
            else:
                axs[0].set_xlim(cvlims)
            axs[0].set_ylim([ylower, yupper])
            axs[0].axhline(y=0, xmin=0, xmax=1, linestyle='--', color='k', linewidth=1)

            if len(self.macrostates)>0:
                assert len(self.macrostates)==2, 'The plotter assumes two macrostates (if any), i.e. R and P, but found %i' %(len(self.macrostates))
                if do_latex:
                    fig.text(0.65, 0.88, r'\textit{Thermodynamic properties}', fontsize=16)
                    fig.text(0.65, 0.86, '-------------------------------------', fontsize=16)
                    fig.text(0.65, 0.82, r'$F_{R}     = $ %s'                          %(self.R.text('f'))    , fontsize=16)
                    fig.text(0.65, 0.78, r'$\left\langle CV \right\rangle_{R}  = $ %s' %(self.R.text('cv'))   , fontsize=16)
                    fig.text(0.65, 0.74, r'$\Delta(CV)_{R} = $ %s'                     %(self.R.text('cvstd')), fontsize=16)
                    fig.text(0.65, 0.70, r'$F_{P}     = $ %s'                          %(self.P.text('f'))    , fontsize=16)
                    fig.text(0.65, 0.66, r'$\left\langle CV \right\rangle_{P}  = $ %s' %(self.P.text('cv'))   , fontsize=16)
                    fig.text(0.65, 0.62, r'$\Delta(CV)_{P} =  $ %s'                    %(self.P.text('cvstd')), fontsize=16)
                    fig.text(0.65, 0.58, r'$F_{TS}    = $ %s'                          %(self.ts.text('f'))   , fontsize=16)
                    fig.text(0.65, 0.54, r'$CV_{TS}   = $ %s'                          %(self.ts.text('cv'))  , fontsize=16)

                else:
                    fig.text(0.65, 0.88, 'Thermodynamic properties', fontsize=16)
                    fig.text(0.65, 0.86, '-------------------------------------', fontsize=16)
                    fig.text(0.65, 0.82, 'F_R     = %s' %(self.R.text('f'))    , fontsize=16)
                    fig.text(0.65, 0.78, '<CV>_R  = %s' %(self.R.text('cv'))   , fontsize=16)
                    fig.text(0.65, 0.74, 'd(CV)_R = %s' %(self.R.text('cvstd')), fontsize=16)
                    fig.text(0.65, 0.70, 'F_P     = %s' %(self.P.text('f'))    , fontsize=16)
                    fig.text(0.65, 0.66, '<CV>_P  = %s' %(self.P.text('cv'))   , fontsize=16)
                    fig.text(0.65, 0.62, 'd(CV)_P = %s' %(self.P.text('cvstd')), fontsize=16)
                    fig.text(0.65, 0.58, 'F_TS    = %s' %(self.ts.text('f'))   , fontsize=16)
                    fig.text(0.65, 0.54, 'CV_TS   = %s' %(self.ts.text('cv'))  , fontsize=16)

            if rate is not None:
                #TODO: update to new style already implemented for thermodynamic properties, code below will now still give errors
                A, A_dist = rate.A, rate.A_dist
                k_for, k_for_dist, dF_for, dF_for_dist, k_back, k_back_dist, dF_back, dF_back_dist = rate.compute_rate(self)
                Aunit = '%s/s' %self.cv_output_unit
                if do_latex:
                    if k_for_dist is None:
                        fig.text(0.65, 0.50, r'\textit{Kinetic properties}', fontsize=16)
                        fig.text(0.65, 0.48, '-------------------------------------', fontsize=16)
                        fig.text(0.65, 0.44, r'$A     = $ %s $ %s.s^{-1}$' %(format_scientific(A/(parse_unit(self.cv_output_unit)/second)), self.cv_output_unit), fontsize=16)
                        fig.text(0.65, 0.40, r'$k_{F} = $ %s $ s^{-1}$' %(format_scientific(k_for*second)), fontsize=16)
                        fig.text(0.65, 0.36, r'$k_{B} = $ %s $ s^{-1}$' %(format_scientific(k_back*second)), fontsize=16)
                        fig.text(0.65, 0.32, r'\textit{Phenomenological barrier}'   , fontsize=16)
                        fig.text(0.65, 0.28, '-------------------------------------', fontsize=16)
                        fig.text(0.65, 0.24, r'$\Delta F_{F}  = %.3f\ \ $ %s' %(dF_for/parse_unit(self.f_output_unit),self.f_output_unit), fontsize=16)
                        fig.text(0.65, 0.20, r'$\Delta F_{B}  = %.3f\ \ $ %s' %(dF_back/parse_unit(self.f_output_unit),self.f_output_unit), fontsize=16)
                    else:
                        fig.text(0.65, 0.48, r'\textit{Kinetic properties}', fontsize=16)
                        fig.text(0.65, 0.46, '-------------------------------------', fontsize=16)
                        fig.text(0.65, 0.42, r'$A     = $ %s' %A_dist.print(unit=Aunit, do_scientific=True), fontsize=16)
                        fig.text(0.65, 0.38, r'$k_{F} = $ %s' %k_for_dist.print(unit='1/s', do_scientific=True), fontsize=16)
                        fig.text(0.65, 0.34, r'$k_{B} = $ %s' %k_back_dist.print(unit='1/s', do_scientific=True), fontsize=16)
                        fig.text(0.65, 0.28, r'\textit{Phenomenological barrier}'   , fontsize=16)
                        fig.text(0.65, 0.26, '-------------------------------------', fontsize=16)
                        fig.text(0.65, 0.22, r'$\Delta F_{F}  = $ %s' %dF_for_dist.print(unit=self.f_output_unit), fontsize=16)
                        fig.text(0.65, 0.18, r'$\Delta F_{B}  = $ %s' %dF_back_dist.print(unit=self.f_output_unit), fontsize=16)
                else:
                    if k_for_dist is None:
                        fig.text(0.65, 0.48, 'Kinetic properties', fontsize=16)
                        fig.text(0.65, 0.46, '-------------------------------------', fontsize=16)
                        fig.text(0.65, 0.42, 'A  = %s %s/s' %(format_scientific(rate.A/(parse_unit(self.cv_output_unit)/second)), self.cv_output_unit), fontsize=16)
                        fig.text(0.65, 0.38, 'kF = %s 1/s' %(format_scientific(k_for*second)), fontsize=16)
                        fig.text(0.65, 0.34, 'kB = %s 1/s' %(format_scientific(k_back*second)), fontsize=16)
                        fig.text(0.65, 0.28, 'Phenomenological barrier', fontsize=16)
                        fig.text(0.65, 0.26, '-------------------------------------'  , fontsize=16)
                        fig.text(0.65, 0.22, 'dF_F = %.3f %s' %(dF_for/parse_unit(self.f_output_unit),self.f_output_unit) , fontsize=16)
                        fig.text(0.65, 0.18, 'dF_B = %.3f %s' %(dF_back/parse_unit(self.f_output_unit),self.f_output_unit), fontsize=16)
                    else:
                        fig.text(0.65, 0.48, 'Kinetic properties', fontsize=16)
                        fig.text(0.65, 0.46, '-------------------------------------', fontsize=16)
                        fig.text(0.65, 0.42, 'A  = %s' %A_dist.print(unit='1e12*%s' %Aunit), fontsize=16)
                        fig.text(0.65, 0.38, 'kF = %s' %k_for_dist.print(unit='1e8/s'), fontsize=16)
                        fig.text(0.65, 0.34, 'kB = %s' %k_back_dist.print(unit='1e8/s'), fontsize=16)
                        fig.text(0.65, 0.28, 'Phenomenological barrier', fontsize=16)
                        fig.text(0.65, 0.26, '-------------------------------------'  , fontsize=16)
                        fig.text(0.65, 0.22, 'dF_F = %s' %dF_for_dist.print(unit=self.f_output_unit), fontsize=16)
                        fig.text(0.65, 0.18, 'dF_B = %s' %dF_back_dist.print(unit=self.f_output_unit), fontsize=16)
        #save
        fig.set_size_inches(fig_size)
        if fn is not None:
            pp.savefig(fn)
        else:
            pp.show()
        return



class FreeEnergySurface2D(object):
    '''
        Class implementing a 2D free energy surface F(CV1,CV2) (stored in self.fs) as function of two collective variables (CV) denoted by CV1 (stored in self.cv1s) and CV2 (stored in self.cv2s).
    '''
    def __init__(self, cv1s, cv2s, fs, temp, error=None, cv1_output_unit='au', cv2_output_unit='au', f_output_unit='kjmol', cv1_label='CV1', cv2_label='CV2', f_label='F'):
        '''
            :param cv1s: array containing the values for the first collective variable CV1. Should be given in atomic units. If you need help properly converting to atomic units, we refer to the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type cv1s: np.ndarray

            :param cv2s: array the values for the second collective variable CV2. Should be given in atomic units. If you need help properly converting to atomic units, we refer to the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.

            :param fs: 2D array containing the free energy values corresponding to the given values of CV1 and CV2 in xy indexing. Should be given in atomic units. If you need help properly converting to atomic units, we refer to the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type fs: np.ndarray

            :param temp: temperature at which the free energy is given. Should be given in atomic units. Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type temp: float

            :param error: error distribution on the free energy profile, defaults to None
            :type error: child of :py:class:`Distribution <thermolib.error.Distribution>`, optional, default=None

            :param cv1_output_unit: unit in which the CV1 values will be printed/plotted, not the unit in which the input array is given (which is assumed to be atomic units). Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type cv1_output_unit: str, optional, defaults to 'au'

            :param cv2_output_unit: unit in which the CV2 values will be printed/plotted, not the unit in which the input array is given (which is assumed to be atomic units). Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type cv2_output_unit: str, optional, defaults to 'au'

            :param f_output_unit: unit in which the free energy values will be printe/plotted, not the unit in which the input array f is given (which is assumed to be kjmol). Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type f_output_unit: str, optional, default='kjmol'

            :param cv1_label: label of CV1 for printing and plotting
            :type cv1_label: str, optional, default='CV1'

            :param cv2_label: label of CV2 for printing and plotting
            :type cv2_label: str, optional, default='CV2'

            :param f_label: label for the free energy for printing and plotting
            :type f_label: str, optional, default='F'
        '''
        self.cv1s = cv1s.copy()
        self.cv2s = cv2s.copy()
        self.fs   = fs.copy()
        self.error = None
        if error is not None:
            self.error = error.copy()
        self.T = temp
        self.cv1_output_unit = cv1_output_unit
        self.cv2_output_unit = cv2_output_unit
        self.f_output_unit = f_output_unit
        self.cv1_label = cv1_label
        self.cv2_label = cv2_label
        self.f_label = f_label

    def flower(self, nsigma=2):
        '''
            Return the lower limit of an n-sigma error bar, i.e. :math:`\\mu - n\\sigma` with :math:`\\mu` the mean and :math:`\\sigma` the standard deviation.

            :param nsigma: defines the n-sigma error bar
            :type nsigma: int, optional, default=2

            :return: the lower limit of the n-sigma error bar
            :rtype: np.ndarray with dimensions determined by self.error

            :raises AssertionError: if self.error is not defined.
        '''
        assert self.error is not None, 'Flower cannot be computed because no error distribution was defined in the error attribute'
        flower, fupper = self.error.nsigma_conf_int(nsigma)
        return flower

    def fupper(self, nsigma=2):
        '''
            Return the upper limit of an n-sigma error bar, i.e. :math:`\\mu + n\\sigma` with :math:`\\mu` the mean and :math:`\\sigma` the standard deviation.

            :param nsigma: defines the n-sigma error bar
            :type nsigma: int, optional, default=2
            
            :return: the upper limit of the n-sigma error bar
            :rtype: np.ndarray with dimensions determined by self.error

            :raises AssertionError: if self.error is not defined.
        '''
        assert self.error is not None, 'Fupper cannot be computed because no error distribution was defined in the error attribute'
        flower, fupper = self.error.nsigma_conf_int(nsigma)
        return fupper

    @property
    def beta(self):
        return 1.0/(boltzmann*self.T)

    def copy(self):
        '''
            Make and return a copy of the current FreeEnergySurface2D instance.
        '''
        fupper, flower = None, None
        if self.fupper is not None:
            fupper = self.fupper.copy()
        if self.flower is not None:
            flower = self.flower.copy()
        fes = FreeEnergySurface2D(
            self.cv1s.copy(), self.cv2s.copy(), self.fs.copy(), self.T, fupper=fupper, flower=flower, 
            cv1_output_unit=self.cv1_output_unit, cv2_output_unit=self.cv2_output_unit, f_output_unit=self.f_output_unit, cv1_label=self.cv1_label, cv2_label=self.cv2_label
        )
        return fes

    @classmethod
    def from_txt(cls, fn, temp, cv1_col=0, cv2_col=1, f_col=2, cv1_input_unit='au', cv1_output_unit='au', cv2_input_unit='au', cv2_output_unit='au', f_output_unit='kjmol', f_input_unit='kjmol', cv1_label='CV1', cv2_label='CV2', f_label='F', delimiter=None, verbose=False):
        '''
            Read the free energy surface on a 2D grid as function of two collective variables from a txt file. 

            :param fn: the name of the txt file (assumed to be readable by numpy.loadtxt) containing the data.
            :type fn: str

            :param temp: the temperature at which the free energy is given
            :type temp: float

            :param cv1_col: the column in which the first collective variable is stored
            :type cv1_col: int, optional, default=0

            :param cv2_col: the column in which the second collective variable is stored
            :type cv2_col: int, optional, default=1

            :param f_col: the column in which the free energy is stored
            :type f_col: int, optional, default=2

            :param cv1_input_unit: the unit in which the first CV values are stored in the input file
            :type cv1_unit: str, optional, default='au'

            :param cv1_output_unit: unit in which the CV1 values will be printed/plotted, not the unit in which the input array is given (which is given by cv1_input_unit). Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type cv1_output_unit: str, optional, default='au'

            :param cv2_input_unit: the unit in which the second CV values are stored in the input file
            :type cv2_input_unit: str, optional, default='au'

            :param cv2_output_unit: unit in which the CV2 values will be printed/plotted, not the unit in which the input array is given (which is given by cv2_input_unit). Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type cv2_output_unit: str, optional, default='au'

            :param f_input_unit: the unit in which the free energy values are stored in the input file
            :type f_input_unit: str, optional, default='kjmol'

            :param f_output_unit: unit in which the free energy values will be printed/plotted, not the unit in which the input array is given (which is given by f_input_unit). Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type cv2_output_unit: str, optional, default='kjmol'

            :param cv1_label: the label for the CV1 axis in plots
            :type cv1_label: str, optional, default='CV1'

            :param cv2_label: the label for the CV2 axis in plots, defaults to 'CV2'
            :type cv2_label: str, optional, default='CV2'

            :param f_label: label for the free energy for printing and plotting
            :type f_label: str, optional, default='F'

            :param delimiter: the delimiter used in the numpy input file, this argument is parsed to the numpy.loadtxt routine.
            :type delimiter: str, optional, default=None

            :param verbose: If True, increase logging verbosity
            :type verbose: bool, optional, default=False

            :return: 2D free energy surface
            :rtype: FreeEnergySurface2D
        '''
        data = np.loadtxt(fn, delimiter=delimiter, dtype=float)
        cv1s = data[:,cv1_col]*parse_unit(cv1_input_unit)
        cv2s = data[:,cv2_col]*parse_unit(cv2_input_unit)
        fs = data[:,f_col]*parse_unit(f_input_unit)
        #unravel data so that cv1s and cv2s are 1D arrays and fs is a 2D array
        #with F[i,j] the free energy corresponding to cv1s[i] and cv2s[j]
        cv1us = np.unique(cv1s)
        cv2us = np.unique(cv2s)
        N1, N2 = len(cv1us), len(cv2us)
        assert len(cv1s)==N1*N2, 'Something went wrong in unraveling input data'
        assert len(cv2s)==N1*N2, 'Something went wrong in unraveling input data'
        assert len(fs)==N1*N2, 'Something went wrong in unraveling input data'
        fus = np.zeros([N2,N1], float)*np.nan
        for index, (cv1, cv2, f) in enumerate(zip(cv1s,cv2s,fs)):
            i1 = int(index//N2)
            i2 = int(index%N2)
            assert abs(cv1us[i1]-cv1)<1e-6, 'Internal CV consistency check failed'
            assert abs(cv2us[i2]-cv2)<1e-6, 'Internal CV consistency check failed'
            if np.isinf(f):
                fus[i2,i1] = np.nan
            else:
                fus[i2,i1] = f
        cv1s = cv1us.copy()
        cv2s = cv2us.copy()
        fs = fus.copy()
        if verbose:
            print('CV grid specification')
            print('---------------------')
            cv1_min, cv1_max, cv1_delta, cv1_num = cv1s.min(), cv1s.max(), (cv1s[1:]-cv1s[:-1]).mean(), len(cv1s)
            cv2_min, cv2_max, cv2_delta, cv2_num = cv2s.min(), cv2s.max(), (cv2s[1:]-cv2s[:-1]).mean(), len(cv2s)
            print('CV1 grid [%s]: start = %.3e    end = %.3e    delta = %.3e    N = %i' %(cv1_output_unit, cv1_min/parse_unit(cv1_output_unit), cv1_max/parse_unit(cv1_output_unit), cv1_delta/parse_unit(cv1_output_unit), cv1_num))
            print('CV2 grid [%s]: start = %.3e    end = %.3e    delta = %.3e    N = %i' %(cv2_output_unit, cv2_min/parse_unit(cv2_output_unit), cv2_max/parse_unit(cv2_output_unit), cv2_delta/parse_unit(cv2_output_unit), cv2_num))
        return cls(cv1s, cv2s, fs, temp, cv1_output_unit=cv1_output_unit, cv2_output_unit=cv2_output_unit, f_output_unit=f_output_unit, cv1_label=cv1_label, cv2_label=cv2_label, f_label=f_label)

    @classmethod
    def from_histogram(cls, histogram, temp, cv1_output_unit=None, cv2_output_unit=None, cv1_label=None, cv2_label=None, f_output_unit='kjmol', f_label='F', propagator=Propagator()):
        '''
            Use the 2D probability histogram :math:`p(CV1,CV2)` to construct the corresponding 2D free energy surface at the given temperature using the following formula

            .. math:: 

                F(CV1,CV2) = -k_BT\\log\\left(p(CV1,CV2)\\right)
        
            :param histogram: histogram from which the free energy profile is computed
            :type histogram: :py:class:`Histogram2D <thermolib.thermodynamics.histogram.Histogram2D>`

            :param temp: the temperature at which the histogram input data was simulated
            :type temp: float

            :param cv1_output_unit: the units for printing and plotting of CV1 values (not the unit of the input array, that is assumed to be in atomic units). Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module. If None is given, the cv1_output_unit attribute of the histogram instance is used.
            :type cv1_output_unit: str, default=None

            :param cv2_output_unit: the units for printing and plotting of CV2 values (not the unit of the input array, that is assumed to be in atomic units). Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module. If None is given, the cv2_output_unit attribute of the histogram instance is used.
            :type cv2_output_unit: str, default=None
                
            :param f_output_unit: the units for printing and plotting of free energy values (not the unit of the input array, that is assumed to be in atomic units). Units are defined using the molmod `units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_ module.
            :type f_output_unit: str, default='kjmol'

            :param cv1_label: label for the CV1 for printing and plotting. If None is given, the cv1_label attribute of the histogram instance is used.
            :type cv1_label: str, optional, default=None

            :param cv2_label: label for the CV2 for printing and plotting. If None is given, the cv2_label attribute of the histogram instance is used.
            :type cv2_label: str, optional, default=None

            :param f_label: label for the free energy for printing and plotting
            :type f_label: str, optional, default='F'

            :param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken)
            :type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator()

            :raises RuntimeError: if the histogram.error could not be properly interpretated.
        '''
        fs = np.zeros(histogram.ps.shape)*np.nan
        fs[histogram.ps>0] = -boltzmann*temp*np.log(histogram.ps[histogram.ps>0])
        error = None
        if histogram.error is not None:
            beta = 1.0/(boltzmann*temp)
            def function(ps):
                result = np.zeros(ps.shape)*np.nan
                result[ps>0] = -np.log(ps[ps>0])/beta
                return result
            if isinstance(histogram.error, LogGaussianDistribution):
                error = GaussianDistribution.log_from_loggaussian(histogram.error, scale=-1/beta)
            elif isinstance(histogram.error, GaussianDistribution):
                error = propagator(function, histogram.error)
            elif isinstance(histogram.error, MultiLogGaussianDistribution):
                error = MultiGaussianDistribution.log_from_loggaussian(histogram.error, scale=-1/beta)
            elif isinstance(histogram.error, MultiGaussianDistribution):
                error = propagator(function, histogram.error, target_distribution=MultiGaussianDistribution, flattener=histogram.error.flattener)
            else:
                raise RuntimeError('Something went wrong!')
        if cv1_output_unit is None:
            cv1_output_unit = histogram.cv1_output_unit
        if cv2_output_unit is None:
            cv2_output_unit = histogram.cv2_output_unit
        if cv1_label is None:
            cv1_label = histogram.cv1_label
        if cv2_label is None:
            cv2_label = histogram.cv2_label
        fep = cls(histogram.cv1s, histogram.cv2s, fs, temp, error=error, cv1_output_unit=cv1_output_unit, cv2_output_unit=cv2_output_unit, cv1_label=cv1_label, cv2_label=cv2_label, f_output_unit=f_output_unit, f_label=f_label)
        if hasattr(histogram.error, 'flattener'):
            fep.error.flattener = histogram.error.flattener
        return fep

    def savetxt(self, fn_txt):
        '''
            Save the free energy profile to a txt file using numpy.savetxt. The units in which the CVs and free energy are written is specified in the attributes cv1_output_unit, cv2_output_unit and f_output_unit.

            :param fn_txt: name of the file to write fes to
            :type fn_txt: str
        '''
        header = '%s [%s]\t %s [%s]\t Free energy [%s]' %(self.cv1_label, self.cv1_output_unit,self.cv2_label, self.cv2_output_unit, self.f_output_unit)
        xv,yv = np.meshgrid(self.cv1s,self.cv2s)
        np.savetxt(fn_txt, np.vstack((yv.flatten()/parse_unit(self.cv1_output_unit),xv.flatten()/parse_unit(self.cv2_output_unit), self.fs.flatten()/parse_unit(self.f_output_unit))).T, header=header,fmt='%f')

    def set_ref(self, ref='min'):
        '''
            Set the energy reference of the free energy surface.

            :param ref: the choice for the energy reference. Currently only one possibility is implemented, i.e. *m* or *min* for the global minimum.
            :type ref: str, default='min'
            
            :raises IOError: if invalid value for keyword argument ref is given. See doc above for choices.
        '''
        #find reference
        if ref.lower() in ['m', 'min']:
            data = self.fs.copy()
            data[np.isnan(data)] = np.inf
            index = np.unravel_index(np.argmin(data), data.shape)
        else:
            raise IOError('Invalid REF specification, recieved %s and should be min' %ref)
        #set reference
        self.fs -= self.fs[index]
        if self.error is not None:
            self.error.set_ref(index=index)

    def detect_clusters(self, eps=1.5, min_samples=8, metric='euclidean', fn_plot=None, delete_clusters=[-1]):
        '''
            Routine to apply `the DBSCAN clustering algoritm as implemented in the Scikit Learn package <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_ to the (CV1,CV2) grid points that correspond to finite free energies (i.e. not nan or inf) to detect clusters of neighboring points.

            The DBSCAN algorithm first identifies the core samples, defined as samples for which at least ``min_samples`` other samples are within a distance of ``eps``. Next, the data is divided into clusters based on these core samples. A cluster is defined as a set of core samples that can be built by recursively taking a core sample, finding all of its neighbors that are core samples, finding all of their neighbors that are core samples, and so on. A cluster also has a set of non-core samples, which are samples that are neighbors of a core sample in the cluster but are not themselves core samples. Intuitively, these samples are on the fringes of a cluster. Each cluster is given an integer as label.

            Any sample that is not a core sample, and is at least ``eps`` in distance from any core sample, is considered an outlier by the algorithm and is what we here consider an isolated point/region. These points get the cluster label of -1.

            Finally, all data points belonging to a cluster with label specified in ``delete_clusters`` will have theire free energy set to nan. A safe choice here is to just delete isolated regions, i.e. the point in cluster with label -1 (which is the default).

            :param eps: DBSCAN parameter representing maximum distance between two samples for them to be considered neighbors (for more, see `DBSCAN documentation <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_)
            :type eps: float, optional, default=1.5

            :param min_samples: DBSCAN parameter representing the number of samples in a neighborhood for a point to be considered a core point (for more, see `DBSCAN documentation <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_)
            :type min_samples: int, optional, default=8

            :param metric: DBSCAN parameter representing the metric used when calculating distance (for more, see `DBSCAN documentation <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_)
            :type metric: str or callable, optional, default='euclidean'

            :param fn_plot: if specified, a plot will be made (and written to ``fn_plot``) visualizing the resulting clusters
            :type fn_plot: str, optional, default=None

            :param delete_clusters: list of cluster names whos members will be deleted from the free energy surface data. If set to to [-1], only isolated points (not belonging to a cluster) will be deleted.
            :type delete_clusters: list, optional, default=[-1]
        '''
        #collect data
        data = []
        for i1,cv1 in enumerate(self.cv1s):
            for i2,cv2 in enumerate(self.cv2s):
                f = self.fs[i2,i1]
                if not (np.isnan(f) or np.isinf(f)):
                    data.append([i1,i2])
        data = np.array(data)
        #perform DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        cl = db.fit(data)
        labels = cl.labels_
        unique_labels = set(labels)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        #some quick statistics.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print('Estimated number of noise points: %d' % n_noise_)
        print('Estimated number of clusters: %d' % n_clusters_)
        for label in unique_labels:
            if label==-1: continue
            n = list(labels).count(label)
            print("  cluster %i has %i points" %(label, n))
        #plot clusters
        if fn_plot is not None:
            colors = [pp.cm.jet(each) for each in np.linspace(0, 1, n_clusters_)]
            for label in sorted(unique_labels):
                if label==-1:
                    col = [0, 0, 0, 1] #Black used for noise.
                else:
                    col = colors[label]
                class_member_mask = (labels == label)
                xy = data[class_member_mask & core_samples_mask]
                pp.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor=tuple(col), markersize=12)
                xy = data[class_member_mask & ~core_samples_mask]
                pp.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor=tuple(col), markersize=6)
            pp.title('Clustered data points')
            pp.gcf().set_size_inches([12,12])
            pp.savefig(fn_plot)
        #for each point belonging to one of the clusters specified in
        #delete_clusters: set free energy to nan and probability to zero
        for dlabel in delete_clusters:
            cluster = data[labels==dlabel]
            for i1,i2 in cluster:
                self.fs[i2,i1] = np.nan

    def crop(self, cv1range=[-np.inf,np.inf], cv2range=[-np.inf,np.inf], return_new_fes=False):
        '''
            Crop the free energy surface by removing all data for which either cv1 or cv2 is beyond a given range.
            
            :param cv1range: range of cv1 (along x-axis) that will remain after cropping
            :type cv1range: list, optional, default=[-np.inf,np.inf]

            :param cv2range: range of cv2 (along y-axis) that will remain after cropping
            :type cv2range: list, optional, default=[-np.inf,np.inf]

            :param return_new_fes: if set to false, the cropping process will be applied on the existing instance, otherwise a copy will be returned
            :type return_new_fes: bool, optional, default=False

            :return: new cropped FES if ``return_new_fes=True``
            :rtype: None or :py:class:`FreeEnerySurface2D <thermolib.thermodynamics.fep.FreeEnergySurface2D>`
        '''
        #cut off some unwanted regions
        cv1s = self.cv1s.copy()
        cv2s = self.cv2s.copy()
        fs = self.fs.copy()
        
        indexes1 = []
        for i, cv1 in enumerate(cv1s):
            if cv1range[0]<=cv1<=cv1range[1]:
                indexes1.append(i)
        if len(indexes1)>0:
            cv1s = cv1s[np.array(indexes1)]
            fs = fs[np.array(indexes1),:]
        else:
            raise ValueError('All grid points cropped because of specified cv1range!')
        
        indexes2 = []
        for i, cv2 in enumerate(cv2s):
            if cv2range[0]<=cv2<=cv2range[1]:
                indexes2.append(i)
        if len(indexes2)>0:
            cv2s = cv2s[np.array(indexes2)]
            fs = fs[:,np.array(indexes2)]
        else:
            raise ValueError('All grid points cropped because of specified cv2range!')
        
        error = None
        if self.error is not None:
            if isinstance(self.error, GaussianDistribution):
                i1s, i2s = np.meshgrid(indexes1, indexes2)
                error = GaussianDistribution(self.error.means[i1s,i2s], self.error.stds[i1s,i2s])
            elif isinstance(self.error, LogGaussianDistribution):
                i1s, i2s = np.meshgrid(indexes1, indexes2)
                error = LogGaussianDistribution(self.error.lmeans[i1s,i2s], self.error.lstds[i1s,i2s])
            elif isinstance(self.error, MultiGaussianDistribution) or isinstance(self.error, MultiLogGaussianDistribution):
                assert not isinstance(self.error.flattener, DummyFlattener)
                indexes = []
                for i1 in indexes1:
                    for i2 in indexes2:
                        indexes.append(self.error.flattener.flatten_index(i1,i2))
                ixs, iys = np.meshgrid(indexes, indexes)
                flattener = Flattener(len(indexes1),len(indexes2))
                if isinstance(self.error, MultiGaussianDistribution):
                    error = MultiGaussianDistribution(self.error.means[indexes], self.error.covariance[ixs,iys], flattener=flattener)
                elif isinstance(self.error, MultiLogGaussianDistribution):
                    error = MultiLogGaussianDistribution(self.error.lmeans[indexes], self.error.lcovariance[ixs,iys], flattener=flattener)
                else:
                    raise RuntimeError('One of previous if statements should have been satisfied!')
            else:
                raise NotImplementedError('Type of error distribution (%s) not supported for cropping.' %(error.__class__.__name__))

        if return_new_fes:
            return FreeEnergySurface2D(cv1s, cv2s, fs, self.T, error=error, cv1_output_unit=self.cv1_output_unit, cv2_output_unit=self.cv2_output_unit, f_output_unit=self.f_output_unit, cv1_label=self.cv1_label, cv2_label=self.cv2_label)
        else:
            self.cv1s = cv1s.copy()
            self.cv2s = cv2s.copy()
            self.fs = fs.copy()
            self.error = error

    def transform(self, *args, **kwargs):
        raise NotImplementedError('Transform function not yet implemented for 2D FES!')

    def project_difference(self, sign=1, delta=None, cv_output_unit='au', return_class=BaseFreeEnergyProfile, error_distribution=MultiGaussianDistribution, propagator=Propagator()):
        '''
            Construct a 1D free energy profile representing the projection of the 2D FES onto the difference of collective variables using the following formula:

            .. math::
            
                F1(q) = -k_B T \\log\\left( \\int_{-\\infty}^{+\\infty} e^{-\\beta F2(x,x+q)}dx \\right)

            with :math:`q=CV2-CV1`. This routine is a wrapper around the more general :py:meth:`project_function <thermolib.thermodynamics.fep.FreeEnergySurface2D.project_function>`.

            :param sign: If sign is set to 1, the projection is done on q=CV2-CV1, if it is set to -1, projection is done to q=CV1-CV2 instead
            :type sign: int, optional, default=1

            :param delta: width of the single-bin approximation of the delta function applied in the projection formula. The delta function is one whenever abs(function(cv1,cv2)-q)<delta/2. Hence, delta has the same units as the new collective variable q. If None, the average bin width of the new CV is used.
            :type delta: float, optional, default=None

            :param cv_output_unit: unit for the new CV for printing and plotting purposes
            :type cv_output_unit: str, optional, default='au'
            
            :param return_class: The class of which an instance will finally be returned.
            :type return_class: python class object, optional, default=BaseFreeEnergyProfile

            :param error_distribution: the model for the error distribution of the projected free energy profile
            :type error_distribution: class from :py:mod:`error <thermolib.error>` module, optional, default= :py:class:`MultiGaussianDistribution <thermolib.error.MultiGaussianDistribution>`

            :param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken)
            :type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator()

            :returns: projected 1D free energy profile
            :rtype: see return_class argument

            :raises ValueError: if an invalid value for sign is given.
        '''
        if sign==1:
            x = self.cv1s.copy()
            y = self.cv2s.copy()
            cv_label = '%s-%s' %(self.cv2_label, self.cv1_label)
            def function(q1,q2):
                return q2-q1
        elif sign==-1:
            x = self.cv2s.copy()
            y = self.cv1s.copy()
            cv_label = '%s-%s' %(self.cv1_label, self.cv2_label)
            def function(q1,q2):
                return q1-q2
        else:
            raise ValueError('Recieved invalid sign, should be 1 or -1 but got %s' %str(sign))
        #construct grid for projected degree of freedom
        cvs = []
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                q = yj-xi
                if (abs(np.array(cvs)-q)>1e-6).all():
                    cvs.append(q)
        cvs = np.array(sorted(cvs))
        return self.project_function(function, cvs, cv_label=cv_label, cv_output_unit=cv_output_unit, return_class=return_class, error_distribution=error_distribution, propagator=propagator)

    def project_average(self, delta=None, cv_output_unit='au', return_class=BaseFreeEnergyProfile, error_distribution=MultiGaussianDistribution, propagator=Propagator()):
        '''
            Construct a 1D free energy profile representing the projection of the 2D free energy surface F2(CV1,CV2) onto the average q=(CV1+CV2)/2 of the collective variables using the following formula:

            .. math::
                
                F1(q) = -k_B T \\log\\left( 2\\int_{-\infty}^{+\infty} e^{-\\beta F2(x,2q-x}dx \\right)

            with :math:`q=0.5\\dot(CV1+CV2)`. This routine is a wrapper around the more general :py:meth:`project_function <thermolib.thermodynamics.fep.FreeEnergySurface2D.project_function>`.

            :param delta: width of the single-bin approximation of the delta function applied in the projection formula. The delta function is one whenever abs(function(cv1,cv2)-q)<delta/2. Hence, delta has the same units as the new collective variable q. If None, the average bin width of the new CV is used.
            :type delta: float, optional, default=None
            
            :param cv_output_unit: unit for the new CV for plotting and printing purposes
            :type cv_output_unit: str, optional, default='au'
            
            :param return_class: The class of which an instance will finally be returned.
            :type return_class: python class object, optional, default=BaseFreeEnergyProfile

            :param error_distribution: the model for the error distribution of the projected free energy profile
            :type error_distribution: class from :py:mod:`error <thermolib.error>` module, optional, default= :py:class:`MultiGaussianDistribution <thermolib.error.MultiGaussianDistribution>`

            :param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken)
            :type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator()

            :returns: projected 1D free energy profile
            :rtype: see return_class argument
        '''
        cvs = []
        for i, xi in enumerate(self.cv1s):
            for j, yj in enumerate(self.cv2s):
                q = 0.5*(xi+yj)
                if q not in cvs:
                    cvs.append(q)
        cvs = np.array(sorted(cvs))
        def function(q1,q2):
            return 0.5*(q1+q2)
        return self.project_function(function, cvs, cv_label='0.5*(%s+%s)' %(self.cv1_label,self.cv1_label), cv_output_unit=cv_output_unit, return_class=return_class, error_distribution=error_distribution, propagator=propagator)

    def project_cv1(self, delta=None, cv_output_unit='au', return_class=BaseFreeEnergyProfile, error_distribution=MultiGaussianDistribution, propagator=Propagator()):
        '''
            Construct a 1D free energy profile representing the projection of the 2D free energy surface F2(CV1,CV2) onto q=CV1 using the formula:

            .. math::
            
                F1(q) = -k_B T \\log\\left( \\int_{-\infty}^{+\infty} e^{-\\beta F2(q,y}dy \\right)

            with :math:`q=CV1`. This routine is a wrapper around the more general :py:meth:`project_function <thermolib.thermodynamics.fep.FreeEnergySurface2D.project_function>`
            
            :param delta: width of the single-bin approximation of the delta function applied in the projection formula. The delta function is one whenever abs(function(cv1,cv2)-q)<delta/2. Hence, delta has the same units as the new collective variable q. If None, the average bin width of the new CV is used.
            :type delta: float, optional, default=None

            :param cv_output_unit: unit for the new CV for printing and plotting purposes
            :type cv_output_unit: str, optional, default='au'

            :param return_class: The class of which an instance will finally be returned.
            :type return_class: python class object, optional, default=BaseFreeEnergyProfile

            :param error_distribution: the model for the error distribution of the projected free energy profile
            :type error_distribution: class from :py:mod:`error <thermolib.error>` module, optional, default= :py:class:`MultiGaussianDistribution <thermolib.error.MultiGaussianDistribution>`

            :param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken)
            :type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator()

            :returns: projected 1D free energy profile
            :rtype: see return_class argument
        '''
        def function(q1,q2):
            return q1
        return self.project_function(function, self.cv1s.copy(), delta=delta, cv_label=self.cv1_label, cv_output_unit=self.cv1_output_unit, return_class=return_class, error_distribution=error_distribution, propagator=propagator)

    def project_cv2(self, delta=None, cv_output_unit='au', return_class=BaseFreeEnergyProfile, error_distribution=MultiGaussianDistribution, propagator=Propagator()):
        '''
            Construct a 1D free energy profile representing the projection of the 2D FES F2(CV1,CV2) onto q=CV2. This is implemented as follows:

                F1(q) = -k_B T \\log\\left( \\int_{-\infty}^{+\infty} e^{-\\beta F2(x,q}dx \\right)

            with :math:`q=CV2`. This routine is a wrapper around the more general :py:meth:`project_function <thermolib.thermodynamics.fep.FreeEnergySurface2D.project_function>`

            :param delta: width of the single-bin approximation of the delta function applied in the projection formula. The delta function is one whenever abs(function(cv1,cv2)-q)<delta/2. Hence, delta has the same units as the new collective variable q. If None, the average bin width of the new CV is used.
            :type delta: float, optional, default=None

            :param cv_output_unit: unit for the new CV for printing and plotting purposes
            :type cv_output_unit: str, optional, default='au'

            :param return_class: The class of which an instance will finally be returned.
            :type return_class: python class object, optional, default=BaseFreeEnergyProfile

            :param error_distribution: the model for the error distribution of the projected free energy profile
            :type error_distribution: class from :py:mod:`error <thermolib.error>` module, optional, default= :py:class:`MultiGaussianDistribution <thermolib.error.MultiGaussianDistribution>`

            :param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken)
            :type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator()

            :returns: projected 1D free energy profile
            :rtype: see return_class argument
        '''
        def function(q1,q2):
            return q2
        return self.project_function(function, self.cv2s.copy(), delta=delta, cv_label=self.cv2_label, cv_output_unit=self.cv2_output_unit, return_class=return_class, error_distribution=error_distribution, propagator=propagator)

    def project_function(self, function, qs, delta=None, cv_label='CV', cv_output_unit='au', return_class=BaseFreeEnergyProfile, error_distribution=MultiGaussianDistribution, propagator=Propagator()):
        '''
            Routine to implement the general projection of a 2D FES onto a collective variable defined by the given function (which takes the original two CVs as arguments).

            :param function: function in terms of the original CVs to define the new CV to project upon
            :type function: callable

            :param qs: grid for the new CV
            :type qs: np.ndarray

            :param delta: width of the single-bin approximation of the delta function applied in the projection formula. The delta function is one whenever abs(function(cv1,cv2)-q)<delta/2. Hence, delta has the same units as the new collective variable q. If None, the average bin width of the new CV is used.
            :type delta: float, optional, default=None
            
            :param cv_label: label for the new CV
            :type cv_label: str, optional, default='CV'

            :param cv_output_unit: unit for the new CV for printing and plotting purposes
            :type cv_output_unit: str, optional, default='au'

            :param return_class: The class of which an instance will finally be returned
            :type return_class: python class object, optional, default=BaseFreeEnergyProfile

            :param error_distribution: the model for the error distribution of the projected free energy profile
            :type error_distribution: class from :py:mod:`error <thermolib.error>` module, optional, default= :py:class:`MultiGaussianDistribution <thermolib.error.MultiGaussianDistribution>`

            :param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken)
            :type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator()

            :returns: projected 1D free energy profile
            :rtype: see return_class argument
        '''
        CV1s, CV2s = np.meshgrid(self.cv1s, self.cv2s, indexing='ij')
        if delta is None:
            delta = (qs[1:]-qs[:-1]).mean()
        def dirac(cv1, cv2, q):
            assert cv1.shape==cv2.shape
            result = np.zeros(cv1.shape, float)
            result[:] = abs(function(cv1,cv2)-q)<delta/2
            return result
        delta1 = (self.cv1s[1:]-self.cv1s[:-1]).mean()
        delta2 = (self.cv2s[1:]-self.cv2s[:-1]).mean()
        def project(f12s):
            P12s = np.exp(-self.beta*f12s)
            P12s[np.isnan(P12s)] = 0.0
            pqs = np.zeros(len(qs), float)
            for i,q in enumerate(qs):
                Hs = dirac(CV1s, CV2s, q)
                pqs[i] = integrate2d(P12s*Hs, dx=delta1, dy=delta2)/delta
            fqs = np.zeros(len(qs))*np.nan
            fqs[pqs>0] = -np.log(pqs[pqs>0])/self.beta
            return fqs
        if self.error is None:
            fs = project(self.fs)
            error = None
        else:
            error = propagator(project, self.error, target_distribution=error_distribution, samples_are_flattened=True)
            fs = error.mean()
        return return_class(qs, fs, self.T, error=error, cv_output_unit=cv_output_unit, f_output_unit=self.f_output_unit, cv_label=cv_label)

    def plot(self, fn=None, slicer=[slice(None),slice(None)], obss=['value'], linestyles=None, linewidths=None, colors=None, cv1_lims=None, cv2_lims=None, flims=None, ncolors=8, plot_additional_function_contours=None, **plot_kwargs):
        '''
            Make either a 2D contour plot of F(CV1,CV2) or a 1D sliced plot of F along a slice in the direction specified by the slicer argument. Appart from the value of the free energy itself, other (statistical) related properties can be plotted as defined in the obbs argument. At the end of the argument list, you can also specify any matplotlib keyword arguments you wish to parse to the matplotlib plotter. E.g. if you want to specify the colormap, you can just add at the end of the arguments ``cmap='rainbow'``.

            :param fn: name of a file to save plot to. If None, the plot will not be saved to a file.
            :type fn: str, optional, default=None

            :param slicer: determines which degrees of freedom (CV1/CV2) vary/stay fixed in the plot. If slice(none) is specified, the free energy will be plotted as function of the corresponding CV. If an integer `i` is specified, that corresponding CV will be kept fixed at its `i-th` value. Some examples:

                - [slice(None),slice(Nonne)] -- a 2D contour plot will be made of F as function of both CVs
                - [slice(None),10] -- a 1D plot will be made of F as function of CV1 with CV2 fixed at self.cv2s[10]
                - [23,slice(None)] -- a 1D plot will be made of F as function of CV2 with CV1 fixed at self.cv1s[23]
            :type slicer: list of `slices <https://www.w3schools.com/python/ref_func_slice.asp>`_ or integers, optional, default=[slice(None),slice(None)]

            :param obss: Specify which statistical property/properties to plot. Multiple values are allowed, which will be plotted on the same figure. Following options are supported:

                - **value** - the values stored in self.fs
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

            :param flims: limits to the plotting range of the free energy. If None, no limits are enforced
            :type flims: list of strings or None, optional, default=None

            :param ncolors: only relevant for 2D contour plot, represents the number of contours (and hence colors) to be used in plot.
            :type ncolors: int, optional, default=8

            :param plot_additional_function_contours: allows to specify function f(CV1,CV2) and a list of contour values [c_1, c_2, ...]. This will add contours of the form :math:`f(CV1,CV2)=c_i` to the plot.
            :type plot_additional_function_contours: [callable, list(float)], optional, default=None
        '''
        #preprocess
        assert isinstance(slicer, list) or isinstance(slicer, np.ndarray), 'Slicer should be list or array, instead got %s' %(slicer.__class__.__name__)
        assert len(slicer)==2, 'Slicer should be list of length 2, instead got list of length %i' %( len(slicer))

        if isinstance(slicer[0], slice) and isinstance(slicer[1], slice):
            ndim = 2
            xs = self.cv1s[slicer[0]]/parse_unit(self.cv1_output_unit)
            xlabel = '%s [%s]' %(self.cv1_label, self.cv1_output_unit)
            xlims = cv1_lims
            ys = self.cv2s[slicer[1]]/parse_unit(self.cv2_output_unit)
            ylabel = '%s [%s]' %(self.cv2_label, self.cv2_output_unit)
            ylims = cv2_lims
            title = 'F(:,:)'
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
                title = 'F(%s[%i]=%.3f %s,:)' %(self.cv1_label,slicer[0],self.cv1s[slicer[0]]/parse_unit(self.cv1_output_unit),self.cv1_output_unit)
        else:
            raise ValueError('At least one of the two elements in slicer should be a slice instance!')
        funit = parse_unit(self.f_output_unit)

        #read data 
        data = []
        labels = []
        for obs in obss:
            values = None
            if obs.lower() in ['value']:
                values = self.fs.copy()[slicer[0],slicer[1]]
            elif obs.lower() in ['mean']:
                assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
                values = self.error.mean()[slicer[0],slicer[1]]
            elif obs.lower() in ['lower']:
                assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
                values = self.error.nsigma_conf_int(2)[0][slicer[0],slicer[1]]
            elif obs.lower() in ['upper']:
                assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
                values = self.error.nsigma_conf_int(2)[1][slicer[0],slicer[1]]
            elif obs.lower() in ['error']:
                assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
                lower = self.error.nsigma_conf_int(2)[0][slicer[0],slicer[1]]
                upper = self.error.nsigma_conf_int(2)[1][slicer[0],slicer[1]]
                values = 0.5*np.abs(upper - lower)
            elif obs.lower() in ['sample']:
                assert self.error is not None, 'Observable %s can only be plotted if error is defined!' %obs
                values = self.error.sample()[slicer[0],slicer[1]]
            if values is None: raise ValueError('Could not interpret observable %s' %obs)
            assert ndim==len(values.shape), 'Observable data has inconsistent dimensions!'

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
                pp.plot(xs, zs/funit, label=label, linestyle=linestyle, linewidth=linewidth, color=color, **plot_kwargs)
            if self.error is not None:
                pp.fill_between(xs, lower/funit, upper/funit, **plot_kwargs, alpha=0.33)
            pp.xlabel(xlabel, fontsize=16)
            pp.ylabel('%s [%s]' %(self.f_label, self.f_output_unit), fontsize=16)
            pp.title('Derived profiles from %s' %title, fontsize=16)
            if xlims is not None: pp.xlim(xlims)
            if flims is not None: pp.ylim(flims)
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
                    contourf = ax.contourf(xs, ys, data[i].T/funit, levels=levels, **plot_kwargs) #transpose data to convert ij indexing (internal) to xy indexing (for plotting)
                    contour = ax.contour(xs, ys, data[i].T/funit, levels=levels, colors='gray') #transpose data to convert ij indexing (internal) to xy indexing (for plotting)
                else:
                    contourf = ax.contourf(xs, ys, data[i].T/funit, **plot_kwargs) #transpose data to convert ij indexing (internal) to xy indexing (for plotting)
                    contour = ax.contour(xs, ys, data[i].T/funit, colors='gray') #transpose data to convert ij indexing (internal) to xy indexing (for plotting)
                ax.set_xlabel(xlabel, fontsize=16)
                ax.set_ylabel(ylabel, fontsize=16)
                ax.set_title('%s of %s' %(labels[i],title), fontsize=16)
                if xlims is not None: ax.set_xlim(xlims)
                if ylims is not None: ax.set_ylim(ylims)
                cbar = pp.colorbar(contourf, ax=ax, extend='both')
                cbar.set_label('%s [%s]' %(self.f_label, self.f_output_unit), fontsize=16)
                pp.clabel(contour, inline=1, fontsize=10)
                if plot_additional_function_contours is not None:
                    function, levels = plot_additional_function_contours
                    Xs, Ys = np.meshgrid(xs, ys)
                    fvals = function(Xs,Ys)
                    contour = ax.contour(xs, ys, fvals, colors='red', linewidth=2.0)
            fig.set_size_inches(size)
        else:
            raise ValueError('Can only plot 1D or 2D pcond data, but received %i-d data. Make sure that the combination of qslice and cvslice results in 1 or 2 dimensional data.' %(len(data.shape)))

        fig.tight_layout()
        if fn is not None:
            pp.savefig(fn)
        else:
            pp.show()
        return



def plot_profiles(profiles, fn=None, labels=None, flims=None, colors=None, linestyles=None, linewidths=None, do_latex=False):
    '''
        Make a plot to compare multiple 1D free energy profiles

        :param profiles: list of profiles to plot
        :type profiles: list of :py:class:`BaseProfile <thermolib.thermodynamics.fep.BaseProfile>` or child classes

        :param fn: file name to save the figure to. If None, the plot will not be saved.
        :type fn: str, optional, default=None
        
        :param labels: list of labels for the legend. Order is assumed to be consistent with profiles.
        :type labels: list(str), optional, default=None

        :param flims: [lower,upper] limits of the free energy axis in plots.
        :type flims: list/np.ndarray, optional, default=None

        :param colors: List of matplotlib color definitions for each entry in profile. If an entry is None, a color will be chosen internally. If colors=None, implying all colors are chosen internally.
		:type colors: List(str), optional, default=None

		:param linestyles: List of matplotlib line style definitions for each entry in histograms. If an entry is None, the default line style of '-' will be chosen . If linestyles=None, implying all line styles are set to the default of '-'.
		:type linestyles: List(str), optional, default=None

		:param linewidths: List of matplotlib line width definitions for each entry in histograms. If an entry is None, the default line width of 1 will be chosen. If linewidths=None, implying all line widths are set to the default of 2.
		:type linewidths: List(str), optional, default=None

        :param do_latex: Use LaTeX to do formatting of text in plot, requires working LaTeX installation.
        :type do_latex: bool, optional, default=False
    '''
    rc('text', usetex=do_latex)
    #initialize
    linewidth_default = 2
    linestyle_default = '-'
    pp.clf()
    fig, axs = pp.subplots(nrows=1, ncols=1, squeeze=False)
    cmap = cm.get_cmap('tab10')
    cv_unit, f_unit = profiles[0].cv_output_unit, profiles[0].f_output_unit
    cv_label = profiles[0].cv_label
    fmax = -np.inf
    #add free energy plots and possibly probability histograms
    for iprof, prof in enumerate(profiles):
        label = 'FEP %i' %iprof
        if labels is not None:
            label = labels[iprof]
        #set color, linestyles, ...
        color = None
        if colors is not None:
            color = colors[iprof]
        if color is None:
            color = cmap(iprof)
        linewidth = None
        if linewidths is not None:
            linewidth = linewidths[iprof]
        if linewidth is None:
            linewidth = linewidth_default
        linestyle = None
        if linestyles is not None:
            linestyle = linestyles[iprof]
        if linestyle is None:
            linestyle = linestyle_default
        #plot free energy
        fmax = max(fmax, np.ceil(max(prof.fs[~np.isnan(prof.fs)]/kjmol)/10)*10)
        axs[0,0].plot(prof.cvs/parse_unit(cv_unit), prof.fs/parse_unit(f_unit), linewidth=linewidth, linestyle=linestyle, color=color, label=label)
        if prof.error is not None:
            axs[0,0].fill_between(prof.cvs/parse_unit(cv_unit), prof.flower()/parse_unit(f_unit), prof.fupper()/parse_unit(f_unit), color=color, alpha=0.33)
    #decorate
    cv_range = [min(profiles[0].cvs/parse_unit(cv_unit)), max(profiles[0].cvs/parse_unit(cv_unit))]
    axs[0,0].set_xlabel('%s [%s]' %(cv_label, cv_unit), fontsize=14)
    axs[0,0].set_ylabel('F [%s]' %f_unit, fontsize=14)
    axs[0,0].set_title('Free energy profile', fontsize=14)
    axs[0,0].set_xlim(cv_range)
    if flims is None:
        if not np.isinf(fmax):
            axs[0,0].set_ylim([-1,fmax])
    else:
        axs[0,0].set_ylim(flims)
    axs[0,0].legend(loc='best')
    #save
    fig.set_size_inches([8,8])
    if fn is not None:
        pp.savefig(fn)
    else:
        pp.show()



#an alias for backward compatibility
def plot_feps(fn, feps, temp, labels=None, flims=None, colors=None, linestyles=None, linewidths=None, do_latex=False):
    '''
        .. deprecated:: 1.7

            Use :py:meth:`plot_profiles <thermolib.thermodynamics.fep.plot_profiles>` instead. Current plot_feps routine is just an alias that ignores temp argument.
    '''
    print('WARNING: plot_feps is a depricated routine, use plot_profiles instead. Current plot_feps routine is just an alias that ignores temp argument from now on.')
    plot_profiles(feps, fn=fn, labels=labels, flims=flims, colors=colors, linestyles=linestyles, linewidths=linewidths, do_latex=do_latex)