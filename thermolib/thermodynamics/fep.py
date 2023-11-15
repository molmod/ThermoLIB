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
from __future__ import annotations

from typing import List
from molmod.units import *
from molmod.constants import *

import numpy as np, sys

from thermolib.tools import integrate, integrate2d, format_scientific

import matplotlib.pyplot as pp
import matplotlib.cm as cm
from matplotlib import gridspec, rc
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})

from sklearn.cluster import DBSCAN #clustering algorithm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from thermolib.thermodynamics.histogram import HistogramND


__all__ = ['BaseFreeEnergyProfile', 'SimpleFreeEnergyProfile', 'FreeEnergySurface2D', 'plot_feps']


class BaseFreeEnergyProfile(object):
    '''
        Base class to define a free energy profile F(q) (stored in self.fs) as function of a certain collective variable (CV) denoted by q (stored in self.cvs).
    '''
    def __init__(self, cvs, fs, temp, fupper=None, flower=None, cv_output_unit='au', f_output_unit='kjmol', cv_label='CV'):
        """
            :param cv: the collective variable values, which should be in atomic units!
            :type cv: np.ndarray

            :param f: the free energy values, which should be in atomic units!
            :type f: np.ndarray

            :param temp: the temperature at which the free energy is constructed, which should be in atomic units!
            :type temp: float

            :param flower: the lower limit of the error bar on the free energy values, which should be in atomic units!
            :type flower: np.ndarray

            :param fupper: the upper limit of the error bar on the free energy values, which should be in atomic units!
            :type fupper: np.ndarray

            :param cv_output_unit: the units for printing and plotting of CV values (not the unit of the input array, that is assumed to be in atomic units). Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_.
            :type cv_output_unit: str, default=au
                
            :param f_output_unit: the units for printing and plotting of free energy values (not the unit of the input array, that is assumed to be in atomic units). Units are defined using `the molmod routine molmod.units <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_.
            :type f_output_unit: str, default=kjmol

            :param cv_label: label for the new CV
            :type cv_label: str, optional, default='CV'
        """
        assert len(cvs)==len(fs), "cvs and fs array should be of same length"
        self.cvs = cvs.copy()
        self.fs = fs.copy()
        self.T = temp
        if fupper is not None:
            self.fupper = fupper.copy()
        else:
            self.fupper = None
        if flower is not None:
            self.flower = flower.copy()
        else:
            self.flower = None
        self.cv_output_unit = cv_output_unit
        self.f_output_unit = f_output_unit
        self.cv_label = cv_label
        self.microstates = []
        self.macrostates = []
        self.compute_probdens()

    def _beta(self):
        return 1.0/(boltzmann*self.T)

    beta = property(_beta)

    @classmethod
    def from_txt(cls, fn, temp, cvcol=0, fcol=1, flowercol=None, fuppercol=None, cv_input_unit='au', f_input_unit='kjmol', cv_output_unit='au', f_output_unit='kjmol', cv_label='CV', cvrange=None, delimiter=None, reverse=False, cut_constant=False):
        '''
            Read the free energy profile as function of a collective variable from a txt file.

            :param fn: the name of the txt file containing the data
            :type fn: str

            :param temp: the temperature at which the free energy is constructed
            :type temp: float

            :param cvcol: the column in which the collective variable is stored
            :type cvcol: int, default=0

            :param fcol: the column in which the free energy is stored.
            :type fcol: int, default=1

            :param cv_input_unit: the units in which the CV values are stored in the file. Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_. 
            :type cv_input_unit: str or float, default=au

            :param cv_output_unit: the units for printing and plotting of CV values (not the unit of the input array, that is defined by cv_input_unit). Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_. 
            :type cv_output_unit: str or float, default=au

            :param f_input_unit: the units in which the free energy values are stored in the file. Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_
            :type f_input_unit: str or float, default=kjmol

            :param f_output_unit: the units for printing and plotting of free energy values (not the unit of the input array, that is defined by f_input_unit). Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_
            :type f_output_unit: str or float, default=kjmol

            :param cv_label: label for the new CV
            :type cv_label: str, optional, default='CV'

            :param cvrange: only read free energy for CVs in the given range
            :type cvrange: tuple or list, default=None
            
            :param delimiter: The delimiter used in the txt input file to separate columns. The default is set to None, corresponding to the default of `the numpy.loadtxt routine <https://numpy.org/doc/1.20/reference/generated/numpy.loadtxt.html>`_ (i.e. whitespace).
            :type delimiter: str, default=None
            
            :param reverse: if set to True, reverse the X axis (usefull to make sure reactant is on the left)
            :type reverse: bool, default=False

            :param cut_constant: if set to True, the data points at the start and end of the data array that are constant will be cut. Usefull to cut out unsampled areas for large and small CV values. 
            :type cut_constant: bool, default=False
        '''
        #TODO: deal with inf values in input file

        data = np.loadtxt(fn, delimiter=delimiter, dtype=float)
        cvs = data[:,cvcol]*parse_unit(cv_input_unit)
        fs = data[:,fcol]*parse_unit(f_input_unit)
        fupper, flower = None, None
        if fuppercol is not None and flowercol is not None:
            flower = data[:,flowercol]*parse_unit(f_input_unit)
            fupper = data[:,fuppercol]*parse_unit(f_input_unit)
        elif (fuppercol is None and flowercol is not None) or (fuppercol is not None and flowercol is None):
            raise ValueError("For errorbars, you need to define both column indexes of the upper error limit, fuppercol, and the lower limit, flowercol.")
        if reverse:
            cvs = cvs[::-1]
            fs = fs[::-1]
            if fupper is not None:
                fupper = fupper[::-1]
                flower = flower[::-1]
        if cvrange is not None:
            indexes = []
            for i, cv in enumerate(cvs):
                if cvrange[0]<=cv and cv<=cvrange[1]:
                    indexes.append(i)
            cvs = cvs[np.array(indexes)]
            fs = fs[np.array(indexes)]
            if fupper is not None:
                fupper = fupper[np.array(indexes)]
                flower = flower[np.array(indexes)]
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
            if fupper is not None:
                fupper = fupper[mask]
                flower = flower[mask]
        return cls(cvs, fs, temp, fupper=fupper, flower=flower, cv_output_unit=cv_output_unit, f_output_unit=f_output_unit)

    @classmethod
    def from_histogram(cls, histogram, temp, cv_output_unit=None, cv_label=None, f_output_unit='kjmol'):
        '''
            Use the estimated probability histogram to construct the corresponding free energy profile at the given temperature.
        
            :param histogram: histogram from which the free energy profile is computed
            :type histogram: histogram.Histogram1D

            :param temp: the temperature at which the histogram input data was simulated, in atomic units.
            :type temp: float

            :param cv_output_unit: the units for printing and plotting of CV values (not the unit of the input array, that is defined by cv_input_unit). Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_. Defaults to the cv_output_unit of the given histogram
            :type cv_output_unit: str or float, optional

            :param cv_label: label for the new CV. Defaults to the cv_label of the given histogram
            :type cv_label: str, optional

            :param f_output_unit: the units for printing and plotting of free energy values (not the unit of the input array, that is defined by f_input_unit). Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_.
            :type f_output_unit: str or float, optional, default=kjmol

            :return: free energy profile corresponding to the estimated probability histogram
            :rtype: BaseFreeEnergyProfile (or its inheriting child class)
        '''
        fupper = None
        flower = None
        fs = np.zeros(len(histogram.ps))*np.nan
        fs[histogram.ps>0] = -boltzmann*temp*np.log(histogram.ps[histogram.ps>0])
        if histogram.pupper is not None and histogram.plower is not None:
            fupper = np.zeros(len(histogram.ps))*np.nan
            flower = np.zeros(len(histogram.ps))*np.nan
            fupper[histogram.plower>0] = -boltzmann*temp*np.log(histogram.plower[histogram.plower>0])
            flower[histogram.pupper>0] = -boltzmann*temp*np.log(histogram.pupper[histogram.pupper>0])
        if cv_output_unit is None:
            cv_output_unit = histogram.cv_output_unit
        if cv_label is None:
            cv_label = histogram.cv_label
        return cls(histogram.cvs, fs, temp, fupper=fupper, flower=flower, cv_output_unit=cv_output_unit, f_output_unit=f_output_unit, cv_label=cv_label)

    @classmethod
    def from_average(cls, feps, cv_output_unit=None, cv_label=None, f_output_unit=None, error_estimate=None, nsigma=2):
        '''
            Start from a set of free energy profiles and compute and return the averaged free energy profile. If error_estimate is set to 'std', an error on the freee energy profile will be computed from the standard deviation within the set of profiles. 

            :param feps: set of free energy profiles to be averaged
            :type histograms: list(BaseFreeEnergyProfile)

            :param error_estimate: indicate if and how to perform error analysis. One of following options is available:
            
                *  **std** -- compute error from the standard deviation within the set of profiles.
                *  **None** -- do not estimate the error.
            
            :type error_estimate: str, optional, default=None

            :param cv_output_unit: the units for printing and plotting of CV values. Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_. Defaults to the cv_output_unit of the first free energy profile given.
            :type cv_output_unit: str or float, optional

            :param cv_label: label for the collective variable in plots. Defaults to the cv_label of the first given free energy profile.
            :type cv_label: str, optional

            :param f_output_unit: the units for printing and plotting of free energy values (not the unit of the input array, that is defined by f_input_unit). Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_. Defaults to the f_output_unit of the first free energy profile given.
            :type f_output_unit: str or float, optional.
            
            :param nsigma: only relevant when error estimation is turned on (i.e. when keyword ``error_estimate`` is not None), this option defines how large the error interval should be in terms of the standard deviation sigma. A ``nsigma=2`` implies a 2-sigma error bar (corresponding to 95% confidence interval) will be returned.
            :type nsigma: int, optional, default=2

            :return: averages free energy profile
            :rtype: identical as the class of the current routine
        '''
        #sanity checks
        cv_label = None
        cvs, temp = None, None
        for fep in feps:
            if temp is None:
                temp = fep.T
            else:
                assert temp==fep.T, 'Cannot average histograms corresponding to non-identical temperatures.'
            if cvs is None:
                cvs = fep.cvs.copy()
            else:
                assert (abs(cvs-fep.cvs)<1e-12*np.abs(cvs.mean())).all(), 'Cannot average the histograms as they do not have consistent CV grids'
            if cv_output_unit is None:
                cv_output_unit = fep.cv_output_unit
            if f_output_unit is None:
                f_output_unit = fep.f_output_unit
            if cv_label is None:
                cv_label = fep.cv_label
            else:
                assert cv_label==fep.cv_label, 'Inconsistent CV label definition in histograms (%s!=%s)' %(cv_label,fep.cv_label)
        #collect free energy value
        fss = np.array([fep.fs for fep in feps])
        #average histograms
        fs = fss.mean(axis=0)
        #compute error if requested
        flower, fupper = None, None
        if error_estimate is not None and error_estimate.lower() in ['std']:
            ferr = fss.std(axis=0,ddof=1)
            flower = fs - nsigma*ferr
            fupper = fs + nsigma*ferr
        return cls(cvs, fs, temp, flower=flower, fupper=fupper, cv_output_unit=cv_output_unit, cv_label=cv_label)

    def savetxt(self, fn_txt):
        '''
            Save the free energy profile as txt file. The values of CV and free energy are written in units specified by the cv_output_unit and f_output_unit attributes of the self instance. If both flower and fupper are defined, they will also be written as the third and fourth column respectively.

            :param fn_txt: name for the output file
            :type fn_txt: str
        '''
        
        if self.flower is None or self.fupper is None:
            header = '%s [%s]\tFree energy [%s]' %(self.cv_label, self.cv_output_unit, self.f_output_unit)
            data = np.vstack((self.cvs/parse_unit(self.cv_output_unit), self.fs/parse_unit(self.f_output_unit))).T
        else:
            header = '%s [%s]\tFree energy [%s]\tLower error [%s]\tUpper error [%s]' %(self.cv_label, self.cv_output_unit, self.f_output_unit, self.f_output_unit, self.f_output_unit)
            data = np.vstack((self.cvs/parse_unit(self.cv_output_unit), self.fs/parse_unit(self.f_output_unit), self.flower/parse_unit(self.f_output_unit), self.fupper/parse_unit(self.f_output_unit))).T
        np.savetxt(fn_txt, data, header=header)

    def process_states(self, *args, **kwargs):
        raise NotImplementedError('BaseFreeEnergyProfile has no routine process_states implemented. Try to convert the free energy profile to a SimpleFreeEnergyProfile first.')

    def set_ref(self, ref='min'):
        '''
            Set the energy reference of the free energy profile.

            :param ref: the choice for the energy reference, currently only 'min' or 'm' is implemented resulting in setting the reference to the minimum in the free energy profile. Defaults to 'min'
            :type ref: str
        
            :raises IOError: invalid value for keyword argument ref is given. See doc above for choices.
        '''
        if ref.lower() in ['m', 'min']:
            fmin = self.fs[~np.isnan(self.fs)].min()
            self.fs -= fmin
            if self.fupper is not None:
                self.fupper -= fmin
            if self.flower is not None:
                self.flower -= fmin
        else:
            raise IOError('Invalid REF specification, recieved %s and should be min, r, ts or p' %ref)

    def set_microstates(self, **kwargs):
        raise NotImplementedError

    def set_macrostates(self, **kwargs):
        raise NotImplementedError

    def compute_probdens(self):
        '''
            Compute the probability density profile associated with the free energy profile:
        
            .. math:: p(q) = \\frac{1}{q_0Z}\\exp\\left(-\\beta F(q)\\right)

            with

            .. math:: Z = \\frac{1}{q_0}\\int_{-\\infty}^{+\\infty}\\exp\\left(-\\beta F(q)\\right)dq

            Herein, :math:`q_0` represents an arbitrary constant to account for the fact that the partition function should in principle be dimensionless. However, it can be chosen freely and has no impact on the final free energy profile (appart from meaningless a vertical shift). Therefore, in this implementation it is chosen as :math:`q_0=1`.
        '''
        mask = ~np.isnan(self.fs)
        self.ps = np.zeros(self.fs.shape)
        self.ps[mask] = np.exp(-self.beta*self.fs[mask])
        self.ps /= integrate(self.cvs[mask], self.ps[mask])

    def macrostate(self, cvrange=None, indexes=None, verbose=False, do_error=False, nsigma=2, Ncycles=100):
        '''
            Return the contribution to the partition function and free energy corresponding to the macrostate in the given range of cvs. This contribution is computed as follows. 

            .. math::

                \\begin{aligned}
                    Z_A &= \\frac{1}{q_0}\\int_A \\exp\\left(-\\beta F(q)\\right)dq \\\\

                        &= Z\\cdot\\int_A p(q)dq \\\\

                    F_A &= -k_B T\\log{Z_A}
                \\end{aligned}

            :param cvrange: the range of the collective variable defining the macrostate. Either cvrange or indices should be defined, defaults to None
            :type cvrange: np.ndarray, optional

            :param indexes: the indexes of the collective variable defining the macrostate. Either cvrange or indices should be defined, defaults to None
            :type indexes: list, optional

            :param verbose: set to true to turn on verbosity, defaults to False
            :type verbose: bool, optional

            :param do_error: set to true to use the error bars defined in self.fupper and self.flower to estimate lower and upper error bars on the macrostate partition function and free energy. First a the error bar on the free energy will be reduced to a normal distribution standard deviation (std = (self.fupper-self.flower)/nsigma)). Next, the value of the partition function and free energy for the macrostates will be computed Ncycle times, where in each cycle normally distributed errors will be added to the self.fs profile. Finally, the mean and std on the resulting Ncycle estimates for Z and F will be used as mean and error bar.
            :type verbose: bool, optional

            :param nsigma: the number of sigmas used in the n-sigma interval to compute the error bars self.flower and self.fupper. This nsigma is used to recompute the normally distributed standard deviation from (self.fupper-self.flower)/nsigma, which is in term required for the error estimation on the macrostate Z and F.

            :return: 
                *  **mean** (float) - the expected (=mean) value of the collective variable in the macrostate
                *  **std** (float) - the thermal fluctuation (=standard deviation) of the CV in the macrostate
                *  **Z** (float) - the contribution of the macrostate to the partition function
                *  **F** (float) - the free energy of the given macrostate
                *  **Zerr** (float) - only returned if do_error is set to True, the estimated error bar on Z
                *  **Ferr** (float) - only returned if do_error is set to True, the estimated error bar on F
        '''
        if do_error:
            assert self.flower is not None, "flower attribute needs to be defined to be able to perform error estimation of macrostate"
            assert self.fupper is not None, "fupper attribute needs to be defined to be able to perform error estimation of macrostate"
        if verbose: print('Processing macrostate(indices=%s, cvrange=%s)' %(indexes, cvrange))
        if indexes is None:
            if cvrange is not None:
                assert cvrange[0] is not None, "Lower limit of CV range in macrostate cannot be None"
                assert cvrange[1] is not None, "Upper limit of CV range in macrostate cannot be None"
                indexes = []
                for i, cv in enumerate(self.cvs):
                    if not np.isnan(self.fs[i]) and cvrange[0]<=cv<cvrange[1]:
                        indexes.append(i)
            else:
                print('ERROR: either indexes or cvrange should be given')
                sys.exit()
            if not len(indexes)>1:
                print('ERROR: cvrange should contain at least 2 data points')
                sys.exit()
        else:
            if cvrange is not None:
                print('WARNING: both indexes and cvrange given, cvrange was ignored')
        cvs = self.cvs[np.array(indexes)]
        fs = self.fs[np.array(indexes)]
        if do_error:
            flowers = self.flower[np.array(indexes)]
            fuppers = self.fupper[np.array(indexes)]
        ps = self.ps[np.array(indexes)]
        if do_error:
            Ps, Zs, Fs = np.zeros(Ncycles, dtype=float)
            for icycle in range(Ncycles):
                fs_err =  fs + np.random.normal(fs, (fuppers-flowers)/nsigma)
                Zs[i] = integrate(cvs, np.exp(-self.beta*fs_err))
                Fs[i] = -np.log(Zs[i])/self.beta
                Ps[i] = Zs[i]/(np.exp(-self.beta*fs_err).sum())
            P, Perr = Ps.mean(), Ps.std()
            Z, Zerr = Zs.mean(), Zs.std()
            F, Ferr = Fs.mean(), Fs.std()
        else:
            P = integrate(cvs, ps)
            Z = integrate(cvs, np.exp(-self.beta*fs))
            F = -np.log(Z)/self.beta
        mean = integrate(cvs, ps*cvs)/P
        std = np.sqrt(integrate(cvs, ps*(cvs-mean)**2)/P)
        if verbose:
            print('VALUES:')
            print('  CV Mean [%s] = ' %self.cv_output_unit, mean/parse_unit(self.cv_output_unit))
            print('  CV Min  [%s] = ' %self.cv_output_unit, min(cvs)/parse_unit(self.cv_output_unit))
            print('  CV Max  [%s] = ' %self.cv_output_unit, max(cvs)/parse_unit(self.cv_output_unit))
            print('  CV Std  [%s] = ' %self.cv_output_unit, std/parse_unit(self.cv_output_unit))
            print('')
            print('  F(CV min) [%s] = ' %self.f_output_unit, fs[np.where(cvs==min(cvs))[0][0]]/parse_unit(self.f_output_unit))
            print('  F(CV max) [%s] = ' %self.f_output_unit, fs[np.where(cvs==max(cvs))[0][0]]/parse_unit(self.f_output_unit))
            print('  F min     [%s] = ' %self.f_output_unit, min(fs)/parse_unit(self.f_output_unit))
            print('  F max     [%s] = ' %self.f_output_unit, max(fs)/parse_unit(self.f_output_unit))
            print('')
            if do_error:
                raise NotImplementedError
                print('  F [%s] = %.3f +- %.3f' %(F/parse_unit(self.f_output_unit), Ferr/parse_unit(self.f_output_unit)))
                print('  Z [-] = %.3e +- %.3e', Z)
                print('  P [-] = %.3f +- %.3f', P*100)
            else:
                print('  F [%s] = ' %self.f_output_unit, F/parse_unit(self.f_output_unit))
                print('  Z [-] = ', Z)
                print('  P [-] = ', P*100)
        if not do_error:
            return mean, std, Z, F
        else:
            return mean, std, Z, F, Zerr, Ferr

    def plot(self, fn, flim=None):
        '''
            Make a plot of the free energy profile. The values of CV and free energy are plotted in units specified by the cv_output_unit and f_output_unit attributes of the self instance.

            :param fn: Name of the file of the figure. Supported file formats are determined by the supported formats of the `matplotlib.pyplot.savefig routine <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html>`_
            :type fn: str

            :param flim: plot range for the free energy
            :type flim: float, optional, defaults to min-max.
        '''
        rc('text', usetex=False)
        pp.clf()
        fig, axs = pp.subplots(nrows=1, ncols=1)
        axs = [axs]
        #make free energy plot
        axs[0].plot(self.cvs/parse_unit(self.cv_output_unit), self.fs/parse_unit(self.f_output_unit), linewidth=1, color='0.2')
        if self.flower is not None and self.fupper is not None:
            axs[0].fill_between(self.cvs/parse_unit(self.cv_output_unit), self.flower/parse_unit(self.f_output_unit), self.fupper/parse_unit(self.f_output_unit), alpha=0.33)
        #decorate
        axs[0].set_xlabel('%s [%s]' %(self.cv_label, self.cv_output_unit))
        axs[0].set_ylabel('F [%s]' %self.f_output_unit)
        axs[0].set_title('Free energy profile')
        axs[0].set_xlim([min(self.cvs/parse_unit(self.cv_output_unit)), max(self.cvs/parse_unit(self.cv_output_unit))])
        if flim is not None:
            axs[0].set_ylim([0, flim/parse_unit(self.f_output_unit)])
        #save
        fig.set_size_inches([len(axs)*8,8])
        pp.savefig(fn)
        return

    def crop(self, cvrange, return_new_fes=False):
        '''
            Crop the free energy profile to the limits given by cvrange and throw away cropped data. If return_new_fes is set to false, a copy of the cropped profile will be returns, otherwise the current profile will be cropped and overwritten.

            :param cvrange: the range of the collective variable defining the new range to which the FEP will be cropped.
            :type cvrange: tuple
            
            :param return_new_fes: If set to False, the cropped data will be written to the current instance (overwritting the original data). If set to True, a new instance will be initialized with the cropped data, defaults to False
            :type return_new_fes: bool, optional
            
            :return: an instance of the current class if the keyword argument `return_new_fes` is set to True. Otherwise, None is returned.
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
        if return_new_fes:
            return self.__class__(cvs, fs, self.T, cv_output_unit=self.cv_output_unit, f_output_unit=self.f_output_unit, cv_label=self.cv_label)
        else:
            self.cvs = cvs.copy()
            self.fs = fs.copy()
            self.compute_probdens()

    def recollect(self, new_cvs, fn_plt=None, return_new_fes=False):
        '''
            Redefine the CV array to the new given array. For each interval of new CV values, collect all old free energy values for which the corresponding CV value falls in this new interval and average out. As such, this routine can be used to filter out noise on a given free energy profile by means of averaging.

            :param new_cvs:  Array of new CV values
            :type new_cvs: np.ndarray
            
            :param fn_plt: File name for comparison plot of old and new profile., defaults to None
            :type fn_plt: str, optional
            
            :param return_new_fes: If set to False, the recollected data will be written to the current instance (overwritting the original data). If set to True, a new instance will be initialized with the recollected data., defaults to False
            :type return_new_fes: bool, optional
            
            :return: Returns an instance of the current class if the keyword argument `return_new_fes` is set to True. Otherwise, None is returned.
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
        if fn_plt is not None:
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
            pp.savefig(fn_plt)
        if return_new_fes:
            return self.__class__(new_cvs, new_fs, self.T, cv_output_unit=self.cv_output_unit, f_output_unit=self.f_output_unit, cv_label=self.cv_label)
        else:
            self.cvs = new_cvs[~np.isnan(new_fs)].copy()
            self.fs = new_fs[~np.isnan(new_fs)].copy()
            self.compute_probdens()

    def transform_function(self, function, derivative=None, cv_label='Q', cv_output_unit='au'):
        '''
            Routine to transform the current free energy profile in terms of the original CV towards a free energy profile in terms of the new collective variable Q.

            :param function: The transformation function from the old CV towards the new Q
            :type function: callable

            :param derivative: The analytical derivative of the transformation function. If set to None, the derivative will be estimated through numerical differentiation. Defaults to None
            :type derivative: callable, optional

            :param cv_label: The label of the new collective variable used in plotting etc, defaults to 'Q'
            :type cv_label: str, optional

            :param cv_output_unit: The unit of the new collective varaible used in plotting and printing, defaults to 'au'
            :type cv_output_unit: str, optional

            :return: transformed free energy profile
            :rtype: the same class as the instance this routine is called upon
        '''
        #TODO: not tested yet!
        qs = function(self.cvs)
        if derivative is None:
            eps = min(qs[1:]-qs[:-1])*0.001
            def derivative(q):
                return (function(q+eps/2)-function(q-eps/2))/eps
        dfs = derivative(qs)
        fs = self.fs - np.log(dfs)/self.beta
        fupper, flower = None, None
        if self.fupper is not None:
            fupper = self.fupper - np.log(dfs)/self.beta
        if self.flower is not None:
            flower = self.flower - np.log(dfs)/self.beta
        return self.__class__(qs, fs, self.T, fupper=fupper, flower=flower, cv_output_unit=cv_output_unit, f_output_unit=self.f_output_unit, cv_label=cv_label)


class SimpleFreeEnergyProfile(BaseFreeEnergyProfile):
    '''
        Class implementing a 1D FEP representing a simple bi-stable profile with 2 minima representing the reactant and process states and 1 local maximum representing the transition state. 
            
        As such, this class offers all features of the parent class :meth:`BaseFreeEnergyProfile` as well as the additional feature to automatically identify the micro/macrostates corresponding to reactant state, transition state and product state (the :meth:`process_states` routine). 
            
        See :meth:`BaseFreeEnergyProfile` for constructor arguments and documentation.
    '''
    def __init__(self, cvs, fs, temp, fupper=None, flower=None, cv_output_unit='au', f_output_unit='kjmol', cv_label='CV'):
        BaseFreeEnergyProfile.__init__(self, cvs, fs, temp, fupper=fupper, flower=flower, cv_output_unit=cv_output_unit, f_output_unit=f_output_unit, cv_label=cv_label)
        self.ir  = None
        self.its = None
        self.ip  = None

    def process_states(self, ts_range=[-np.inf,np.inf], verbose=False):
        '''
            Routine to find the reactant (R), transition state (TS) and product state (P) through application of the :meth:`_find_R_TS_P` routine and add the corresponding micro and macrostates. These will afterwards be shown on the free energy plot using the :meth:`plot` routine.

            :param ts_range: range for the cv in which to look for the transition state as a local maximum.
            :type ts_range: list, optional, default=[-np.inf,np.inf]

            :param verbose: set to True to increase verbosity.
            :type verbose: bool, optional, default=False
        '''
        self._find_R_TS_P(ts_range=ts_range)
        self.set_microstates([self.ir, self.its, self.ip])
        self.set_macrostates([-np.inf, self.cvs[self.its], np.inf], verbose=verbose)

    def _find_R_TS_P(self, ts_range=[-np.inf,np.inf]):
        '''
            Internal routine called by :meth:`process_states` to find:
            
            *  the transition state (TS) as the local maximum within the given ts_range
            *  the reactant (R) as local minimum left of TS
            *  the product (P) as local minimum right of TS

            :param ts_range: range for the cv in which to look for the transition state as a local maximum
            :type ts_range: list, optional, default=[-np.inf,np.inf]

            :raises ValueError: the transition state cannot be found in the range defined by ts_range.
        '''
        self.ir  = None
        self.its = None
        self.ip  = None
        cv_upper = None
        #find transition state
        for i, (cv, f) in enumerate(zip(self.cvs, self.fs)):
            if ts_range[0]<=cv<=ts_range[1]:
                cv_upper = cv
                if self.its is None or f>self.fs[self.its]:
                    self.its = i
        if self.its is None:
            raise ValueError('Could not find transition state, are you sure it is within the given range %.6f<=CV<=%.6f?' %(ts_range[0], ts_range[1]))
        #check that self.its is not simply the largest cv in the given range (indicating it is not a local maximum)
        if self.cvs[self.its]==cv_upper:
            print('WARNING: The transition state corresponds with the upper boundary of the given ts_range, hence it might not be a local maximum!')
        #find reactant
        for i, (cv, f) in enumerate(zip(self.cvs, self.fs)):
            if cv<self.cvs[self.its]:
                if self.ir is None or np.isnan(self.fs[self.ir]) or f<self.fs[self.ir]:
                    self.ir = i
        #find product
        for i, (cv, f) in enumerate(zip(self.cvs, self.fs)):
            if self.cvs[self.its]<cv:
                if self.ip is None or np.isnan(self.fs[self.ip]) or f<self.fs[self.ip]:
                    self.ip = i

    def set_ref(self, ref='min'):
        ''' 
            Set the energy reference of the free energy profile.

            :param ref: the choice for the energy reference, should be one of:

                *  *m* or *min* for the global minimum
                *  *r* or *reactant* for the reactant minimum       
                *  *ts*, *trans_state* or *transition* for the transition state maximum
                *  *p* or *product* for the product minimum

                The options r, ts and p are only available if the reactant, transition state and product have already been found by the routine find_states.
            :type ref: str, optional, default=min
            
            :raises IOError: invalid value for keyword argument ref is given. See doc above for choices.
        '''
        fref = 0.0
        if ref.lower() in ['r', 'reactant']:
            assert self.ir is not None, 'Reactant state not defined yet, did you already apply the find_states routine?'
            fref = self.fs[self.ir]
        elif ref.lower() in ['p', 'product']:
            assert self.ip is not None, 'Product state not defined yet, did you already apply the find_states routine?'
            fref = self.fs[self.ip]
        elif ref.lower() in ['ts', 'trans_state', 'transition']:
            assert self.its is not None, 'Transition state not defined yet, did you already apply the find_states routine?'
            fref = self.fs[self.its]
        elif ref.lower() in ['m', 'min']:
            fref = self.fs[~np.isnan(self.fs)].min()
        else:
            raise IOError('Invalid REF specification, recieved %s and should be min, r, ts or p' %ref)
        self.fs -= fref
        if self.fupper is not None:
            self.fupper -= fref
        if self.flower is not None:
            self.flower -= fref
        #Micro and macrostates need to be updated
        if self.ir is not None and self.its is not None and self.ip is not None:
            self.microstates = []
            self.macrostates = []
            self.set_microstates([self.ir, self.its, self.ip])
            self.set_macrostates([-np.inf, self.cvs[self.its], np.inf])

    def set_microstates(self, indices):
        '''
            Routine to define microstates, i.e. points on the 1D FEP. This routine is called by :meth:`process_states` to add the microstates found by the latter. These microstates will be visualized as points with corresponding free energy value on a plot made by the :meth:`plot` routine.

            :param indices: list of indices corresponding to the index of the microstates in the self.cvs and self.fs arrays 
            :type indices: list(int)
        '''
        for index in indices:
            if index is not None:
                self.microstates.append([index, self.cvs[index], self.fs[index]])

    def set_macrostates(self, cvs, verbose=False):
        '''
            Routine to define macrostates, i.e. regions on the 1D FEP. This routine is called by :meth:`process_states` to add the macrostates found by the latter. These macrostates will be visualized as horizontal linesegments on a plot made by the :meth:`plot` routine with free energy contributions as computed by the :meth:`macrostate` routine.

            :param cvs: list of integers giving defining the index of the boundaries between subsequent macrostates.
            :type cvs: list(int)

            :param verbose: set to True to increase verbosity
            :type verbose: bool, optional, default=False
        '''
        for i in range(len(cvs)-1):
            self.macrostates.append(self.macrostate(cvrange=cvs[i:i+2], verbose=verbose))

    def plot(self, fn, rate=None, micro_marker='s', micro_color='r', micro_size='4', macro_linestyle='-', macro_color='b', do_latex=False, fig_size=[16,8]):
        '''
            Plot the free energy profile including visualization of the microstates (markers) and macrostates (lines) defined by :meth:`set_microstates` and :meth:`set_macrostates` respectively. The values of CV and free energy are plotted in units specified by the cv_unit and f_output_unit attributes of the self instance.

            :param fn: file name of the resulting plot, the extension of the file name will determine the format (png or pdf)
            :type fn: str

            :param rate: rate factor instance from :mod:`thermolib.kinetics` module which will allow to include indication of reaction rate and phenomenological free energy barriers, defaults to None
            :type rate: thermolib.kinetics.rate.RateFactorEquilibrium or thermolib.kinetics.rate.RateFactorAlternative, optional

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
        '''
        rc('text', usetex=do_latex)
        pp.clf()
        fig = pp.gcf()
        gs  = gridspec.GridSpec(1,2, width_ratios=[2,1])
        ax = fig.add_subplot(gs[0])
        axs = [ax]
        #make free energy plot
        axs[0].plot(self.cvs/parse_unit(self.cv_output_unit), self.fs/parse_unit(self.f_output_unit), linewidth=1, color='0.2')
        if self.flower is not None and self.fupper is not None:
            axs[0].fill_between(self.cvs/parse_unit(self.cv_output_unit), self.flower/parse_unit(self.f_output_unit), self.fupper/parse_unit(self.f_output_unit), alpha=0.33)
        #plot vline for transition state if defined
        cv_width = max(self.cvs)-min(self.cvs)
        ylower = -1+min([state[3]/kjmol for state in self.macrostates]+[0])
        if self.its is not None:
            axs[0].axvline(x=self.cvs[self.its]/parse_unit(self.cv_output_unit), linestyle='--', color='k', linewidth=1)
            axs[0].text((self.cvs[self.its]+0.01*cv_width)/parse_unit(self.cv_output_unit), 0, '%.3f' %(self.cvs[self.its]/parse_unit(self.cv_output_unit)), color='k', fontsize=12)
            #plot lines for the limits defining the TS region
            if rate is not None:
                axs[0].fill_betweenx([ylower, max(self.fs)/kjmol], x1=rate.CV_TS_lims[0]/parse_unit(self.cv_output_unit), x2=rate.CV_TS_lims[1]/parse_unit(self.cv_output_unit), alpha=0.33, color='k')
        #plot microstates
        for i, cv, f in self.microstates:
            axs[0].plot(cv/parse_unit(self.cv_output_unit), f/parse_unit(self.f_output_unit), linestyle='none', marker=micro_marker, color=micro_color, markersize=micro_size)
            axs[0].text((cv+0.01*cv_width)/parse_unit(self.cv_output_unit), f/parse_unit(self.f_output_unit), '%.1f' %(f/parse_unit(self.f_output_unit)), color=micro_color, fontsize=12)
        #plot macrostates
        for mean, std, Z, F in self.macrostates:
            xcen = (mean-min(self.cvs))/cv_width
            axs[0].axhline(y=F/parse_unit(self.f_output_unit), xmin=xcen-0.075, xmax=xcen+0.075, linestyle=macro_linestyle, color=macro_color, linewidth=2)
            axs[0].text((mean+0.075*cv_width)/parse_unit(self.cv_output_unit), F/parse_unit(self.f_output_unit), '%.1f' %(F/parse_unit(self.f_output_unit)), color=macro_color, fontsize=12)
        #cv_output_unit
        axs[0].set_xlabel('%s [%s]' %(self.cv_label, self.cv_output_unit), fontsize=14)
        axs[0].set_ylabel('Energy [%s]' %self.f_output_unit, fontsize=14)
        axs[0].set_title('Free energy profile F(CV)', fontsize=16)
        axs[0].set_xlim([min(self.cvs/parse_unit(self.cv_output_unit)), max(self.cvs/parse_unit(self.cv_output_unit))])
        axs[0].set_ylim([ylower, max(self.fs[~np.isnan(self.fs)])/kjmol+10])
        axs[0].axhline(y=0, xmin=0, xmax=1, linestyle='--', color='k', linewidth=1)

        if len(self.macrostates)>0:
            assert len(self.macrostates)==2, 'The plotter assumes two macrostates (if any), i.e. R and P, but found %i' %(len(self.macrostates))
            Zr,Fr = self.macrostates[0][2], self.macrostates[0][3]
            Zp,Fp = self.macrostates[1][2], self.macrostates[1][3]
            Fts = self.fs[self.its]
            if do_latex:
                fig.text(0.65, 0.88, r'\textit{Thermodynamic properties}', fontsize=16)
                fig.text(0.65, 0.86, '-------------------------------------', fontsize=16)
                fig.text(0.65, 0.82, r'$Z_{R} =\ $'+format_scientific(Zr/parse_unit(self.cv_output_unit))+'  %s' %(self.cv_output_unit), fontsize=16)
                fig.text(0.65, 0.78, r'$F_{R} = -k_B T \log(Z_{R}) = %.3f\ \ $ kJ.mol$^{-1}$' %(Fr/kjmol), fontsize=16)
                fig.text(0.65, 0.74, r'$Z_{P} =\ $'+format_scientific(Zp/parse_unit(self.cv_output_unit))+'  %s' %(self.cv_output_unit), fontsize=16)
                fig.text(0.65, 0.70, r'$F_{P} = -k_B T \log(Z_{P}) = %.3f\ \ $ kJ.mol$^{-1}$' %(Fp/kjmol), fontsize=16)
                fig.text(0.65, 0.66, r'$F(q_{TS}) = %.3f\ \ $ kJ.mol$^{-1}$' %(Fts/kjmol), fontsize=16)
            else:
                fig.text(0.65, 0.88, 'Thermodynamic properties', fontsize=16)
                fig.text(0.65, 0.86, '-------------------------------------', fontsize=16)
                fig.text(0.65, 0.82, 'ZR = ' + format_scientific(Zr/parse_unit(self.cv_output_unit))+'%s' %(self.cv_output_unit), fontsize=16)
                fig.text(0.65, 0.78, 'FR = -kT log(ZR) = %.3f kJ/mol' %(Fr/kjmol), fontsize=16)
                fig.text(0.65, 0.74, 'ZP = '+format_scientific(Zp/parse_unit(self.cv_output_unit))+'%s' %(self.cv_output_unit), fontsize=16)
                fig.text(0.65, 0.70, 'FP = -kT log(ZP) = %.3f kJ/mol' %(Fp/kjmol), fontsize=16)
                fig.text(0.65, 0.66, 'F(q_TS) = %.3f kJ/mol' %(Fts/kjmol), fontsize=16)
        if rate is not None:
            k_forward = rate.A*np.exp(-Fts/(boltzmann*self.T))/Zr
            k_backward = rate.A*np.exp(-Fts/(boltzmann*self.T))/Zp
            dF_forward = Fts+boltzmann*self.T*np.log(boltzmann*self.T*Zr/(planck*rate.A))
            dF_backward = Fts+boltzmann*self.T*np.log(boltzmann*self.T*Zp/(planck*rate.A))
            if do_latex:
                fig.text(0.65, 0.58, r'\textit{Kinetic properties}', fontsize=16)
                fig.text(0.65, 0.56, '-------------------------------------', fontsize=16)
                fig.text(0.65, 0.50, r'$A  =\ $' + format_scientific(rate.A/(parse_unit(self.cv_output_unit)/second)) + r' %s.s$^{-1}$' %(self.cv_output_unit), fontsize=16)
                fig.text(0.65, 0.44, r'$k_{F} = A \frac{e^{-\beta\cdot F(q_{TS})}}{Z_{R} } =\ $ ' +format_scientific(k_forward*second)  + r' s$^{-1}$', fontsize=16)
                fig.text(0.65, 0.38, r'$k_{B} = A \frac{e^{-\beta\cdot F(q_{TS})}}{Z_{P} } =\ $ ' +format_scientific(k_backward*second) + r' s$^{-1}$', fontsize=16)
                fig.text(0.65, 0.30, r'\textit{Phenomenological barrier}', fontsize=16)
                fig.text(0.65, 0.28, '-------------------------------------', fontsize=16)
                fig.text(0.65, 0.22, r'$k = \frac{k_B T}{h}e^{-\beta\cdot\Delta F}$', fontsize=16)
                fig.text(0.65, 0.18, r'$\Delta F_{F}  = %.3f\ \ $ kJ.mol$^{-1}$' %(dF_forward/kjmol), fontsize=16)
                fig.text(0.65, 0.14, r'$\Delta F_{B}  = %.3f\ \ $ kJ.mol$^{-1}$' %(dF_backward/kjmol), fontsize=16)
            else:
                fig.text(0.65, 0.58, 'Kinetic properties', fontsize=16)
                fig.text(0.65, 0.56, '-------------------------------------', fontsize=16)
                fig.text(0.65, 0.50, 'A  = ' + format_scientific(rate.A/(parse_unit(self.cv_output_unit)/second)) + r' %s/s' %(self.cv_output_unit), fontsize=16)
                fig.text(0.65, 0.44, 'kF = A exp(-F(q_TS)/kT)/ZR = ' +format_scientific(k_forward*second)  + r' 1/s', fontsize=16)
                fig.text(0.65, 0.38, 'kB = A exp(-F(q_TS)/kT)/ZP = ' +format_scientific(k_backward*second) + r' 1/s', fontsize=16)
                fig.text(0.65, 0.30, 'Phenomenological barrier', fontsize=16)
                fig.text(0.65, 0.28, '-------------------------------------', fontsize=16)
                fig.text(0.65, 0.22, 'k = kT/h exp(-dF/kT)', fontsize=16)
                fig.text(0.65, 0.18, 'dF_F = %.3f kJ/mol' %(dF_forward/kjmol), fontsize=16)
                fig.text(0.65, 0.14, 'dF_B = %.3f kJ/mol' %(dF_backward/kjmol), fontsize=16)
        #save
        fig.set_size_inches(fig_size)
        pp.savefig(fn)
        return


class FreeEnergySurface2D(object):
    def __init__(self, cv1s, cv2s, fs, temp, fupper=None, flower=None, cv1_output_unit='au', cv2_output_unit='au', f_output_unit='kjmol', cv1_label='CV1', cv2_label='CV2'):
        '''
            Class implementing a 2D free energy surface F(cv1,cv2) (stored in self.fs) as function of two collective variables (CV) denoted by cv1 (stored in self.cv1s) and cv2 (stored in self.cv2s).

            :param cv1s: array containing the values for the first collective variable CV1 in atomic units.
            :type cv1s: np.ndarray

            :param cv2s: array the values for the second collective variable CV2 in atomic units.
            :type cv2s: np.ndarray

            :param fs: 2D array containing the free energy values corresponding to the given values of CV1 and CV2  in atomic units. 
            :type fs: np.ndarray

            :param temp: temperature at which the free energy is given in atomic units.
            :type temp: float

            :param fupper: upper value of error bar on free energy in atomic units.
            :type fupper: np.ndarray

            :param flower: lower value of error bar on free energy in atomic units.
            :type flower: np.ndarray

            :param cv1_output_unit: unit in which the CV1 values will be printed/plotted, not the unit in which the input array is given (which is assumed to be atomic units). Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_.
            :type cv1_output_unit: str, optional, defaults to 'au'

            :param cv2_output_unit: unit in which the CV2 values will be printed/plotted, not the unit in which the input array is given (which is assumed to be atomic units). Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_
            :type cv2_output_unit: str, optional, defaults to 'au'

            :param f_output_unit: unit in which the free energy values will be printe/plotted, not the unit in which the input array f is given (which is assumed to be kjmol). Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_.
            :type f_output_unit: str, optional, default='kjmol'

            :param cv1_label: label for CV1 axis on plots
            :type cv1_label: str, optional, default='CV1'

            :param cv2_label: label for CV2 axis on plots
            :type cv2_label: str, optional, default='CV2'
        '''
        self.cv1s = cv1s.copy()
        self.cv2s = cv2s.copy()
        self.fs   = fs.copy()
        if fupper is not None:
            self.fupper = fupper.copy()
        else:
            self.fupper = None
        if flower is not None:
            self.flower = flower.copy()
        else:
            self.flower = None
        self.T = temp
        self.cv1_output_unit = cv1_output_unit
        self.cv2_output_unit = cv2_output_unit
        self.f_output_unit = f_output_unit
        self.cv1_label = cv1_label
        self.cv2_label = cv2_label
        self.compute_probdens()

    def _beta(self):
        return 1.0/(boltzmann*self.T)

    beta = property(_beta)

    def copy(self):
        '''
            Make and return a copy of the current FreeEnergySurface2D instance.

            :return: a copy of the current instance
            :rtype: FreeEnergySurface2D
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
    def from_txt(cls, fn, temp, cv1_col=0, cv2_col=1, f_col=2, cv1_input_unit='au', cv1_output_unit='au', cv2_input_unit='au', cv2_output_unit='au', f_output_unit='kjmol', f_input_unit='kjmol', cv1_label='CV1', cv2_label='CV2', cv1_range=None, cv2_range=None, delimiter=None, verbose=False):
        '''
            Read the free energy surface on a 2D grid as function of two collective variables from a txt file. 

            :param fn: the name of the txt file containing the data. It is assumed this file can be read by the numpyt.loadtxt routine.
            :type fn: str

            :param temp: the temperature at which the free energy is constructed
            :type temp: float

            :param cv1_col: the column in which the first collective variable is stored
            :type cv1_col: int, optional, default=0

            :param cv2_col: the column in which the second collective variable is stored
            :type cv2_col: int, optional, default=1

            :param f_col: the column in which the free energy is stored
            :type f_col: int, optional, default=2

            :param cv1_input_unit: the unit in which the first CV values are stored in the input file
            :type cv1_unit: str, optional, default='au'

            :param cv1_output_unit: unit in which the CV1 values will be printed/plotted, not the unit in which the input array is given (which is given by cv1_input_unit). Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_.
            :type cv1_output_unit: str, optional, default='au'

            :param cv2_input_unit: the unit in which the second CV values are stored in the input file
            :type cv2_input_unit: str, optional, default='au'

            :param cv2_output_unit: unit in which the CV2 values will be printed/plotted, not the unit in which the input array is given (which is given by cv2_input_unit). Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_
            :type cv2_output_unit: str, optional, default='au'

            :param f_input_unit: the unit in which the free energy values are stored in the input file
            :type f_input_unit: str, optional, default='kjmol'

            :param f_output_unit: unit in which the free energy values will be printed/plotted, not the unit in which the input array is given (which is given by f_input_unit). Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_
            :type cv2_output_unit: str, optional, default='kjmol'

            :param cv1_label: the label for the CV1 axis in plots
            :type cv1_label: str, optional, default='CV1'

            :param cv2_label: the label for the CV2 axis in plots, defaults to 'CV2'
            :type cv2_label: str, optional, default='CV2'

            :param cv1_range: [CVmin,CVmax] indicating to only read free energy for which the first CV in the given range
            :type cv1_range: tuple/list, optional, default=None

            :param cv2_range: [CVmin,CVmax] indicating to only read free energy for which the second CV in the given range
            :type cv2_range: tuple/list, optional, default=None

            :param delimiter: the delimiter used in the numpy input file, this argument is parsed to the numpy.loadtxt routine.
            :type delimiter: str, optional, default=None

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
        return cls(cv1s, cv2s, fs, temp, cv1_output_unit=cv1_output_unit, cv2_output_unit=cv2_output_unit, f_output_unit=f_output_unit, cv1_label=cv1_label, cv2_label=cv2_label)

    @classmethod
    def from_histogram(cls, histogram, temp):
        '''
            Use the estimated 2D probability histogram to construct the corresponding 2D free energy surface at the given temperature.
        
            :param histogram: histogram from which the free energy profile is computed
            :type histogram: histogram.Histogram2D

            :param temp: the temperature at which the histogram input data was simulated
            :type temp: float

            :return: free energy profile corresponding to the estimated probability histogram
            :rtype: cls
        '''
        fupper = None
        flower = None
        fs = np.zeros(histogram.ps.shape)*np.nan
        fs[histogram.ps>0] = -boltzmann*temp*np.log(histogram.ps[histogram.ps>0])
        if histogram.pupper is not None and histogram.plower is not None:
            fupper = np.zeros(histogram.ps.shape)*np.nan
            flower = np.zeros(histogram.ps.shape)*np.nan
            fupper[histogram.plower>0] = -boltzmann*temp*np.log(histogram.plower[histogram.plower>0])
            flower[histogram.pupper>0] = -boltzmann*temp*np.log(histogram.pupper[histogram.pupper>0])
        return cls(histogram.cv1s, histogram.cv2s, fs, temp, fupper=fupper, flower=flower, cv1_output_unit=histogram.cv1_output_unit, cv2_output_unit=histogram.cv2_output_unit, cv1_label=histogram.cv1_label, cv2_label=histogram.cv2_label)

    def savetxt(self, fn_txt):
        '''
            Save the free energy profile as txt file. The units in which the CVs and free energy are written is specified in the attributes cv1_output_unit, cv2_output_unit and f_output_unit of the self instance.
        '''
        header = '%s [%s]\t %s [%s]\t Free energy [%s]' %(self.cv1_label, self.cv1_output_unit,self.cv2_label, self.cv2_output_unit, self.f_output_unit)
        xv,yv = np.meshgrid(self.cv1s,self.cv2s)
        np.savetxt(fn_txt, np.vstack((yv.flatten()/parse_unit(self.cv1_output_unit),xv.flatten()/parse_unit(self.cv2_output_unit), self.fs.flatten()/parse_unit(self.f_output_unit))).T, header=header,fmt='%f')

    def compute_probdens(self):
        '''
            Compute the probability density profile associated with the free energy profile as given below and store internally in `self.ps`

            .. math:: p(q) = \\frac{\\exp\\left(-\\beta F(q)\\right)}{\\int_{-\\infty}^{+\\infty}\\exp\\left(-\\beta F(q)\\right)dq}
        '''
        self.ps = np.exp(-self.beta*self.fs)
        self.ps[np.isnan(self.ps)] = 0.0
        self.ps /= integrate2d(self.ps, x=self.cv1s, y=self.cv2s)

    def set_ref(self, ref='min'):
        '''
            Set the energy reference of the free energy surface.

            :param ref: the choice for the energy reference. Currently only one possibility is implemented, i.e. *m* or *min* for the global minimum.
            :type ref: str, default='min'
            
            :raises IOError: invalid value for keyword argument ref is given. See doc above for choices.
        '''
        if ref.lower() in ['m', 'min']:
            fref = self.fs[~np.isnan(self.fs)].min()
        else:
            raise IOError('Invalid REF specification, recieved %s and should be min' %ref)
        self.fs -= fref
        if self.fupper is not None:
            self.fupper -= fref
        if self.flower is not None:
            self.flower -= fref

    def detect_clusters(self, eps=1.5, min_samples=8, metric='euclidean', fn_plot=None, delete_clusters=[-1]):
        '''
            Routine to apply `the DBSCAN clustering algoritm as implemented in the Scikit Learn package <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_ to the (CV1,CV2) grid points that correspond to finite free energies (i.e. not nan or inf) to detect clusters of neighboring points.

            The DBSCAN algorithm first identifies the core samples, defined as samples for which at least ``min_samples`` other samples are within a distance of ``eps``. Next, the data is divided into clusters based on these core samples. A cluster is defined as a set of core samples that can be built by recursively taking a core sample, finding all of its neighbors that are core samples, finding all of their neighbors that are core samples, and so on. A cluster also has a set of non-core samples, which are samples that are neighbors of a core sample in the cluster but are not themselves core samples. Intuitively, these samples are on the fringes of a cluster. Each cluster is given an integer as label.

            Any sample that is not a core sample, and is at least ``eps`` in distance from any core sample, is considered an outlier by the algorithm and is what we here consider an isolated point/region. These points get the cluster label of -1.

            Finally, all data points belonging to a cluster with label specified in ``delete_clusters`` will have theire free energy set to nan. A safe choice here is to just delete isolated regions, i.e. the point in cluster with label -1 (which is the default).

            :param eps: DBSCAN parameter representing maximum distance between two samples for them to be considered neighbors (for more, see `DBSCAN documentation <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_), defaults to 1.5
            :type eps: float, optional

            :param min_samples: DBSCAN parameter representing the number of samples in a neighborhood for a point to be considered a core point (for more, see `DBSCAN documentation <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_), defaults to 8
            :type min_samples: int, optional

            :param metric: DBSCAN parameter representing the metric used when calculating distance (for more, see `DBSCAN documentation <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_), defaults to 'euclidean'
            :type metric: str or callable, optional

            :param fn_plot: if specified, a plot will be made (and written to ``fn_plot``) visualizing the resulting clusters, defaults to None
            :type fn_plot: str, optional

            :param delete_clusters: list of cluster names whos members will be deleted from the free energy surface data, defaults to [-1] meaning only isolated points (not belonging to a cluster) will be deleted.
            :type delete_clusters: list, optional
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
            self.compute_probdens()

    def crop(self, cv1range=None, cv2range=None, return_new_fes=False):
        '''
            Crop the free energy surface by removing all data for which either cv1 (along x-axis) or cv2 (along y-axis) is beyond a given range.
            
            :param cv1range: range of cv1 (along x-axis) that will remain after cropping, defaults to None
            :type cv1range: [type], optional

            :param cv2range: range of cv2 (along y-axis) that will remain after cropping, defaults to None
            :type cv2range: [type], optional

            :param return_new_fes: if set to false, the cropping process will be applied on the existing instance, otherwise a copy will be returned, defaults to False
            :type return_new_fes: bool, optional

            :return: None or new instance of :class:`FreeEnergySurface2D` representing cropped FES, depending on ``return_new_fes``
        '''
        #cut off some unwanted regions
        cv1s = self.cv1s.copy()
        cv2s = self.cv2s.copy()
        fs = self.fs.copy()
        if cv1range is not None:
            indexes = []
            for i, cv1 in enumerate(cv1s):
                if cv1range[0]<=cv1<=cv1range[1]:
                    indexes.append(i)
            cv1s = cv1s[np.array(indexes)]
            fs = fs[:,np.array(indexes)]
        if cv2range is not None:
            indexes = []
            for i, cv2 in enumerate(cv2s):
                if cv2range[0]<=cv2<=cv2range[1]:
                    indexes.append(i)
            cv2s = cv2s[np.array(indexes)]
            fs = fs[np.array(indexes),:]
        if return_new_fes:
            return FreeEnergySurface2D(cv1s, cv2s, fs, self.T, cv1_output_unit=self.cv1_output_unit, cv2_output_unit=self.cv2_output_unit, f_output_unit=self.f_output_unit, cv1_label=self.cv1_label, cv2_label=self.cv2_label)
        else:
            self.cv1s = cv1s.copy()
            self.cv2s = cv2s.copy()
            self.fs = fs.copy()
            self.compute_probdens()

    def rotate(self, interpolate=True):
        '''
            Transform the free energy profile in terms of the following two new collective variables:

            .. math::

                \\begin{aligned}
                    u &= \\frac{CV_1+CV_2}{2} \\\\
                    v &= CV_2 - CV_1
                \\end{aligned}

            This transformation represents a simple rotation (and mirroring). From probability theory we find the transformation formula:

            .. math:: F_\\text{rot}(u,v) = F\\left(u-\\frac{v}{2}, u+\\frac{v}{2}\\right))

            The uniform (u,v)-grid introduces new grid points in between the original (cv1,cv2) grid points. If interpolate is True, the free energy for these points is interpolated if all four neighbors have defined (i.e. not nan) free energies.

            :param interpolate: if set to true, interpolate undefined grid points (arrising due to rotation) between neighbors, defaults to True
            :type interpolate: bool, optional

            :return: rotated free energy surface
            :rtype: FreeEnergySurface2D
        '''
        #First make unique list of us and vs
        print('Making unique arrays for u=0.5*(cv1+cv2) and v=cv2-cv1')
        us = []
        vs = []
        for i1, cv1 in enumerate(self.cv1s):
            for i2, cv2 in enumerate(self.cv2s):
                u = 0.5*(cv1+cv2)
                v = cv2-cv1
                if len(np.where(abs(u-np.array(us))<1e-6)[0])==0:
                    us.append(u)
                v = cv2-cv1
                if len(np.where(abs(v-np.array(vs))<1e-6)[0])==0:
                    vs.append(v)
        us = np.array(sorted(us))
        vs = np.array(sorted(vs))
        #Second make dictionairy to look up i1,i2 corresponding to cv1,cv2 from
        #iu,iv corresponding to u,v
        print('Making look-up dictionairy for u,v values')
        dic = {}
        for i1, cv1 in enumerate(self.cv1s):
            for i2, cv2 in enumerate(self.cv2s):
                #find iu
                u = 0.5*(cv1+cv2)
                indexes = np.where(abs(us-u)<1e-6)[0]
                assert len(indexes)==1, 'Len(indexes) should be 1, got %i' %(len(indexes))
                iu = indexes[0]
                #find iv
                v = cv2-cv1
                indexes = np.where(abs(vs-v)<1e-6)[0]
                assert len(indexes)==1, 'Len(indexes) should be 1, got %i' %(len(indexes))
                iv = indexes[0]
                #store
                dic['%i,%i' %(iu,iv)] = (i1,i2)
        #Third, construct rotated free energy
        print('Constructing rotated free energy')
        fs = np.zeros([len(us), len(vs)], float)*np.nan
        fupper, flower = None, None
        if self.fupper is not None and self.flower is not None:
            fupper = np.zeros([len(us), len(vs)], float)*np.nan
            flower = np.zeros([len(us), len(vs)], float)*np.nan
        for iu, u in enumerate(us):
            for iv, v in enumerate(vs):
                key = '%i,%i' %(iu,iv)
                if key in dic.keys():
                    (i1,i2) = dic[key]
                    assert abs(0.5*(self.cv1s[i1]+self.cv2s[i2]) - u)<1e-8
                    assert abs(    (self.cv2s[i2]-self.cv1s[i1]) - v)<1e-8
                    fs[iu,iv] = self.fs[i2,i1]
                    if fupper is not None and flower is not None:
                        fupper[iu,iv] = self.fupper[i2,i1]
                        flower[iu,iv] = self.flower[i2,i1]
        #Fourth, do interpolate
        if interpolate:
            print('Interpolating extra intermediate grid points')
            for iu, u in enumerate(us):
                for iv, v in enumerate(vs):
                    if np.isnan(fs[iu,iv]) and (iu not in [0,len(us)-1]) and (iv not in [0,len(vs)-1]):
                        fl = fs[iu-1,iv]
                        fr = fs[iu+1,iv]
                        fb = fs[iu,iv-1]
                        fa = fs[iu,iv+1]
                        if not (np.isnan(fl) or np.isnan(fr) or np.isnan(fb) or np.isnan(fa)):
                            fs[iu,iv] = 0.25*(fl+fr+fb+fa)
                            if fupper is not None and flower is not None:
                                fupper[iu,iv] = 0.25*(fupper[iu-1,iv] + fupper[iu+1,iv] + fupper[iu,iv-1] + fupper[iu,iv+1])
                                flower[iu,iv] = 0.25*(flower[iu-1,iv] + flower[iu+1,iv] + flower[iu,iv-1] + flower[iu,iv+1])
        if self.cv1_output_unit==self.cv2_output_unit:
            u_unit = self.cv1_output_unit
            v_unit = self.cv1_output_unit
        else:
            u_unit = 'au'
            v_unit = 'au'
        return FreeEnergySurface2D(vs, us, fs, self.T, fupper=fupper, flower=flower, cv1_output_unit=v_unit, cv2_output_unit=u_unit, f_output_unit=self.f_output_unit, cv1_label='CV2-CV1', cv2_label='0.5*(CV1+CV2)')

    def transform(self, function, derivative=None, Q1_label='Q1', Q2_label='Q2', Q1_output_unit='au', Q2_output_unit='au'):
        '''
            Routine to transform the current free energy surface in terms of the original (CV1,CV2) towards a free energy profile in terms of the new collective variables (Q1,Q2).

            :param function: The transformation function (Q1,Q2) = f_1(CV1,CV2),f_2(CV1,CV2)
            :type function: callable

            :param derivative: The analytical derivative of the transformation function, i.e. the Jacobian J = [[J_11(CV1,CV2),J_12(CV1,CV2)],[J_21(CV1,CV2),J_22(CV1,CV2)]] given by J_mn(CV1,CV2) = df_m/dCV_n. If set to None, the derivative will be estimated through numerical differentiation. Defaults to None
            :type derivative: np.array(callable), optional

            :param Q1_label: The label of the new collective variable Q1 used in plotting etc, defaults to 'Q1'
            :type Q1_label: str, optional

            :param Q2_label: The label of the new collective variable Q2 used in plotting etc, defaults to 'Q2'
            :type Q2_label: str, optional

            :param Q1_output_unit: The unit of the new collective variable Q1 used in plotting and printing, defaults to 'au'
            :type Q1_output_unit: str, optional

            :param Q2_output_unit: The unit of the new collective variable Q2 used in plotting and printing, defaults to 'au'
            :type Q2_output_unit: str, optional

            :return: transformed free energy surface
            :rtype: the same class as the instance this routine is called upon
        '''
        raise NotImplementedError
        #TODO: not tested yet!
        f1,f2 = function
        q1s, q2s = f1(self.cv1s,self.cv2s), f2(self.cv1s,self.cv2s)
        if derivative is None:
            eps = min(qs[1:]-qs[:-1])*0.001
            def J11(cv1,cv2):
                return (f1(cv1+eps/2,cv2)-f1(cv1-eps/2,cv2))/eps
            def J12(cv1,cv2):
                return (f1(cv1,cv2+eps/2)-f1(cv1,cv2-eps/2))/eps
            def J21(cv1,cv2):
                return (f2(cv1+eps/2,cv2)-f2(cv1-eps/2,cv2))/eps
            def J22(cv1,cv2):
                return (f2(cv1,cv2+eps/2)-f2(cv1,cv2-eps/2))/eps
            def jacobian(cv1,cv2):
                raise NotImplementedError
        dfs = derivative(qs)
        fs = self.fs - np.log(dfs)/self.beta
        fupper, flower = None, None
        if self.fupper is not None:
            fupper = self.fupper - np.log(dfs)/self.beta
        if self.flower is not None:
            flower = self.flower - np.log(dfs)/self.beta
        return self.__class__(qs, fs, self.T, fupper=fupper, flower=flower, cv_output_unit=cv_output_unit, f_output_unit=self.f_output_unit, cv_label=cv_label)

    def project_difference(self, sign=1, cv_output_unit='au', return_class=BaseFreeEnergyProfile):
        '''
            Construct a 1D free energy profile representing the projection of the 2D FES onto the difference of collective variables:

            .. math:: F1(q) = -k_B T \\log\\left( \\int_{-\\infty}^{+\\infty} e^{-\\beta F2(x,x+q)}dx \\right)

            with :math:`q=CV2-CV1`. This projection is implemented by first projecting the probability density and afterwards reconstructing the free energy.

            :param sign: If sign is set to 1, the projection is done on q=CV2-CV1, if it is set to -1, projection is done to q=CV1-CV2 instead. Defaults to 1
            :type sign: int, optional

            :param cv_output_unit: unit for the new CV for printing and plotting purposes
            :type cv_output_unit: str, optional, default='au'
            
            :param return_class: The class of which an instance will finally be returned.
            :type return_class: python class object, optional, default=BaseFreeEnergyProfile

            :returns: projected 1D free energy profile
            :rtype: see return_class argument
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
        return self.project_function(function, cvs, cv_label=cv_label, cv_output_unit=cv_output_unit, return_class=return_class)

    def project_average(self, cv_output_unit='au', return_class=BaseFreeEnergyProfile):
        '''
            Construct a 1D free energy profile representing the projection of the 2D FES F2(CV1,CV2) onto the average q=(CV1+CV2)/2 of the collective variables:

            .. math::
                
                F1(q) = -k_B T \\log\\left( 2\\int_{-\infty}^{+\infty} e^{-\\beta F2(x,2q-x}dx \\right)

            with :math:`q=0.5\\dot(CV1+CV2)`. This projection is implemented by first projecting the probability density and afterwards reconstructing the free energy.

            :param cv_output_unit: unit for the new CV for plotting and printing purposes
            :type cv_output_unit: str, optional, default='au'
            
            :param return_class: The class of which an instance will finally be returned.
            :type return_class: python class object, optional, default=BaseFreeEnergyProfile

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
        return self.project_function(function, cvs, cv_label='0.5*(%s+%s)' %(self.cv1_label,self.cv1_label), cv_output_unit=cv_output_unit, return_class=return_class)

    def project_cv1(self, return_class=BaseFreeEnergyProfile):
        '''
            Construct a 1D free energy profile representing the projection of the 2D FES F2(CV1,CV2) onto q=CV1. This is implemented as follows:

            .. math:: F1(q) = -k_B T \\log\\left( \\int_{-\infty}^{+\infty} e^{-\\beta F2(q,y}dy \\right)

            :param return_class: The class of which an instance will finally be returned.
            :type return_class: python class object, optional, default=BaseFreeEnergyProfile

            :returns: projected 1D free energy profile
            :rtype: see return_class argument
        '''
        def function(q1,q2):
            return q1
        return self.project_function(function, self.cv1s.copy(), cv_label=self.cv1_label, cv_output_unit=self.cv1_output_unit, return_class=return_class)

    def project_cv2(self, return_class=BaseFreeEnergyProfile):
        '''
            Construct a 1D free energy profile representing the projection of the 2D FES F2(CV1,CV2) onto q=CV2. This is implemented as follows:

                F1(q) = -k_B T \\log\\left( \\int_{-\infty}^{+\infty} e^{-\\beta F2(x,q}dx \\right)

            :param return_class: The class of which an instance will finally be returned.
            :type return_class: python class object, optional, default=BaseFreeEnergyProfile

            :returns: projected 1D free energy profile
            :rtype: see return_class argument
        '''
        def function(q1,q2):
            return q2
        return self.project_function(function, self.cv2s.copy(), cv_label=self.cv2_label, cv_output_unit=self.cv2_output_unit, return_class=return_class)

    def project_function(self, function, cvs, delta=1e-3, cv_label='CV', cv_output_unit='au', return_class=BaseFreeEnergyProfile):
        '''
            Routine to implement the general projection of a 2D FES onto a collective variable defined by the given function (which takes the original two CVs as arguments).

            :param function: function in terms of the original CVs to define the new CV to project upon
            :type function: callable

            :param cvs: grid for the new CV
            :type cvs: np.ndarray

            :param delta: width of the single-bin approximation of the delta function applied in the projection formula. The delta function is one whenever abs(function(cv1,cv2)-q)<delta/2. Hence, delta has the same units as the new collective variable q.
            :type delta: float, optional, default=1e-3
            
            :param cv_label: label for the new CV
            :type cv_label: str, optional, default='CV'

            :param cv_output_unit: unit for the new CV for printing and plotting purposes
            :type cv_output_unit: str, optional, default='au'

            :param return_class: The class of which an instance will finally be returned
            :type return_class: python class object, optional, default=BaseFreeEnergyProfile

            :returns: projected 1D free energy profile
            :rtype: see return_class argument
        '''
        CV1s, CV2s = np.meshgrid(self.cv1s, self.cv2s, indexing='xy')
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
            pqs = np.zeros(len(cvs), float)
            for i,q in enumerate(cvs):
                Hs = dirac(CV1s, CV2s, q)
                pqs[i] = integrate2d(P12s*Hs, dx=delta1, dy=delta2)/delta
            fqs = np.zeros(len(cvs))*np.nan
            fqs[pqs>0] = -np.log(pqs[pqs>0])/self.beta
            return fqs
        fs = project(self.fs)
        flower, fupper = None, None
        if self.fupper is not None:
            fupper = project(self.fupper)
        if self.flower is not None:
            flower = project(self.flower)
        return return_class(cvs, fs, self.T, fupper=fupper, flower=flower, cv_output_unit=cv_output_unit, f_output_unit=self.f_output_unit, cv_label=cv_label)

    def plot(self, fn_png, obs='F', cv1_lims=None, cv2_lims=None, lims=None, ncolors=8, scale='lin'):
        '''
            Simple routine to make a 2D contour plot of either the free energy F or probability distribution P as specified in ``obs``. The values of the CVs and the free energy will be plotted in units specified by the CV1_output_unit, CV2_output_unit and f_output_unit attributes of the self instance.

            :param fn_png: File name to write the plot to. The extension determines the format (PNG or PDF).
            :type fn_png: str

            :param obs: Specification which observable should be plotted, should be 'F' for free energy or 'P' for probability.
            :type obs: str, optional, choices=('F','P'), default='F'

            :param cv1_lims: range defining the plot limits of CV1
            :type cv1_lims: tupple/list, optional, default=None

            :param cv2_lims: range defining the plot limits of CV1
            :type cv2_lims: tupple/list, optional, default=None

            :param lims: range defining the plot limits for the observable
            :type lims: tupple/list, optional, default=None

            :param ncolors: number of different colors included in contour plot
            :type ncolors: int, optional, default=8

            :param scale: scal for the observable, should be 'lin' for linear or 'log' for logarithmic.
            :type scale: str, optional, choices=('lin', 'log'), default='lin'

            :raises IOError: if invalid observable is given, see doc above for possible choices.

            :raises IOError: if invalid scale is given, see doc above for possible choices.
        '''
        pp.clf()
        if obs.lower() in ['f', 'free energy']:
            obs = self.fs/parse_unit(self.f_output_unit)
            label = 'Free energy [%s]' %self.f_output_unit
        elif obs.lower() in ['p', 'probability']:
            obs = self.ps*(parse_unit(self.cv1_output_unit)*parse_unit(self.cv2_output_unit))
            label = r'Probability density [(%s*%s)$^{-1}$]' %(self.cv1_output_unit, self.cv2_output_unit)
        else:
            raise IOError('Recieved invalid observable. Should be f, free energy, p or probability but got %s' %obs)
        plot_kwargs = {}
        if lims is None:
            lims = [min(obs[~np.isnan(obs)]), max(obs[~np.isnan(obs)])]
        if lims is not None:
            if scale.lower() in ['lin', 'linear']:
                plot_kwargs['levels'] = np.linspace(lims[0], lims[1], ncolors+1)
            elif scale.lower() in ['log', 'logarithmic']:
                plot_kwargs['levels'] = np.logspace(lims[0], lims[1], ncolors+1)
                plot_kwargs['norm'] = LogNorm()
                plot_kwargs['locator'] = LogLocator()
            else:
                raise IOError('recieved invalid scale value, got %s, should be lin or log' %scale)
        contourf = pp.contourf(self.cv1s/parse_unit(self.cv1_output_unit), self.cv2s/parse_unit(self.cv2_output_unit), obs, cmap=pp.get_cmap('rainbow'), **plot_kwargs)
        contour = pp.contour(self.cv1s/parse_unit(self.cv1_output_unit), self.cv2s/parse_unit(self.cv2_output_unit), obs, **plot_kwargs)
        if cv1_lims is not None:
            pp.xlim(cv1_lims)
        if cv2_lims is not None:
            pp.ylim(cv2_lims)
        pp.xlabel('%s [%s]' %(self.cv1_label, self.cv1_output_unit), fontsize=16)
        pp.ylabel('%s [%s]' %(self.cv2_label, self.cv2_output_unit), fontsize=16)
        cbar = pp.colorbar(contourf, extend='both')
        cbar.set_label(label, fontsize=16)
        pp.clabel(contour, inline=1, fontsize=10)
        fig = pp.gcf()
        fig.set_size_inches([12,8])
        pp.savefig(fn_png)

def plot_feps(fn, feps, temp=None, labels=None, flims=None, colors=None, linestyles=None, linewidths=None, do_latex=False):
    '''
        Make a plot to compare multiple free energy profiles

        :param fn: file name to write the figure to, the extension determines the format (PNG or PDF).
        :type fn: str

        :param feps: list of free energy profiles to plot
        :type feps: list(BaseFreeEnergyProfile)

        :param temp: if temp is defined, an additional pane will be added to the plot containing the corresponding histograms at the specified temperature.
        :type temp: float, optional, default=None
        
        :param labels: list of labels for the legend, one for each histogram.
        :type labels: list(str), optional, default=None

        :param flims: [lower,upper] limits of the free energy axis in plots.
        :type flims: list/np.ndarray, optional, default=None

        :param colors: List of matplotlib color definitions for each entry in histograms. If an entry is None, a color will be chosen internally. Defaults to None, implying all colors are chosen internally.
		:type colors: List(str), optional

		:param linestyles: List of matplotlib line style definitions for each entry in histograms. If an entry is None, the default line style of '-' will be chosen . Defaults to None, implying all line styles are set to the default of '-'.
		:type linestyles: List(str), optional

		:param linewidths: List of matplotlib line width definitions for each entry in histograms. If an entry is None, the default line width of 1 will be chosen. Defaults to None, implying all line widths are set to the default of 2.
		:type linewidths: List(str), optional
    '''
    rc('text', usetex=do_latex)
    if temp is not None:
        from .histogram import Histogram1D
    #initialize
    linewidth_default = 2
    linestyle_default = '-'
    nfeps = len(feps)
    pp.clf()
    if temp is None:
        fig, axs = pp.subplots(nrows=1, ncols=1, squeeze=False)
    else:
        fig, axs = pp.subplots(nrows=1, ncols=2, squeeze=False)
    cmap = cm.get_cmap('tab10')
    cv_unit, f_unit = feps[0].cv_output_unit, feps[0].f_output_unit
    cv_label = feps[0].cv_label
    fmax = -np.inf
    #add free energy plots and possibly probability histograms
    for ifep, fep in enumerate(feps):
        label = 'FEP %i' %ifep
        if labels is not None:
            label = labels[ifep]
        #set color, linestyles, ...
        color = None
        if colors is not None:
            color = colors[ifep]
        if color is None:
            color = cmap(ifep)
        linewidth = None
        if linewidths is not None:
            linewidth = linewidths[ifep]
        if linewidth is None:
            linewidth = linewidth_default
        linestyle = None
        if linestyles is not None:
            linestyle = linestyles[ifep]
        if linestyle is None:
            linestyle = linestyle_default
        #plot free energy
        fmax = max(fmax, np.ceil(max(fep.fs[~np.isnan(fep.fs)]/kjmol)/10)*10)
        axs[0,0].plot(fep.cvs/parse_unit(cv_unit), fep.fs/parse_unit(f_unit), linewidth=linewidth, linestyle=linestyle, color=color, label=label)
        if fep.flower is not None and fep.fupper is not None:
            axs[0,0].fill_between(fep.cvs/parse_unit(cv_unit), fep.flower/parse_unit(f_unit), fep.fupper/parse_unit(f_unit), color=color, alpha=0.33)
        #histogram if requested
        if temp is not None:
            hist = Histogram1D.from_fep(fep, temp)
            axs[0,1].plot(hist.cvs/parse_unit(cv_unit), hist.ps/max(hist.ps), linewidth=linewidth, color=color, linestyle=linestyle, label=label)
            if hist.plower is not None and hist.pupper is not None:
                axs[0,1].fill_between(hist.cvs/parse_unit(cv_unit), hist.plower/max(hist.ps), hist.pupper/max(hist.ps), color=color, alpha=0.33)
    #decorate
    cv_range = [min(feps[0].cvs/parse_unit(cv_unit)), max(feps[0].cvs/parse_unit(cv_unit))]
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
    if temp is not None:
        axs[0,1].set_xlabel('%s [%s]' %(cv_label, cv_unit), fontsize=14)
        axs[0,1].set_ylabel('Relative probability [-]', fontsize=14)
        axs[0,1].set_title('Probability histogram', fontsize=14)
        axs[0,1].set_xlim(cv_range)
        axs[0,1].legend(loc='best')
    #save
    if temp is not None:
        fig.set_size_inches([16,8])
    else:
        fig.set_size_inches([8,8])
    pp.savefig(fn)
