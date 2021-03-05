#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 - 2019 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
# Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium;
# all rights reserved unless otherwise stated.
#
# This file is part of a library developed by Louis Vanduyfhuys at
# the Center for Molecular Modeling under supervision of prof. Veronique
# Van Speybroeck. Usage of this package should be authorized by prof. Van
# Van Speybroeck.

from molmod.units import *
from molmod.constants import *

from ..tools import integrate, integrate2d, format_scientific, free_energy_from_histogram_with_error

import matplotlib.pyplot as pp
import matplotlib.cm as cm
from matplotlib import gridspec, rc
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

#clustering algorithm
#from sklearn.cluster import DBSCAN

import numpy as np

import sys

__all__ = ['BaseFreeEnergyProfile', 'SimpleFreeEnergyProfile']

class BaseFreeEnergyProfile(object):
    def __init__(self, cv, f, temp, cv_unit='au', f_unit='kjmol', cv_label='CV'):
        '''
            Class to define a free energy profile as function of a certain
            collective variable (CV).

            The corresponding probability density
            can also be extracted:

                p(cv) = exp(-beta*F(cv))/Z

                with

                Z = int(exp(-beta*F(cv)), cv=-inf...+inf)

            Finally, one can also extract the partition function and free
            energy of a certain macrostate defined by its range in the cv. This
            partition function is normalized like a probability density.

                Z_A = int(exp(-beta*F(cv)), cv in A)/Z
                    = int(p(cv), cv in A)/Z

                F_A = -kT*log(Z_A)

            **Arguments**

            cv
                numpy array representing the collective variable axis

            f
                numpy array representing the free energy axis

            temp
                the temperature at which the free energy is constructed

            **Optional Arguments**

            cv_unit
                the units for printing of CV values

            f_unit
                the units for printing of free energy values
        '''
        assert len(cv)==len(f), "cv and f array should be of same length"
        self.cvs = cv.copy()
        self.fs = f.copy()
        self.T = temp
        self.cv_unit = cv_unit
        self.f_unit = f_unit
        self.cv_label = cv_label
        self.microstates = []
        self.macrostates = []
        self.compute_probdens()

    def _beta(self):
        return 1.0/(boltzmann*self.T)

    beta = property(_beta)

    @classmethod
    def from_txt(cls, fn, temp, cvcol=0, fcol=1, cv_unit='au', f_unit='kjmol', cvrange=None, delimiter=None, reverse=False, cut_constant=False):
        '''
            Read the free energy profile as function of a collective variable
            from a txt file.

            **Arguments**

            fn
                the name of the txt file containing the data

            temp
                the temperature at which the free energy is constructed

            **Optional Arguments**

            cvcol
                the column in which the collective variable is stored

            fcol
                the column in which the free energy is stored

            cv_unit
                the units in which the CV are stored [default=atomic units]

            f_unit
                the units in which the free energy are stored [default=kjmol]

            cvrange
                a tuple/list [CVmin,CVmax], only read free energy for CVs in the
                given range.

            reverse
                if set to True, reverse the X axis (usefull to make sure
                reactant is on the left)

            cut_constant
                if set to True, the data points at the start and end of the
                data array that are constant will be cut. Usefull to cut out
                unsampled areas for large and small CV values.

        '''
        #TODO: deal with inf values in input file

        data = np.loadtxt(fn, delimiter=delimiter, dtype=float)
        cvs = data[:,cvcol]*parse_unit(cv_unit)
        fs = data[:,fcol]*parse_unit(f_unit)
        if reverse:
            cvs = cvs[::-1]
            fs = fs[::-1]
        if cvrange is not None:
            indexes = []
            for i, cv in enumerate(cvs):
                if cvrange[0]<=cv and cv<=cvrange[1]:
                    indexes.append(i)
            cvs = cvs[np.array(indexes)]
            fs = fs[np.array(indexes)]
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
        return cls(cvs, fs, temp, cv_unit=cv_unit, f_unit=f_unit)

    @classmethod
    def from_trajectory_histogram(cls, cv_data, temp, cv_bins=None, n_bins=100, stride=1, nblocks=1, fn_plot=None, cv_unit='au', f_unit='kjmol', cv_label='CV'):
        '''
            Derive a free energy profile in term of the given CV from the
            histogram of a molecular trajectory.

            **Arguments**

                cv_data     a numpy array containig the CV values along the
                            trajectory. This array can be computed from an
                            XYZ trajectory file using the <trajectory_xyz_to_CV>
                            routine in the thermolib.tools module.
                            
                temp        temperature of the trajectory

            **Optional Arguments**

                cv_bins     CV bin edges for histogram construction. If not
                            given a histogram with n_bins between min and max
                            will be constructed.

                n_bins      The number of bins used between min and max if
                            cv_bins is not specified.

                stride      If stride is larger than 1, not the entire trajectory
                            is used to construct the histogram. Instead, samples 
                            are taken every <stride> frames. In this way, we hope
                            to minimize correlations.
                
                nblocks     If nblocks is larger than 1, divide the entire trajectory
                            in a number of blocks equal to nblocks and construct a
                            profile (with error estimate) on each block. 
                
                fn_plot     If a file name is given, a plot will be made of the
                            resulting probability histogram and corresponding free 
                            energy with error estimates as obtained from Bayesian 
                            error propagation and Gamma distributions. The error bars 
                            represent 95% confidence intervals (i.e. 2 sigma).
                
                cv_unit     CV unit for plotting

                cv_label    CV label in plots

                f_unit      free energy unit for plotting
        '''
        cv_data = cv_data[::stride]
        if cv_bins is None:
            start, end, num = min(cv_data), max(cv_data), n_bins
            delta = (end-start)/(num-1)
            cv_bins = np.arange(start, end+delta, delta)
        cvs, ps, plower, pupper, fs, flower, fupper = free_energy_from_histogram_with_error(cv_data, cv_bins, temp)
        flower -= min(fs)
        fupper -= min(fs)
        fs -= min(fs)
        fep = cls(cvs.copy(), fs.copy(), temp, cv_unit=cv_unit, f_unit=f_unit, cv_label=cv_label)
        if fn_plot is not None:
            pp.clf()
            fig, axs = pp.subplots(nrows=1+int(nblocks>1), ncols=2, squeeze=False)
            #make free energy plot
            axs[0,0].plot(cvs/parse_unit(cv_unit), ps, linewidth=1, color='b')
            axs[0,0].fill_between(cvs/parse_unit(cv_unit), plower, pupper, color='0.7')
            axs[0,1].plot(cvs/parse_unit(cv_unit), fs/parse_unit(f_unit), linewidth=1, color='b')
            axs[0,1].fill_between(cvs/parse_unit(cv_unit), flower/parse_unit(f_unit), fupper/parse_unit(f_unit), color='0.7')
            #decorate
            axs[0,0].set_xlabel('%s [%s]' %(cv_label, cv_unit), fontsize=14)
            axs[0,0].set_ylabel('P [-]', fontsize=14)
            axs[0,0].set_title('Probability profile', fontsize=14)
            axs[0,0].set_xlim([min(cvs/parse_unit(cv_unit)), max(cvs/parse_unit(cv_unit))])
            axs[0,1].set_xlabel('%s [%s]' %(cv_label, cv_unit), fontsize=14)
            axs[0,1].set_ylabel('F [%s]' %f_unit, fontsize=14)
            axs[0,1].set_title('Free energy profile', fontsize=14)
            axs[0,1].set_xlim([min(cvs/parse_unit(cv_unit)), max(cvs/parse_unit(cv_unit))])
            axs[0,1].set_ylim([-1, np.ceil(max(fs/kjmol)/10)*10])
            if nblocks>1:
                blocksize = len(cv_data)//nblocks
                cmap = cm.get_cmap('tab10')
                for iblock in range(nblocks):
                    block = cv_data[iblock*blocksize:(iblock+1)*blocksize]
                    cvs, ps, plower, pupper, fs, flower, fupper = free_energy_from_histogram_with_error(block, cv_bins, temp)
                    flower -= min(fs[~np.isnan(fs)])
                    fupper -= min(fs[~np.isnan(fs)])
                    fs -= min(fs[~np.isnan(fs)])
                    axs[1,0].plot(cvs/parse_unit(cv_unit), ps, linewidth=2, color=cmap(iblock), label='Block %i' %iblock)
                    axs[1,0].fill_between(cvs/parse_unit(cv_unit), plower, pupper, color=cmap(iblock, alpha=0.2))
                    axs[1,1].plot(cvs/parse_unit(cv_unit), fs/parse_unit(f_unit), linewidth=2, color=cmap(iblock), label='Block %i' %iblock)
                    axs[1,1].fill_between(cvs/parse_unit(cv_unit), flower/parse_unit(f_unit), fupper/parse_unit(f_unit), color=cmap(iblock, alpha=0.2))
                #decorate
                axs[1,0].legend(loc='best')
                axs[1,0].set_xlabel('%s [%s]' %(cv_label, cv_unit), fontsize=14)
                axs[1,0].set_ylabel('P [-]', fontsize=14)
                axs[1,0].set_xlim([min(cvs/parse_unit(cv_unit)), max(cvs/parse_unit(cv_unit))])
                axs[1,1].set_xlabel('%s [%s]' %(cv_label, cv_unit), fontsize=14)
                axs[1,1].set_ylabel('F [%s]' %f_unit, fontsize=14)
                axs[1,1].set_xlim([min(cvs/parse_unit(cv_unit)), max(cvs/parse_unit(cv_unit))])
                axs[1,1].set_ylim([-1, np.ceil(max(fs[~np.isnan(fs)+~np.isinf(fs)]/kjmol)/10)*10])
            #save
            fig.set_size_inches([16,8*(1+int(nblocks>1))])
            pp.savefig(fn_plot)
        return fep

    def process_states(self, **kwargs):
        raise NotImplementedError

    def set_ref(self, ref='min'):
        '''
            Set the energy reference to ref, which should be one of

                * m, min                        the global minimum
        '''
        if ref.lower() in ['m', 'min']:
            self.fs -= self.fs[~np.isnan(self.fs)].min()
        else:
            raise IOError('Invalid REF specification, recieved %s and should be min' %ref)

    def set_microstates(self, **kwargs):
        raise NotImplementedError

    def set_macrostates(self, **kwargs):
        raise NotImplementedError

    def compute_probdens(self):
        '''
            Compute the probability density profile associated with the
            free energy profile

                p(q) = exp(-beta*F(q))/int(exp(-beta*F(q)))
        '''
        mask = ~np.isnan(self.fs)
        self.ps = np.zeros(self.fs.shape)
        self.ps[mask] = np.exp(-self.beta*self.fs[mask])
        self.ps /= integrate(self.cvs[mask], self.ps[mask])

    def macrostate(self, cvrange=None, indexes=None, verbose=False):
        '''
            Return the contribution to the partition function corresponding to
            the macrostate in the given range of cvs.

            **Arguments**

            cvrange
                the range of the collective variable defining the macrostate.
                Either cvrange or indices should be defined.

            indexes
                the indexes of the collective variable defining the macrostate.
                Either cvrange or indices should be defined.


            **Returns**

            mean
                the expected (=mean) value of the collective variable in the
                macrostate

            std
                the thermal fluctuation (=standard deviation) of the CV in the
                macrostate

            Z
                the contribution of the macrostate to the partition function

            F
                the free energy of the given macrostate

        '''
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
                print('WARNING: both indexes and cvrange given, I ignored cvrange')
        cvs = self.cvs[np.array(indexes)]
        fs = self.fs[np.array(indexes)]
        ps = self.ps[np.array(indexes)]
        P = integrate(cvs, ps)
        Z = integrate(cvs, np.exp(-self.beta*fs))
        F = -np.log(Z)/self.beta
        mean = integrate(cvs, ps*cvs)/P
        std = np.sqrt(integrate(cvs, ps*(cvs-mean)**2)/P)
        if verbose:
            print('VALUES:')
            print('  CV Mean [%s] = ' %self.cv_unit, mean/parse_unit(self.cv_unit))
            print('  CV Min  [%s] = ' %self.cv_unit, min(cvs)/parse_unit(self.cv_unit))
            print('  CV Max  [%s] = ' %self.cv_unit, max(cvs)/parse_unit(self.cv_unit))
            print('  CV Std  [%s] = ' %self.cv_unit, std/parse_unit(self.cv_unit))
            print('')
            print('  F(CV min) [%s] = ' %self.f_unit, fs[np.where(cvs==min(cvs))[0][0]]/parse_unit(self.f_unit))
            print('  F(CV max) [%s] = ' %self.f_unit, fs[np.where(cvs==max(cvs))[0][0]]/parse_unit(self.f_unit))
            print('  F min     [%s] = ' %self.f_unit, min(fs)/parse_unit(self.f_unit))
            print('  F max     [%s] = ' %self.f_unit, max(fs)/parse_unit(self.f_unit))
            print('')
            print('  F [%s] = ', F/parse_unit(self.f_unit))
            print('  Z [-] = ', Z)
            print('  P [-] = ', P*100)
        return mean, std, Z, F
    
    def plot(self, fn_png, micro_marker='s', micro_color='r', micro_size='4', macro_linestyle='-', macro_color='b'):
        '''
            Plot the free energy profile
        '''
        pp.clf()
        fig, axs = pp.subplots(nrows=1, ncols=1)
        axs = [axs]
        #make free energy plot
        axs[0].plot(self.cvs/parse_unit(self.cv_unit), self.fs/parse_unit(self.f_unit), linewidth=1, color='0.2')
        #decorate
        axs[0].set_xlabel('%s [%s]' %(self.cv_label, self.cv_unit))
        axs[0].set_ylabel('F [%s]' %self.f_unit)
        axs[0].set_title('Free energy profile')
        axs[0].set_xlim([min(self.cvs), max(self.cvs)])
        #save
        fig.set_size_inches([len(axs)*8,8])
        pp.savefig(fn_png)
        return

    def crop(self, cvrange=None, return_new_fes=False):
        '''
            Crop the free energy profile to the limits given by cvrange. If
            return_new_fes is set to false, a copy of the cropped profile
            will be returns, otherwise the current profile will be cropped
            and overwritten.
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
            return self.__class__(cvs, fs, self.T, cv_unit=self.cv_unit, f_unit=self.f_unit, cv_label=self.cv_label)
        else:
            self.cvs = cvs.copy()
            self.fs = fs.copy()
            self.compute_probdens()

    def recollect(self, new_cvs, fn_plt=None, return_new_fes=False):
        '''
            Redefine the CV array to the new given array. For each interval of
            new CV values, collect all old f values that fall in this new
            interval and average out. As such, this routine can be used to
            filter out noise on a given free energy profile by means of
            averaging.

            **Arguments**

            new_cvs     Array of new CV values

            **Optional Arguments**

            fn_plot     File name for comparison plot of old and new profile.
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
                data.append(self.fs[iold])
                iold += 1
            #print('  no further old cvs found')
            if len(data)>0:
                new_fs[inew] = sum(data)/len(data)
                #print('  ==> averaged previous data=',data,' to f=%.3f' %(new_fs[inew]/kjmol))
            if iold>=len(self.cvs):
                #print('reached end of old cvs, breaking')
                break
        if fn_plt is not None:
            pp.clf()
            fig, ax = pp.subplots(nrows=1, ncols=1)
            #make free energy plot
            ax.plot(self.cvs/parse_unit(self.cv_unit), self.fs/parse_unit(self.f_unit), linewidth=1, color='0.3', label='Original')
            ax.plot(new_cvs/parse_unit(self.cv_unit), new_fs/parse_unit(self.f_unit), linewidth=3, color='sandybrown', label='Recollected')
            #decorate
            ax.set_xlabel('%s [%s]' %(self.cv_label, self.cv_unit))
            ax.set_ylabel('F [%s]' %self.f_unit)
            ax.set_title('Free energy profile')
            ax.set_xlim([min(new_cvs), max(new_cvs)])
            ax.legend(loc='best')
            #save
            fig.set_size_inches([8,8])
            pp.savefig(fn_plt)
        if return_new_fes:
            return self.__class__(new_cvs, new_fs, self.T, cv_unit=self.cv_unit, f_unit=self.f_unit, cv_label=self.cv_label)
        else:
            self.cvs = new_cvs[~np.isnan(new_fs)].copy()
            self.fs = new_fs[~np.isnan(new_fs)].copy()
            self.compute_probdens()



class SimpleFreeEnergyProfile(BaseFreeEnergyProfile):
    '''
        Free Energy Profile consisting of single reactant, transition state
        and product state.
    '''
    def __init__(self, cvs, fs, temp, cv_unit='au', f_unit='kjmol', cv_label='CV'):
        BaseFreeEnergyProfile.__init__(self, cvs, fs, temp, cv_unit=cv_unit, f_unit=f_unit, cv_label=cv_label)
        self.ir  = None
        self.its = None
        self.ip  = None

    def process_states(self, ts_range=[-np.inf,np.inf], verbose=False):
        '''
            Routine to find the reactant (R), transition state (TS) and product
            state (P) through application of the ``_find_R_TS_P'' routine and
            add the corresponding micro and macrostates. These will afterwards
            be shown on the free energy plot using the ``plot'' routine.
        '''
        self._find_R_TS_P(ts_range=ts_range)
        self.add_microstates([self.ir, self.its, self.ip])
        self.add_macrostates([-np.inf, self.cvs[self.its], np.inf], verbose=verbose)

    def _find_R_TS_P(self, ts_range=[-np.inf,np.inf]):
        '''
            Routine to find

            * the transition state (TS) as the local maximum within the given
              TS_RANGE (which is by default [-inf,inf])
            * the reactant (R) as local minimum left of TS
            * the product (P) as local minimum right of TS
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
            Set the energy reference to ref, which should be one of

                * m, min                        the global minimum
                * r, reactant                   the reactant minimum
                * ts, trans_state, transition   the transition state maximum
                * p, product                    the product minimum

            The options r, ts and p are only available if the reactant,
            transition state and product have already been found by the routine
            find_states.
        '''
        if ref.lower() in ['r', 'reactant']:
            assert self.ir is not None, 'Reactant state not defined yet, did you already apply the find_states routine?'
            self.fs -= self.fs[self.ir]
        elif ref.lower() in ['p', 'product']:
            assert self.ip is not None, 'Product state not defined yet, did you already apply the find_states routine?'
            self.fs -= self.fs[self.ip]
        elif ref.lower() in ['ts', 'trans_state', 'transition']:
            assert self.its is not None, 'Transition state not defined yet, did you already apply the find_states routine?'
            self.fs -= self.fs[self.its]
        elif ref.lower() in ['m', 'min']:
            self.fs -= self.fs[~np.isnan(self.fs)].min()
        else:
            raise IOError('Invalid REF specification, recieved %s and should be min, r, ts or p' %ref)
        #Micro and macrostates need to be updated
        if self.ir is not None and self.its is not None and self.ip is not None:
            self.microstates = []
            self.macrostates = []
            self.add_microstates([self.ir, self.its, self.ip])
            self.add_macrostates([-np.inf, self.cvs[self.its], np.inf])

    def add_microstates(self, indices):
        for index in indices:
            if index is not None:
                self.microstates.append([index, self.cvs[index], self.fs[index]])

    def add_macrostates(self, cvs, verbose=False):
        for i in range(len(cvs)-1):
            self.macrostates.append(self.macrostate(cvrange=cvs[i:i+2], verbose=verbose))

    def plot(self, fn_png, rate=None, micro_marker='s', micro_color='r', micro_size='4', macro_linestyle='-', macro_color='b'):
        '''
            Plot the free energy profile as well as the microstates and
            macrostates identified earlier.
        '''
        pp.clf()
        fig = pp.gcf()
        gs  = gridspec.GridSpec(1,2, width_ratios=[2,1])
        ax = fig.add_subplot(gs[0])
        axs = [ax]
        #make free energy plot
        axs[0].plot(self.cvs/parse_unit(self.cv_unit), self.fs/parse_unit(self.f_unit), linewidth=1, color='0.2')
        #plot vline for transition state if defined
        cv_width = max(self.cvs)-min(self.cvs)
        ylower = -1+min([state[3]/kjmol for state in self.macrostates]+[0])
        if self.its is not None:
            axs[0].axvline(x=self.cvs[self.its]/parse_unit(self.cv_unit), linestyle='--', color='k', linewidth=1)
            axs[0].text((self.cvs[self.its]+0.01*cv_width)/parse_unit(self.cv_unit), 0, '%.3f' %(self.cvs[self.its]/parse_unit(self.cv_unit)), color='k', fontsize=12)
            #plot lines for the limits defining the TS region
            if rate is not None:
                axs[0].fill_betweenx([ylower, max(self.fs)/kjmol], x1=rate.CV_TS_lims[0]/parse_unit(self.cv_unit), x2=rate.CV_TS_lims[1]/parse_unit(self.cv_unit), alpha=0.3, color='k')
        #plot microstates
        for i, cv, f in self.microstates:
            axs[0].plot(cv/parse_unit(self.cv_unit), f/parse_unit(self.f_unit), linestyle='none', marker=micro_marker, color=micro_color, markersize=micro_size)
            axs[0].text((cv+0.01*cv_width)/parse_unit(self.cv_unit), f/parse_unit(self.f_unit), '%.1f' %(f/parse_unit(self.f_unit)), color=micro_color, fontsize=12)
        #plot macrostates
        for mean, std, Z, F in self.macrostates:
            xcen = (mean-min(self.cvs))/cv_width
            axs[0].axhline(y=F/parse_unit(self.f_unit), xmin=xcen-0.075, xmax=xcen+0.075, linestyle=macro_linestyle, color=macro_color, linewidth=2)
            axs[0].text((mean+0.075*cv_width)/parse_unit(self.cv_unit), F/parse_unit(self.f_unit), '%.1f' %(F/parse_unit(self.f_unit)), color=macro_color, fontsize=12)
        #decorate
        axs[0].set_xlabel('%s [%s]' %(self.cv_label, self.cv_unit), fontsize=14)
        axs[0].set_ylabel('Energy [%s]' %self.f_unit, fontsize=14)
        axs[0].set_title('Free energy profile F(CV)', fontsize=16)
        axs[0].set_xlim([min(self.cvs/parse_unit(self.cv_unit)), max(self.cvs/parse_unit(self.cv_unit))])
        axs[0].set_ylim([ylower, max(self.fs[~np.isnan(self.fs)])/kjmol+10])
        axs[0].axhline(y=0, xmin=0, xmax=1, linestyle='--', color='k', linewidth=1)

        if len(self.macrostates)>0:
            assert len(self.macrostates)==2, 'The plotter assumes two macrostates (if any), i.e. R and P, but found %i' %(len(self.macrostates))
            Zr,Fr = self.macrostates[0][2], self.macrostates[0][3]
            Zp,Fp = self.macrostates[1][2], self.macrostates[1][3]
            Fts = self.fs[self.its]
            fig.text(0.65, 0.88, r'\textit{Thermodynamic properties}', fontsize=16)
            fig.text(0.65, 0.86, '-------------------------------------', fontsize=16)
            fig.text(0.65, 0.82, r'$Z_{R} =\ $'+format_scientific(Zr/parse_unit(self.cv_unit))+'  %s' %(self.cv_unit), fontsize=16)
            fig.text(0.65, 0.78, r'$F_{R} = -k_B T \log(Z_{R}) = %.3f\ \ $ kJ.mol$^{-1}$' %(Fr/kjmol), fontsize=16)
            fig.text(0.65, 0.74, r'$Z_{P} =\ $'+format_scientific(Zp/parse_unit(self.cv_unit))+'  %s' %(self.cv_unit), fontsize=16)
            fig.text(0.65, 0.70, r'$F_{P} = -k_B T \log(Z_{P}) = %.3f\ \ $ kJ.mol$^{-1}$' %(Fp/kjmol), fontsize=16)
            fig.text(0.65, 0.66, r'$F(q_{TS}) = %.3f\ \ $ kJ.mol$^{-1}$' %(Fts/kjmol), fontsize=16)
        if rate is not None:
            k_forward = rate.A*np.exp(-Fts/(boltzmann*self.T))/Zr
            k_backward = rate.A*np.exp(-Fts/(boltzmann*self.T))/Zp
            dF_forward = Fts+boltzmann*self.T*np.log(boltzmann*self.T*Zr/(planck*rate.A))
            dF_backward = Fts+boltzmann*self.T*np.log(boltzmann*self.T*Zp/(planck*rate.A))
            fig.text(0.65, 0.58, r'\textit{Kinetic properties}', fontsize=16)
            fig.text(0.65, 0.56, '-------------------------------------', fontsize=16)
            fig.text(0.65, 0.50, r'$A  =\ $' + format_scientific(rate.A/(parse_unit(self.cv_unit)/second)) + r' %s.s$^{-1}$' %(self.cv_unit), fontsize=16)
            fig.text(0.65, 0.44, r'$k_{F} = A \frac{e^{-\beta\cdot F(q_{TS})}}{Z_{R} } =\ $ ' +format_scientific(k_forward*second)  + r' s$^{-1}$', fontsize=16)
            fig.text(0.65, 0.38, r'$k_{B} = A \frac{e^{-\beta\cdot F(q_{TS})}}{Z_{P} } =\ $ ' +format_scientific(k_backward*second) + r' s$^{-1}$', fontsize=16)
            fig.text(0.65, 0.30, r'\textit{Phenomenological barrier}', fontsize=16)
            fig.text(0.65, 0.28, '-------------------------------------', fontsize=16)
            fig.text(0.65, 0.22, r'$k = \frac{k_B T}{h}e^{-\beta\cdot\Delta F}$', fontsize=16)
            fig.text(0.65, 0.18, r'$\Delta F_{F}  = %.3f\ \ $ kJ.mol$^{-1}$' %(dF_forward/kjmol), fontsize=16)
            fig.text(0.65, 0.14, r'$\Delta F_{B}  = %.3f\ \ $ kJ.mol$^{-1}$' %(dF_backward/kjmol), fontsize=16)
        #save
        fig.set_size_inches([12,8])
        pp.savefig(fn_png)
        return


class FreeEnergySurface2D(object):
    def __init__(self, cv1s, cv2s, fs, temp, cv1_unit='au', cv2_unit='au', f_unit='kjmol', cv1_label='CV1', cv2_label='CV2'):
        self.cv1s = cv1s.copy()
        self.cv2s = cv2s.copy()
        self.fs   = fs.copy()
        self.T = temp
        self.cv1_unit = cv1_unit
        self.cv2_unit = cv2_unit
        self.f_unit = f_unit
        self.cv1_label = cv1_label
        self.cv2_label = cv2_label
        self.compute_probdens()

    def _beta(self):
        return 1.0/(boltzmann*self.T)

    beta = property(_beta)

    def copy(self):
        fes = FreeEnergySurface2D(
            self.cv1s.copy(), self.cv2s.copy(), self.fs.copy(), self.T,
            self.cv1_unit, self.cv2_unit, self.f_unit,
            self.cv1_label, self.cv2_label
        )
        return fes

    @classmethod
    def from_txt(cls, fn, temp, cv1_col=0, cv2_col=1, f_col=2, cv1_unit='au', cv2_unit='au', f_unit='kjmol', cv1_label='CV1', cv2_label='CV2', cv1_range=None, cv2_range=None, delimiter=None):
        '''
            Read the free energy profile as function of a collective variable
            from a txt file.

            **Arguments**

            fn
                the name of the txt file containing the data

            temp
                the temperature at which the free energy is constructed

            **Optional Arguments**

            cv1_col

                the column in which the first collective variable is stored

            cv2_col
                the column in which the second collective variable is stored

            f_col
                the column in which the free energy is stored

            cv1_label
                the label of CV1 for labels in plots

            cv2_label
                the label of CV2 for labels in plots

            cv1_unit
                the units in which the first CV are stored [default=atomic units]

            cv2_unit
                the units in which the second CV are stored [default=atomic units]

            f_unit
                the units in which the free energy are stored [default=kjmol]

            cv1_range
                a tuple/list [CVmin,CVmax], only read free energy for which the
                first CV in the given range.

            cv2_range
                a tuple/list [CVmin,CVmax], only read free energy for which the
                second CV in the given range.
        '''
        data = np.loadtxt(fn, delimiter=delimiter, dtype=float)
        cv1s = data[:,cv1_col]*parse_unit(cv1_unit)
        cv2s = data[:,cv2_col]*parse_unit(cv2_unit)
        fs = data[:,f_col]*parse_unit(f_unit)
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
            if np.isinf(f):
                fus[i2,i1] = np.nan
            else:
                fus[i2,i1] = f
        cv1s = cv1us.copy()
        cv2s = cv2us.copy()
        fs = fus.copy()
        return cls(cv1s, cv2s, fs, temp, cv1_label=cv1_label, cv2_label=cv2_label)

    def compute_probdens(self):
        '''
            Compute the probability density profile associated with the
            free energy profile

                p(q) = exp(-beta*F(q))/int(exp(-beta*F(q)))
        '''
        self.ps = np.exp(-self.beta*self.fs)
        self.ps[np.isnan(self.ps)] = 0.0
        self.ps /= integrate2d(self.ps, x=self.cv1s, y=self.cv2s)

    def set_ref(self, ref='min'):
        '''
            Set the energy reference to ref, which should be one of

                * m, min                        the global minimum
        '''
        if ref.lower() in ['m', 'min']:
            self.fs -= self.fs[~np.isnan(self.fs)].min()
        else:
            raise IOError('Invalid REF specification, recieved %s and should be min' %ref)

    def detect_clusters(self, eps=1.5, min_samples=8, metric='euclidean', fn_plot=None, delete_clusters=[-1]):
        '''
            Routine to apply the DBSCAN clustering algoritm to the (CV1,CV2)
            grid points that correspond to finite free energies (i.e. not nan
            or inf) to detect clusters of neighboring points.

            The DBSCAN algorithm first identifies the core samples,there exist
            MIN_SAMPLES other samples within a distance of EPS, which are
            defined as neighbors of the core sample. Next, the data is divided
            into clusters based on these core samples.

            A cluster is defined a set of core samples that can be built by
            recursively taking a core sample, finding all of its neighbors that
            are core samples, finding all of their neighbors that are core
            samples, and so on. A cluster also has a set of non-core samples,
            which are samples that are neighbors of a core sample in the cluster
            but are not themselves core samples. Intuitively, these samples are
            on the fringes of a cluster. Each cluster is given an integer as
            label.

            Any sample that is not a core sample, and is at least eps in
            distance from any core sample, is considered an outlier by the
            algorithm and is what we here consider an isolated point/region.
            These points get the cluster label of -1.

            Finally, all data points belonging to a cluster with label specified
            in DELETE_CLUSTERS will have theire free energy set to nan. A safe
            choice here is to just delete isolated regions, i.e. the point in
            cluster with label -1 (which is the default).
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
            Crop the free energy surface by removing all data for which either
            cv1 is beyond the range defined in CV1RANGE or cv2 is beyond the
            range defined in CV1RANGE. Furthermore, if RETURN_NEW_FES is set to
            False, the cropping will be performed on the existing instance,
            otherwise a copy will be returned.
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
            return FreeEnergySurface2D(cv1s, cv2s, fs, self.T, cv1_unit=self.cv1_unit, cv2_unit=self.cv2_unit, f_unit=self.f_unit, cv1_label=self.cv1_label, cv2_label=self.cv2_label)
        else:
            self.cv1s = cv1s.copy()
            self.cv2s = cv2s.copy()
            self.fs = fs.copy()
            self.compute_probdens()

    def rotate(self, interpolate=True):
        '''
            Transform the free energy profile in terms of the following two
            new collective variables:

                u = 0.5*(cv1+cv2)
                v = (cv2-)cv1)

            This transformation represents a simple rotation (and mirroring).
            From probability theory we easily find the transformation formula:

                Ft(u,v) = F(u-0.5*v,u+0.5*v)

            The uniform (u,v)-grid introduces new grid points in between the
            original (cv1,cv2) grid points. If interpolate is True, the free
            energy for these points is interpolated if all four neighbors
            have defined (i.e. not nan) free energies.
        '''
        #First make unique list of us and vs
        print('Making unique arrays for u=0.5*(cv1+cv2) and v=cv2-cv1')
        us = []
        vs = []
        for i1, cv1 in enumerate(self.cv1s):
            for i2, cv2 in enumerate(self.cv2s):
                u = 0.5*(cv1+cv2)
                v = cv2-cv1
                if len(np.where(abs(u-us)<1e-6)[0])==0:
                    us.append(u)
                v = cv2-cv1
                if len(np.where(abs(v-vs)<1e-6)[0])==0:
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
        for iu, u in enumerate(us):
            for iv, v in enumerate(vs):
                key = '%i,%i' %(iu,iv)
                if key in dic.keys():
                    (i1,i2) = dic[key]
                    assert abs(0.5*(self.cv1s[i1]+self.cv2s[i2]) - u)<1e-8
                    assert abs(    (self.cv2s[i2]-self.cv1s[i1]) - v)<1e-8
                    fs[iu,iv] = self.fs[i2,i1]
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
        if self.cv1_unit==self.cv2_unit:
            u_unit = self.cv1_unit
            v_unit = self.cv1_unit
        else:
            u_unit = 'au'
            v_unit = 'au'
        return FreeEnergySurface2D(vs, us, fs, self.T, cv1_unit=v_unit, cv2_unit=u_unit, f_unit='kjmol', cv1_label='CV2-CV1', cv2_label='0.5*(CV1+CV2)')

    def project_difference(self, sign=1):
        '''
            Construct a 1D free energy profile representing the projection of
            the 2D FES onto the difference of collective variables:

                F1(q) = -kT*log( int( e(-beta*F2(x,x+q)), x=-inf...+inf) )

                with q = CV2-CV1

            This projection is implemented by first projecting the probability
            density and afterwards reconstructing the free energy.

            If sign is set to -1, the projection is done on q=CV1-CV2 instead.
        '''
        #Get the 2D probability density
        if sign==1:
            x = self.cv1s.copy()
            y = self.cv2s.copy()
            Pxy = self.ps.copy()
            label = 'CV2-CV1'
        elif sign==-1:
            x = self.cv2s.copy()
            y = self.cv1s.copy()
            Pxy = self.ps.T.copy()
            label = 'CV1-CV2'
        else:
            raise ValueError('Recieved invalid sign, should be 1 or -1 but got %s' %str(sign))
        #construct grid for projected degree of freedom q
        Q = []
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                q = yj-xi
                if (abs(Q-q)>1e-6).all():
                    Q.append(q)
        Q = np.array(sorted(Q))
        #construct probability on this grid
        Pq = np.zeros(len(Q), float)
        for iq, q in enumerate(Q):
            lower = max(x[0], y[0]-q)
            upper = min(x[-1], y[-1]-q)
            if lower>upper:
                print('For q=%.6f:  Lower=%.6f  Upper=%.6f' %(q,lower,upper))
                print('             Pq set to zero')
                #raise ValueError('Upper limit below lower limit in projection')
            delta = (x[-1]-x[0])/(len(x)-1)
            x_cropped = np.arange(lower, upper, delta)
            if not upper in x_cropped:
                x_cropped = np.array(list(x_cropped)+[upper])
            #print('q=',q,'  xcropped=',x_cropped)
            integrandum = np.zeros(x_cropped.shape, float)
            for index_cropped, xi in enumerate(x_cropped):
                x_indices = np.where(abs(x-xi)<1e-6)[0]
                if len(x_indices)>1:
                    print('ERROR: in project for q=%.3f and xi=%.3f: found more than 1 x_index' %(q,xi))
                    sys.exit()
                elif len(x_indices)==0:
                    integrandum[index_cropped] = 0
                else:
                    ix = x_indices[0]
                    y_indices = np.where(abs(y-(xi+q))<1e-6)[0]
                    if len(y_indices)>1:
                        print('ERROR: in project for q=%.3f and xi=%.3f: found more than 1 y_index' %(q,xi))
                        sys.exit()
                    elif len(y_indices)==0:
                        integrandum[index_cropped] = 0
                    else:
                        iy = y_indices[0]
                        integrandum[index_cropped] = Pxy[iy,ix] #x is stored in columns of Pxy
            Pq[iq] = np.trapz(integrandum,x=x_cropped)
        #Derive corresponding free energy and return free energy profile
        if self.cv1_unit==self.cv2_unit:
            cv_unit = self.cv1_unit
        else:
            cv_unit = 'au'
        fs = np.zeros(len(Pq), float)*np.nan
        fs[Pq>0] = -np.log(Pq[Pq>0])/self.beta
        return SimpleFreeEnergyProfile(Q, fs, self.T, cv_unit=cv_unit, f_unit='kjmol', cv_label=label)

    def project_average(self):
        '''
            Construct a 1D free energy profile representing the projection of
            the 2D FES F2(CV1,CV2) onto the average q=(CV1+CV2)/2 of the
            collective variables:

                F1(q) = -kT*log( 2*int( e(-beta*F2(x,2*q-x)), x=-inf...+inf) )

                with q = CV2-CV1

            This projection is implemented by first projecting the probability
            density and afterwards reconstructing the free energy.
        '''
        x = self.cv1s.copy()
        y = self.cv2s.copy()
        Pxy = self.ps.copy()
        #construct grid for projected degree of freedom q
        Q = []
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                q = 0.5*(xi+yj)
                if q not in Q:
                    Q.append(q)
        Q = np.array(sorted(Q))
        #construct probability on this grid
        Pq = np.zeros(len(Q), float)
        for iq, q in enumerate(Q):
            lower = max(x[0], 2*q-y[-1])
            upper = min(x[-1], 2*q-y[0])
            if lower>upper:
                print('For q=%.6f:  Lower=%.6f  Upper=%.6f' %(q,lower,upper))
                print('             Pq set to zero')
                #raise ValueError('For projection on q=%.6f: upper limit (= %.6f) of integration variable lower than lower limit (= %.6f).' %(q, upper, lower))
            delta = (x[-1]-x[0])/(len(x)-1)
            x_cropped = np.arange(lower, upper, delta)
            if not upper in x_cropped:
                x_cropped = np.array(list(x_cropped)+[upper])
            integrandum = np.zeros(x_cropped.shape, float)
            for index_cropped, xi in enumerate(x_cropped):
                x_indices = np.where(abs(x-xi)<1e-6)[0]
                if len(x_indices)>1:
                    print('ERROR: in project for q=%.3f and xi=%.3f: found more than 1 x_index' %(q,xi))
                    sys.exit()
                elif len(x_indices)==0:
                    integrandum[index_cropped] = 0
                else:
                    ix = x_indices[0]
                    y_indices = np.where(abs(y-(2*q-xi))<1e-6)[0]
                    if len(y_indices)>1:
                        print('ERROR: in project for q=%.3f and xi=%.3f: found more than 1 y_index' %(q,xi))
                        sys.exit()
                    elif len(y_indices)==0:
                        integrandum[index_cropped] = 0
                    else:
                        iy = y_indices[0]
                        integrandum[index_cropped] = 2*Pxy[iy,ix] #the extra 2 comes from delta((x+y)/2-q)=2*delta(x+y-2*q). Basically, it is a jacobian. Also x is stored in columns of Pxy
            Pq[iq] = np.trapz(integrandum,x=x_cropped)
        #Derive corresponding free energy and return free energy profile
        if self.cv1_unit==self.cv2_unit:
            cv_unit = self.cv1_unit
        else:
            cv_unit = 'au'
        fs = np.zeros(len(Pq), float)*np.nan
        fs[Pq>0] = -np.log(Pq[Pq>0])/self.beta
        return SimpleFreeEnergyProfile(Q, fs, self.T, cv_unit=cv_unit, f_unit='kjmol', cv_label='0.5*(CV1+CV2)')

    def project_cv1(self):
        '''
            Construct a 1D free energy profile representing the projection of
            the 2D FES F2(CV1,CV2) onto q=CV1. This is implemented by simply
            integrating CV2 out of P2(CV1,CV2).
        '''
        cvs = self.cv1s.copy()
        ps = np.zeros(len(cvs), float)*np.nan
        for i, cv in enumerate(cvs):
            ys = self.ps[:,i].copy()
            mask = ~np.isnan(ys)
            ps[i] = np.trapz(ys[mask], x=self.cv2s[mask])
            #print(i, cv, ys, self.fs[:,i]/kjmol, ps[i], -np.log(ps[i])/self.beta/kjmol if ps[i]>0 else np.nan)
        fs = np.zeros(len(cvs), float)*np.nan
        fs[ps>0] = -np.log(ps[ps>0])/self.beta
        return SimpleFreeEnergyProfile(cvs, fs, self.T, cv_unit=self.cv1_unit, f_unit='kjmol', cv_label=self.cv1_label)

    def project_cv2(self):
        '''
            Construct a 1D free energy profile representing the projection of
            the 2D FES F2(CV1,CV2) onto q=CV2. This is implemented by simply
            integrating CV1 out of P2(CV1,CV2).
        '''
        cvs = self.cv2s.copy()
        ps = np.zeros(len(cvs), float)*np.nan
        for i, cv in enumerate(cvs):
            ys = self.ps[i,:].copy()
            mask = ~np.isnan(ys)
            ps[i] = np.trapz(ys[mask], x=self.cv1s[mask])
        fs = np.zeros(len(cvs), float)*np.nan
        fs[ps>0] = -np.log(ps[ps>0])/self.beta
        return SimpleFreeEnergyProfile(cvs, fs, self.T, cv_unit=self.cv2_unit, f_unit='kjmol', cv_label=self.cv2_label)

    def plot(self, fn_png, obs='F', cv1_lims=None, cv2_lims=None, lims=None, ncolors=8, scale='lin'):
        '''
            Simple routine to make a 2D contour plot of either the free energy F
            or probability distribution P as specified in OBS.
        '''
        pp.clf()
        if obs.lower() in ['f', 'free energy']:
            obs = self.fs/parse_unit(self.f_unit)
            label = 'Free energy [kJ/mol]'
        elif obs.lower() in ['p', 'probability']:
            obs = self.ps*(parse_unit(self.cv1_unit)*parse_unit(self.cv2_unit))
            label = r'Probability density [(%s*%s)$^{-1}$]' %(self.cv1_unit, self.cv2_unit)
        else:
            raise IOError('Recieved invalid observable. Should be f, free energy, p or probability but got %s' %obs)
        plot_kwargs = {}
        if lims is not None:
            if scale.lower() in ['lin', 'linear']:
                plot_kwargs['levels'] = np.linspace(lims[0], lims[1], ncolors+1)
            elif scale.lower() in ['log', 'logarithmic']:
                plot_kwargs['levels'] = np.logspace(lims[0], lims[1], ncolors+1)
                plot_kwargs['norm'] = LogNorm()
                plot_kwargs['locator'] = LogLocator()
            else:
                raise IOError('recieved invalid scale value, got %s, should be lin or log' %scale)
        contourf = pp.contourf(self.cv1s/parse_unit(self.cv1_unit), self.cv2s/parse_unit(self.cv2_unit), obs, cmap=pp.get_cmap('rainbow'), **plot_kwargs)
        contour = pp.contour(self.cv1s/parse_unit(self.cv1_unit), self.cv2s/parse_unit(self.cv2_unit), obs, **plot_kwargs)
        if cv1_lims is not None:
            pp.xlim(cv1_lims)
        if cv2_lims is not None:
            pp.ylim(cv2_lims)
        pp.xlabel('%s [%s]' %(self.cv1_label, self.cv1_unit), fontsize=16)
        pp.ylabel('%s [%s]' %(self.cv2_label, self.cv2_unit), fontsize=16)
        cbar = pp.colorbar(contourf, extend='both')
        cbar.set_label(label, fontsize=16)
        pp.clabel(contour, inline=1, fontsize=10)
        fig = pp.gcf()
        fig.set_size_inches([12,8])
        pp.savefig(fn_png)
