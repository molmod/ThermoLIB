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
from molmod.io.xyz import XYZReader

from ..tools import integrate, integrate2d, format_scientific
from .fep import BaseFreeEnergyProfile, FreeEnergySurface2D

import matplotlib.pyplot as pp
from matplotlib import gridspec
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import numpy as np

import sys

__all__ = ['ConditionalProbability1D1D', 'ConditionalProbability1D2D']


class ConditionalProbability1D1D(object):
    '''
        Routine to store and compute conditional probabilities of the form

            p(q1|cv)
    '''
    def __init__(self, q1s, cvs, q1_label='Q', cv_label='CV'):
        self.q1s = q1s.copy()
        self.cvs = cvs.copy()
        self.q1_label = q1_label
        self.cv_label = cv_label
        # Some initialization
        self.q1min, self.q1max, self.q1num = min(q1s), max(q1s), len(q1s)
        self.cvmin, self.cvmax, self.cvnum = min(cvs), max(cvs), len(cvs)
        self.pconds = np.zeros([self.q1num, self.cvnum])
        self.norms = np.zeros(self.cvnum)
        self.pqs = np.zeros(self.q1num)
        self.pcvs = np.zeros(self.cvnum)

    def process_trajectory_xyz(self, fns, Q1, CV):
        '''
            Compute the conditional probability p(q1|cv) (and norm for final
            normalisation) by processing a series of XYZ trajectory files. The
            final probability is estimated as the average over all given files.
            These files may also contain data from biased simulations as long
            as the bias is constant over the simulation. For example, data from
            Umbrella Sampling is OK, while data from metadynamica itself is not.
            Data obtained from a regular MD with the final MTD profile as bias
            is OK.
        '''
        if not isinstance(fns, list): fns = [fns]
        for fn in fns:
            xyz = XYZReader(fn)
            for title, coords in xyz:
                cvi = CV.compute(coords, deriv=False)
                if not self.cvmin<=cvi<=self.cvmax:
                    print("Found frame with cv beyond range of current FEP")
                    continue
                else:
                    icv = int((cvi-self.cvmin)/(self.cvmax-self.cvmin)*(self.cvnum-1))
                    assert self.cvs[icv]<=cvi<=self.cvs[icv+1], "Inconsistency detected in cvs"
                q1i = Q1.compute(coords, deriv=False)
                if not self.q1min<=q1i<=self.q1max:
                    raise ValueError("Found frame with q1 beyond given bounds, aborting")
                else:
                    iq1 = int((q1i-self.q1min)/(self.q1max-self.q1min)*(self.q1num-1))
                    assert self.q1s[iq1]<=q1i<=self.q1s[iq1+1], "Inconsistency detected in q1s"
                self.pconds[iq1,icv] += 1
                self.norms[icv] += 1

    def process_trajectory_cvs(self, fns, col_q1=1, col_cv=2):
        '''
            Routine to update conditional probability p(q1|cv) (and norm for
            final normalisation) by processing a series of CV trajectory file.
            Each CV trajectory file contains rows of the form

                time q1 cv

            If the trajectory file contains this data in a different order, it
            can be accounted for using the col_xx keyword arguments. Similar
            constraints apply to these CV trajectory files as specified in the
            routine ´´process_trajectory_xyz´´.
        '''
        if not isinstance(fns, list): fns = [fns]
        print('Constructing conditional probability...')
        for fn in fns:
            print('  Reading data from %s' %fn)
            data = np.loadtxt(fn)
            for row in data:
                q1i, cvi = row[col_q1], row[col_cv]
                if not self.cvmin<=cvi<=self.cvmax:
                    #print("Found frame with cv(=%.6f) beyond range of current FEP (=[%.6f,%.6f]), skipping" %(cvi, self.cvmin, self.cvmax))
                    continue
                else:
                    icv = int((cvi-self.cvmin)/(self.cvmax-self.cvmin)*(self.cvnum-1))
                    assert self.cvs[icv]<=cvi<=self.cvs[icv+1], "Inconsistency detected in cvs: for cv=%.6f we found icv=%i, however: cvs[%i]=%.6f and cvs[%i]=%.6f" %(cvi,icv,icv,self.cvs[icv],icv+1,self.cvs[icv+1])
                if not self.q1min<=q1i<=self.q1max:
                    raise ValueError("Found frame with q1 beyond given bounds")
                else:
                    iq1 = int((q1i-self.q1min)/(self.q1max-self.q1min)*(self.q1num-1))
                    if q1i-self.q1s[iq1]<-1e-6 or 1e-6<q1i-self.q1s[iq1+1]:
                        raise AssertionError("Inconsistency detected in q1 values: assigned index %i to value %.6f but q1s[%i]=%.6f and q1s[%i]=%.6f" %(iq1,q1i,iq1,self.q1s[iq1],iq1+1,self.q1s[iq1+1]))
                self.pconds[iq1,icv] += 1
                self.norms[icv] += 1

    def finish(self, fn_plt=None, plot_cvs=None):
        # Normalize conditional probability as well as some additional probability densities
        self.pqs = np.sum(self.pconds, axis=1)
        self.pconds[:, self.norms>0] /= self.norms[self.norms>0]
        assert (self.pconds[:, self.norms<=0]==0).all(), 'Inconsistency in normalisation of conditional probability'

        self.pqs /= np.trapz(self.pqs, x=self.q1s)
        self.pcvs = self.norms.copy()/self.norms.sum()

        # Plot if requested
        if fn_plt is not None:
            if plot_cvs is None:
                pp.clf()
                contourf = pp.contourf(self.cvs, self.q1s, self.pconds, cmap=pp.get_cmap('rainbow'))
                pp.xlabel('%s [au]' %self.cv_label, fontsize=16)
                pp.ylabel('%s [au]' %self.q1_label, fontsize=16)
                pp.colorbar(contourf)
                pp.tight_layout()
                pp.savefig(fn_plt)
                pp.close()

            else:
                pp.clf()
                if   len(plot_cvs)<=2: nrows,ncols = 1,2
                elif len(plot_cvs)<=4: nrows,ncols = 2,2
                elif len(plot_cvs)<=6: nrows,ncols = 2,3
                elif len(plot_cvs)<=9: nrows,ncols = 3,3
                else:
                    raise ValueError('Requested to plot too much cv values, limited to maximum 9')
                fig, axs = pp.subplots(nrows=nrows,ncols=ncols, sharex=True, sharey=True)
                for i, cv in enumerate(plot_cvs):
                    icv = int((cv-self.cvmin)/(self.cvmax-self.cvmin)*(self.cvnum-1))
                    iax = i//ncols
                    jax = i%ncols
                    ax = axs[iax,jax]
                    ax.plot(self.q1s, self.pconds[:,icv])
                    if iax==nrows-1:
                        ax.set_xlabel('%s [au]' %self.q1_label, fontsize=16)
                    if jax==0:
                        ax.set_ylabel('Conditional probability p(%s||%s)' %(self.q1_label,self.cv_label), fontsize=16)
                    ax.set_title('CV = %.3f au' %(self.cvs[icv]))
                fig.set_size_inches([8*ncols,8*nrows])
                fig.tight_layout()
                pp.savefig(fn_plt)
                pp.close()


    def transform(self, fep, cv_unit='au', f_unit='kjmol', cv_label=None):
        '''
            Transform the provided 1D FEP to a different 1D FES using the current
            conditional probability according to the formula

                FES(q1) = -kT*log(P(q1))

                P(q1) = int(condprob(q1|cv)*exp(-beta*F(cv), cv)
        '''
        assert isinstance(fep, BaseFreeEnergyProfile), 'Input argument should be instance of BaseFreeEnergyProfile, instead received %s' %fep.__class__.__name__
        assert len(fep.cvs)==len(self.cvs), 'Dimension of 1D CV in conditional probability inconsistent with 1D FEP'
        assert (abs(fep.cvs-self.cvs)<1e-6).all(), 'Values of 1D CV in conditional probability not identical to those of 1D FEP'
        if cv_label is None: cv_label = self.q1_label
        # Construct 1D FES
        ps = np.sum(self.pconds[:, ~np.isnan(fep.fs)]*np.exp(-fep.beta*fep.fs[~np.isnan(fep.fs)]), axis=1)
        ps /= np.trapz(ps, x=self.q1s)
        fs = np.zeros([self.q1num], float)*np.nan
        fs[ps>0] = -np.log(ps[ps>0])/fep.beta
        return BaseFreeEnergyProfile(self.q1s, fs, fep.T, cv_unit=cv_unit, f_unit=f_unit, cv_label=cv_label)


class ConditionalProbability1D2D(object):
    '''
        Routine to store and compute conditional probabilities of the form

            p(q1,q2|cv)
    '''
    def __init__(self, q1s, q2s, cvs, q1_label='Q1', q2_label='Q2', cv_label='CV'):
        self.q1s = q1s.copy()
        self.q2s = q2s.copy()
        self.cvs = cvs.copy()
        self.q1_label = q1_label
        self.q2_label = q2_label
        self.cv_label = cv_label
        #some initialization
        self.q1min, self.q1max, self.q1num = min(q1s), max(q1s), len(q1s)
        self.q2min, self.q2max, self.q2num = min(q2s), max(q2s), len(q2s)
        self.cvmin, self.cvmax, self.cvnum = min(cvs), max(cvs), len(cvs)
        self.pconds = np.zeros([self.q2num, self.q1num, self.cvnum])
        self.norms = np.zeros(self.cvnum)
        self.pqs = np.zeros([self.q2num, self.q1num])
        self.pcvs = np.zeros(self.cvnum)

    def process_trajectory_xyz(self, fns, Q1, Q2, CV):
        '''
            Compute the conditional probability p(q1,q2|cv) (and norm for final
            normalisation) by processing a series of XYZ trajectory files. The
            final probability is estimated as the average over all given files.
            These files may also contain data from biased simulations as long
            as the bias is constant over the simulation. For example, data from
            Umbrella Sampling is OK, while data from metadynamica itself is not.
            Data obtained from a regular MD with the final MTD profile as bias
            is OK.
        '''
        if not isinstance(fns, list): fns = [fns]
        for fn in fns:
            xyz = XYZReader(fn)
            for title, coords in xyz:
                cvi = CV.compute(coords, deriv=False)
                if not self.cvmin<=cvi<=self.cvmax:
                    print("Found frame with cv beyond range of current FEP")
                    continue
                else:
                    icv = int((cvi-self.cvmin)/(self.cvmax-self.cvmin)*(self.cvnum-1))
                    assert self.cvs[icv]<=cvi<=self.cvs[icv+1], "Inconsistency detected in cvs"
                q1i = Q1.compute(coords, deriv=False)
                if not self.q1min<=q1i<=self.q1max:
                    raise ValueError("Found frame with q1 beyond given bounds, aborting")
                else:
                    iq1 = int((q1i-self.q1min)/(self.q1max-self.q1min)*(self.q1num-1))
                    assert self.q1s[iq1]<=q1i<=self.q1s[iq1+1], "Inconsistency detected in q1s"
                q2i = Q2.compute(coords, deriv=False)
                if not self.q1min<=q1i<=self.q1max:
                    raise ValueError("Found frame with q2 beyond given bounds, aborting")
                else:
                    iq2 = int((q2i-self.q2min)/(self.q2max-self.q2min)*(self.q2num-1))
                    assert self.q2s[iq2]<=q2i<=self.q2s[iq2+1], "Inconsistency detected in q2s"
                self.pconds[iq2,iq1,icv] += 1
                self.norms[icv] += 1

    def process_trajectory_cvs(self, fns, col_q1=1, col_q2=2, col_cv=3):
        '''
            Routine to update conditional probability p(q1,q2|cv) (and norm for
            final normalisation) by processing a series of CV trajectory file.
            Each CV trajectory file contains rows of the form

                time q1 q2 cv

            If the trajectory file contains this data in a different order, it
            can be accounted for using the col_xx keyword arguments. Similar
            constraints apply to these CV trajectory files as specified in the
            routine ´´process_trajectory_xyz´´.
        '''
        if not isinstance(fns, list): fns = [fns]
        print('Constructing conditional probability...')
        for fn in fns:
            print('  Reading data from %s' %fn)
            data = np.loadtxt(fn)
            for row in data:
                q1i, q2i, cvi = row[col_q1], row[col_q2], row[col_cv]
                if not self.cvmin<=cvi<=self.cvmax:
                    #print("Found frame with cv(=%.6f) beyond range of current FEP (=[%.6f,%.6f]), skipping" %(cvi, self.cvmin, self.cvmax))
                    continue
                else:
                    icv = int((cvi-self.cvmin)/(self.cvmax-self.cvmin)*(self.cvnum-1))
                    assert self.cvs[icv]<=cvi<=self.cvs[icv+1], "Inconsistency detected in cvs: for cv=%.6f we found icv=%i, however: cvs[%i]=%.6f and cvs[%i]=%.6f" %(cvi,icv,icv,self.cvs[icv],icv+1,self.cvs[icv+1])
                if not self.q1min<=q1i<=self.q1max:
                    raise ValueError("Found frame with q1 beyond given bounds")
                else:
                    iq1 = int((q1i-self.q1min)/(self.q1max-self.q1min)*(self.q1num-1))
                    if q1i-self.q1s[iq1]<-1e-6 or 1e-6<q1i-self.q1s[iq1+1]:
                        raise AssertionError("Inconsistency detected in q1 values: assigned index %i to value %.6f but q1s[%i]=%.6f and q1s[%i]=%.6f" %(iq1,q1i,iq1,self.q1s[iq1],iq1+1,self.q1s[iq1+1]))
                if not self.q2min<=q2i<=self.q2max:
                    raise ValueError("Found frame with q2 beyond given bounds")
                else:
                    iq2 = int((q2i-self.q2min)/(self.q2max-self.q2min)*(self.q2num-1))
                    if q2i-self.q2s[iq2]<-1e-6 or 1e-6<q2i-self.q2s[iq2+1]:
                        raise AssertionError("Inconsistency detected in q2 values: assigned index %i to value %.6f but q2s[%i]=%.6f and q2s[%i]=%.6f" %(iq2,q2i,iq2,self.q2s[iq2],iq2+1,self.q2s[iq2+1]))
                self.pconds[iq2,iq1,icv] += 1
                self.norms[icv] += 1

    def finish(self, fn_plt=None, plot_cvs=None):
        # normalize conditional probability as well as some additional probability densities
        for icv, norm in enumerate(self.norms):
            self.pqs += self.pconds[:,:,icv]
            if norm>0:
                self.pconds[:,:,icv] /= norm
            else:
                assert (self.pconds[:,:,icv]==0).all(), 'Inconsistency in normalisation of conditional probability'
        self.pqs /= integrate2d(self.pqs, x=self.q1s, y=self.q2s)
        self.pcvs = self.norms.copy()/self.norms.sum()
        # plot if requested
        if fn_plt is not None:
            if plot_cvs is None:
                raise IOError('If a plot is to be made (as indicated by giving a fn_plot different from None), plot_cvs should be specified as a list of cv values for which the conditional probability will be plotted')
            pp.clf()
            if   len(plot_cvs)<=2: nrows,ncols = 1,2
            elif len(plot_cvs)<=4: nrows,ncols = 2,2
            elif len(plot_cvs)<=6: nrows,ncols = 2,3
            elif len(plot_cvs)<=9: nrows,ncols = 3,3
            else:
                raise ValueError('Requested to plot too much cv values, limited to maximum 9')
            fig, axs = pp.subplots(nrows=nrows,ncols=ncols, sharex=True, sharey=True)
            #fig.suptitle(r'Log of conditional probability distribution')
            for i, cv in enumerate(plot_cvs):
                icv = int((cv-self.cvmin)/(self.cvmax-self.cvmin)*(self.cvnum-1))
                iax = i//ncols
                jax = i%ncols
                ax = axs[iax,jax]
                contourf = ax.contourf(self.q1s, self.q2s, self.pconds[:,:,icv], cmap=pp.get_cmap('rainbow'))
                if iax==nrows-1:
                    ax.set_xlabel('%s [au]' %self.q1_label, fontsize=16)
                if jax==0:
                    ax.set_ylabel('%s [au]' %self.q2_label, fontsize=16)
                ax.set_title('CV = %.3f au' %(self.cvs[icv]))
                cbar = pp.colorbar(contourf, ax=ax)
            #ax1.plot(self.cvs, np.log(self.pcvs)/np.log(10))
            #ax1.set_xlabel('%s [au]' %self.cv_label)
            #ax1.set_ylabel(r'Log of probabilty density [au$^{-1}$]')
            #ax1.set_title('Probability distrubution of old variable in trajectory')
            fig.set_size_inches([8*ncols,8*nrows])
            fig.tight_layout()
            pp.savefig(fn_plt)
            pp.close()

    def transform(self, fep, cv1_unit='au', cv2_unit='au', f_unit='kjmol', label1=None, label2=None):
        '''
            Transform the provided 1D FEP to a 2D FES using the current
            conditional probability according to the formula

                FES(q1,q2) = -kT*log(P(q1,q2))

                P(q1,q2) = int(condprob(q1,q2|cv)*exp(-beta*F(cv), cv)
        '''
        assert isinstance(fep, BaseFreeEnergyProfile), 'Input argument should be instance of BaseFreeEnergyProfile, instead received %s' %fep.__class__.__name__
        assert len(fep.cvs)==len(self.cvs), 'Dimension of 1D CV in conditional probability inconsistent with 1D FEP'
        assert (abs(fep.cvs-self.cvs)<1e-6).all(), 'Values of 1D CV in conditional probability not identical to those of 1D FEP'
        if label1 is None: label1 = self.q1_label
        if label2 is None: label2 = self.q2_label
        #construct 2D FES
        ps = np.zeros([self.q2num, self.q1num], float)
        for icv, fcv in enumerate(fep.fs):
            if not np.isnan(fcv):
                ps += self.pconds[:,:,icv]*np.exp(-fep.beta*fcv)
        ps /= integrate2d(ps, x=self.q1s, y=self.q2s)
        fs = np.zeros([self.q2num, self.q1num], float)*np.nan
        fs[ps>0] = -np.log(ps[ps>0])/fep.beta
        return FreeEnergySurface2D(self.q1s, self.q2s, fs, fep.T, cv1_unit=cv1_unit, cv2_unit=cv2_unit, f_unit=f_unit, cv1_label=label1, cv2_label=label2)
