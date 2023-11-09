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

from typing import DefaultDict
from molmod.units import *
from molmod.constants import *
from molmod.io.xyz import XYZReader

from ..tools import integrate, integrate2d, format_scientific
from .fep import BaseFreeEnergyProfile, FreeEnergySurface2D

import matplotlib.pyplot as pp
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
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
        Routine to store and compute conditional probabilities of the form :math:`p(q_1|cv)` which allow to convert a free energy profile in terms of the collective variable :math:`cv` to a free energy profile in terms of the collective variable :math:`q_1`.

    '''
    def __init__(self, q1s, cvs, q1_label='Q1', cv_label='CV'):
        '''
            :param q1s: array representing the grid point values of the collective variable Q1, which should be in atomic units!
            :type q1s: numpy.ndarray

            :param cvs: array representing the grid point values of the collective variable CV, which should be in atomic units!
            :type cvs: numpy.ndarray

            :param q1_label: 
            :type q1_label: str, optional, default='Q1'
            
            :param cv_label:
            :type cv_label: str, optional, default='CV'
        '''
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
        self._finished = False

    def process_trajectory_xyz(self, fns, Q1, CV, sub=slice(None,None,None), finish=True):
        '''
            Compute the conditional probability p(q1|cv) (and norm for final normalisation) by processing a series of XYZ trajectory files. The final probability is estimated as the average over all given files. These files may also contain data from biased simulations.

            :param fns: file name (or list of file names) which contain trajectories that are used to compute the conditional probability.
            :type fns: str or list(str)

            :param Q1: collective variable definition used to compute the Q1 value, should be an object with the *compute* routine to compute the value of the collective variable given a set of molecular coordinates.
            :type Q1: classes defined in :mod:`thermolib.thermodynamics.cv`

            :param CV: collective variable definition used to compute the CV value, should be an object with the *compute* routine to compute the value of the collective variable given a set of molecular coordinates.
            :type CV: classes defined in :mod:`thermolib.thermodynamics.cv`
            
            :param sub: slice object to subsample the trajectory, for more information see https://molmod.github.io/molmod/reference/io.html#module-molmod.io.xyz
            :type sub: slice, optional, default=slice(None, None, None)

            :param finish: set this to True if the given file name(s) are the only relevant trajectories and hence the conditional probability should be computed from only these trajectories. Setting it to True will therefore trigger propper normalisation of the conditional probability. Set this to False if you intend to call the routine *process_trajectory_xyz* again later on with additional trajectory files.
            :type finish: bool, optional, default=True
        '''
        if self._finished:
            raise RuntimeError("Cannot read additional XYZ trajectory because current conditional probability has already been finished.")
        print('Constructing/updating conditional probability with input from XYZ trajectory files ...')
        if not isinstance(fns, list): fns = [fns]
        for fn in fns:
            xyz = XYZReader(fn, sub=sub)
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
        if finish:            
            self.finish()

    def process_trajectory_cvs(self, fns, col_q1=1, col_cv=2, sub=slice(None,None,None), finish=True):
        '''
            Compute the conditional probability p(q1|cv) (and norm for final normalisation) by processing a series of CV trajectory files. Each CV trajectory file contains rows of the form

                time q1 cv

            If the trajectory file contains this data in a different order, it can be accounted for using the col_xx keyword arguments.

            :param fns: file name (or list of file names) which contain trajectories that are used to compute the conditional probability.
            :type fns: str or list(str)

            :param col_q1: column index of the collective variable Q1 in the given input file.
            :type col_q1: int, optional, default=1

            :param col_cv: column index of the collective variable CV in the given input file.
            :type col_cv: int, optional, default=2
            
            :param sub: slice object to subsample the file.
            :type sub: slice, optional, default=slice(None, None, None)

            :param finish: set this to True if the given file name(s) are the only relevant trajectories and hence the conditional probability should be computed from only these trajectories. Setting it to True will therefore trigger propper normalisation of the conditional probability. Set this to False if you intend to call the routine *process_trajectory_xyz* again later on with additional trajectory files.
            :type finish: bool, optional, default=True
        '''
        if self._finished:
            raise RuntimeError("Cannot read additional XYZ trajectory because current conditional probability has already been finished.")
        if not isinstance(fns, list): fns = [fns]
        print('Constructing/updating conditional probability with input from CV trajectory files ...')
        for fn in fns:
            print('  Reading data from %s' %fn)
            data = np.loadtxt(fn)
            data = data[sub]
            self.pconds[:-1, :-1] += np.histogram2d(data[:, col_q1], data[:, col_cv], bins=(self.q1s, self.cvs))[0]
            self.norms[:-1] += np.histogram(data[:, col_cv], bins=self.cvs)[0]
        if finish:
            self.finish()

    def finish(self):
        if self._finished:
            raise RuntimeError("Current conditional probability has already been finished.")
        # Normalize conditional probability as well as some additional probability densities
        self.pqs = np.sum(self.pconds, axis=1)
        self.pconds[:, self.norms>0] /= self.norms[self.norms>0]
        assert (self.pconds[:, self.norms<=0]==0).all(), 'Inconsistency in normalisation of conditional probability'
        self.pqs /= np.trapz(self.pqs, x=self.q1s)
        self.pcvs = self.norms.copy()/self.norms.sum()
        self._finished = True

    def plot(self, fn='condprob.png', icvs=None):
        pp.clf()
        if icvs is None:
            contourf = pp.contourf(self.cvs, self.q1s, self.pconds, cmap=pp.get_cmap('rainbow'))
            pp.xlabel('%s [au]' %self.cv_label, fontsize=16)
            pp.ylabel('%s [au]' %self.q1_label, fontsize=16)
            pp.colorbar(contourf)
        else:
            for icv in icvs:
                pp.plot(self.q1s, self.pconds[:,icv], label='%s=%.3f' %(self.cv_label,self.cvs[icv]))
                pp.xlabel('%s [au]' %self.q1_label, fontsize=16)
                pp.ylabel('Conditional probability p(%s|%s)' %(self.q1_label,self.cv_label), fontsize=16)
                pp.title('CV = %.3f au' %(self.cvs[icv]))
        fig = pp.gcf()
        fig.set_size_inches([8,8])
        fig.tight_layout()
        pp.savefig(fn)
        return

    def transform(self, fep, cv_output_unit='au', f_output_unit=None, cv_label='CV'):
        '''
            Transform the provided 1D FES to a different 1D FES using the current conditional probability according to the formula. 
            
            **WARNING**: The error on the original fep is propagated to an error on the new transformed fep, but it is assumed that the conditional probability used for this transformation is exactly known (i.e. has no error, which is offcourse an approximation as the conditional probability is itself estimated from the simulation).

            .. math:: F(q) &= -kT \\ln\\left(\\int p(q|v)\\cdot e^{-\\beta F(v)} dv\\right)
            
            :param fep: the free energy profile F(v) which will be transformed
            :type fep: (child of) BaseFreeEnergyProfile

            :param cv_output_unit: the unit to be used in plotting and printing of the new cv
            :type cv_output_unit: str, optional, default='au'

            :param f_output_unit: the unit of the the transformed free energy profile to be used in plotting and printing of energies. If set to None, the f_output_unit of the given free energy profile will be use.
            :type f_output_unit: str, optional, default=None

            :param cv_label: the label of the new collective variable to be used in plot labels
            :type cv_label: str, optional, default='CV'
        '''
        assert self._finished, "Conditional probability needs to be finished before applying at in transformations."
        assert isinstance(fep, BaseFreeEnergyProfile), 'Input argument should be instance of BaseFreeEnergyProfile, instead received %s' %fep.__class__.__name__
        assert len(fep.cvs)==len(self.cvs), 'Dimension of 1D CV in conditional probability inconsistent with 1D FEP'
        assert (abs(fep.cvs-self.cvs)<1e-6).all(), 'Values of 1D CV in conditional probability not identical to those of 1D FEP'
        if cv_label is None: cv_label = self.q1_label
        # Construct 1D FES
        def transform(fs_old):
            mask = ~np.isnan(fs_old)
            ps = np.trapz(self.pconds[:,mask]*np.exp(-fep.beta*fs_old[mask]), x=self.cvs[mask])
            ps /= np.trapz(ps, x=self.q1s)
            fs_new = np.zeros([self.q1num], float)*np.nan
            fs_new[ps>0] = -np.log(ps[ps>0])/fep.beta
            return fs_new
        fs = transform(fep.fs)
        fupper, flower = None, None
        if fep.fupper is not None or fep.flower is None:
            print('WARNING: the error on the original fep is propagated to an error on the new transformed fep, but it is assumed that the conditional probability use for this transformation is exactly known (i.e. has no error), which is an approximation as the conditional probability is itself estimated from the simulation.')
        if fep.fupper is not None:
            fupper = transform(fep.fupper)
        if fep.flower is not None:
            flower = transform(fep.flower)
        if f_output_unit is None:
            f_output_unit = fep.f_output_unit
        return BaseFreeEnergyProfile(self.q1s, fs, fep.T, fupper=fupper, flower=flower, cv_output_unit=cv_output_unit, f_output_unit=f_output_unit, cv_label=cv_label)

    def deproject(self, fep, cv1_output_unit='au', cv2_output_unit='au', f_output_unit=None, cv1_label='Q1', cv2_label='Q2'):
        '''
            Deproject the provided 1D FEP F(q) to a 2D FES F(q,v) using the current conditional probability according to the formula

            .. math:: F(q_1,q_2) &= F(q_2)-kT \\ln\\left(p(q_1|q_2)\\right)
            
            **WARNING**: the error on the 1D FEP is propagated towards the deprojected 2D FES, but under the assumption that the underlying conditional probability is exactly known (i.e. it has no error, which is offcourse an approximation as the conditional probability is itself estimated from a molecular simulation).

            :param fep: the free energy profile F(q_2) which will be transformed
            :type fep: (child of) BaseFreeEnergyProfile

            :param cv1_output_unit: the unit to be used in plotting and printing of the collective variable :math:`q_1` in :math:`F(q_1,q_2)`
            :type cv1_output_unit: str, optional, default='au'

            :param cv2_output_unit: the unit to be used in plotting and printing of the collective variable :math:`q_2` in :math:`F(q_1,q_2)`
            :type cv2_output_unit: str, optional, default='au'

            :param f_output_unit: the unit of the the transformed free energy profile to be used in plotting and printing of energies. If set to None, the f_output_unit of the given free energy profile will be use.
            :type f_output_unit: str, optional, default=None

            :param cv1_label: the label of the collective variable :math:`q_1` in :math:`F(q_1,q_2)` to be used in plot labels
            :type cv1_label: str, optional, default='Q1'

            :param cv2_label: the label of the collective variable :math:`q_2` in :math:`F(q_1,q_2)` to be used in plot labels
            :type cv2_label: str, optional, default='Q2'
        '''
        assert self._finished, "Conditional probability needs to be finished before applying at in transformations."
        assert isinstance(fep, BaseFreeEnergyProfile), 'Input argument should be instance of BaseFreeEnergyProfile, instead received %s' %fep.__class__.__name__
        assert len(fep.cvs)==len(self.cvs), 'Dimension of collective variable v in conditional probability p(q|v) inconsistent with collective variable in 1D FEP'
        assert (abs(fep.cvs-self.cvs)<1e-6).all(), 'Values of collective variable v in conditional probability p(q|v) not identical to those of collective variable in 1D FEP'
        # Construct 1D FES
        fs = np.zeros([len(self.cvs), len(self.q1s)], float)*np.nan
        fupper, flower = None, None
        if fep.fupper is not None:
            fupper = np.zeros([len(self.cvs), len(self.q1s)], float)*np.nan
        if fep.flower is not None:
            flower = np.zeros([len(self.cvs), len(self.q1s)], float)*np.nan
        kT = boltzmann*fep.T
        for iq1 in range(len(self.q1s)):
            for icv in range(len(self.cvs)):
                if self.pconds[iq1,icv]>0:
                    fs[icv,iq1] = fep.fs[icv] - kT*np.log(self.pconds[iq1,icv])
                    if fep.fupper is not None:
                        fupper[icv,iq1] = fep.fupper[icv] - kT*np.log(self.pconds[iq1,icv])
                    if fep.flower is not None:
                        flower[icv,iq1] = fep.flower[icv] - kT*np.log(self.pconds[iq1,icv])
        if fep.fupper is not None or fep.flower is None:
            print('WARNING: the error on the 1D FEP is propagated to an error on the new deprojected 2D FES, but it is assumed that the conditional probability use for this transformation is exactly known (i.e. has no error), which is an approximation as the conditional probability is itself estimated from the simulation.')
        if f_output_unit is None:
            f_output_unit = fep.f_output_unit
        return FreeEnergySurface2D(self.q1s, self.cvs, fs, fep.T, fupper=fupper, flower=flower, cv1_output_unit=cv1_output_unit, cv2_output_unit=cv2_output_unit, f_output_unit=f_output_unit, cv1_label=cv1_label, cv2_label=cv2_label)

        

class ConditionalProbability1D2D(object):
    '''
        Class to store and compute conditional probabilities of the form :math:`p(q_1,q_2|cv)` which can be used to transform a 1D free energy profile in terms of the collective variable *cv* towards a 2D free energy surface in terms of the collective variables :math:`q_{1}` and :math:`q_{2}`.

    '''
    def __init__(self, q1s, q2s, cvs, q1_label='Q1', q2_label='Q2', cv_label='CV'):
        '''
            :param q1s: array representing the grid point values of the collective variable Q1, which should be in atomic units!
            :type q1s: numpy.ndarray
            .
            :param q2s: array representing the grid point values of the collective variable Q2, which should be in atomic units!
            :type q2s: numpy.ndarray

            :param cvs: array representing the grid point values of the collective variable CV, which should be in atomic units!
            :type cvs: numpy.ndarray

            :param q1_label: 
            :type q1_label: str, optional, default='Q1'

            :param q2_label: 
            :type q2_label: str, optional, default='Q2'
            
            :param cv_label:
            :type cv_label: str, optional, default='CV'
        '''
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
        self._finished = False

    def process_trajectory_xyz(self, fns, Q1, Q2, CV, sub=slice(None,None,None), finish=True):
        '''
            Compute the conditional probability :math:`p(q_1,q_2|v)` (and norm for final normalisation) by processing a series of XYZ trajectory files. The final probability is estimated as the average over all given files. These files may also contain data from biased simulations as long as the bias is constant over the simulation. For example, data from Umbrella Sampling is OK, while data from metadynamica itself is not. Data obtained from a regular MD with the final MTD profile as bias is OK.

            :param fns: file name (or list of file names) which contain trajectories that are used to compute the conditional probability.
            :type fns: str or list(str)

            :param Q1: collective variable definition used to compute the Q1 value, should be an object with the *compute* routine to compute the value of the collective variable Q1 given a set of molecular coordinates.
            :type Q1: classes defined in :mod:`thermolib.thermodynamics.cv`

            :param Q2: collective variable definition used to compute the Q2 value, should be an object with the *compute* routine to compute the value of the collective variable Q2 given a set of molecular coordinates.
            :type Q2: classes defined in :mod:`thermolib.thermodynamics.cv`

            :param CV: collective variable definition used to compute the CV value, should be an object with the *compute* routine to compute the value of the collective variable given a set of molecular coordinates.
            :type CV: classes defined in :mod:`thermolib.thermodynamics.cv`
            
            :param sub: slice object to subsample the trajectory, for more information see https://molmod.github.io/molmod/reference/io.html#module-molmod.io.xyz
            :type sub: slice, optional, default=slice(None, None, None)

            :param finish: set this to True if the given file name(s) are the only relevant trajectories and hence the conditional probability should be computed from only these trajectories. Setting it to True will therefore trigger propper normalisation of the conditional probability. Set this to False if you intend to call the routine *process_trajectory_xyz* again later on with additional trajectory files.
            :type finish: bool, optional, default=True
        '''
        if self._finished:
            raise RuntimeError("Cannot read additional XYZ trajectory because current conditional probability has already been finished.")
        if not isinstance(fns, list): fns = [fns]
        for fn in fns:
            xyz = XYZReader(fn, sub=sub)
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
        if finish:            
            self.finish()

    def process_trajectory_cvs(self, fns, col_q1=1, col_q2=2, col_cv=3, sub=slice(None,None,None), finish=True, tolerance=1e-6):
        '''
            Routine to update conditional probability :math:`p(q_1,q_2|v)` (and norm for final normalisation) by processing a series of CV trajectory file. Each CV trajectory file contains rows of the form

                time q1 q2 v

            If the trajectory file contains this data in a different order, it can be accounted for using the col_xx keyword arguments. Similar constraints apply to these CV trajectory files as specified in the routine :meth:`process_trajectory_xyz`.

            :param fns: file name (or list of file names) which contain trajectories that are used to compute the conditional probability.
            :type fns: str or list(str)

            :param col_q1: column index of the collective variable Q1 in the given input file.
            :type col_q1: int, optional, default=1

            :param col_q2: column index of the collective variable Q2 in the given input file.
            :type col_q2: int, optional, default=2

            :param col_cv: column index of the collective variable CV in the given input file.
            :type col_cv: int, optional, default=3
            
            :param sub: slice object to subsample the file.
            :type sub: slice, optional, default=slice(None, None, None)

            :param finish: set this to True if the given file name(s) are the only relevant trajectories and hence the conditional probability should be computed from only these trajectories. Setting it to True will therefore trigger propper normalisation of the conditional probability. Set this to False if you intend to call the routine *process_trajectory_xyz* again later on with additional trajectory files.
            :type finish: bool, optional, default=True
        '''
        if self._finished:
            raise RuntimeError("Cannot read additional XYZ trajectory because current conditional probability has already been finished.")
        if not isinstance(fns, list): fns = [fns]
        print('Constructing conditional probability...')
        for fn in fns:
            print('  Reading data from %s' %fn)
            data = np.loadtxt(fn)
            data = data[sub]
            for row in data:
                q1i, q2i, cvi = row[col_q1], row[col_q2], row[col_cv]
                if not self.cvmin<=cvi<=self.cvmax:
                    #print("Found frame with cv(=%.6f) beyond range of current FEP (=[%.6f,%.6f]), skipping" %(cvi, self.cvmin, self.cvmax))
                    continue
                else:
                    icv = int((cvi-self.cvmin)/(self.cvmax-self.cvmin)*(self.cvnum-1))
                    if icv<self.cvnum-1:
                        assert self.cvs[icv]-tolerance<=cvi<=self.cvs[icv+1]+tolerance, "Inconsistency detected in cvs: for cv=%f we found icv=%i, however: cvs[%i]=%f and cvs[%i]=%f" %(cvi,icv,icv,self.cvs[icv],icv+1,self.cvs[icv+1])
                if not self.q1min<=q1i<=self.q1max:
                    raise ValueError("Found frame with q1 beyond given bounds")
                else:
                    iq1 = int((q1i-self.q1min)/(self.q1max-self.q1min)*(self.q1num-1))
                    if iq1<self.q1num-1:
                        if q1i-self.q1s[iq1]<-tolerance or tolerance<q1i-self.q1s[iq1+1]:
                            raise AssertionError("Inconsistency detected in q1 values: assigned index %i to value %.6f but q1s[%i]=%.6f and q1s[%i]=%.6f" %(iq1,q1i,iq1,self.q1s[iq1],iq1+1,self.q1s[iq1+1]))
                if not self.q2min<=q2i<=self.q2max:
                    raise ValueError("Found frame with q2 beyond given bounds")
                else:
                    iq2 = int((q2i-self.q2min)/(self.q2max-self.q2min)*(self.q2num-1))
                    if iq2<self.q2num-1:
                        if q2i-self.q2s[iq2]<-tolerance or tolerance<q2i-self.q2s[iq2+1]:
                            raise AssertionError("Inconsistency detected in q2 values: assigned index %i to value %.6f but q2s[%i]=%.6f and q2s[%i]=%.6f" %(iq2,q2i,iq2,self.q2s[iq2],iq2+1,self.q2s[iq2+1]))
                self.pconds[iq2,iq1,icv] += 1
                self.norms[icv] += 1
        if finish:            
            self.finish()

    def finish(self):
        if self._finished:
            raise RuntimeError("Current conditional probability has already been finished.")
        # normalize conditional probability as well as some additional probability densities
        for icv, norm in enumerate(self.norms):
            self.pqs += self.pconds[:,:,icv]
            if norm>0:
                self.pconds[:,:,icv] /= norm
            else:
                assert (self.pconds[:,:,icv]==0).all(), 'Inconsistency in normalisation of conditional probability'
        self.pqs /= integrate2d(self.pqs, x=self.q1s, y=self.q2s)
        self.pcvs = self.norms.copy()/self.norms.sum()
        self._finished = True

    def plot(self, icvs, fn='condprob.png', scale='log', ncolors=8, mode='2D', slice_1D_indices=None):
        '''
            Plot the conditional probability in of several possible modes, defined by the mode argument:
            
                * **mode=2d** -- a 2D contour plot for each CV-value in a given set of indices (defined in the argument **icvs**). In other words, this makes a 2D contour plot of condprob(:,:|icv) for each icv in icvs.
                * **mode=1d-q1** --  a 1D crossection plot of the conditional probability as function of Q1 for a given set of Q2 indices (defined in the argument **slice_1D_inidices**) as well as a given set of CV indices (defined in the argument **icvs**).
                * **mode=1d-q2** --  a 1D crossection plot of the conditional probability as function of Q2 for a given set of Q1 indices (defined in the argument **slice_1D_inidices**) as well as a given set of CV indices (defined in the argument **icvs**).

            :param icvs: list of indices corresponding to CV values in p(Q1,Q2|CV) for which to plot the conditional probability
            :type icvs: list

            :param fn: the name of the file to write the plot to
            :type fn: str, optional, default='condprob.png'

            :param scale: either lin (or linear) for a linear plotting scale or log (or logarithm) for a logarithmic plotting scale.
            :type scale: str, optional, default='log'

            :param ncolors: the number of colors in the color bar for 2D plotting. Only applicable when mode is set to 2D.
            :type ncolors: int, optional, default=8

            :param mode: the plot mode, see description above for more details.
            :type mode: str, optional, default='2D'

            :param slice_1D_indices: only applicable when mode is set to 1d-q1 or 1d-q2, see description above for more details.
            :type slice_1D_indices: list(int), optional, default=None
        '''
        if   len(icvs)<=2: nrows,ncols = 1,2
        elif len(icvs)<=4: nrows,ncols = 2,2
        elif len(icvs)<=6: nrows,ncols = 2,3
        elif len(icvs)<=9: nrows,ncols = 3,3
        else:
            raise ValueError('Requested to plot too much cv values, limited to maximum 9')
        pp.clf()
        fig, axs = pp.subplots(nrows=nrows,ncols=ncols, sharex=False, sharey=True)
        for index, icv in enumerate(icvs):
            iax = index//ncols
            jax = index%ncols
            ax = axs[iax,jax]
            obs = self.pconds[:,:,icv].copy()
            if scale.lower() in ['log', 'logarithmic']:
                obs[obs<=0.0] = np.log(obs[obs>0.0].min())
                obs[obs>0.0] = np.log(obs[obs>0.0])
                ylabel_1D = 'log(P) [-]'
            elif scale.lower() in ['lin', 'linear']:
                ylabel_1D = 'P [au]'
            else:
                raise IOError('recieved invalid scale value, got %s, should be lin or log' %scale)
            if mode.lower()=='2d':
                lims = [min(obs[~np.isnan(obs)]), max(obs[~np.isnan(obs)])]
                plot_kwargs = {'levels': np.linspace(lims[0], lims[1], ncolors+1)}
                contourf = ax.contourf(self.q1s, self.q2s, obs, cmap=pp.get_cmap('rainbow'), **plot_kwargs)
                cbar = pp.colorbar(contourf, ax=ax)
                ax.set_xlabel('%s [au]' %self.q1_label, fontsize=16)
                if jax==0:
                    ax.set_ylabel('%s [au]' %self.q2_label, fontsize=16)
            elif mode.lower().startswith('1d'):
                assert slice_1D_indices is not None, 'When mode is set to %s, the indices of the collective variable %s for which 1D slices will be plotted need to be specified using the slice_1D_indices argument' %(mode, mode.lower().lstrip('1d-'))
                obs_sliced = []
                cvs = None
                if mode.lower() in ['1d-q1', '1d-cv1']:
                    for icv2 in slice_1D_indices:
                        obs_sliced.append(('%s=%.3e au' %(self.q2_label,self.q2s[icv2]),obs[icv2,:]))
                    cvs = self.q1s.copy()
                    xlabel_1D = '%s [au]' %(self.q1_label)
                elif mode.lower() in ['1d-q2', '1d-cv2']:
                    for icv1 in slice_1D_indices:
                        obs_sliced.append(('%s=%.3e au' %(self.q1_label,self.q1s[icv1]),obs[:,icv1]))
                    cvs = self.q2s.copy()
                    xlabel_1D = '%s [au]' %(self.q2_label)
                else:
                    raise ValueError('Invalid mode specification, received %s' %mode)
                for label,slice in obs_sliced:
                    ax.plot(cvs, slice, label=label)
                ax.set_xlabel(xlabel_1D, fontsize=16)
                if jax==0:
                    ax.set_ylabel(ylabel_1D, fontsize=16)
                    if iax==0:
                        ax.legend(loc='best', fontsize=14)
            else:
                raise ValueError('Invalid mode specification, received %s' %mode)
            ax.set_title('CV[%i] = %.3f au' %(icv, self.cvs[icv]), fontsize=16)
        if scale.lower() in ['log', 'logarithmic']:
            fig.suptitle('Logarithm of conditional probability distribution p(Q1,Q2|CV)', fontsize=22)
        elif scale.lower() in ['lin', 'linear']:
            fig.suptitle('Conditional probability distribution p(Q1,Q2|CV)', fontsize=22)
        fig.set_size_inches([8*ncols,8*nrows+2])
        fig.tight_layout(rect=(0,0,1,0.98))
        pp.savefig(fn)
        return

    def transform(self, fep, cv1_output_unit='au', cv2_output_unit='au', f_output_unit='kjmol'):
        '''
            Transform the provided 1D FEP to a 2D FES using the current conditional probability according to the formula

            .. math:: F(q_1,q_2) &= -kT\cdot\\ln\\left(\\int p(q_1,q_2|v)\\cdot e^{-\\beta F(v)}dv\\right)

            **WARNING**: the error on the 1D FEP is propagated towards the 2D FES, but under the assumption that the underlying conditional probability is exactly known (i.e. it has no error, which is offcourse an approximation as the conditional probability is itself estimated from a molecular simulation).
            
            :param fep: The free energy profile that will be transformed (:math:`F(v)` in the equation above).
            :type fep: BaseFreeEnergyProfile or inheriting child class
            
            :param cv1_output_unit: unit in which the CV1 values will be printed/plotted. Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_.
            :type cv1_output_unit: str, optional, defaults to 'au'

            :param cv2_output_unit: unit in which the CV2 values will be printed/plotted. Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_
            :type cv2_output_unit: str, optional, defaults to 'au'
            
            :param f_output_unit: unit in which the free energy values will be printe/plotted. Units are defined using `the molmod routine <https://molmod.github.io/molmod/reference/const.html#module-molmod.units>`_.
            :type f_output_unit: str, optional, default='kjmol'
        '''
        assert self._finished, "Conditional probability needs to be finished before applying at in transformations."
        assert isinstance(fep, BaseFreeEnergyProfile), 'Input argument should be instance of BaseFreeEnergyProfile, instead received %s' %fep.__class__.__name__
        assert len(fep.cvs)==len(self.cvs), 'Dimension of 1D CV in conditional probability inconsistent with 1D FEP'
        assert (abs(fep.cvs-self.cvs)<1e-6).all(), 'Values of 1D CV in conditional probability not identical to those of 1D FEP'
        #construct 2D FES
        def deproject(fs_old):
            ps = np.zeros([self.q2num, self.q1num], float)
            for icv, fcv in enumerate(fs_old):
                if not np.isnan(fcv):
                    ps += self.pconds[:,:,icv]*np.exp(-fep.beta*fcv)
            ps /= integrate2d(ps, x=self.q1s, y=self.q2s)
            fs_new = np.zeros([self.q2num, self.q1num], float)*np.nan
            fs_new[ps>0] = -np.log(ps[ps>0])/fep.beta
            return fs_new
        fs = deproject(fep.fs)
        fupper, flower = None, None
        if fep.fupper is not None or fep.flower is None:
            print('WARNING: the error on the 1D FEP is propagated to an error on the new deprojected 2D FES, but it is assumed that the conditional probability use for this transformation is exactly known (i.e. has no error), which is an approximation as the conditional probability is itself estimated from the simulation.')
        if fep.fupper is not None:
            fupper = deproject(fep.fupper)
        if fep.flower is not None:
            flower = deproject(fep.flower)
        return FreeEnergySurface2D(self.q1s, self.q2s, fs, fep.T, fupper=fupper, flower=flower, cv1_output_unit=cv1_output_unit, cv2_output_unit=cv2_output_unit, f_output_unit=f_output_unit, cv1_label=self.q1_label, cv2_label=self.q2_label)
