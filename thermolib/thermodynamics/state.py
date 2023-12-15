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

from thermolib.error import Propagator, GaussianDistribution, SampleDistribution
from thermolib.tools import integrate

from molmod.units import *

import numpy as np, sys


__all__ = ['Minimum', 'Maximum', 'MinInf', 'PlusInf', 'Integrate']


class State(object):
    '''Abstract class for inheriting classes Microstate and Macrostate'''
    def __init__(self, name, cv_unit='au', f_unit='kjmol'):
        self.name = name
        self.cycles = {'cv': None, 'f': None}
        self.cv = None
        self.cv_dist = None
        self.F = None
        self.F_dist = None
        self.cv_unit = cv_unit
        self.f_unit = f_unit
    
    def get_cv(self):
        if self.cv_dist is None:
            return self.cv
        else:
            return self.cv_dist.mean()

    def get_F(self):
        if self.F_dist is None:
            return self.F
        else:
            return self.F_dist.mean()
    
    def text(self, obs, fmt='%.3f', do_scientific=False):
        if obs.lower()=='f':
            if self.F_dist is None:
                if do_scientific:
                    return fmt %do_scientific(self.F/parse_unit(self.f_unit)) + ' ' + self.f_unit
                else:
                    return fmt %(self.F/parse_unit(self.f_unit)) + ' ' + self.f_unit
            else:
                return self.F_dist.print(fmt=fmt, unit=self.f_unit, do_scientific=do_scientific)
        elif obs.lower()=='cv':
            if self.cv_dist is None:
                if do_scientific:
                    return fmt %do_scientific(self.cv/parse_unit(self.cv_unit)) + ' ' + self.cv_unit
                else:
                    return fmt %(self.cv/parse_unit(self.cv_unit)) + ' ' + self.cv_unit
            else:
                return self.cv_dist.print(fmt=fmt, unit=self.cv_unit, do_scientific=do_scientific)
        else:
            raise ValueError('Invalid definition for obs, should be f or cv, received %s' %obs)


class Microstate(State):
    '''Abstract parent class for defining microstates on a free energy profile. The routine _get_index_fep needs to be implemented in child classes.'''
    def __init__(self, name, cv_unit='au', f_unit='kjmol'):
        State.__init__(self, name, cv_unit=cv_unit, f_unit=f_unit)
        self.cycles['indexes'] = None
        self.index = None
    
    def _get_index_fep(self, cvs, fs):
        raise NotImplementedError

    def _get_state_fep(self, cvs, fs):
        index = self._get_index_fep(cvs, fs)
        return index, cvs[index], fs[index]

    def compute(self, cvs, fs, fdist=None):
        if fdist is not None:
            def fun_cv(fs):
                return self._get_state_fep(cvs, fs)[1]
            def fun_F(fs):
                return self._get_state_fep(cvs, fs)[2]
            propagator = Propagator(ncycles=fdist.ncycles, target_distribution=SampleDistribution)
            cv_dist = propagator(fun_cv, fdist)
            F_dist  = propagator(fun_F , fdist)
            #cvmean = cv_dist.means
            self.index = None #np.argmin(np.abs(cvs-cvmean))
            self.cv = cvs[self.index]
            self.F = fs[self.index]
            self.cv_dist = cv_dist
            self.F_dist = F_dist
        else:
            self.index, self.cv, self.F = self._get_state_fep(cvs, fs)

    def print(self):
        print('MICROSTATE %s:' %self.name)
        print('--------------')
        if self.F_dist is not None:
            print('  index = ',self.index)
            print('  F     = %s' %(self.F_dist.print(unit=self.f_unit)))
            print('  CV    = %s' %(self.cv_dist.print(unit=self.cv_unit)))
        else:
            print('  index = %i' %(self.index))
            print('  F     = %.3f  %s' %(self.F/parse_unit(self.f_unit)  , self.f_unit ))
            print('  CV    = %.3f  %s' %(self.cv/parse_unit(self.cv_unit), self.cv_unit))
        print('')


class Minimum(Microstate):
    '''Microstate class that identifies the minimum in a certain range defined by either cv values, indexes or other microstates.'''
    def __init__(self, name, cv_range=None, index_range=None, ms_range=None, cv_unit='au', f_unit='kjmol'):
        self.check_in_range = self._get_check_in_range_function(cv_range=cv_range, index_range=index_range, ms_range=ms_range)
        Microstate.__init__(self, name, cv_unit=cv_unit, f_unit=f_unit)

    def _get_check_in_range_function(self, cv_range=None, index_range=None, ms_range=None):
        if cv_range is not None:
            assert index_range is None and ms_range is None, 'Only one of cv_range, index_range or ms_range should be defined!'
            def check(index, cvs, fs):
                return cv_range[0]<=cvs[index]<=cv_range[1]
        elif index_range is not None:
            assert cv_range is None and ms_range is None, 'Only one of cv_range, index_range or ms_range should be defined!'
            def check(index, cvs, fs):
                return index_range[0]<=index<=index_range[1]
        elif ms_range is not None:
            assert cv_range is None and index_range is None, 'Only one of cv_range, index_range or ms_range should be defined!'
            def check(index, cvs, fs):
                start = ms_range[0]._get_index_fep(cvs, fs)
                end   = ms_range[1]._get_index_fep(cvs, fs)
                return start<=index<=end
        else:
            #if no range is defined, return an always-passing function.
            def check(index, cvs, fs):
                return True
        return check

    def _get_index_fep(self, cvs, fs):
        index = np.nan
        fcurrent = np.nan
        #print("Search for Minimum microstate %s" %self.name)
        for i, f in enumerate(fs):
            if self.check_in_range(i, cvs, fs):
                if np.isnan(index):
                    if not np.isnan(f):
                        index = i 
                        fcurrent = f
                elif f<fcurrent:
                    index = i
                    fcurrent = f
        assert not np.isnan(index), "No valid minimum found for %s microstate in range" %(self.name)
        return index
   

class Maximum(Microstate):
    '''Microstate class that identifies the maximum in a certain range defined by either cv values, indexes or other microstates.'''
    def __init__(self, name, cv_range=None, index_range=None, ms_range=None, cv_unit='au', f_unit='kjmol'):
        self.check_in_range = self._get_check_in_range_function(cv_range=cv_range, index_range=index_range, ms_range=ms_range)
        Microstate.__init__(self, name, cv_unit=cv_unit, f_unit=f_unit)

    def _get_check_in_range_function(self, cv_range=None, index_range=None, ms_range=None):
        if cv_range is not None:
            assert index_range is None and ms_range is None, 'Only one of cv_range, index_range or ms_range should be defined!'
            def check(index, cvs, fs):
                return cv_range[0]<=cvs[index]<=cv_range[1]
        elif index_range is not None:
            assert cv_range is None and ms_range is None, 'Only one of cv_range, index_range or ms_range should be defined!'
            def check(index, cvs, fs):
                return index_range[0]<=index<=index_range[1]
        elif ms_range is not None:
            assert cv_range is None and index_range is None, 'Only one of cv_range, index_range or ms_range should be defined!'
            def check(index, cvs, fs):
                start = ms_range[0]._get_index_fep(cvs, fs)
                end   = ms_range[1]._get_index_fep(cvs, fs)
                return start<=index<=end
        else:
            #if no range is defined, return an always-passing function.
            def check(index, cvs):
                return True
        return check

    def _get_index_fep(self, cvs, fs):
        index = None
        fcurrent = None
        for i, f in enumerate(fs):
            if self.check_in_range(i, cvs, fs):
                if index is None:
                    if not np.isnan(f):
                        index = i 
                        fcurrent = f
                elif f>fcurrent:
                    index = i
                    fcurrent = f
        assert index is not None, "No valid maximum found for %s microstate in range" %(self.name)
        return index


class MinInf(Microstate):
    '''Microstate defining a CV value of -inf. Usefull for later defining macrostates as an integral from -inf to a given microstate.'''
    def __init__(self):
        Microstate.__init__(self, 'mininf')

    def _get_index_fep(self, cvs, fs):
        return 0


class PlusInf(Microstate):
    '''Microstate defining a CV value of +inf. Usefull for later defining macrostates as an integral from a given microstate to +inf.'''
    def __init__(self):
        Microstate.__init__(self, 'plusinf')

    def _get_index_fep(self, cvs, fs):
        return len(cvs)-1
    

class Macrostate(State):
    '''Abstract parent class for defining macrostates on a free energy profile. The routine _get_state_fep needs to be implemented in child classes.'''
    def __init__(self, name, cv_unit='au', f_unit='kjmol'):
        State.__init__(self, name, cv_unit=cv_unit, f_unit=f_unit)
        self.cvstd = None
        self.cvstd_dist = None
        self.Z = None
        self.Z_dist = None

    def _get_state_fep(self, cvs, fs):
        raise NotImplementedError

    def compute(self, cvs, fs, fdist=None):
        if fdist is not None:
            def fun_cv(fs):
                return self._get_state_fep(cvs, fs)[0]
            def fun_cvstd(fs):
                return self._get_state_fep(cvs, fs)[1]
            def fun_Z(fs):
                return self._get_state_fep(cvs, fs)[2]
            def fun_F(fs):
                return self._get_state_fep(cvs, fs)[3]
            propagator = Propagator(ncycles=fdist.ncycles)
            cv_dist    = propagator(fun_cv   , fdist)
            cvstd_dist = propagator(fun_cvstd, fdist)
            Z_dist     = propagator(fun_Z    , fdist)
            F_dist     = propagator(fun_F    , fdist)
            self.cv         = cv_dist.mean()
            self.cv_dist    = cv_dist
            self.cvstd      = cvstd_dist.mean()
            self.cvstd_dist = cvstd_dist
            self.Z          = Z_dist.mean()
            self.Z_dist     = Z_dist
            self.F          = F_dist.mean()
            self.F_dist     = F_dist
        else:
            self.cv, self.cvstd, self.Z, self.F, P = self._get_state_fep(cvs, fs)

    def print(self):
        print('MACROSTATE %s:' %self.name)
        print('--------------')
        if self.F_dist is not None:
            print('  F       = %s' %(self.F_dist.print(unit=self.f_unit)))
            print('  CV Mean = %s' %(self.cv_dist.print(unit=self.cv_unit)))
            print('  CV StD  = %s' %(self.cvstd_dist.print(unit=self.cv_unit)))
        else:
            print('  F       = %.3f  %s' %(self.f/parse_unit(self.f_unit), self.f_unit))
            print('  CV Mean = %.3f  %s' %(self.cv/parse_unit(self.cv_unit), self.cv_unit))
            print('  CV StD  = %.3f  %s' %(self.cvstd/parse_unit(self.cv_unit), self.cv_unit))
        print('')

    def text(self, obs, fmt='%.3f'):
        if obs.lower()=='f':
            if self.F_dist is None:
                return fmt %(self.F/parse_unit(self.f_unit)) + ' ' + self.f_unit
            else:
                return self.F_dist.print(fmt=fmt, unit=self.f_unit)
        elif obs.lower()=='cv':
            if self.cv_dist is None:
                return fmt %(self.cv/parse_unit(self.cv_unit)) + ' ' + self.cv_unit
            else:
                return self.cv_dist.print(fmt=fmt, unit=self.cv_unit)
        elif obs.lower()=='cvstd':
            if self.cvstd_dist is None:
                return fmt %(self.cvstd/parse_unit(self.cv_unit)) + ' ' + self.cv_unit
            else:
                return self.cvstd_dist.print(fmt=fmt, unit=self.cv_unit)
        else:
            raise ValueError('Invalid definition for obs, should be f, cv or cvstd, received %s' %obs)


class Integrate(Macrostate):
    def __init__(self, name, microstates, beta, cv_unit='au', f_unit='kjmol'):
        self.microstates = microstates
        self.beta = beta
        Macrostate.__init__(self, name, cv_unit=cv_unit, f_unit=f_unit)
    
    def _get_indices_fep(self, cvs, fs):
        start = self.microstates[0]._get_index_fep(cvs, fs)
        end = self.microstates[1]._get_index_fep(cvs, fs)
        return np.array(range(start,end+1))
    
    def _get_state_fep(self, cvs, fs):
        indexes = self._get_indices_fep(cvs, fs)
        mask = ~np.isnan(fs)
        ps = np.exp(-self.beta*fs)/integrate(cvs[mask], np.exp(-self.beta*fs[mask]))
        cvs_indexed = cvs[indexes]
        ps_indexed = ps[indexes]
        fs_indexed = fs[indexes]
        mask2 = ~np.isnan(fs_indexed)
        P = integrate(cvs_indexed[mask2], ps_indexed[mask2])
        Z = integrate(cvs_indexed[mask2], np.exp(-self.beta*fs_indexed[mask2]))
        F = -np.log(Z)/self.beta
        mean = integrate(cvs_indexed[mask2], ps_indexed[mask2]*cvs_indexed[mask2])/P
        std = np.sqrt(integrate(cvs_indexed[mask2], ps_indexed[mask2]*(cvs_indexed[mask2]-mean)**2)/P)
        return mean, std, Z, F, P
