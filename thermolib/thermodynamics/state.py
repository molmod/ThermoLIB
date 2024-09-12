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
# Vanduyfhuys or prof. Van Van Speybroeck.

from __future__ import annotations

from thermolib.error import Propagator, GaussianDistribution, LogGaussianDistribution, SampleDistribution, ncycles_default
from thermolib.tools import integrate

from molmod.units import *

import numpy as np, sys
from numpy.ma import masked_array
import time

__all__ = ['Minimum', 'Maximum', 'Integrate']



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
            r = self.cv_dist.mean()
            if isinstance(r, masked_array):
                r = r.filled(np.nan)
            if isinstance(r, np.ndarray) and len(r)==1: r = r[0]
            return r

    def get_F(self):
        if self.F_dist is None:
            return self.F
        else:
            r = self.F_dist.mean()
            if isinstance(r, masked_array):
                r = r.filled(np.nan)
            if isinstance(r, np.ndarray) and len(r)==1: r = r[0]
            return r
    
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
        cv, f = cvs[index], fs[index]
        if isinstance(cv, masked_array): cv = cv.filled(np.nan)
        if isinstance(f , masked_array): f  = f.filled(np.nan)
        return index, cv, f
    
    def compute(self, fep, dist_prop='montecarlo'):
        '''
            :param dist_prop: method of how the distribution of the error on the free energy profile is propagated to the error on the cv value and free energy value of each micro and macrostate. Currently only 'montecarlo' and 'stationairy' is implemented. Stationairy assumes the position of the minimum/maximum is given by that of the mean profile (i.e. with no errr on the cv value) and the value of the minimum/maximum is then extracted from the corresponding free energy value of the free energy profile (with its associated error bar). Montecarlo takes random samples from the fep, computes the microstate and uses these to estimate the distribution of the microstate. Defaults to 'montecarlo'
            :type dist_prop: str, optional
        '''
        if fep.error is not None:
            self.index, self.cv, self.F = None, None, None
            if dist_prop=='montecarlo':
                def fun_cv(fs):
                    return self._get_state_fep(fep.cvs, fs)[1]
                def fun_F(fs):
                    return self._get_state_fep(fep.cvs, fs)[2]
                self.propagator = Propagator(ncycles=ncycles_default)
                self.propagator.gen_args_samples(fep.error)
                self.propagator.calc_fun_values(fun_cv)
                self.cv_dist = self.propagator.get_distribution()
                self.propagator.calc_fun_values(fun_F)
                self.F_dist = self.propagator.get_distribution()
            elif dist_prop=='stationary':
                index, cv, F = self._get_state_fep(fep.cvs, fep.fs)
                self.cv_dist = GaussianDistribution(means=np.float64(cv), stds=np.float64(0.0))
                self.F_dist = GaussianDistribution(means=np.float64(F), stds=fep.error.std()[index])
            else:
                raise NotImplementedError('Distribution propagation method %s not implemented' %dist_prop)
        else:
            self.index, self.cv, self.F = self._get_state_fep(fep.cvs, fep.fs)
            self.cv_dist, self.F_dist = None, None

    def print(self):
        print('MICROSTATE %s:' %self.name)
        print('--------------')
        if self.F_dist is not None:
            print('  index = ',self.index)
            print('  F     = %s' %(self.F_dist.print(unit=self.f_unit)))
            print('  CV    = %s' %(self.cv_dist.print(unit=self.cv_unit)))
        else:
            print('  index = ', self.index)
            print('  F     = %.3f  %s' %(self.F/parse_unit(self.f_unit)  , self.f_unit ))
            print('  CV    = %.3f  %s' %(self.cv/parse_unit(self.cv_unit), self.cv_unit))
        print('')


class Minimum(Microstate):
    '''Microstate class that identifies the minimum in a certain range defined by either cv values, indexes or other microstates.'''
    def __init__(self, name, cv_range=[-np.inf, np.inf], cv_unit='au', f_unit='kjmol'):
        self.cv_range = cv_range
        Microstate.__init__(self, name, cv_unit=cv_unit, f_unit=f_unit)

    def _get_index_fep(self, cvs, fs):
        index = np.nan
        fcurrent = np.nan
        for i, (cv, f) in enumerate(zip(cvs,fs)):
            if self.cv_range[0]<=cv and cv<=self.cv_range[1]:
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
    def __init__(self, name, cv_range=None, cv_unit='au', f_unit='kjmol'):
        self.cv_range = cv_range
        Microstate.__init__(self, name, cv_unit=cv_unit, f_unit=f_unit)
    
    def _get_index_fep(self, cvs, fs):
        index = None
        fcurrent = None
        for i, (cv, f) in enumerate(zip(cvs,fs)):
            if self.cv_range[0]<=cv and cv<=self.cv_range[1]:
                if index is None:
                    if not np.isnan(f):
                        index = i 
                        fcurrent = f
                elif f>fcurrent:
                    index = i
                    fcurrent = f
        assert index is not None, "No valid maximum found for %s microstate in range" %(self.name)
        return index


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

    def compute(self, fep, dist_prop='montecarlo', verbose=False):
        '''
            :param dist_prop: method of how the distribution of the error on the free energy profile is propagated to the error on the cv value and free energy value of each micro and macrostate. Currently only 'analytical' is implemented, which applies the pythagorian sum of error bars valid for the sum (= discrete integral).
            :type dist_prop: str, optional
        '''
        if fep.error is not None:
            if dist_prop=='montecarlo':
                #method below underestimates error because of neglect of covariance between free energy values of neighboring points and the 'filtering' behavior of the integral.
                def fun_cv(fs):
                    return self._get_state_fep(fep.cvs, fs)[0]
                def fun_cvstd(fs):
                    return self._get_state_fep(fep.cvs, fs)[1]
                def fun_Z(fs):
                    return self._get_state_fep(fep.cvs, fs)[2]
                def fun_F(fs):
                    return self._get_state_fep(fep.cvs, fs)[3]
                #init propagator
                self.propagator = Propagator(ncycles=ncycles_default)
                self.propagator.gen_args_samples(fep.error)
                self.propagator.calc_fun_values(fun_cv)
                self.cv_dist = self.propagator.get_distribution()
                self.propagator.calc_fun_values(fun_cvstd)
                self.cvstd_dist = self.propagator.get_distribution()
                self.propagator.calc_fun_values(fun_F)
                self.F_dist     = self.propagator.get_distribution()
                self.propagator.calc_fun_values(fun_Z)
                self.Z_dist = self.propagator.get_distribution(target_distribution=LogGaussianDistribution)
                self.cv     = self.cv_dist.mean()
                self.cvstd  = self.cvstd_dist.mean()
                self.Z      = self.Z_dist.mean()
                self.F      = self.F_dist.mean()
            else:
                raise NotImplementedError('Distribution propagation method %s not implemented' %dist_prop)
        else:
            self.cv, self.cvstd, self.Z, self.F = self._get_state_fep(fep.cvs, fep.fs)

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
    def __init__(self, name, cv_range, beta, cv_unit='au', f_unit='kjmol'):
        self.cv_range = cv_range
        self.beta = beta
        Macrostate.__init__(self, name, cv_unit=cv_unit, f_unit=f_unit)
    
    def _get_state_fep(self, cvs, fs, fdist=None):
        #get cv_range in case a microstate was used in the cv_range argument
        cv_range = [None, None]
        if isinstance(self.cv_range[0], Microstate):
            index = self.cv_range[0]._get_index_fep(cvs, fs)
            cv_range[0] = cvs[index]
        else:
            cv_range[0] = self.cv_range[0]
        if isinstance(self.cv_range[1], Microstate):
            index = self.cv_range[1]._get_index_fep(cvs, fs)
            cv_range[1] = cvs[index]
        else:
            cv_range[1] = self.cv_range[1]    
        index_mask = (cv_range[0]<=cvs)*(cvs<=cv_range[1])
        #get state
        mask = ~np.isnan(fs)
        ps = np.exp(-self.beta*fs)/integrate(cvs[mask], np.exp(-self.beta*fs[mask]))
        cvs_indexed = cvs[index_mask]
        ps_indexed = ps[index_mask]
        fs_indexed = fs[index_mask]
        mask2 = ~np.isnan(fs_indexed)
        P = integrate(cvs_indexed[mask2], ps_indexed[mask2])
        Z = integrate(cvs_indexed[mask2], np.exp(-self.beta*fs_indexed[mask2]))
        F = -np.log(Z)/self.beta
        mean = integrate(cvs_indexed[mask2], ps_indexed[mask2]*cvs_indexed[mask2])/P
        std = np.sqrt(integrate(cvs_indexed[mask2], ps_indexed[mask2]*(cvs_indexed[mask2]-mean)**2)/P)
        if fdist is None:
            return mean, std, Z, F,
        else:
            raise NotImplementedError
            ferrs_indexed = fdist.std()[index_mask]

