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

from thermolib.error import Propagator, GaussianDistribution, LogGaussianDistribution
from thermolib.tools import integrate

from ..units import *

import numpy as np
from numpy.ma import masked_array
import time

__all__ = ['Minimum', 'Maximum', 'Integrate']



class State(object):
    '''
        Abstract class serving as a parent for Microstate and Macrostate classes
    '''
    def __init__(self, name, cv_unit='au', f_unit='kjmol', propagator=Propagator()):
        '''
            :param name: a name for the state to be used in printing/loging
            :type name: str

            :param cv_unit: a unit for the cv properties in the state used in printing/loging
            :type cv_unit: str, optional, default='au'

            :param f_unit: a unit for the (free) energy properties in the state used in printing/loging
            :type f_unit: str, optional, default='kjmol'

            :param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken)
            :type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator()
        '''
        self.name = name
        self.cv = None
        self.cv_dist = None
        self.F = None
        self.F_dist = None
        self.cv_unit = cv_unit
        self.f_unit = f_unit
        self.propagator = propagator
    
    def get_cv(self):
        '''
            If computation of the state value(s) is done on a FEP which doesn't have error bars, then the cv value of the current state will be stored in self.cv. If the FEP includes error bars, then the cv value and its propagated error bar will be stored as a distribution in self.cv_dist. The get_cv routine allows to extract the cv value itself (not its error bar) from the proper internal attribute indepedent on the error bar was propagated or not.

            :returns: CV value of current state (without error bar)
            :rtype: float
        '''
        if self.cv_dist is None:
            return self.cv
        else:
            r = self.cv_dist.mean()
            if isinstance(r, masked_array):
                r = r.filled(np.nan)
            if isinstance(r, np.ndarray) and len(r)==1: r = r[0]
            return r

    def get_F(self):
        '''
            If computation of the state value(s) is done on a FEP which doesn't have error bars, then the free energy value of the current state will be stored in self.F. If the FEP includes error bars, then the F value and its propagated error bar will be stored as a distribution in self.F_dist. The get_F routine allows to extract the F value itself (not its error bar) from the proper internal attribute indepedent on the error bar was propagated or not.

            :returns: free energy value of current state (without error bar)
            :rtype: float
        '''
        if self.F_dist is None:
            return self.F
        else:
            r = self.F_dist.mean()
            if isinstance(r, masked_array):
                r = r.filled(np.nan)
            if isinstance(r, np.ndarray) and len(r)==1: r = r[0]
            return r
    
    def text(self, obs, fmt='%.3f', do_scientific=False):
        '''
            Routine to format the observable defined in obs (cv or f) into text for printing.

            :param obs: the observable of the current state that needs to be converted to text. Is not case sensitive.
            :type obs: str, either 'cv' or 'f'

            :param fmt: python formatting string of a float to be used in the notation 
            :type fmt: str, optional, default='%.3f'

            :param do_scientific: format the float in scientific notation, e.g. 2.31 10^2
            :type do_scientific: bool, optional, default=False

            :raises ValueError: if obs is neither 'cv' nor 'f'

            :return: formatted string for printing containing the specified observable in the current state
            :rtype: str
        '''
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
    '''
        Abstract parent class for defining microstates on a free energy profile (fep), i.e. a single point on the fep. The routine _get_index_fep needs to be implemented in child classes.
    '''
    def __init__(self, name, cv_unit='au', f_unit='kjmol', propagator=Propagator()):
        '''
            :param name: a name for the state to be used in printing/loging
            :type name: str

            :param cv_unit: a unit for the cv properties in the state used in printing/loging
            :type cv_unit: str, optional, default='au'

            :param f_unit: a unit for the (free) energy properties in the state used in printing/loging
            :type f_unit: str, optional, default='kjmol'

            :param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken, or the desired distribution of the targeted error). See documentation on the :py:class:`Propagator <thermolig.error.Propagator>` class for more info.
            :type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator()
        '''
        State.__init__(self, name, cv_unit=cv_unit, f_unit=f_unit, propagator=propagator)
        self.index = None

    def _get_index_fep(self, cvs, fs):
        '''
            Routine for computing the index of the current microstate cv. This routine needs to be implemented in the child classes.
        '''
        raise NotImplementedError

    def _get_state_fep(self, cvs, fs):
        '''
            Routine that will use the index defining the current microstate as obtained by the _get_index_fep routine and compute the corresponding cv and f value.

            :param cvs: cvs argument representing the cv grid to be parsed to the _get_index_fep routine
            :type cvs: np.ndarray

            :param fs: fs argument representing the f grid to be parsed to the _get_index_fep routine
            :type fs: np.ndarray

            :return: index, cv value , f value of microstate
            :rtype: (int, float, float)
        '''
        index = self._get_index_fep(cvs, fs)
        cv, f = cvs[index], fs[index]
        if isinstance(cv, masked_array): cv = cv.filled(np.nan)
        if isinstance(f , masked_array): f  = f.filled(np.nan)
        return index, cv, f
    
    def compute(self, fep, dist_prop='montecarlo'):
        '''
            Routine to compute the current microstate values for the given profile (fep).

            :param fep: the profile of an observable for which the current microstate will be computed
            :type fep: :py:class:`BaseProfile <thermolib.thermodynamics.fep.BaseProfile>`

            :param dist_prop: method of how the distribution of the error on the free energy profile is propagated to the error on the cv and free energy value of the microstate. Currently only 'montecarlo' and 'stationairy' is implemented. 
            
                - Stationairy assumes the microstate cv value (e.g. the cv of the minimum/maximum of the fep) is given by that of the mean fep (i.e. with no error on the cv value) and the microstate f value (e.g. the min/max free energy) is then extracted by evaluating the fep at the microstate cv value (with an error bar on the f value taken from the fep error at the microstate cv value).
                - Montecarlo takes random samples from the entire fep, computes the microstate cv and f values and uses all these samples to estimate the distribution of the microstate. 
            
            Stationairy is used for very specific purposes (when setting the plotting reference of feps) and is not recommended in general.
            :type dist_prop: str, optional, default='montecarlo'

            :raises NotImplementedError: if the given dist_prop is not supported (see above for allowed values).
        '''
        if fep.error is not None:
            self.index, self.cv, self.F = None, None, None
            if dist_prop=='montecarlo':
                def fun_cv(fs):
                    return self._get_state_fep(fep.cvs, fs)[1]
                def fun_F(fs):
                    return self._get_state_fep(fep.cvs, fs)[2]
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
        '''
            Print the current thermodynamic microstate properties
        '''
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
    '''
        Microstate class that identifies the minimum in a certain range defined by either cv values, indexes or other microstates.
    '''
    def __init__(self, name, cv_range=[-np.inf, np.inf], cv_unit='au', f_unit='kjmol', propagator=Propagator()):
        '''
            :param name: a name for the state to be used in printing/loging
            :type name: str

            :param cv_range: the microstate cv value will be looked for only in this range 
            :type cv_range: list, optional, default=[-np.inf, np.inf]
            
            :param cv_unit: a unit for the cv properties in the state used in printing/loging
            :type cv_unit: str, optional, default='au'

            :param f_unit: a unit for the (free) energy properties in the state used in printing/loging
            :type f_unit: str, optional, default='kjmol'

            :param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken)
            :type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator()
        '''
        self.cv_range = cv_range
        Microstate.__init__(self, name, cv_unit=cv_unit, f_unit=f_unit, propagator=propagator)

    def _get_index_fep(self, cvs, fs):
        '''
            Routine for computing the index of the current microstate cv for the profile defined by cvs and fs.
        
            :param cvs: cv grid on which to look for the microstate cv value
            :type cvs: np.ndarray

            :param fs: f values corresponding to the cv grid
            :type fs: np.ndarray

            :returns: the index of the microstate cv value
            :rtype: int
        '''
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
    '''
        Microstate class that identifies the maximum in a certain range defined by either cv values, indexes or other microstates.
    '''
    def __init__(self, name, cv_range=None, cv_unit='au', f_unit='kjmol', propagator=Propagator()):
        '''
            :param name: a name for the state to be used in printing/loging
            :type name: str

            :param cv_range: the microstate cv value will be looked for only in this range 
            :type cv_range: list, optional, default=[-np.inf, np.inf]
            
            :param cv_unit: a unit for the cv properties in the state used in printing/loging
            :type cv_unit: str, optional, default='au'

            :param f_unit: a unit for the (free) energy properties in the state used in printing/loging
            :type f_unit: str, optional, default='kjmol'

            :param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken)
            :type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator()
        '''
        self.cv_range = cv_range
        Microstate.__init__(self, name, cv_unit=cv_unit, f_unit=f_unit, propagator=propagator)
    
    def _get_index_fep(self, cvs, fs):
        '''
            Routine for computing the index of the current microstate cv for the profile defined by cvs and fs.
        
            :param cvs: cv grid on which to look for the microstate cv value
            :type cvs: np.ndarray

            :param fs: f values corresponding to the cv grid
            :type fs: np.ndarray

            :returns: the index of the microstate cv value
            :rtype: int
        '''
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
    '''
        Abstract parent class for defining macrostates on a free energy profile. The routine _get_state_fep needs to be implemented in child classes.
    '''
    def __init__(self, name, cv_unit='au', f_unit='kjmol', propagator=Propagator()):
        '''
            :param name: a name for the state to be used in printing/loging
            :type name: str

            :param cv_unit: a unit for the cv properties in the state used in printing/loging
            :type cv_unit: str, optional, default='au'

            :param f_unit: a unit for the (free) energy properties in the state used in printing/loging
            :type f_unit: str, optional, default='kjmol'

            :param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken, or the desired distribution of the targeted error). See documentation on the :py:class:`Propagator <thermolig.error.Propagator>` class for more info.
            :type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator()
        '''
        State.__init__(self, name, cv_unit=cv_unit, f_unit=f_unit, propagator=propagator)
        self.cvstd = None
        self.cvstd_dist = None
        self.Z = None
        self.Z_dist = None

    def _get_state_fep(self, cvs, fs):
        '''
            Routine to compute the macrostate properties for the profile defined by cvs and fs. This routine needs to be implementd in the child classes.
        '''
        raise NotImplementedError

    def compute(self, fep, dist_prop='montecarlo'):
        '''
            Routine to compute the current macrostate properties values for the given profile (fep).

            :param fep: the profile of an observable for which the current macrostate will be computed
            :type fep: :py:class:`BaseProfile <thermolib.thermodynamics.fep.BaseProfile>`

            :param dist_prop: method of how the distribution of the error on the free energy profile is propagated to the error on the macrostate properties. Currently only 'montecarlo' is implemented. 
            
                - Montecarlo takes random samples from the entire fep, computes the microstate cv and f values and uses all these samples to estimate the distribution of the microstate. 
            
            :type dist_prop: str, optional, default='montecarlo'

            :raises NotImplementedError: if the given dist_prop is not supported (see above for allowed values).
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
        '''
            Print the current thermodynamic macrostate properties
        '''
        print('MACROSTATE %s:' %self.name)
        print('--------------')
        if self.F_dist is not None:
            print('  F       = %s' %(self.F_dist.print(unit=self.f_unit)))
            print('  CV Mean = %s' %(self.cv_dist.print(unit=self.cv_unit)))
            print('  CV StD  = %s' %(self.cvstd_dist.print(unit=self.cv_unit)))
        else:
            print('  F       = %.3f  %s' %(self.F/parse_unit(self.f_unit), self.f_unit))
            print('  CV Mean = %.3f  %s' %(self.cv/parse_unit(self.cv_unit), self.cv_unit))
            print('  CV StD  = %.3f  %s' %(self.cvstd/parse_unit(self.cv_unit), self.cv_unit))
        print('')

    def text(self, obs, fmt='%.3f'):
        '''
            Routine to format the observable defined in obs (cv, cvstd f) into text for printing.

            :param obs: the observable of the current state that needs to be converted to text. This should either be 'cv' for the mean CV in the macrostate, 'cvstd' for the std on the cv in the macrostate or 'f' for the macrostate free energy. Is not case sensitive.
            :type obs: str

            :param fmt: python formatting string of a float to be used in the notation 
            :type fmt: str, optional, default='%.3f'

            :param do_scientific: format the float in scientific notation, e.g. 2.31 10^2
            :type do_scientific: bool, optional, default=False

            :raises ValueError: if obs is neither 'cv' nor 'f'

            :return: formatted string for printing containing the specified observable in the current state
            :rtype: str
        '''
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
    '''
        Definie macrostate as the boltzmann weighted integral over a range of microstates (i.e. range of points on the fep). The macrostate properties (mean cv, cvstd and macrostate free energy) are defined as follows

        .. math::

            \\left\\langle cv \\right\\rangle   &= \\frac{\\int q e^{-\\beta F(q)} dq}{\\int e^{-\\beta F(q)} dq} \\\\
            \\sigma^2                           &= \\frac{\\int \\left(q-\\left\\langle cv \\right\\rangle\\right)^2 e^{-\\beta F(q)} dq}{\\int e^{-\\beta F(q)} dq} \\\\
            f                                   &= -k_B T \\ln\\left[\\int e^{-\\beta F(q)} dq\\right]

        Herein, all integrals are done over a given cv range.
    '''
    
    def __init__(self, name, cv_range, beta, cv_unit='au', f_unit='kjmol', propagator=Propagator()):
        '''
            :param name: a name for the state to be used in printing/loging
            :type name: str

            :param cv_range: the range for the cv defining the macrostate in the integrals above
            :type cv_range: list of either floats or :py:class:`MicroStates <thermolib.thermodynamics.state.MicrosState>`

            :param beta: the beta (i.e. 1/kT) value used in the boltzmann weighting
            :type beta: float

            :param cv_unit: a unit for the cv properties in the state used in printing/loging
            :type cv_unit: str, optional, default='au'

            :param f_unit: a unit for the (free) energy properties in the state used in printing/loging
            :type f_unit: str, optional, default='kjmol'

            :param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken)
            :type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator()
        '''
        self.cv_range = cv_range
        self.beta = beta
        Macrostate.__init__(self, name, cv_unit=cv_unit, f_unit=f_unit, propagator=propagator)
    
    def _get_state_fep(self, cvs, fs):
        '''
            Routine to compute the macrostate properties for the profile defined by cvs and fs.

            :param cvs: cv grid on which to look for the microstate cv value
            :type cvs: np.ndarray

            :param fs: f values corresponding to the cv grid
            :type fs: np.ndarray
        '''
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
        return mean, std, Z, F,
