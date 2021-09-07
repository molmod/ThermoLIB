#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 - 2021 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
# Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium;
# all rights reserved unless otherwise stated.
#
# This file is part of a library developed by Louis Vanduyfhuys at
# the Center for Molecular Modeling under supervision of prof. Veronique
# Van Speybroeck. Usage of this package should be authorized by prof. Van
# Van Speybroeck.


from molmod.units import *

import numpy as np
import matplotlib.pyplot as pp

__all__ = ['Parabola1D', 'Polynomial1D', 'Parabola2D', 'bias_dict']

class BiasPotential1D(object):
    def __init__(self, name, inverse_cv=False):
        self.name = name
        self.sign_q = 1.0
        if inverse_cv:
            self.sign_q = -1.0
    
    def print(self, **pars_units):
        return '%s (%s): %s' %(self.__class__.__name__, self.name, self.print_pars(**pars_units))

    def print_pars(self, **pars_units):
        raise NotImplementedError

    def __call__(self, q):
        raise NotImplementedError
    
    def plot(self, fn):
        raise NotImplementedError


class Parabola1D(BiasPotential1D):
    def __init__(self, name, q0, kappa, inverse_cv=False):
        BiasPotential1D.__init__(self, name, inverse_cv=inverse_cv)
        self.q0 = q0
        self.kappa = kappa
    
    def print_pars(self, kappa_unit='kjmol', q0_unit='au'):
        return 'K=%.0f %s  q0=%.3e %s' %(self.kappa/parse_unit(kappa_unit), kappa_unit, self.q0/parse_unit(q0_unit), q0_unit)
    
    def __call__(self, q):
        return 0.5*self.kappa*(q*self.sign_q-self.q0)**2


class Polynomial1D(BiasPotential1D):
    def __init__(self, name, coeffs, inverse_cv=False, unit='au'):
        '''
            Bias potential given by general polynomial of any degree.

            :param name: name of the bias
            :type name: str

            :param coeffs: list of expansion coefficients of the polynomial in increasing order starting with the coefficient of power 0. The degree of the polynomial is given by len(coeffs)-1
            :type coeffs: list/np.ndarray
        '''
        BiasPotential1D.__init__(self, name, inverse_cv=inverse_cv)
        self.coeffs = coeffs
        self.unit = unit
    
    def print_pars(self, e_unit='kjmol', q_unit='au'):
        return ' '.join(['a%i=%.3e %s/%s^%i' %(n,an/(e_unit/q_unit**n), e_unit, q_unit, n) for n,an in enumerate(self.coeffs)])

    def __call__(self, q):
        result = 0.0
        for n, an in enumerate(self.coeffs):
            result += an*(q*self.sign_q)**n
        return result*parse_unit(self.unit)

class AddMultiplePotentials1D(BiasPotential1D):
    def __init__(self, biasses, coeffs=None):
        '''
            Add multiple bias potentials together, possibly weighted by the given coefficients. If coefficients is not specified, all potentials are simply added without prefactor.

            :param biasses: list of bias potentials
            :type biasses: list(BiasPotential1D)

            :param coeffs: array of weigth coefficients. If not given, defaults to an array of ones.
            :type coeffs: list/np.ndarray, optional
        '''
        assert isinstance(biasses, list), 'Biasses should be a list'
        for bias in biasses:
            assert isinstance(bias, BiasPotential1D), 'Each bias in the list should be a (child) instance of the BiasPotential1D class. However, member %s is of class %s' %(bias.name, bias.__class__.__name__)
        if coeffs is None:
            coeffs = np.ones(len(biasses))
        else:
            if isinstance(coeffs, list):
                coeffs = np.array(coeffs)
            assert isinstance(coeffs, np.ndarray), 'Coefficients should be list or numpy array'
            assert len(coeffs)==len(biasses), 'Coefficients should be array of same length as biasses.'
        BiasPotential1D.__init__(self, 'MultipleBias')
        self.biasses = biasses
        self.coeffs = coeffs.copy()
    
    def print_pars(self, **par_units):
        string = 'MultipleBias:\n'
        for bias in self.biasses:
            string += '  %s\n' %(bias.print(**par_units))
        return string
    
    def __call__(self, q):
        result = 0.0
        for bias in self.biasses:
            result += bias(q)
        return result

class BiasPotential2D(object):
    def __init__(self, name, inverse_cv1=False, inverse_cv2=False):
        self.name = name
        self.sign_q1 = 1.0
        if inverse_cv1:
            self.sign_q1 = -1.0
        self.sign_q2 = 1.0
        if inverse_cv2:
            self.sign_q2 = -1.0
    
    def __call__(self, q1, q2):
        raise NotImplementedError
    
    def print(self, *pars_units):
        return '%s (%s): %s' %(self.__class__.__name__, self.name, self.print_pars(*pars_units))

    def print_pars(self, *pars_units):
        raise NotImplementedError
    
    def plot(self, fn, cv1s, cv2s, obs=None, obs_unit=None, cv1_unit='au', cv2_unit='au', cv1_label='CV1', cv2_label='CV2', levels=None):
        if obs is None:
            CV1, CV2 = np.meshgrid(cv1s, cv2s, indexing='ij')
            obs = self(CV1,CV2)
            obs_unit = 'kjmol'
            levels = np.linspace(0.0, 100, 11)
        pp.clf()
        contourf = pp.contourf(cv1s/parse_unit(cv1_unit), cv2s/parse_unit(cv2_unit), obs.T/parse_unit(obs_unit), cmap=pp.get_cmap('rainbow'), levels=levels)
        contour = pp.contour(cv1s/parse_unit(cv1_unit), cv2s/parse_unit(cv2_unit), obs.T/parse_unit(obs_unit), levels=levels)
        pp.xlabel('%s [%s]' %(cv1_label, cv1_unit), fontsize=16)
        pp.ylabel('%s [%s]' %(cv2_label, cv2_unit), fontsize=16)
        cbar = pp.colorbar(contourf, extend='both')
        cbar.set_label('Bias boltzmann factor [-]', fontsize=16)
        pp.clabel(contour, inline=1, fontsize=10)
        pp.title('Bias %s (%s):\n %s' %(self.__class__.__name__, self.name.replace('_','\_'), self.print_pars()), fontsize=14)
        fig = pp.gcf()
        fig.set_size_inches([12,8])
        pp.savefig(fn)


class Parabola2D(BiasPotential2D):
    def __init__(self, name, q01, q02, kappa1, kappa2, inverse_cv1=False, inverse_cv2=False):
        BiasPotential2D.__init__(self, name, inverse_cv1=inverse_cv1, inverse_cv2=inverse_cv2)
        self.q01 = q01
        self.q02 = q02
        self.kappa1 = kappa1
        self.kappa2 = kappa2
    
    def print_pars(self, kappa1_unit='kjmol', kappa2_unit='kjmol', q01_unit='au', q02_unit='au'):
        return 'K1=%.0f %s  q01=%.3e %s  K2=%.0f %s  q02=%.3e %s' %(self.kappa1/parse_unit(kappa1_unit), kappa1_unit, self.q01/parse_unit(q01_unit), q01_unit, self.kappa2/parse_unit(kappa2_unit), kappa2_unit, self.q02/parse_unit(q02_unit), q02_unit)
    
    def __call__(self, q1, q2):
        return 0.5*self.kappa1*(q1*self.sign_q1-self.q01)**2 + 0.5*self.kappa2*(q2*self.sign_q2-self.q02)**2

bias_dict = {
    'Parabola1D'  : Parabola1D,
    'Polynomial1D': Polynomial1D,
    'Parabola2D'  : Parabola2D,
}
