#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 - 2026 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
# Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium;
# all rights reserved unless otherwise stated.
#
# This file is part of a library developed by Louis Vanduyfhuys at
# the Center for Molecular Modeling. Usage of this package should be 
# authorized by prof. Van Vanduyfhuys.

from typing import List
from ..units import *
from ..constants import *
from scipy import interpolate

import numpy as np
import matplotlib.pyplot as pp

__all__ = [
    'BiasPotential1D', 'Parabola1D', 'Polynomial1D', 'PlumedSplinePotential1D',
    'MultipleBiasses1D', 'BiasPotential2D', 'Parabola2D'
]

class BiasPotential1D(object):
    '''
        Abstract class for 1-dimensional bias potentials. Its inheriting child classes should implement the ``__call__`` and ``print_pars`` routines.
    '''
    def __init__(self, name, inverse_cv=False):
        '''
            :param name: name for the bias which will also be given in the title of plots
            :type name: string

            :param inverse_cv: If set to True, the CV axix will be inverted prior to bias potential evaluation. WARNING: possible rest value parameters of the potential (such as the rest value of the Parabola1D potential) will not be multiplied with -1!
            :type inverse_cv: bool, optional, default=False
        '''
        self.name = name
        self.sign_q = 1.0
        if inverse_cv:
            self.sign_q = -1.0
    
    def print(self, **pars_units):
        '''
            Routine to return a formatted string defining the current bias for logging purposes. The keywords arguments in ``pars_units`` are defined in the ``print_pars`` routine defined in the child classes.
        '''
        return '%s (%s): %s' %(self.__class__.__name__, self.name, self.print_pars(**pars_units))

    def print_pars(self, **pars_units):
        '''
            Routine to format a string containing the parameters of the bias for printing/logging purposes. This routine needs to be implemented in each child class.
        '''
        raise NotImplementedError

    def __call__(self, q):
        '''
            Routine to compute the value of the bias as function of the given CV value. This routine needs to be implemented in each child class.
        '''
        raise NotImplementedError
    
    def plot(self, cvs, fn=None, e_unit='kjmol', cv_unit='au', cv_label='CV1'):
        '''
            Routine to make a plot of the bias potential as function of the CV.

            :param cvs: array representing the cv grid on which the bias is evaluated
            :type cvs: np.ndarray
            
            :param fn: file name to write the plot to. If None, the plot is not written to a file.
            :type fn: str, optional, default=None

            :param e_unit: energy unit in which the bias potential will be plotted
            :type e_unit: str, optional, default='kjmol'

            :param cv_unit: unit in which the CV will be plotted
            :type cv_unit: str, optional, default='au'

            :param cv_label: Label of the CV to be used in the x-label of the plot
            :type cv_label: str, optional, default='CV1'
        '''
        obs = self(cvs)
        pp.clf()
        pp.plot(cvs/parse_unit(cv_unit), obs/parse_unit(e_unit))
        pp.xlabel('%s [%s]' %(cv_label, cv_unit), fontsize=16)
        pp.ylabel('Energy [%s]' %(e_unit), fontsize=16)
        pp.title('Bias %s (%s):\n %s' %(self.__class__.__name__, self.name.replace('_','\_'), self.print_pars()), fontsize=14)
        fig = pp.gcf()
        fig.set_size_inches([8,8])
        if fn is not None:
            pp.savefig(fn)


class Parabola1D(BiasPotential1D):
    '''
        A 1-dimensional parabolic bias potential

        .. math:: V(q) = \\frac{\\kappa}{2}\\left(s*q-q_0\\right)^2

        The potential can be computed for a given CV value using its ``__call__`` routine, the bias definition can be printed in a formated way using its ``print`` routine and a plot can be made using its ``plot`` routine.
    '''
    def __init__(self, name, q0, kappa, inverse_cv=False):
        '''
            :param name: name for the bias which will also be given in the title of plots
            :type name: string

            :param q0: the value of the parabola equilibruim (i.e. its minimum)
            :type q0: float

            :param kappa: the force constant of the parabola
            :type kappa: float

            :param inverse_cv: If set to True, the CV-axis will be inverted prior to bias evaluation (sets :math:`s=-1` in the above formula for the bias potential). WARNING: the rest value parameter q0 of the potential will not be multiplied with -1!
            :type inverse_cv: bool, optional, defaults to False
        '''
        BiasPotential1D.__init__(self, name, inverse_cv=inverse_cv)
        self.q0 = q0
        self.kappa = kappa
    
    def print_pars(self, kappa_unit='kjmol', q0_unit='au'):
        '''
            Routine used by the :py:meth:`print <thermolib.thermodynamics.bias.BiasPotential1D.print>` routine defined in the parent :py:class:`BiasPotential1D <thermolib.thermodynamics.bias.BiasPotential1D>` to print the parameters of the bias potential in a formatted way. The keyword arguments defined below, can be specified upon calling the :py:meth:`print <thermolib.thermodynamics.bias.BiasPotential1D.print>` routine.

            :param kappa_unit: Unit of the parabola force constant kappa
            :type kappa_unit: str, optional, default='kjmol'

            :param q0_unit: Unit of the parabola equilibrium CV value q0
            :type q0_unit: str, optional, default='au'
        '''
        return 'K=%.0f %s  q0=%.3e %s' %(self.kappa/parse_unit(kappa_unit), kappa_unit, self.q0/parse_unit(q0_unit), q0_unit)
    
    def __call__(self, q):
        '''
            Compute the value of the bias potential for the given CV value

            :param q: CV value at which the bias potential will be evaluated
            :type q: float
        '''
        return 0.5*self.kappa*(q*self.sign_q-self.q0)**2


class Polynomial1D(BiasPotential1D):
    '''
        Bias potential given by general polynomial of any degree:

        .. math:: V(q) = \\sum_{n}a_n\\left(s*q\\right)^n

        The potential can be computed for a given CV value using its ``__call__`` routine, the bias definition can be printed in a formated way using its ``print`` routine and a plot can be made using its ``plot`` routine.
    '''
    def __init__(self, name, coeffs, inverse_cv=False, unit='au'):
        '''
            :param name: name for the bias which will also be given in the title of plots
            :type name: string

            :param coeffs: list of expansion coefficients of the polynomial in increasing order starting with the coefficient of power 0. The degree of the polynomial is given by len(coeffs)-1
            :type coeffs: list/np.ndarray

            :param inverse_cv: If True, the CV-axis will be inverted prior to bias evaluation (sets :math:`s=-1` in the above formula for the bias potential).
            :type inverse_cv: bool, optional, default=False

            :param unit: unit in which the bias is given
            :type unit: str, optional, default='au'
        '''
        BiasPotential1D.__init__(self, name, inverse_cv=inverse_cv)
        self.coeffs = coeffs
        self.unit = unit
    
    def print_pars(self, e_unit='kjmol', q_unit='au'):
        '''
            Routine used by the :py:meth:`print <thermolib.thermodynamics.bias.BiasPotential1D.print>` routine defined in the parent :py:class:`BiasPotential1D <thermolib.thermodynamics.bias.BiasPotential1D>` to print the parameters of the bias potential in a formatted way. The keyword arguments defined below, can be specified upon calling the :py:meth:`print <thermolib.thermodynamics.bias.BiasPotential1D.print>` routine.

            :param e_unit: energy unit to be used in defining the units of the coefficients
            :type kappa_unit: str, optional, default='kjmol'

            :param q0_unit: cv unit to be used in defining the units of the coefficients
            :type q0_unit: str, optional, default='au'
        '''
        return ' '.join(['a%i=%.3e %s/%s^%i' %(n,(an/(parse_unit(e_unit)/parse_unit(q_unit)))**n, e_unit, q_unit, n) for n,an in enumerate(self.coeffs)])

    def __call__(self, q):
        '''
            Compute the value of the bias potential for the given CV value

            :param q: CV value at which the bias potential will be evaluated
            :type q: float
        '''
        result = 0.0
        for n, an in enumerate(self.coeffs):
            result += an*(q*self.sign_q)**n
        return result*parse_unit(self.unit)


class PlumedSplinePotential1D(BiasPotential1D):
    '''
        A bias potential read from a PLUMED file, which is spline-interpolated. The potential can be computed for a given CV value using its ``__call__`` routine, the bias definition can be printed in a formated way using its ``print`` routine and a plot can be made using its ``plot`` routine.
    '''
    def __init__(self, name, fn, inverse_cv=False, unit='au', scale=1.0):
        '''
            :param name: name for the bias which will also be given in the title of plots
            :type name: string

            :param fn: specifies the filename of an external potential written on a grid and acting on the collective variable, as used with the EXTERNAL keyword in PLUMED.
            :type fn: str

            :param inverse_cv: If set to True, the CV-axis will be inverted prior to bias evaluation.
            :type inverse_cv: bool, optional, default=False
            
            :param unit: unit used to express the external potential
            :type unit: str, optional, default='au'

            :param scale: scaling factor for the bias potential (useful to invert free energy surfaces)
            :type scale: float, optional, default=1.0
        '''
        BiasPotential1D.__init__(self, name, inverse_cv=inverse_cv)
        pars = np.loadtxt(fn).T
        self.splint = interpolate.splrep(pars[0], pars[1])
        self.unit = unit
        self.scale = scale
    
    def print_pars(self, **kwargs):
        '''
            Not implemented, will just return empty string.
        '''
        #TODO
        return ''

    def __call__(self, q):
        '''
            Compute the value of the bias potential for the given CV value

            :param q: CV value at which the bias potential will be evaluated
            :type q: float
        '''
        value = interpolate.splev(q*self.sign_q, self.splint, der=0)
        return value*self.scale*parse_unit(self.unit)


class MultipleBiasses1D(BiasPotential1D):
    '''
        A class to add multiple bias potentials together, possibly weighted by given coefficients.
    '''
    def __init__(self, biasses, coeffs=None):
        '''
            :param biasses: list of bias potentials
            :type biasses: list of instances of child classes of :py:class:`BiasPotential1D <thermolib.thermodynamics.bias.BiasPotential1D>`

            :param coeffs: array of weigth coefficients. If None, coeffs is set to an array of ones (i.e. no weighting is applied).
            :type coeffs: list/np.ndarray, optional, default=None

            :raises AssertionError: if biasses argument is not a list
            :raises AssertionError: if a bias in the biasses list is not an instance of :py:class:`BiasPotential1D <thermolib.thermodynamics.bias.BiasPotential1D>`
            :raises AssertionError: if coeffs is not a np.ndarray
            :raises AssertionError: if biasses and coeffs are not of equal length
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
        '''
            Routine used by the :py:meth:`print <thermolib.thermodynamics.bias.BiasPotential1D.print>` routine defined in the parent :py:class:`BiasPotential1D <thermolib.thermodynamics.bias.BiasPotential1D>` to print the parameters of the bias potential in a formatted way. The keyword arguments defined below, can be specified upon calling the :py:meth:`print <thermolib.thermodynamics.bias.BiasPotential1D.print>` routine  and will be parsed to the individual ``print`` routines of each bias. Hence, you can append the unit definitions of each bias in the list as keyword arguments of the current routine.
        '''
        string = 'MultipleBias:\n'
        for bias in self.biasses:
            string += '  %s\n' %(bias.print(**par_units))
        return string
    
    def __call__(self, q):
        '''
            Compute the value of the bias potential for the given CV value

            :param q: CV value at which the bias potential will be evaluated
            :type q: float
        '''
        result = 0.0
        for idx,bias in enumerate(self.biasses):
            result += self.coeffs[idx]*bias(q)
        return result


class BiasPotential2D(object):
    '''
        A base class for 2-dimensional bias potentials. Its inheriting child classes should implement the ``__call__`` and ``print_pars`` routines.
    '''
    def __init__(self, name, inverse_cv1=False, inverse_cv2=False):
        '''
            :param name: name for the bias which will also be given in the title of plots
            :type name: string
            
            :param inverse_cv1: If set to True, the CV1-axis will be inverted prior to bias evaluation. WARNING: possible rest value parameters of the potential (such as the rest value q01 of the Parabola2D potential) will not be multiplied with -1!
            :type inverse_cv1: bool, optional, defaults to False

            :param inverse_cv2: If set to True, the CV2-axis will be inverted prior to bias evaluation. WARNING: possible rest value parameters of the potential (such as the rest value q02 of the Parabola2D potential) will not be multiplied with -1!
            :type inverse_cv2: bool, optional, defaults to False
        '''
        self.name = name
        self.sign_q1 = 1.0
        if inverse_cv1:
            self.sign_q1 = -1.0
        self.sign_q2 = 1.0
        if inverse_cv2:
            self.sign_q2 = -1.0
    
    def __call__(self, q1, q2):
        '''
            Compute the value of the bias potential for the given CV value

            :param q1: value of first CV at which the bias potential will be evaluated
            :type q1: float

            :param q2: value of second CV at which the bias potential will be evaluated
            :type q2: float
        '''
        raise NotImplementedError
    
    def print(self, *pars_units):
        '''
            Routine to return a formatted string defining the current bias for logging purposes. The keywords arguments in ``pars_units`` are defined in the ``print_pars`` routine defined in the child classes.
        '''
        return '%s (%s): %s' %(self.__class__.__name__, self.name, self.print_pars(*pars_units))

    def print_pars(self, *pars_units):
        '''
            Routine to format a string containing the parameters of the bias for printing/logging purposes. This routine needs to be implemented in each child class.
        '''
        raise NotImplementedError
    
    def plot(self, cv1s, cv2s, fn=None, temp=None, e_unit='kjmol', cv1_unit='au', cv2_unit='au', cv1_label='CV1', cv2_label='CV2', levels=np.linspace(0.0, 100, 11)):
        '''
            Routine to make a plot of the bias potential as function of the CV.

            :param cv1s: array representing the cv1 grid on which the bias is evaluated
            :type cv1s: np.ndarray

            :param cv2s: array representing the cv2 grid on which the bias is evaluated
            :type cv2s: np.ndarray
            
            :param fn: file name to write the plot to. If None, the plot is not written to a file.
            :type fn: str, optional, default=None

            :param temp: If None, the bias potential itself is plotted. Else, the boltzmann factor exp(-V/kT) corresponding with the bias potential V is computed at the corresponding temperature given by temp.
            :type temp: float or None, optional, default=None

            :param e_unit: energy unit in which the bias potential will be plotted. This parameter is ignored if temp is not None
            :type e_unit: str, optional, default='kjmol'

            :param cv1_unit: unit in which the CV1 will be plotted
            :type cv1_unit: str, optional, default='au'

            :param cv2_unit: unit in which the CV2 will be plotted
            :type cv2_unit: str, optional, default='au'

            :param cv1_label: Label of the CV1 to be used in the x-label of the plot
            :type cv1_label: str, optional, default='CV1'

            :param cv2_label: Label of the CV2 to be used in the x-label of the plot
            :type cv2_label: str, optional, default='CV2'
        '''
        CV1, CV2 = np.meshgrid(cv1s, cv2s, indexing='ij')
        if temp is None:
            obs = self(CV1,CV2)
            obs_unit = e_unit
            obs_label = 'Bias energy [%s]' %e_unit
        else:
            beta = 1.0/(boltzmann*temp)
            obs = np.exp(-beta*self(CV1,CV2))
            obs_unit = 'au'
            obs_label = 'Bias boltzmann factor [-]'
        pp.clf()
        contourf = pp.contourf(cv1s/parse_unit(cv1_unit), cv2s/parse_unit(cv2_unit), obs.T/parse_unit(obs_unit), cmap=pp.get_cmap('rainbow'), levels=levels)
        contour = pp.contour(cv1s/parse_unit(cv1_unit), cv2s/parse_unit(cv2_unit), obs.T/parse_unit(obs_unit), levels=levels)
        pp.xlabel('%s [%s]' %(cv1_label, cv1_unit), fontsize=16)
        pp.ylabel('%s [%s]' %(cv2_label, cv2_unit), fontsize=16)
        cbar = pp.colorbar(contourf, extend='both')
        cbar.set_label(obs_label, fontsize=16)
        pp.clabel(contour, inline=1, fontsize=10)
        pp.title('Bias: %s' %(self.print()), fontsize=14)
        fig = pp.gcf()
        fig.set_size_inches([12,8])
        pp.savefig(fn)


class Parabola2D(BiasPotential2D):
    '''
        A 2-dimensional parabolic bias potential.

        .. math:: V(q_1,q_2) = \\frac{\\kappa_1}{2}\\left(s_1*q_1-q_{1,0}\\right)^2 + \\frac{\\kappa_2}{2}\\left(s_2*q_2-q_{2,0}\\right)^2

        The potential can be computed for a given CV value using its ``__call__`` routine, the bias definition can be printed in a formated way using its ``print`` routine and a plot can be made using its ``plot`` routine.
    '''
    def __init__(self, name, q01, q02, kappa1, kappa2, inverse_cv1=False, inverse_cv2=False):
        '''
            :param name: name for the bias which will also be given in the title of plots
            :type name: string

            :param q01: the value of the first collective variable corresponding to the parabola minimum
            :type q01: float

            :param q02: the value of the second collective variable corresponding to the parabola minimum
            :type q02: float

            :param kappa1: the force constant of the parabola in the direction of the first collective variable
            :type kappa: float

            :param kappa2: the force constant of the parabola in the direction of the second collective variable
            :type kappa: float

            :param inverse_cv1: If set to True, the CV1-axis will be inverted prior to bias evaluation (sets :math:`s_1=-1` in the expression above). WARNING: the rest value parameter q01 will not be multiplied with -1!
            :type inverse_cv1: bool, optional, defaults to False

            :param inverse_cv2: If set to True, the CV2-axis will be inverted prior to bias evaluation (sets :math:`s_2=-1` in the expression above). WARNING: the rest value parameter q02 will not be multiplied with -1!
            :type inverse_cv2: bool, optional, defaults to False
        '''
        BiasPotential2D.__init__(self, name, inverse_cv1=inverse_cv1, inverse_cv2=inverse_cv2)
        self.q01 = q01
        self.q02 = q02
        self.kappa1 = kappa1
        self.kappa2 = kappa2
    
    def print_pars(self, kappa1_unit='kjmol', kappa2_unit='kjmol', q01_unit='au', q02_unit='au'):
        '''
            Routine used by the :py:meth:`print <thermolib.thermodynamics.bias.BiasPotential1D.print>` routine defined in the parent :py:class:`BiasPotential1D <thermolib.thermodynamics.bias.BiasPotential1D>` to print the parameters of the bias potential in a formatted way. The keyword arguments defined below, can be specified upon calling the :py:meth:`print <thermolib.thermodynamics.bias.BiasPotential2D.print>` routine.

            :param kappa1_unit: Unit of the parabola force constant kappa along the first CV
            :type kappa1_unit: str, optional, default='kjmol'

            :param kappa2_unit: Unit of the parabola force constant kappa along the second CV
            :type kappa2_unit: str, optional, default='kjmol'

            :param q01_unit: Unit of the CV1 value corresponding to the parabola equilibrium (minimum)
            :type q01_unit: str, optional, default='au'

            :param q02_unit: Unit of the CV2 value corresponding to the parabola equilibrium (minimum)
            :type q02_unit: str, optional, default='au'
        '''
        return 'K1=%.0f %s  q01=%.3e %s  K2=%.0f %s  q02=%.3e %s' %(self.kappa1/parse_unit(kappa1_unit), kappa1_unit, self.q01/parse_unit(q01_unit), q01_unit, self.kappa2/parse_unit(kappa2_unit), kappa2_unit, self.q02/parse_unit(q02_unit), q02_unit)
    
    def __call__(self, q1, q2):
        '''
            Compute the value of the bias potential for the given values of CV1 and CV2

            :param q1: CV1 value at which the bias potential will be evaluated
            :type q1: float

            :param q2: CV2 value at which the bias potential will be evaluated
            :type q2: float
        '''
        return 0.5*self.kappa1*(q1*self.sign_q1-self.q01)**2 + 0.5*self.kappa2*(q2*self.sign_q2-self.q02)**2



class MultipleBiasses2D(BiasPotential2D):
    '''
        A class to add multiple bias potentials (either 1D or 2D) together, possibly weighted by given coefficients.
    '''
    def __init__(self, biasses, additional_bias_dimension='cv1', coeffs=None):
        '''
            :param biasses: list of bias potentials to be added
            :type biasses: list(BiasPotential1D,BiasPotential2D)

            :param additional_bias_dimension: For each 1D potential in the list of biasses, apply the bias along the cv defined by this parameter.
            :type additional_bias_dimension: str, optional, default='cv1'
            
            :param coeffs: array of weigth coefficients. If not given, defaults to an array of ones (i.e. no weighting).
            :type coeffs: list/np.ndarray, optional

            :raises AssertionError: if argument biasses is not a list
            :raises AssertionError: if a memeber of biasses is not an instance of the :py:class:`BiasPotential1D <thermolib.thermodynamics.bias.BiasPotential1D>` or :py:class:`BiasPotential2D <thermolib.thermodynamics.bias.BiasPotential2D>`class
            :raises AssertionError: if argument coeffs is not a np.ndarray
            :raises AssertionError: if argument coeffs is given and is not of same length as argument biasses
        '''
        assert isinstance(biasses, list), 'Biasses should be a list'
        for bias in biasses:
            assert isinstance(bias, BiasPotential1D) or isinstance(bias, BiasPotential2D), 'Each bias in the list should be a (child) instance of the BiasPotential1D class or BiasPotential2D class. However, member %s is of class %s' %(bias.name, bias.__class__.__name__)
        if coeffs is None:
            coeffs = np.ones(len(biasses))
        else:
            if isinstance(coeffs, list):
                coeffs = np.array(coeffs)
            assert isinstance(coeffs, np.ndarray), 'Coefficients should be list or numpy array'
            assert len(coeffs)==len(biasses), 'Coefficients should be array of same length as biasses.'
        BiasPotential2D.__init__(self, 'MultipleBias')
        self.biasses = biasses
        self.additional_bias_dimension = additional_bias_dimension
        self.coeffs = coeffs.copy()
    
    def print_pars(self, **par_units):
        '''
            Routine used by the :py:meth:`print <thermolib.thermodynamics.bias.BiasPotential2D.print>` routine defined in the parent :py:class:`BiasPotential2D <thermolib.thermodynamics.bias.BiasPotential2D>` to print the parameters of the bias potential in a formatted way. The keyword arguments defined below, can be specified upon calling the :py:meth:`print <thermolib.thermodynamics.bias.BiasPotential2D.print>` routine and will be parsed to the individual ``print`` routines of each bias. Hence, you can append the unit definitions of each bias in the list as keyword arguments of the current routine.
        '''
        string = 'MultipleBias:\n'
        for bias in self.biasses:
            string += '  %s\n' %(bias.print(**par_units))
        return string
    
    def __call__(self, q1, q2):
        '''
            Compute the value of the bias potential for the given values of CV1 and CV2

            :param q1: CV1 value at which the bias potential will be evaluated
            :type q1: float

            :param q2: CV2 value at which the bias potential will be evaluated
            :type q2: float
        '''
        result = 0.0
        for idx,bias in enumerate(self.biasses):
            if isinstance(bias, BiasPotential1D):
                if self.additional_bias_dimension.lower() in ['q1', 'cv1']:
                    result += self.coeffs[idx]*bias(q1)
                elif self.additional_bias_dimension.lower() in ['q2', 'cv2']:
                    result += self.coeffs[idx]*bias(q2)
                else:
                    raise ValueError('additional bias_dimension is either cv1(/q1) or cv2(/q2) but you specified %s' % self.additional_bias_dimension)
            elif isinstance(bias, BiasPotential2D):
                result += self.coeffs[idx]*bias(q1,q2)
            else:
                raise ValueError('Additional bias is not recognized correctly')
        return result
