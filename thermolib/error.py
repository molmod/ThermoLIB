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

import numpy as np, numpy.ma as ma
np.seterr(divide='ignore', invalid='ignore')

from molmod.units import *

import matplotlib.pyplot as pp

from .tools import format_scientific, multivariate_normal
from .flatten import DummyFlattener

import time

__all__ = [
    'SampleDistribution', 'GaussianDistribution', 'LogGaussianDistribution', 
    'MultiDistribution', 'MultiGaussianDistribution', 'MultiLogGaussianDistribution', 
    'ErrorArray', 'Propagator'
]


ncycles_default = 50


class Distribution(object):
    '''
        Abstract parent class for probability distributions used for error propagation. Most of the routines here are implemented in the child classes.
    '''
    def __init__(self, shape, flattener=None):
        '''
            :param shape: the dimensions of the property for which the distribution is stored
            :type shape: list of integers

            :param flattener: Only required for properties represented by nD-arrays with n>=2. The flattener encodes how to flatten these multidimensional properties stored in nD-arrays into a longer 1D-array for easiear error propagation. This flattener also allows to do the inverse transformation, i.e. deflatten the flattened 1D-array back to its original nD-array format.
            :type flattener: :py:class:`Flattener <thermolib.flatten.Flattener>`, optional, default=None
        '''
        self.shape = shape
        self.flattener = flattener

    def shift(self, ref):
        raise NotImplementedError

    def set_ref(self, index=None, value=None):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError
    
    def mean(self):
        raise NotImplementedError

    def std(self):
        raise NotImplementedError

    def nsigma_conf_int(self, nsigma):
        raise NotImplementedError
    
    @classmethod
    def from_samples(cls, samples):
        raise NotImplementedError
    
    def print(self, fmt='%.3f', unit='au', do_scientific=False, nsigma=2):
        '''
            Routine to print the statistical properties of a scalar quantity.

            :param fmt: python string formatting for how to format floats (such as the mean and error). This argument is ignored if ``do_scientific`` is set tot True.
            :type fmt: str, optional, default='%.3f'

            :param unit: unit in which to print the current quantity.
            :type unit: str, optional, default='au'

            :param do_scientific: use scientific formatting of floats
            :type do_scientific: bool, optional, default=False

            :param nsigma: print error bars as :math:`n\\sigma` error bars. Hence, setting ``nsigma=2`` will return error bars that are twice the standard deviation resulting from the error distribution.
            :type nsigma: int, optional, default=2
        '''
        assert len(self.shape)==1 and self.shape[0]==1, "Can't print information for multiple statistical variables"
        lower, upper = self.nsigma_conf_int(nsigma)
        lower /= parse_unit(unit)
        upper /= parse_unit(unit)
        mean = self.mean()/parse_unit(unit)
        if do_scientific:
            lower = format_scientific(lower)
            upper = format_scientific(upper)
            mean = format_scientific(mean)
        else:
            lower = fmt % lower
            upper = fmt % upper
            mean  = fmt % mean
        return '%s <= %s <= %s %s' %(lower, mean, upper, unit)


class SampleDistribution(Distribution):
    '''
        Define the distribution by explicitly supplying it with a population of samples. All statistical properties will be derived from this population.
    '''
    def __init__(self, samples):
        '''
            :param samples: the samples that define the population
            :type samples: np.ndarray
        '''
        self.samples = samples.copy()
        if len(samples.shape)==1:
            shape = (1,)
        else:
            shape = samples.shape[1:]
        Distribution.__init__(self, shape)

    def shift(self, ref):
        '''
            Shift all samples in the population with the given ref.

            :param ref: reference with which each sample will be shifted
            :type ref: type consistent with a sample from the population
        '''
        self.samples += ref
    
    def set_ref(self, index=None, value=None):
        '''
            Set reference of population samples by shifting all population samples over an amount equal to -value.

            :param index: invalid keyword for SampleDistribution. Only valid value is None.

            :param value: old population value that will become new reference (i.e. become zero)
            :type value: type consistent with population samples, optional, default=None

            :raises AssertionError: if index is not None
            :raises AssertionError: if value is None
        '''
        assert index is None, 'SampleDistribution.set_ref can only be called with value keyword'
        assert value is not None, 'SampleDistribution.set_ref can only be called with value keyword'
        self.shift(-value)

    @classmethod
    def from_samples(cls, samples):
        '''
            Defines the current SampleDistribution based on a population defined by the given samples. A trivial routine for the current class, but implemented for completeness of Distribution parent class.

            :param samples: the samples that define the population
            :type samples: np.ndarray
        '''
        return cls(samples)
    
    def mean(self):
        '''
            Return the distribution mean

            :return: mean of the distibution
            :rtype: np.ndarray
        '''
        samples_ma = ma.masked_invalid(self.samples)
        return samples_ma.mean(axis=-1)
    
    def std(self):
        '''
            Return the distribution standard deviation

            :return: standard deviation of the distibution
            :rtype: np.ndarray
        '''
        samples_ma = ma.masked_invalid(self.samples)
        return samples_ma.std(axis=-1)
    
    def sample(self):
        '''
            Returns a random sample from the population given at initialization

            :return: a random sample from the population
            :rtype: np.ndarray or float
        '''
        return self.samples[np.random.random_integers(low=0, high=self.shape[0])]


class GaussianDistribution(Distribution):
    '''
        Implementation of a Gaussian distribution defined by its mean and standard deviation.
    '''
    def __init__(self, means, stds):
        '''
            :param means: mean (or means if multidimensional) of the Gaussian distribution
            :type means: int, float or np.ndarray
            
            :param stds: standard deviation (or standard deviations if multidimensional) of the Gaussian distribution
            :type stds: int, float or np.ndarray
            
            :raises ValueError: if means is not an integer, float or np.ndarray
            :raises AssertionError: if means and stds do not have same shape
        '''
        assert type(means)==type(stds), 'Means and stds in GaussianDistribution should be of same type. Got %s and %s' %(str(type(means)), str(type(stds)))
        if isinstance(means, np.ndarray):
            self.means = means.copy()
            self.stds = stds.copy()
            shape = means.shape
            assert stds.shape==shape, 'Means and stds in GaussianDistribution should be of same shape'
        elif isinstance(means, int) or isinstance(means, float):
            self.means = means
            self.stds = stds
            shape = (1,)
        else:
            raise ValueError('Means in GaussianDistribution should be int, float or np.ndarray, got %s' %(type(means).__name__))
        Distribution.__init__(self, shape)

    def copy(self):
        '''
            Return a hard copy of the Gaussian distribution
        '''
        return GaussianDistribution(self.means, self.stds)
    
    def shift(self, ref):
        '''
            Shift the distribution means with the given ref.

            :param ref: reference with which means will be shifted
            :type ref: type consistent with means
        '''
        self.means += ref
    
    def set_ref(self, index=None, value=None):
        '''
            Set reference of distribution means by shifting them over an amount equal to -value or -self.means[index].

            :param index: shift distribution means over an amount equal -self.means[index]. Ignored if set to None.
            :type index: int, optional, default=None

            :param value: shift distribution means over an amount equal to -value. Ignored if index is not None.
            :type value: type consistent with population samples, optional, default=None

            :raises ValueError: if both index and value are set to None
        '''
        if index is not None:
            self.means -= self.means[index]
        elif value is not None:
            self.means -= value
        else:
            raise ValueError('In GaussianDistribution.set_ref either index or value keyword argument should be defined')
    
    def sample(self, nsamples=None):
        '''
            Returns a number of random samples from the population defined by the Gaussian distribution

            :param nsamples: the number of samples returned. If None, a single sample will be returned and the shape of the return value will be identical to self.means and self.stds. If not None, the return value will have an additional dimension with size given by nsamples.
            :type nsamples: int or None, optional, default=None

            :return: a (set of) random sample(s) from the population
            :rtype: np.ndarray
        '''
        if nsamples is None:
            return np.random.normal(loc=self.means, scale=self.stds)
        else:
            result = np.stack([np.random.normal(loc=self.means, scale=self.stds) for i in range(nsamples)], axis=len(self.means.shape))
            return result
    
    def mean(self):
        '''
            Return the distribution mean

            :return: mean of the distibution
            :rtype: np.ndarray
        '''
        return self.means
    
    def std(self):
        '''
            Return the distribution standard deviation

            :return: standard deviation of the distibution
            :rtype: np.ndarray
        '''
        return self.stds

    def nsigma_conf_int(self, nsigma):
        '''
            Compute the n-sigma confidence interval, i.e. :math:`\\mu \\pm n\\cdot\\sigma`
        
            :param nsigma: value of n in the above formula
            :type nsigma: int, optional, default=2
        '''
        lower = self.means - nsigma*self.stds
        upper = self.means + nsigma*self.stds
        return lower, upper
    
    @classmethod
    def from_samples(cls, samples):
        '''
            Defines a GaussianDistribution based on a population defined by the given samples. The routine will derive the mean and std from the population and use that to define the Gaussian Distribution.

            :param samples: the samples that define the population
            :type samples: np.ndarray
        '''
        samples_ma = ma.masked_invalid(samples)
        means = samples_ma.mean(axis=-1)
        stds = samples_ma.std(axis=-1, ddof=1)
        return cls(means.filled(np.nan), stds.filled(np.nan))
    
    @classmethod
    def log_from_loggaussian(cls, logdist, shift=0.0, scale=1.0):
        '''
            Define the Gaussian distribution of a variable :math:`Y=log(X)` in which :math:`X` has a given LogGaussian distribution, potential after imposing a shift and rescaling.

            :param logdist: the LogGaussian distribution of variable X in the expression above.
            :type logdist: :py:class:`LogGaussianDistribution <thermolib.error.LogGaussianDistribution>`

            :param shift: shift to be applied to the mean upon conversion
            :type shift: float, optional, default=0.0

            :param scale: rescaling to be applied to the mean and std upon conversion
            :type scale: float, optional, default=1.0

            :raises AssertionError: if the given logdist is not an instance of :py:class:`LogGaussianDistribution <thermolib.error.LogGaussianDistribution>`
        '''
        assert isinstance(logdist, LogGaussianDistribution), 'Can only make Gaussian distribution from logarithm of LogGaussianDistribution'
        return cls(scale*logdist.lmeans+shift, abs(scale)*logdist.lstds)
    
    def print(self, fmt='%.3f', unit='au', do_scientific=False, nsigma=2):
        '''
            Routine to print the statistical properties of a scalar quantity.

            :param fmt: python string formatting for how to format floats (such as the mean and error). This argument is ignored if ``do_scientific`` is set tot True.
            :type fmt: str, optional, default='%.3f'

            :param unit: unit in which to print the current quantity.
            :type unit: str, optional, default='au'

            :param do_scientific: use scientific formatting of floats
            :type do_scientific: bool, optional, default=False

            :param nsigma: print error bars as :math:`n\\sigma` error bars. Hence, setting ``nsigma=2`` will return error bars that are twice the standard deviation resulting from the error distribution.
            :type nsigma: int, optional, default=2
        '''
        lower, upper = self.nsigma_conf_int(nsigma)
        error = (upper-lower)/2.0
        error /= parse_unit(unit)
        mean = self.mean()/parse_unit(unit)
        if do_scientific:
            error = format_scientific(error)
            mean = format_scientific(mean)
        else:
            error = fmt % error
            mean  = fmt % mean
        return '%s +- %s %s' %(mean, error, unit)


class LogGaussianDistribution(Distribution):
    '''
        Implementation of a Log-Gaussian distribution, i.e. variable X is Log-Gaussian with mean lmu and std lsigma if variable log(X) is Gaussian distributed with mean lmu and std lsigma.
    '''
    def __init__(self, lmeans, lstds):
        '''
            :param lmeans: mean (or means if multidimensional) of the corresponding Gaussian distribution of log(X)
            :type means: int, float or np.ndarray
            
            :param lstds: standard deviation (or standard deviations if multidimensional) of the corresponding Gaussian distribution of log(X)
            :type stds: int, float or np.ndarray
            
            :raises ValueError: if lmeans is not an integer, float or np.ndarray
            :raises AssertionError: if lmeans and lstds do not have same shape
        '''
        if isinstance(lmeans, np.ndarray):
            self.lmeans = lmeans.copy()
        else:
            self.lmeans = lmeans
        if isinstance(lstds, np.ndarray):
            self.lstds = lstds.copy()
        else:
            self.lstds = lstds
        assert type(lmeans)==type(lstds), 'lmeans and lstds in GaussianDistribution should be of same type'
        if isinstance(lmeans, np.ndarray):
            shape = lmeans.shape
            assert lstds.shape==shape, 'lmeans and stds in GaussianDistribution should be of same length'
        elif isinstance(lmeans, int) or isinstance(lmeans, float):
            shape = (1,)
        else:
            raise ValueError('lmeans should be int, float or np.ndarray, got %s' %(type(lmeans).__name__))
        Distribution.__init__(self, shape)

    def shift(self, ref):
        '''
            Shift the distribution lmeans (NOT THE DISTRIBUTION MEAN!!) with the given ref.

            :param ref: reference with which lmeans will be shifted
            :type ref: type consistent with lmeans
        '''
        self.lmeans += ref

    def set_ref(self, index=None, value=None):
        '''
            Set reference of distribution lmeans (NOT THE DISTRIBUTION MEANS!!) by shifting them over an amount equal to -value or -self.lmeans[index].

            :param index: shift distribution lmeans over an amount equal -self.lmeans[index]. Ignored if set to None.
            :type index: int, optional, default=None

            :param value: shift distribution lmeans over an amount equal to -value. Ignored if index is not None.
            :type value: type consistent with population samples, optional, default=None

            :raises ValueError: if both index and value are set to None
        '''
        if index is not None:
            self.lmeans -= self.lmeans[index]
        elif value is not None:
            self.lmeans -= value
        else:
            raise ValueError('In LogGaussianDistribution.set_ref either index or value keyword argument should be defined')

    def mean(self):
        '''
            Return the distribution mean

            :return: mean of the distibution
            :rtype: np.ndarray
        '''
        return np.exp(self.lmeans + 0.5*self.lstds**2)

    def std(self):
        '''
            Return the distribution standard deviation

            :return: standard deviation of the distibution
            :rtype: np.ndarray
        '''
        return np.sqrt(np.exp(self.lstds**2)-1)*self.mean()

    def copy(self):
        '''
            Return a hard copy of the LogGaussian distribution
        '''
        return LogGaussianDistribution(self.lmeans, self.lstds)
    
    def sample(self, nsamples=None):
        '''
            Returns a number of random samples from the population defined by the distribution

            :param nsamples: the number of samples returned. If None, a single sample will be returned and the shape of the return value will be identical to self.means and self.stds. If not None, the return value will have an additional dimension with size given by nsamples.
            :type nsamples: int or None, optional, default=None

            :return: a (set of) random sample(s) from the population
            :rtype: np.ndarray
        '''
        if nsamples is None:
            return np.exp(np.random.normal(loc=self.lmeans, scale=self.lstds))
        else:
            result = np.stack([np.exp(np.random.normal(loc=self.lmeans, scale=self.lstds)) for i in range(nsamples)], axis=len(self.shape))
            return result
    
    def nsigma_conf_int(self, nsigma):
        '''
            Compute the n-sigma confidence interval, i.e. :math:`\\exp\\left(\\mu \\pm n\\cdot\\sigma\\right)`
        
            :param nsigma: value of n in the above formula
            :type nsigma: int, optional, default=2
        '''
        lower = self.lmeans - nsigma*self.lstds
        upper = self.lmeans + nsigma*self.lstds
        return np.exp(lower), np.exp(upper)
    
    @classmethod
    def from_samples(cls, samples):
        '''
            Defines a LogGaussianDistribution based on a population defined by the given samples. The routine will derive lmean and lstd from the log of the given population samples and use that to define the LogGaussian Distribution.

            :param samples: the samples that define the population
            :type samples: np.ndarray
        '''
        lsamples = ma.masked_invalid(np.log(samples))
        lmeans = lsamples.mean(axis=-1)
        lstds = lsamples.std(axis=-1, ddof=1)
        return cls(lmeans, lstds)
    
    @classmethod
    def exp_from_gaussian(cls, gaussdist, shift=0.0, scale=1.0):
        '''
            Define the LogGaussian distribution of a variable :math:`Y=exp(X)` in which :math:`X` has a given Gaussian distribution, potential after imposing a shift and rescaling.

            :param gaussdist: the Gaussian distribution of variable X in the expression above.
            :type gaussdist: :py:class:`GaussianDistribution <thermolib.error.GaussianDistribution>`

            :param shift: shift to be applied to the mean upon conversion
            :type shift: float, optional, default=0.0

            :param scale: rescaling to be applied to the mean and std upon conversion
            :type scale: float, optional, default=1.0

            :raises AssertionError: if the given gaussdist is not an instance of :py:class:`GaussianDistribution <thermolib.error.GaussianDistribution>`
        '''
        assert isinstance(gaussdist, GaussianDistribution), 'Can only make LogGaussian distribution from exponent of GaussianDistribution'
        return cls(scale*gaussdist.means+shift, abs(scale)*gaussdist.stds)
    

class FunctionDistribution(Distribution):
    '''
        Class to implement the distribution of a variable Y=f(X) with a known distribution for X
    '''
    def __init__(self, fun, dist):
        '''
            :param fun: function defining variable Y in the expression above.
            :type fun: callable

            :param dist: the distribution of variable X in the expression above.
            :type dist: instance of child of :py:class:`Distribution <thermolib.error.Distribution>`
        '''
        self.fun = fun
        self.dist = dist
        Distribution.__init__(self, dist.shape)

    #def crop(self, indexes):
    #    '''
    #        Routine only applicable if self.dist is an instance of :py:class:`ErrorArray <thermolib.error.ErrorArray>`. Will crop the array of errors to the given indices.
    #
    #        :param indexes: indices to which the error array needs to be cropped
    #        :type indexes: list, tuple, np.ndarray of integers
    #    '''
    #    assert isinstance(self, ErrorArray)
    #    self.dist.crop(indexes)
    
    def mean(self, nsamples=50):
        '''
            Return the distribution mean through numerical estimation

            :param nsamples: number of samples used for numerical estimation
            :type nsamples: int, optional, default=50

            :return: mean of the distibution
            :rtype: np.ndarray
        '''
        return np.array([self.fun(self.dist.sample()) for i in range(nsamples)]).mean(axis=0)
    
    def std(self, nsamples=50):
        '''
            Return the distribution standard deviation through numerical estimation
            
            :param nsamples: number of samples used for numerical estimation
            :type nsamples: int, optional, default=50

            :return: standard deviation of the distibution
            :rtype: np.ndarray
        '''
        return np.array([self.fun(self.dist.sample()) for i in range(nsamples)]).std(axis=0)

    def sample(self, nsamples=None):
        '''
            Returns a number of random samples from the population defined by the distribution

            :param nsamples: the number of samples returned. If None, a single sample will be returned and the shape of the return value will be identical to self.means and self.stds. If not None, the return value will have an additional dimension with size given by nsamples.
            :type nsamples: int or None, optional, default=None

            :return: a (set of) random sample(s) from the population
            :rtype: np.ndarray
        '''
        return self.fun(self.dist.sample(nsamples=nsamples))
    
    def nsigma_conf_int(self, nsigma):
        '''
            Compute the n-sigma confidence interval, i.e. :math:`[f(l),f(u)]` with ``l,u=self.dist.nsigma_conf_int(nsigma)``
        
            :param nsigma: value of n in the above formula
            :type nsigma: int, optional, default=2
        '''
        lower, upper = self.dist.nsigma_conf_int(nsigma)
        fun_lower, fun_upper = self.fun(lower), self.fun(upper)
        if fun_lower<=fun_upper:
            return fun_lower, fun_upper
        else:
            return fun_upper, fun_lower


class MultiDistribution(object):
    '''
        Abstract parent class for probability distributions of multidimensional stochastic properties used for error propagation. This class will explicitly account for correlation across the dimensions, as such it can be used to account for the correlation between the free energy at two different points in a FEP. Most of the routines here are implemented in the child classes.
    '''
    def __init__(self, shape, flattener=DummyFlattener()):
        '''
            :param shape: the dimensions of the property for which the distribution is stored
            :type shape: list of integers

            :param flattener: The flattener encodes how to flatten the multidimensional properties stored in nD-arrays into a longer 1D-array for easiear error propagation. This flattener also allows to do the inverse transformation, i.e. deflatten the flattened 1D-array back to its original nD-array format.
            :type flattener: :py:class:`Flattener <thermolib.flatten.Flattener>`, optional, default=None
        '''
        self.shape = shape
        self.flattener = flattener

    def mean(self, unflatten=True):
        raise NotImplementedError
    
    def std(self, unflatten=True):
        raise NotImplementedError
    
    def cov(self):
        raise NotImplementedError
    
    def sample(self, nsamples=None, unflatten=True):
        raise NotImplementedError

    def nsigma_conf_int(self, nsigma=2, unflatten=True):
        return NotImplementedError

    def corr(self, unflatten=True):
        '''
            Routine that will convert the covariance matrix computed with self.cov to a correlation matrix according to the formula

            .. math:: 

                Cor[i,j] &= \\frac{Cov[i,j]}{\\sigma_i \\sigma_j}
            
            in which :math:`\\sigma_i` is the standard deviation of the i-th quantity in the multidimensional stochastic property.

            :param unflatten: If True, return the correlation matrix in a shape equal to 'the square' of in the original dimensional shape. If False, return the correlation matrix as a flattened 2D array.
            :type unflatten: bool, optional, default=True
        '''
        #get flattened correlation matrix
        corr = self.cov(unflatten=False)
        std = self.std(unflatten=False)
        for i,stdi in enumerate(std):
            for j,stdj in enumerate(std):
                corr[i,j] /= (stdi*stdj)
        #possibly deflatten the correlation matrix before returning it
        if unflatten:
            return self.flattener.unflatten_matrix(self.corr)
        else:
            return corr.copy()
    
    def set_ref(self, index=None, value=None):
        raise NotImplementedError
    
    def shift(self, ref):
        raise NotImplementedError

    def from_samples(self, samples, flattener=DummyFlattener(), flattened=True):
        raise NotImplementedError


class MultiGaussianDistribution(MultiDistribution):
    '''
        Implementation of a multivariate normal distribution for a given vector of means and a given covariance matrix
    '''
    def __init__(self, means, covariance, flattener=DummyFlattener()):
        '''
            :param means: means of the multivariate normal distribution
            :type means: np.ndarray

            :param covariance: covariance matrix of the multivariate normal distribution
            :type covariance: np.ndarray with a dimension equal tot the square of means, i.e. if means.shape=(N,) then covariance.shape=(N,N) and if means.shape=(K,L) then covariance.shape=(K,L,K,L).

            :param flattener: The flattener encodes how to flatten the multidimensional properties stored in nD-arrays into a longer 1D-array for easiear error propagation. This flattener also allows to do the inverse transformation, i.e. deflatten the flattened 1D-array back to its original nD-array format. Should be specified if len(means.shape)=1 because it represents a flattened 2D array (which also implies len(covariance.shape)=2 because it represents a flattened 4D array).
            :type flattener: :py:class:`Flattener <thermolib.flatten.Flattener>`, optional, default=DummyFlattener() indicating no flattening
        '''
        assert isinstance(means, np.ndarray), 'means should be array, detected %s' %(str(type(means)))
        assert isinstance(covariance, np.ndarray), 'covariance matrix should be array, detected %s' %(str(type(covariance)))
        assert len(means.shape)==1, 'means should be 1D array, but detected array shape %s' %(str(means.shape))
        assert len(covariance.shape)==2, 'covariance matrix should be 2D array, but detected array shape %s' %(str(covariance.shape))
        assert covariance.shape[0]==means.shape[0], 'Inconsistent shapes, means has shape %s while covariance has shape %s' %(str(means.shape),str(covariance.shape))
        assert covariance.shape[1]==means.shape[0], 'Inconsistent shapes, means has shape %s while covariance has shape %s' %(str(means.shape),str(covariance.shape))
        self.means = means.copy()
        self.covariance = covariance.copy()
        self.stds = np.sqrt(covariance.diagonal())
        MultiDistribution.__init__(self, means.shape, flattener=flattener)

    def copy(self):
        '''
            Return a hard copy of the MultiGaussian distribution
        '''
        return MultiGaussianDistribution(self.means, self.covariance, flattener=self.flattener)

    def mean(self, unflatten=True):
        '''
            Return the distribution mean

            :param unflatten: If True, return the distribution mean in the original dimensional shape. If False, return the distribution mean as a flattened 1D array.
            :type unflatten: bool, optional, default=True

            :return: mean of the distibution
            :rtype: np.ndarray
        '''
        if unflatten:
            return self.flattener.unflatten_array(self.means)
        else:
            return self.means.copy()

    def std(self, unflatten=True):
        '''
            Return the distribution standard deviation (std)

            :param unflatten: If True, return the distribution std in the original dimensional shape. If False, return the distribution std as a flattened 1D array.
            :type unflatten: bool, optional, default=True

            :return: standard deviation of the distibution
            :rtype: np.ndarray
        '''
        if unflatten:
            return self.flattener.unflatten_array(self.stds)
        else:
            return self.stds.copy()
    
    def cov(self, unflatten=True):
        '''
            Return the covariance matrix of the MultiGaussian distribution.

            :param unflatten: If True, return the covariance matrix in a shape equal to 'the square' of in the original dimensional shape. If False, return the covariance matrix as a flattened 2D array.
            :type unflatten: bool, optional, default=True
        '''
        if unflatten:
            return self.flattener.unflatten_matrix(self.covariance)
        else:
            return self.covariance.copy()

    def sample(self, nsamples=None, unflatten=True):
        '''
            Returns a number of random samples from the population defined by the distribution

            :param nsamples: the number of samples returned. If None, a single sample will be returned with a shape depending on whether or not it is flattened (see parameter unflatten). If not None, the return value will have an additional dimension with size given by nsamples.
            :type nsamples: int or None, optional, default=None

            :param unflatten: If True, return each distribution sample in the original dimensional shape. If False, return each sample as a flattened 1D array.
            :type unflatten: bool, optional, default=True

            :return: a (set of) random sample(s) from the population
            :rtype: np.ndarray whose dimension depend on the nsamples and unflatten keyword values. If nsamples is None, a single sample is returned whose shape depends on whether or not it is flattened (see parameter unflatten). If nsamples is not None, the return value will have an additinal dimension with size given by nsamples.
        '''
        samples_flattened = multivariate_normal(self.means, self.covariance, size=nsamples).T
        if unflatten and not isinstance(self.flattener, DummyFlattener):
            if nsamples is not None:
                samples_unflattened = np.zeros([self.flattener.dim1, self.flattener.dim2, nsamples])
                for isample in range(nsamples):
                    samples_unflattened[...,isample] = self.flattener.unflatten_array(samples_flattened[:,isample])
            else:
                samples_unflattened = self.flattener.unflatten_array(samples_flattened)
            return samples_unflattened
        else:
            return samples_flattened
    
    def nsigma_conf_int(self, nsigma=2, unflatten=True):
        '''
            Compute the n-sigma confidence interval, i.e. :math:`\\mu \\pm n\\cdot\\sigma`
        
            :param nsigma: value of n in the above formula
            :type nsigma: int, optional, default=2

            :param unflatten: If True, return the interval in the original dimensional shape. If False, return as a flattened 1D array.
            :type unflatten: bool, optional, default=True
        '''
        lower = self.mean(unflatten=unflatten) - nsigma*self.std(unflatten=unflatten)
        upper = self.mean(unflatten=unflatten) + nsigma*self.std(unflatten=unflatten)
        return lower, upper

    def set_ref(self, index=None, value=None):
        '''
            Set reference of distribution means by shifting them over an amount equal to -value or -self.means[index].

            :param index: shift distribution means over an amount equal -self.means[index]. Ignored if set to None.
            :type index: tuple|list, optional, default=None

            :param value: shift distribution means over an amount equal to -value. Ignored if index is not None.
            :type value: type consistent with population samples, optional, default=None

            :raises ValueError: if both index and value are set to None
        '''
        if index is not None:
            if self.flattener is not None:
                if isinstance(index, tuple):
                    index = self.flattener.flatten_index(*index)
            self.means -= self.means[index]
        elif value is not None:
            self.means -= value
        else:
            raise ValueError('In MultiGaussianDistribution.set_ref either index or value keyword argument should be defined')

    @classmethod
    def from_samples(cls, samples, flattener=DummyFlattener(), flattened=True):
        """ 
            Defines a MultiGaussianDistribution based on a population defined by the given samples. The routine will derive the mean and std from the population and use that to define the Gaussian Distribution.

            :param samples: the samples that define the population
            :type samples: np.ndarray

            :param flattened: 
                
                - If flattened is True, the given samples should be a 2D array in which a row represents all observations of a single variable and a column represents a single observation of all variables.
                - If flattened is False, then the the samples array is assumed to be 3 dimensional for which the first two dimensions represent a 2D index and the third index is the sample index. Therefore, the flattener will first be applied to flatten the 2D index into a 1D index and hence convert the samples to a 2D array of the same shape as required if the flattened argument is True.
            
            :type flattened: bool, optional, default=False

            :raises AssertionError: if the dimensions of samples array are invalid
        """
        if not flattened:
            samples_unflattened = samples.copy()
            samples = np.zeros([flattener.dim12, samples_unflattened.shape[-1]])
            for isample in range(samples_unflattened.shape[-1]):
                samples[:,isample] = flattener.flatten_array(samples_unflattened[...,isample])
        assert len(samples.shape)==2, 'Samples should be a 2 dimensional array, got an array with %i dimensions' %(len(samples.shape))
        #mask invalid samples
        samples_ma = ma.masked_invalid(samples)
        #compute means
        means = samples_ma.mean(axis=1).filled(np.nan)
        #compute covariance matrix (and fill rows/columns comming from invalid samples with np.nan)
        cov = ma.cov(samples_ma).filled(np.nan)
        assert cov.shape[0]==len(means)
        return cls(means, cov, flattener)

    @classmethod
    def log_from_loggaussian(cls, logdist, shift=0.0, scale=1.0):
        '''
            Define the MultiGaussian distribution of a variable :math:`Y=log(X)` in which :math:`X` has a given MultiLogGaussian distribution, potential after imposing a shift and rescaling.

            :param logdist: the MultiLogGaussian distribution of variable X in the expression above.
            :type logdist: :py:class:`MultiLogGaussianDistribution <thermolib.error.MultiLogGaussianDistribution>`

            :param shift: shift to be applied to the mean upon conversion
            :type shift: float, optional, default=0.0

            :param scale: rescaling to be applied to the mean and std upon conversion
            :type scale: float, optional, default=1.0

            :raises AssertionError: if logdist is not instance of :py:class:`MultiLogGaussianDistribution <thermolib.error.MultiLogGaussianDistribution>`
        '''
        assert isinstance(logdist, MultiLogGaussianDistribution), 'Can only make multivariate Gaussian distribution from logarithm of Multivariate LogGaussianDistribution'
        return cls(scale*logdist.lmeans+shift, (scale**2)*logdist.lcovariance, logdist.flattener)

    def plot_corr_matrix(self, fn=None, fig_size_inches=[8,8], cmap='bwr', logscale=False, cvs=None, nticks=10, decimals=1):
        '''
            gMake a plot of the correlation matrix of the current multivariate Gaussian distribution

            :param fn: file name to which figure will be saved. If None, figure will note be written to file.
            :type fn: _type_, optional, default=None

            :param fig_size_inches: [x,y]-dimensions of the figure in inches
            :type fig_size_inches: list, optional, default=[8,8]

            :param cmap: color map to be used, see `matplotlib documentation <https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html>`_ to see possibilities.
            :type cmap: str, optional, default='bwr'

            :param logscale: plot correlation in logarithmic scale
            :type logscale: bool, optional, default=False

            :param cvs: set x and y-tick labels to CV values given in cvs
            :type cvs: np.ndarray, optional, default=None

            :param nticks: number of ticks on x and y-axis. Only used when parameter cvs is given.
            :type nticks: int, optional, default=10

            :param decimals: number of decimals to be shown in CV tick labels. Only used when parameter cvs is given.
            :type decimals: int, optional, default=1
        '''
        data = self.corr(unflatten=False)
        vmin, vmax = -1,1
        if logscale:
            def special_log(x, a=10):
                result = np.zeros(x.shape)
                result[x>np.exp(-a/100)] = 100.0
                result[x<-np.exp(-a/100)] = -100.0
                mask = (-np.exp(-a/100)<x) & (x<np.exp(-a/100)) & (np.abs(x)>0)
                result[mask] = -np.sign(x[mask])*a/np.log(np.abs(x[mask]))
                return result
            data = special_log(data)
            vmin, vmax = -100,100
        pp.clf()
        surf = pp.matshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        pp.colorbar(surf)
        if cvs is not None:
            #set xtick and yticks positions and labels to CV values
            delta = (len(cvs)-1)//nticks
            positions = np.arange(0,len(cvs)-1,delta)
            labels = np.array([np.round(cv*10**decimals)/10**decimals for cv in cvs[positions]])
            pp.xticks(positions, labels)
            pp.yticks(positions, labels)
        fig = pp.gcf()
        fig.set_size_inches(fig_size_inches)
        if fn is not None:
            pp.savefig(fn)
        else:
            pp.show()


class MultiLogGaussianDistribution(MultiDistribution):
    '''
        Implementation of a multivariate log-normal distribution for a given vector of means and a given covariance matrix
    '''
    def __init__(self, lmeans, lcovariance, flattener=DummyFlattener()):
        '''
            :param means: lmeans of the multivariate log-normal distribution, i.e. means of the underlying normal distribution of log(X)
            :type means: np.ndarray

            :param covariance: lcovariance matrix of the multivariate log-normal distribution, i.e. covariance matrix of the underlying normal distribution of log(X)
            :type covariance: np.ndarray with a dimension equal tot the square of means, i.e. if means.shape=(N,) then covariance.shape=(N,N) and if means.shape=(K,L) then covariance.shape=(K,L,K,L).

            :param flattener: The flattener encodes how to flatten the multidimensional properties stored in nD-arrays into a longer 1D-array for easiear error propagation. This flattener also allows to do the inverse transformation, i.e. deflatten the flattened 1D-array back to its original nD-array format. Should be specified if len(means.shape)=1 because it represents a flattened 2D array (which also implies len(covariance.shape)=2 because it represents a flattened 4D array).
            :type flattener: :py:class:`Flattener <thermolib.flatten.Flattener>`, optional, default=DummyFlattener() indicating no flattening
        '''
        assert isinstance(lmeans, np.ndarray), 'lmeans should be array, detected %s' %(str(type(lmeans)))
        assert isinstance(lcovariance, np.ndarray), 'lcovariance matrix should be array, detected %s' %(str(type(lcovariance)))
        shape = lmeans.shape
        assert len(lmeans.shape)==1, 'lmeans should be 1D array, but detected array shape %s' %(str(lmeans.shape))
        assert len(lcovariance.shape)==2, 'lcovariance matrix should be 2D array, but detected array shape %s' %(str(lcovariance.shape))
        assert lcovariance.shape[0]==lmeans.shape[0], 'Inconsistent shapes, lmeans has shape %s while lcovariance has shape %s' %(str(lmeans.shape),str(lcovariance.shape))
        assert lcovariance.shape[1]==lmeans.shape[0], 'Inconsistent shapes, lmeans has shape %s while lcovariance has shape %s' %(str(lmeans.shape),str(lcovariance.shape))
        self.lmeans = lmeans.copy()
        self.lcovariance = lcovariance.copy()
        self.lstds = np.sqrt(lcovariance.diagonal())
        MultiDistribution.__init__(self, shape, flattener=flattener)

    def copy(self):
        '''
            Return a hard copy of the MultiLogGaussian distribution
        '''
        return MultiLogGaussianDistribution(self.lmeans, self.lcovariance, self.flattener)

    def mean(self, unflatten=True):
        '''
            Return the distribution mean

            :param unflatten: If True, return the distribution mean in the original dimensional shape. If False, return the distribution mean as a flattened 1D array.
            :type unflatten: bool, optional, default=True

            :return: mean of the distibution
            :rtype: np.ndarray
        '''
        means = np.exp(self.lmeans + 0.5*self.lstds**2)
        if unflatten:
            return self.flattener.unflatten_array(means)
        else:
            return means

    def std(self, unflatten=True):
        '''
            Return the distribution standard deviation (std)

            :param unflatten: If True, return the distribution std in the original dimensional shape. If False, return the distribution std as a flattened 1D array.
            :type unflatten: bool, optional, default=True

            :return: standard deviation of the distibution
            :rtype: np.ndarray
        '''
        stds = np.sqrt(np.exp(self.lstds**2)-1)*self.mean(unflatten=False)
        if unflatten:
            return self.flattener.unflatten_array(stds)
        else:
            return stds
    
    def cov(self, unflatten=True):
        '''
            Compute and and return the covariance matrix of the MultiLogGaussian distribution.

            :param unflatten: If True, return the covariance matrix in a shape equal to 'the square' of in the original dimensional shape. If False, return the covariance matrix as a flattened 2D array.
            :type unflatten: bool, optional, default=True
        '''
        #first construct the flattened covariance matrix
        cov = np.zeros(self.lcovariance.shape)
        for i, mui in enumerate(self.mean(unflatten=False)):
            for j, muj in enumerate(self.mean(unflatten=False)):
                cov[i,j] = mui*muj*(np.exp(self.lcovariance[i,j])-1)
        #possibly deflatten before returning
        if unflatten:
            return self.flattener.unflatten_matrix(cov)
        else:
            return cov
    
    def sample(self, nsamples=None, unflatten=True):
        '''
            Returns a number of random samples from the population defined by the distribution

            :param nsamples: the number of samples returned. If None, a single sample will be returned with a shape depending on whether or not it is flattened (see parameter unflatten). If not None, the return value will have an additional dimension with size given by nsamples.
            :type nsamples: int or None, optional, default=None

            :param unflatten: If True, return each distribution sample in the original dimensional shape. If False, return each sample as a flattened 1D array.
            :type unflatten: bool, optional, default=True

            :return: a (set of) random sample(s) from the population
            :rtype: np.ndarray whose dimension depend on the nsamples and unflatten keyword values. If nsamples is None, a single sample is returned whose shape depends on whether or not it is flattened (see parameter unflatten). If nsamples is not None, the return value will have an additinal dimension with size given by nsamples.
        '''
        samples_flattened = np.exp(multivariate_normal(self.lmeans, self.lcovariance, size=nsamples).T)
        if unflatten and not isinstance(self.flattener, DummyFlattener):
            if nsamples is not None:
                samples_unflattened = np.zeros([self.flattener.dim1, self.flattener.dim2, nsamples])
                for isample in range(nsamples):
                    samples_unflattened[...,isample] = self.flattener.unflatten_array(samples_flattened[:,isample])
            else:
                samples_unflattened = self.flattener.unflatten_array(samples_flattened)
            return samples_unflattened
        else:
            return samples_flattened
    
    def nsigma_conf_int(self, nsigma, unflatten=True):
        '''
            Compute the n-sigma confidence interval, i.e. :math:`\\exp\\left(\\mu \\pm n\\cdot\\sigma\\right)`
        
            :param nsigma: value of n in the above formula
            :type nsigma: int, optional, default=2

            :param unflatten: If True, return the interval in the original dimensional shape. If False, return as a flattened 1D array.
            :type unflatten: bool, optional, default=True
        '''
        #first compute the flattened lower and upper bounds
        lower = np.exp(self.lmeans - nsigma*self.lstds)
        upper = np.exp(self.lmeans + nsigma*self.lstds)
        #possibly deflatten before returning
        if unflatten:
            return self.flattener.unflatten_array(lower), self.flattener.unflatten_array(upper)
        else:
            return lower, upper
    
    @classmethod
    def from_samples(cls, samples, flattener=DummyFlattener(), flattened=True):
        """ 
            Defines a MultiLogGaussianDistribution based on a population defined by the given samples. The routine will derive the lmean and lcovariance from the population and use that to define the MultiLogGaussian Distribution.

            :param samples: the samples that define the population
            :type samples: np.ndarray

            :param flattened: 
                
                - If flattened is True, the given samples should be a 2D array in which a row represents all observations of a single variable and a column represents a single observation of all variables.
                - If flattened is False, then the the samples array is assumed to be 3 dimensional for which the first two dimensions represent a 2D index and the third index is the sample index. Therefore, the flattener will first be applied to flatten the 2D index into a 1D index and hence convert the samples to a 2D array of the same shape as required if the flattened argument is True.
            
            :type flattened: bool, optional, default=False

            :raises AssertionError: if the dimensions of samples array are invalid
        """
        if not flattened:
            samples_unflattened = samples.copy()
            samples = np.zeros([flattener.dim12, samples_unflattened.shape[-1]])
            for isample in range(samples_unflattened.shape[-1]):
                samples[:,isample] = flattener.flatten_array(samples_unflattened[...,isample])
        assert len(samples.shape)==2, 'Samples should be a 2 dimensional array, got an array with %i dimensions' %(len(samples.shape))
        #compute the log of samples and mask invalid values
        lsamples = ma.masked_invalid(np.log(samples))
        #compute means
        lmeans = lsamples.mean(axis=1)
        #construct covariance matrix (and fill columns/rows coming from invalid log samples with np.nan)
        lcov = ma.cov(lsamples).filled(np.nan)
        assert lcov.shape[0]==len(lmeans)
        return cls(lmeans, lcov, flattener)

    @classmethod
    def exp_from_gaussian(cls, gaussdist, shift=0.0, scale=1.0):
        '''
            Define the MultiLogGaussian distribution of a variable :math:`Y=exp(X)` in which :math:`X` has a given MultiGaussian distribution, potential after imposing a shift and rescaling.

            :param gaussdist: the Gaussian distribution of variable X in the expression above.
            :type gaussdist: :py:class:`GaussianDistribution <thermolib.error.GaussianDistribution>`

            :param shift: shift to be applied to the mean upon conversion
            :type shift: float, optional, default=0.0

            :param scale: rescaling to be applied to the mean and std upon conversion
            :type scale: float, optional, default=1.0

            :raises AssertionError: if the given gaussdist is not an instance of :py:class:`MultiGaussianDistribution <thermolib.error.MultiGaussianDistribution>`
        '''
        assert isinstance(gaussdist, MultiGaussianDistribution), 'Can only make MultiLogGaussian distribution from exponent of MultiGaussianDistribution'
        return cls(scale*gaussdist.means+shift, (scale**2)*gaussdist.covariance, gaussdist.flattener)

    def plot_corr_matrix(self, fn=None, fig_size_inches=[8,8], cmap='bwr', logscale=False, cvs=None, nticks=10, decimals=1):
        '''
            gMake a plot of the correlation matrix of the current multivariate Gaussian distribution

            :param fn: file name to which figure will be saved. If None, figure will note be written to file.
            :type fn: _type_, optional, default=None

            :param fig_size_inches: [x,y]-dimensions of the figure in inches
            :type fig_size_inches: list, optional, default=[8,8]

            :param cmap: color map to be used, see `matplotlib documentation <https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html>`_ to see possibilities.
            :type cmap: str, optional, default='bwr'

            :param logscale: plot correlation in logarithmic scale
            :type logscale: bool, optional, default=False

            :param cvs: set x and y-tick labels to CV values given in cvs
            :type cvs: np.ndarray, optional, default=None

            :param nticks: number of ticks on x and y-axis. Only used when parameter cvs is given.
            :type nticks: int, optional, default=10

            :param decimals: number of decimals to be shown in CV tick labels. Only used when parameter cvs is given.
            :type decimals: int, optional, default=1
        '''
        data = self.corr(unflatten=False)
        vmin, vmax = -1,1
        if logscale:
            def special_log(x, a=10):
                result = np.zeros(x.shape)
                result[x>np.exp(-a/100)] = 100.0
                result[x<-np.exp(-a/100)] = -100.0
                mask = (-np.exp(-a/100)<x) & (x<np.exp(-a/100)) & (np.abs(x)>0)
                result[mask] = -np.sign(x[mask])*a/np.log(np.abs(x[mask]))
                return result
            data = special_log(data)
            vmin, vmax = -100,100
        pp.clf()
        surf = pp.matshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        if cvs is not None:
            #set xtick and yticks positions and labels to CV values
            delta = (len(cvs)-1)//nticks
            positions = np.arange(0,len(cvs)-1,delta)
            labels = np.array([np.round(cv*10**decimals)/10**decimals for cv in cvs[positions]])
            pp.xticks(positions, labels)
            pp.yticks(positions, labels)
        pp.colorbar(surf)
        fig = pp.gcf()
        fig.set_size_inches(fig_size_inches)
        if fn is not None:
            pp.savefig(fn)
        else:
            pp.show()


class ErrorArray(object):
    '''
        A class to represent an array of independent error distributions. This class is used to store the distributiosn for errors on the conditional probabilities, i.e. for each value of cv in p(q|cv) there is an error distribution. The error distribution on p(q|cv1) is assumed uncorrelated with that of p(q|cv2), but we still want to store the distribution for the errors for each cv value in one object.
    '''
    def __init__(self, errors, axis=0):
        '''
            :param errors: list of error distirbutions to be stored
            :type errors: list of instances of :py:class:`Distribution <thermolib.error.Distribution>`

            :param axis: the axis index along which the errors are stacked. Only two valid choices: 0 and -1
            :type axis: int, optional, default=0
        '''
        assert axis==0 or axis==-1, 'For axis argument, only value 0 or -1 is supported'
        shape = None
        for error in errors:
            if shape is None: shape = error.shape
            else:
                assert shape==error.shape, 'Detected an error with shape %s and an error with shape %s, but all errors should be of same shape.' (str(shape), str(error.shape))
        self.errors = errors
        self.axis = axis

    #def crop(self, indexes):
    #    for error in self.errors:
    #        error.crop(indexes)

    def mean(self):
        '''
            Return the distribution mean for each element in the errors list

            :return: mean of the distibution
            :rtype: np.ndarray
        '''
        return np.stack([error.mean() for error in self.errors], axis=self.axis)
    
    def std(self):
        '''
            Return the distribution standard deviation for each element in the errors list

            :return: mean of the distibution
            :rtype: np.ndarray
        '''
        return np.stack([error.std() for error in self.errors], axis=self.axis)

    def nsigma_conf_int(self, nsigma):
        '''
            Compute the n-sigma confidence interval, i.e. :math:`\\mu \\pm n\\cdot\\sigma`, for each element in the errors list

            :return: mean of the distibution
            :rtype: np.ndarray
        '''
        lowers = []
        uppers = []
        for error in self.errors:
            lower, upper = error.nsigma_conf_int(nsigma)
            lowers.append(lower)
            uppers.append(upper)
        return np.stack(lowers, axis=self.axis), np.stack(uppers, axis=self.axis)

    def sample(self, nsamples=None):
        '''
            Returns a number of random samples taken according to the error distribution for each element in the error list

            :param nsamples: the number of samples returned. If None, a single sample will be returned and the shape of the return value will be identical to self.means and self.stds. If not None, the return value will have an additional dimension with size given by nsamples.
            :type nsamples: int or None, optional, default=None

            :return: multidimensional numpy array containing random samples for each element in self.errors. Shape of this array depends on nsamples, self.naxis and number of elements in self.errors.
            :rtype: np.ndarray
        '''
        if nsamples is not None:
            if self.axis==0:
                samples = np.zeros([len(self.errors),] + list(self.errors[0].shape) +[nsamples,])
                for i, error in enumerate(self.errors):
                    samples[i,...] = error.sample(nsamples=nsamples)
            elif self.axis==-1:
                samples = np.zeros(list(self.errors[0].shape) +[len(self.errors),nsamples])
                for i, error in enumerate(self.errors):
                    samples[...,i,:] = error.sample(nsamples=nsamples)
            else:
                raise ValueError('axis attribute should be 0 or -1 but found %s' %(str(self.axis)))
        else:
            samples = []
            for error in self.errors:
                samples.append(error.sample())
            samples = np.stack(samples, axis=self.axis)
        return samples


class Propagator(object):
    '''
        A class to propagate the error distribution on a set of arguments towards the error distribution on a given function of those arguments. This routine uses the ``sample`` routine of each of its distrubution arguments, meaning that the resulting error is stochastic (will not give the same repeated result).
    '''
    def __init__(self, ncycles=ncycles_default, target_distribution=GaussianDistribution, flattener=DummyFlattener(), samples_are_flattened=False, verbose=False):
        '''
            :param ncycles: the number of cycles for which a random sample is taken for each argument and corresponding function value is computed.
            :type ncycles: int, optional, default given by value of global variable ncycles_default

            :param target_distribution: the type of distribution to be used for the error of the function value. Can be overwritten in the get_distribution routine.
            :type target_distribution: child class of :py:class:`Distribution <thermolib.error.Distribution>`, optional, default=GaussianDistribution

            :param flattener: Flattener to be parsed to the error distribution of the function value. Can be overwritten in the get_distribution routine.
            :type flattener: instance of child class of :py:class:`Flattener <thermolib.flattener.Flattener>`, optional, default=DummyFlattener()

            :param samples_are_flattened: whether or not the samples generated by ``gen_args_samples`` (and hence the ``sample`` routine of the arguments) are flattened. Can be overwritten in the get_distribution routine.
            :type samples_are_flattened: bool, optional, default=False

            :param verbose: If True, increase verbosity of the propagator logging
            :type verbose: bool, optional, default=False
        '''
        self.ncycles = ncycles
        self.argsamples = None
        self.funsamples = None
        self.target_distribution = target_distribution
        self.flattener = flattener
        self.samples_are_flattened = samples_are_flattened
        self.verbose = verbose
    
    def reset(self, target_distribution=None, flattener=None, samples_are_flattened=None):
        '''
            Reinitialize argsamples  for future reuse. For more info on arguments target_distribution, flattener or samples_are_flattened, see documentation in the initializer. Only if such an argument is explicitly specified (i.e. is not None), will it be overwritten with the given value.
        '''
        self.argsamples = None
        self.funsamples = None
        if target_distribution is not None:
            self.target_distribution = target_distribution
        if flattener is not None:
            self.flattener = flattener
        if samples_are_flattened is not None:
            self.samples_are_flattened = samples_are_flattened

    def gen_args_samples(self, *args):
        '''
            Routine that will generate random samples for each of the given arguments. The total number of samples per argument is defined in self.ncycles. 
            
            :param args: list of arguments of the function for which the error needs to be computed
            :type args: list of distributions
        '''
        if self.verbose: print('Error propagation - generating %i argument samples...' %(self.ncycles))
        #generate samples for arguments        
        self.argsamples = [None,]*len(args)
        for iarg, arg in enumerate(args):
            if self.verbose: print('  ... for argument %i (=%s)' %(iarg,arg.__class__.__name__))
            self.argsamples[iarg] = arg.sample(nsamples=self.ncycles)
            if self.verbose: print('  ..... resulted in argsamples[%i].shape=' %(iarg),self.argsamples[iarg].shape)
    
    def calc_fun_values(self, fun):
        '''
            Routine to compute the function value for each set of random samples stored for the arguments during execution of the ``gen_args_samples`` routine.

            :param fun: function f for which the error needs to be computed
            :type fun: callable
        '''
        if self.verbose: print('Error propagation - calculating function values for %i cycles...' %(self.ncycles))
        #propagate argument samples to function values
        self.funsamples = [None,]*self.ncycles
        for icycle in range(self.ncycles):
            if self.verbose: print('  ... for cycle %i' %(icycle))
            args = [self.argsamples[iarg][...,icycle] for iarg in range(len(self.argsamples))]
            if self.verbose: 
                print('  ..... function has %i arguments' %(len(args)))
                for iarg, arg in enumerate(args):
                    print('  ..... argument %i has shape:' %(iarg), arg.shape)
            self.funsamples[icycle] = fun(*args)
        #Some post process book keeping and logging
        if self.verbose: print('Error propagation - generating distribution from samples ...')
        self.funsamples = np.dstack(self.funsamples)
        if self.funsamples.shape[0]==1:
            self.funsamples = self.funsamples[0]
        if self.verbose: print('funsamples.shape=',self.funsamples.shape)

    def get_distribution(self, target_distribution=None, flattener=None, samples_are_flattened=None):
        '''
            Routine to construct the error distribution of the function value from a population generated by the routines ``gen_args_samples`` and ``calc_fun_values``. For more info on arguments target_distribution, flattener or samples_are_flattened, see documentation in the initializer. Only if such an argument is explicitly specified (i.e. is not None), will it be overwritten with the given value.

            :return: Distribution of the function value
            :rtype: determined by parameter target_distribution
        '''
        if target_distribution is None: target_distribution = self.target_distribution
        if flattener is None: flattener = self.flattener
        if samples_are_flattened is None: samples_are_flattened = self.samples_are_flattened
        if target_distribution in [MultiGaussianDistribution, MultiLogGaussianDistribution]:
            distr = target_distribution.from_samples(self.funsamples, flattener=flattener, flattened=samples_are_flattened)
        else:
            distr = target_distribution.from_samples(self.funsamples)
        return distr
    
    def __call__(self, fun, *args, target_distribution=None, flattener=None, samples_are_flattened=None):
        '''
            Default calling sequence of the various Propagator routines.  For more info on arguments target_distribution, flattener or samples_are_flattened, see documentation of :py:meth:`Propagator.__init__ <thermolib.error.Propoagator.__init__>`. Only if such an argument is explicitly specified (i.e. is not None), will it be overwritten with the given value.

            :param fun: function f for which the error needs to be computed
            :type fun: callable

            :param args: list of arguments of the function for which the error needs to be computed
            :type args: list of distributions

            :return: Distribution of the function value
            :rtype: determined by parameter target_distribution
        '''
        #for backward compatibility
        self.reset()
        self.gen_args_samples(*args)
        self.calc_fun_values(fun)
        return self.get_distribution(target_distribution=target_distribution, flattener=flattener, samples_are_flattened=samples_are_flattened)