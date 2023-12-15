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

import numpy as np
from molmod.units import *

from .tools import format_scientific

__all__ = ['GaussianDistribution', 'LogGaussianDistribution', 'SampleDistribution', 'Propagator']

class Distribution(object):
    '''Abstract parent class for distributions.'''
    def __init__(self, shape, ncycles=100):
        self.shape = shape
        self.ncycles = ncycles

    def shift(self, ref):
        raise NotImplementedError

    def set_ref(self, index=None, value=None):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError
    
    def mean(self, ncycles=None):
        raise NotImplementedError

    def std(self, ncycles=None):
        raise NotImplementedError

    def nsigma_conf_int(self, nsigma):
        raise NotImplementedError
    
    @classmethod
    def from_samples(cls, samples, ncycles=100):
        raise NotImplementedError
    
    def print(self, fmt='%.3f', unit='au', do_scientific=False, nsigma=2):
        template = fmt + ' +/- ' + fmt + ' ' + unit
        if do_scientific:
            return (template.replace('%.3f', '%s')) %(format_scientific(self.mean()/parse_unit(unit)), format_scientific(nsigma*self.std()/parse_unit(unit)))
        else:
            return template %(self.mean()/parse_unit(unit), nsigma*self.std()/parse_unit(unit))


class SampleDistribution(Distribution):
    def __init__(self, samples, ncycles=100):
        self.samples = samples.copy()
        if len(samples.shape)==1:
            shape = (1,)
        else:
            shape = samples.shape[1:]
        #print(samples.shape,shape)
        Distribution.__init__(self, shape, ncycles=ncycles)
    
    def transpose(self):
        self.samples = self.samples.T

    def shift(self, ref):
        self.samples += ref
    
    def set_ref(self, index=None, value=None):
        assert index is None, 'SampleDistribution.set_ref can only be called with value keyword'
        assert value is not None, 'SampleDistribution.set_ref can only be called with value keyword'
        self.means -= value

    @classmethod
    def from_samples(cls, samples, ncycles=100):
        return cls(samples, ncycles=ncycles)
    
    def mean(self):
        return self.samples.mean(axis=0)
    
    def std(self):
        return self.samples.std(axis=0)
    
    def sample(self):
        return self.samples[np.random.random_integers(low=0, high=self.shape[0])]


class GaussianDistribution(Distribution):
    '''Implementation of a Gaussian distribution'''
    def __init__(self, means, stds, ncycles=100):
        assert type(means)==type(stds), 'Means and stds in GaussianDistribution should be of same type'
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
        Distribution.__init__(self, shape, ncycles=ncycles)
    
    def transpose(self):
        self.means = self.means.T
        self.stds = self.stds.T

    def copy(self):
        return GaussianDistribution(self.means, self.stds, ncycles=self.ncycles)

    def shift(self, ref):
        self.means += ref
    
    def set_ref(self, index=None, value=None):
        if index is not None:
            self.means -= self.means[index]
        elif value is not None:
            self.means -= value
        else:
            raise ValueError('In GaussianDistribution.set_ref either index or value keyword argument should be defined')
    
    def sample(self):
        return np.random.normal(loc=self.means, scale=self.stds)
    
    def mean(self):
        return self.means
    
    def std(self):
        return self.stds

    def nsigma_conf_int(self, nsigma):
        lower = self.means - nsigma*self.stds
        upper = self.means + nsigma*self.stds
        return lower, upper
    
    @classmethod
    def from_samples(cls, samples, ncycles=100):
        means = samples.mean(axis=0)
        stds = samples.std(axis=0)
        return cls(means, stds, ncycles=ncycles)
    
    @classmethod
    def exp_from_loggaussian(cls, logdist, shift=0.0, scale=1.0):
        assert isinstance(logdist, LogGaussianDistribution), 'Can only make Gaussian distribution from exponential of LogGaussianDistribution'
        return cls(scale*logdist.means+shift, abs(scale)*logdist.stds, ncycles=logdist.ncycles)


class LogGaussianDistribution(Distribution):
    '''Implementation of a Log-Gaussian distribution, i.e. variable X is Log-Gaussian with mean mu and std sigma if variable log(X) is Gaussian distributed with mean mu and std sigma.'''
    def __init__(self, means, stds, ncycles=100):
        if isinstance(means, np.ndarray):
            self.means = means.copy()
        else:
            self.means = means
        if isinstance(stds, np.ndarray):
            self.stds = stds.copy()
        else:
            self.stds = stds
        assert type(means)==type(stds), 'Means and stds in GaussianDistribution should be of same type'
        if isinstance(means, np.ndarray):
            shape = means.shape
            assert stds.shape==shape, 'Means and stds in GaussianDistribution should be of same length'
        elif isinstance(means, int) or isinstance(means, float):
            shape = (1,)
        else:
            raise ValueError('means should be int, float or np.ndarray, got %s' %(type(means).__name__))
        Distribution.__init__(self, shape, ncycles=ncycles)

    def shift(self, ref):
        self.means += ref
    
    def set_ref(self, index=None, value=None):
        if index is not None:
            self.means -= self.means[index]
        elif value is not None:
            self.means -= value
        else:
            raise ValueError('In LogGaussianDistribution.set_ref either index or value keyword argument should be defined')

    def mean(self):
        return np.exp(self.means + 0.5*self.stds**2)

    def std(self):
        return np.sqrt(np.exp(self.stds**2)-1)*self.mean()

    def transpose(self):
        self.means = self.means.T
        self.stds = self.stds.T

    def copy(self):
        return LogGaussianDistribution(self.means, self.stds, ncycles=self.ncycles)
    
    def sample(self):
        return np.exp(np.random.normal(loc=self.means, scale=self.stds))
    
    def nsigma_conf_int(self, nsigma):
        lower = self.means - nsigma*self.stds
        upper = self.means + nsigma*self.stds
        return np.exp(lower), np.exp(upper)
    
    @classmethod
    def from_samples(cls, samples, ncycles=100):
        means = np.log(samples).mean(axis=0)
        stds = np.log(samples).std(axis=0)
        return cls(means, stds, ncycles=ncycles)
    
    @classmethod
    def log_from_gaussian(cls, gaussdist, shift=0.0, scale=1.0):
        assert isinstance(gaussdist, GaussianDistribution), 'Can only make LogGaussian distribution from logarithm of GaussianDistribution'
        return cls(scale*gaussdist.means+shift, abs(scale)*gaussdist.stds, ncycles=gaussdist.ncycles)
    

class FunctionDistribution(Distribution):
    def __init__(self, fun, ifun, dist, ncycles=None):
        self.fun = fun
        self.ifun = ifun
        self.dist = dist
        if ncycles is None:
            ncycles=dist.ncycles
        Distribution.__init__(self, dist.shape, ncycles=ncycles)

    def sample(self):
        return self.fun(self.dist.sample)
    
    def nsigma_conf_int(self, nsigma):
        lower, upper = self.dist.nsigma_conf_int(nsigma)
        fun_lower, fun_upper = self.fun(lower), self.fun(upper)
        if fun_lower<=fun_upper:
            return fun_lower, fun_upper
        else:
            return fun_upper, fun_lower


class Propagator(object):
    '''A class to propagate the distribution of errors on a set of arguments towards the distribution of error on a given function of those arguments. This routine uses the sample routine of each of its distrubution arguments, meaning that the resulting error is stochastic (will not give the same repeated result)'''
    def __init__(self, ncycles=100, target_distribution=GaussianDistribution):
        self.ncycles = ncycles
        self.target_distribution = target_distribution

    def __call__(self, fun, *args):
        ##consistency check and get argument distribution shape
        #shape = None
        #for arg in args:
        #    if shape is None:
        #        shape = arg.shape
        #    else:
        #        assert arg.shape==shape, 'Argument distributions do not have consistent shapes! Found %s and %s' %(str(arg.shape),str(shape))
        #generate samples for arguments and propagate through function
        samples = []
        for icycle in range(self.ncycles):
            argvals = [arg.sample() for arg in args]
            samples.append(fun(*argvals))
        samples = np.array(samples)
        #define target distribution based on samples
        return self.target_distribution.from_samples(samples, ncycles=self.ncycles)