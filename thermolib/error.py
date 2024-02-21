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

import numpy as np, numpy.ma as ma
np.seterr(divide='ignore', invalid='ignore')

from molmod.units import *

import matplotlib.pyplot as pp

from .tools import format_scientific, multivariate_normal
from .flatten import DummyFlattener

import time

__all__ = [
    'ncycles_default', 
    'Distribution', 'SampleDistribution', 'GaussianDistribution', 'LogGaussianDistribution', 
    'MultiDistribution', 'MultiGaussianDistribution', 'MultiLogGaussianDistribution', 
    'ErrorArray', 'Propagator'
]


ncycles_default = 50


class Distribution(object):
    '''Abstract parent class for distributions.'''
    def __init__(self, shape, flattener=None):
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
    def __init__(self, samples):
        self.samples = samples.copy()
        if len(samples.shape)==1:
            shape = (1,)
        else:
            shape = samples.shape[1:]
        Distribution.__init__(self, shape)

    def shift(self, ref):
        self.samples += ref
    
    def set_ref(self, index=None, value=None):
        assert index is None, 'SampleDistribution.set_ref can only be called with value keyword'
        assert value is not None, 'SampleDistribution.set_ref can only be called with value keyword'
        self.means -= value

    @classmethod
    def from_samples(cls, samples):
        return cls(samples)
    
    def mean(self):
        samples_ma = ma.masked_invalid(self.samples)
        return samples_ma.mean(axis=-1)
    
    def std(self):
        samples_ma = ma.masked_invalid(self.samples)
        return samples_ma.std(axis=-1)
    
    def sample(self):
        return self.samples[np.random.random_integers(low=0, high=self.shape[0])]


class GaussianDistribution(Distribution):
    '''Implementation of a Gaussian distribution'''
    def __init__(self, means, stds):
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
        return GaussianDistribution(self.means, self.stds)

    def shift(self, ref):
        self.means += ref
    
    def set_ref(self, index=None, value=None):
        if index is not None:
            self.means -= self.means[index]
        elif value is not None:
            self.means -= value
        else:
            raise ValueError('In GaussianDistribution.set_ref either index or value keyword argument should be defined')
    
    def sample(self, nsamples=None):
        if nsamples is None:
            return np.random.normal(loc=self.means, scale=self.stds)
        else:
            result = np.stack([np.random.normal(loc=self.means, scale=self.stds) for i in range(nsamples)], axis=len(self.means.shape))
            return result
    
    def mean(self):
        return self.means
    
    def std(self):
        return self.stds

    def nsigma_conf_int(self, nsigma):
        lower = self.means - nsigma*self.stds
        upper = self.means + nsigma*self.stds
        return lower, upper
    
    @classmethod
    def from_samples(cls, samples):
        samples_ma = ma.masked_invalid(samples)
        means = samples_ma.mean(axis=-1)
        stds = samples_ma.std(axis=-1, ddof=1)
        return cls(means.filled(np.nan), stds.filled(np.nan))
    
    @classmethod
    def log_from_loggaussian(cls, logdist, shift=0.0, scale=1.0):
        assert isinstance(logdist, LogGaussianDistribution), 'Can only make Gaussian distribution from logarithm of LogGaussianDistribution'
        return cls(scale*logdist.lmeans+shift, abs(scale)*logdist.lstds)
    
    def print(self, fmt='%.3f', unit='au', do_scientific=False, nsigma=2):
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
    '''Implementation of a Log-Gaussian distribution, i.e. variable X is Log-Gaussian with mean mu and std sigma if variable log(X) is Gaussian distributed with mean mu and std sigma.'''
    def __init__(self, lmeans, lstds):
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
        self.lmeans += ref
    
    def set_ref(self, index=None, value=None):
        if index is not None:
            self.lmeans -= self.means[index]
        elif value is not None:
            self.lmeans -= value
        else:
            raise ValueError('In LogGaussianDistribution.set_ref either index or value keyword argument should be defined')

    def mean(self):
        return np.exp(self.lmeans + 0.5*self.lstds**2)

    def std(self):
        return np.sqrt(np.exp(self.lstds**2)-1)*self.mean()

    def copy(self):
        return LogGaussianDistribution(self.lmeans, self.lstds)
    
    def sample(self, nsamples=None):
        if nsamples is None:
            return np.exp(np.random.normal(loc=self.lmeans, scale=self.lstds))
        else:
            result = np.stack([np.exp(np.random.normal(loc=self.lmeans, scale=self.lstds)) for i in range(nsamples)], axis=len(self.lmeans.shape))
            return result
    
    def nsigma_conf_int(self, nsigma):
        lower = self.lmeans - nsigma*self.lstds
        upper = self.lmeans + nsigma*self.lstds
        return np.exp(lower), np.exp(upper)
    
    @classmethod
    def from_samples(cls, samples):
        lsamples = ma.masked_invalid(np.log(samples))
        lmeans = lsamples.mean(axis=-1)
        lstds = lsamples.std(axis=-1, ddof=1)
        return cls(lmeans, lstds)
    
    @classmethod
    def exp_from_gaussian(cls, gaussdist, shift=0.0, scale=1.0):
        assert isinstance(gaussdist, GaussianDistribution), 'Can only make LogGaussian distribution from exponent of GaussianDistribution'
        return cls(scale*gaussdist.means+shift, abs(scale)*gaussdist.stds)
    

class FunctionDistribution(Distribution):
    def __init__(self, fun, dist):
        self.fun = fun
        self.ifun = ifun
        self.dist = dist
        Distribution.__init__(self, dist.shape)

    def sample(self):
        return self.fun(self.dist.sample)
    
    def nsigma_conf_int(self, nsigma):
        lower, upper = self.dist.nsigma_conf_int(nsigma)
        fun_lower, fun_upper = self.fun(lower), self.fun(upper)
        if fun_lower<=fun_upper:
            return fun_lower, fun_upper
        else:
            return fun_upper, fun_lower


class MultiDistribution(object):
    def __init__(self, shape, flattener=DummyFlattener()):
        self.shape = shape
        self.flattener = flattener

    def mean(self, unflatten=True):
        raise NotImplementedError
    
    def std(self, unflatten=True):
        raise NotImplementedError
    
    def cov(self):
        raise NotImplementedError
    
    def sample(self, unflatten=True):
        raise NotImplementedError

    def nsigma_conf_int(self, nsigma=2, unflatten=True):
        return NotImplementedError

    def corr(self, unflatten=True):
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

    def from_samples(self, samples, flattener, flattened=True):
        raise NotImplementedError


class MultiGaussianDistribution(MultiDistribution):
    '''Implementation of a multivariate normal distribution for a given vector of means and a given covariance matrix
    '''
    def __init__(self, means, covariance, flattener=DummyFlattener()):
        "flattener is a class that should only be specified if the given means array actually represents a 1D flattened array of a corresponding 2D matrix (as well as covariance represents a flattened 2D matrix of a corresponding 4D matrix). In this case, flattener contains the routines to return deflattend array/matrices for the routines: mean, std, cov, sample, nsigma_conf_int, ..."
        assert isinstance(means, np.ndarray), 'means should be array, detected %s' %(str(type(means)))
        assert isinstance(covariance, np.ndarray), 'covariance matrix should be array, detected %s' %(str(type(covariance)))
        shape = means.shape
        assert len(means.shape)==1, 'means should be 1D array, but detected array shape %s' %(str(means.shape))
        assert len(covariance.shape)==2, 'covariance matrix should be 2D array, but detected array shape %s' %(str(covariance.shape))
        assert covariance.shape[0]==means.shape[0], 'Inconsistent shapes, means has shape %s while covariance has shape %s' %(str(means.shape),str(covariance.shape))
        assert covariance.shape[1]==means.shape[0], 'Inconsistent shapes, means has shape %s while covariance has shape %s' %(str(means.shape),str(covariance.shape))
        self.means = means.copy()
        self.covariance = covariance.copy()
        self.stds = np.sqrt(covariance.diagonal())
        MultiDistribution.__init__(self, shape, flattener=flattener)

    def copy(self):
        return MultiGaussianDistribution(self.means, self.covariance, flattener=self.flattener)

    def mean(self, unflatten=True):
        if unflatten:
            return self.flattener.unflatten_array(self.means)
        else:
            return self.means.copy()

    def std(self, unflatten=True):
        if unflatten:
            return self.flattener.unflatten_array(self.stds)
        else:
            return self.stds.copy()
    
    def cov(self, unflatten=True):
        if unflatten:
            return self.flattener.unflatten_matrix(self.covariance)
        else:
            return self.covariance.copy()

    def sample(self, nsamples=None, unflatten=True):
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
        lower = self.mean(unflatten=unflatten) - nsigma*self.std(unflatten=unflatten)
        upper = self.mean(unflatten=unflatten) + nsigma*self.std(unflatten=unflatten)
        return lower, upper

    def set_ref(self, index=None, value=None):
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
    def from_samples(cls, samples, flattener=DummyFlattener(), flattened=False):
        """ 
            If flattened is True, the given samples should be a 2D array in which a row represents all observations of a single variable and a column represents a single observation of all variables.

            If flattened is False, then the the samples array is assumed to be 3 dimensional for which the first two dimensions represent a 2D index and the third index is the sample index. Therefore, the flatter will first be applied to flatten the 2D index into a 1D index and hence convert the samples to a 2D array of the same shape as required if the flattened argument is True.
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
        assert isinstance(logdist, MultiLogGaussianDistribution), 'Can only make multivariate Gaussian distribution from logarithm of Multivariate LogGaussianDistribution'
        return cls(scale*logdist.lmeans+shift, (scale**2)*logdist.lcovariance, logdist.flattener)

    def plot_corr_matrix(self, fn=None, fig_size_inches=[8,8], cmap='bwr', logscale=False, cvs=None, nticks=10, decimals=1):
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
    '''Implementation of a multivariate log-normal distribution for a given vector of means and a given covariance matrix
    '''
    def __init__(self, lmeans, lcovariance, flattener=DummyFlattener()):
        "flattener is a class that should only be specified if the given means array actually represents a 1D flattened array of a corresponding 2D matrix (as well as covariance represents a flattened 2D matrix of a corresponding 4D matrix). In this case, flattener contains the routines to return deflattend array/matrices for the routines: mean, std, cov, sample, nsigma_conf_int, ..."
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
        return MultiLogGaussianDistribution(self.lmeans, self.lcovariance, self.flattener)

    def mean(self, unflatten=True):
        means = np.exp(self.lmeans + 0.5*self.lstds**2)
        if unflatten:
            return self.flattener.unflatten_array(means)
        else:
            return means

    def std(self, unflatten=True):
        stds = np.sqrt(np.exp(self.lstds**2)-1)*self.mean(unflatten=False)
        if unflatten:
            return self.flattener.unflatten_array(stds)
        else:
            return stds
    
    def cov(self, unflatten=True):
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
        #first compute the flattened lower and upper bounds
        lower = np.exp(self.lmeans - nsigma*self.lstds)
        upper = np.exp(self.lmeans + nsigma*self.lstds)
        #possibly deflatten before returning
        if unflatten:
            return self.flattener.unflatten_array(lower), self.flattener.unflatten_array(upper)
        else:
            return lower, upper

    def set_ref(self, index=None, value=None):
        raise NotImplementedError
        if index is not None:
            self.means -= self.means[index]
        elif value is not None:
            self.means -= value
        else:
            raise ValueError('In MultiGaussianDistribution.set_ref either index or value keyword argument should be defined')
    
    @classmethod
    def from_samples(cls, samples, flattener=DummyFlattener(), flattened=False):
        """ 
            If flattened is True, the given samples should be a 2D array in which a row represents all observations of a single variable and a column represents a single observation of all variables.

            If flattened is False, then the the samples array is assumed to be 3 dimensional for which the first two dimensions represent a 2D index and the third index is the sample index. Therefore, the flatter will first be applied to flatten the 2D index into a 1D index and hence convert the samples to a 2D array of the same shape as required if the flattened argument is True.
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
        assert isinstance(gaussdist, MultiGaussianDistribution), 'Can only make MultiLogGaussian distribution from exponent of MultiGaussianDistribution'
        return cls(scale*gaussdist.means+shift, (scale**2)*gaussdist.covariance, gaussdist.flattener)

    def plot_corr_matrix(self, fn=None, fig_size_inches=[8,8], cmap='bwr', logscale=False, cvs=None, nticks=10, decimals=1):
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
    def __init__(self, errors):
        self.errors = errors
    
    def mean(self):
        return np.array([error.mean() for error in self.errors])
    
    def std(self):
        return np.array([error.std() for error in self.errors])

    def nsigma_conf_int(self, nsigma):
        lowers = []
        uppers = []
        for error in self.errors:
            lower, upper = error.nsigma_conf_int(nsigma)
            lowers.append(lower)
            uppers.append(upper)
        return np.array(lowers), np.array(uppers)

    def sample(self, nsamples=None):
        samples = []
        for error in self.errors:
            samples.append(error.sample(nsamples=nsamples))
        return np.array(samples)


class Propagator(object):
    '''A class to propagate the distribution of errors on a set of arguments towards the distribution of error on a given function of those arguments. This routine uses the sample routine of each of its distrubution arguments, meaning that the resulting error is stochastic (will not give the same repeated result)'''
    def __init__(self, ncycles=ncycles_default, verbose=False):
        self.ncycles = ncycles
        self.argsamples = None
        self.funsamples = None
        self.verbose = verbose

    def gen_args_samples(self, *args):
        if self.verbose: print('Error propagation - generating argument samples...')
        #generate samples for arguments        
        self.argsamples = [None,]*len(args)
        for iarg, arg in enumerate(args):
            if self.verbose: print('  ... for argument %i (=%s)' %(iarg,arg.__class__.__name__))
            self.argsamples[iarg] = arg.sample(nsamples=self.ncycles)
            if self.verbose: print('  ..... resulted in argsamples[%i].shape=' %(iarg),self.argsamples[iarg].shape)
    
    def calc_fun_values(self, fun):
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
        #return target distribution based on samples and flattener if required
        if self.verbose: print('Error propagation - generating distribution from samples ...')
        self.funsamples = np.dstack(self.funsamples)
        if self.funsamples.shape[0]==1:
            self.funsamples = self.funsamples[0]
        if self.verbose: print('funsamples.shape=',self.funsamples.shape)

    def get_distribution(self, target_distribution=GaussianDistribution, flattener=DummyFlattener(), samples_are_flattened=False):
        if target_distribution in [MultiGaussianDistribution, MultiLogGaussianDistribution]:
            distr = target_distribution.from_samples(self.funsamples, flattener=flattener, flattened=samples_are_flattened)
        else:
            distr = target_distribution.from_samples(self.funsamples)
        return distr
    
    def __call__(self, fun, *args, target_distribution=GaussianDistribution, flattener=DummyFlattener(), samples_are_flattened=False):
        #for backward compatibility
        self.gen_args_samples(*args)
        self.calc_fun_values(fun)
        return self.get_distribution(target_distribution=target_distribution, flattener=flattener, samples_are_flattened=samples_are_flattened)