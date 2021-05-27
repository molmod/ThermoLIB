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

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as pp
import sys


__all__ = [
    'integrate', 'integrate2d', 'format_scientific',
    'free_energy_from_histogram_with_error'
]

def integrate(xs, ys):
    'A simple integration method using the trapezoid rule'
    assert len(xs)==len(ys)
    result = 0.0
    for i in range(len(xs)-1):
        x = 0.5*xs[i]+0.5*xs[i+1]
        y = 0.5*ys[i]+0.5*ys[i+1]
        dx = xs[i+1]-xs[i]
        result += y*dx
    return result

def integrate2d(z,x=None,y=None,dx=1.,dy=1.):
    ''' Integrates a regularly spaced 2D grid using the composite trapezium rule.
    IN:
       z : 2D array
       x : (optional) grid values for x (1D array)
       y : (optional) grid values for y (1D array)
       dx: if x is not supplied, set it to the x grid interval
       dy: if y is not supplied, set it to the x grid interval
    '''
    if x is not None:
        dx = (x[-1]-x[0])/(np.shape(x)[0]-1)
    if y is not None:
        dy = (y[-1]-y[0])/(np.shape(y)[0]-1)
    s1 = z[0,0] + z[-1,0] + z[0,-1] + z[-1,-1]
    s2 = np.sum(z[1:-1,0]) + np.sum(z[1:-1,-1]) + np.sum(z[0,1:-1]) + np.sum(z[-1,1:-1])
    s3 = np.sum(z[1:-1,1:-1])
    return 0.25*dx*dy*(s1 + 2*s2 + 4*s3)


def format_scientific(x, prec=3, latex=True):
    if np.isnan(x):
        return r'nan'
    #print(x, ('{:.%iE}' %prec).format(x))
    a, b = ('{:.%iE}' %prec).format(x).split('E')
    if latex:
        return r'$%s\cdot 10^{%s}$' %(a,b)
    else:
        return '%s 10^%s' %(a, b)

def free_energy_from_histogram_with_error(data, bins, temp, nsigma=2):
    '''
        Construct probability and free energy profile from histogram
        analysis. Include error estimation based on the asymptotic
        normality of the maximum likelihood estimator.
        
        Upper and lower boundary of an nsigma-confidence interval 
        will be returned.
    '''
    Ntot = len(data)
    ns, bin_edges = np.histogram(data, bins, density=False)
    cvs = 0.5*(bin_edges[:-1]+bin_edges[1:]) # bin centers
    ps = ns/Ntot
    fs = np.zeros(len(cvs), float)*np.nan
    fs[ps>0] = -boltzmann*temp*np.log(ps[ps>0])
    #estimate of upper and lower boundary of 95% confidence interval (2 sigma, i.e. for nsigma=2)
    perrors = np.sqrt(ps*(1-ps)/Ntot)
    plower = ps - nsigma*perrors
    plower[plower<1e-10] = 1e-10
    pupper = ps + nsigma*perrors
    pupper[pupper<1e-10] = 1e-10
    fupper = -boltzmann*temp*np.log(plower)
    flower = -boltzmann*temp*np.log(pupper)
    return cvs, ps, plower, pupper, fs, flower, fupper

def trajectory_xyz_to_CV(fns, CV):
    '''
        Compute the CV along an XYZ trajectory. The XYZ
        trajectory is assumed to be composed out of a 
        (list of) XYZ files.
    '''
    cvs = []
    xyz = XYZReader(fn_xyz)
    for title, coords in xyz:
        cv = CV.compute(coords, deriv=False)
        cvs.append(cv)
    del xyz
    return np.array(cvs)

def blav(data, blocksizes=None, fitrange=[0,-1], fn_plot=None, unit='au', plot_auto_correlation_range=np.arange(0,501,1)):
    'Routine to implement block averaging'
    if blocksizes is None:
        blocksizes = np.arange(1,len(data)+1,1)
    sigmas = np.zeros(len(blocksizes), float)
    errors = np.zeros(len(blocksizes), float)
    for i, blocksize in enumerate(blocksizes):
        nblocks = len(data)//blocksize
        blavs = np.zeros(nblocks, float)
        for iblock in range(nblocks):
            blavs[iblock] = data[iblock*blocksize:(iblock+1)*blocksize].mean()
        #the unbiased estimate on the variance of the block averages
        sigmas[i] = blavs.std(ddof=1) 
        #the estimate on the error on the mean of block averages assuming 
        #uncorrelated block averages (which will only be truly valid for 
        #sufficiently large block size)
        errors[i] = sigmas[i]/np.sqrt(nblocks)
    #fit standard deviations
    def function(blocksize, a, b):
        ##arctan function
        #return a*2/np.pi*np.arctan(b*blocksize)            
        #fit which corresponds to fitting tau=1+(t0-1)/blocksize for integrated correlation time, 
        #resulting in err*blocksize/(blocksize+t0-1) fit for errors in which err represents the 
        #true error and t0 the correlation time for the original time series (blocksize=1)
        return a*blocksize/(blocksize+b-1)
    def fit(blocksizes, xs):
        sol, cov = curve_fit(function, blocksizes[fitrange[0]:fitrange[1]], xs[fitrange[0]:fitrange[1]])
        return sol
    error, corrtime = fit(blocksizes, errors)
    #make plot
    if fn_plot is not None:
        pp.clf()
        if plot_auto_correlation_range is not None:
            fig, axs = pp.subplots(nrows=1,ncols=3)
            iplotblav = 2
            nplots = 3
        else:
            fig, axs = pp.subplots(nrows=1,ncols=2)
            iplotblav = 1
            nplots = 2
        axs[0].plot(data/parse_unit(unit), 'bo', markersize=1)
        axs[0].axhline(y=data.mean()/parse_unit(unit), color='b', linestyle='--', linewidth=1)
        axs[0].set_title('Samples [%s]' %unit, fontsize=12)
        
        if plot_auto_correlation_range is not None:
            N = len(data)
            mu = data.mean()
            sigma = data.std(ddof=1)
            ac = np.zeros(len(plot_auto_correlation_range))
            for k in plot_auto_correlation_range:
                ac[k] = ((data[0:N-k]-mu)*(data[k:N]-mu)).sum()/((N-k)*sigma**2)
            axs[1].plot(ac)
            axs[1].axhline(y=0, color='k', linestyle='--')
            axs[1].set_ylim(-1,1)
            
        axs[iplotblav].plot(blocksizes, errors/parse_unit(unit), color='b', linestyle='none', marker='o', markersize=1)
        axs[iplotblav].plot(blocksizes, function(blocksizes,error,corrtime)/parse_unit(unit), color='r', linestyle='-', linewidth=1)
        axs[iplotblav].axhline(y=error/parse_unit(unit), color='k', linestyle='--', linewidth=1)
        axs[iplotblav].set_title('Error of the estimate on the sample mean [%s]' %unit, fontsize=12)
        
        fig.set_size_inches([6*nplots,8])
        pp.show()
        pp.savefig(fn_plot, dpi=300)
    return blavs.mean(), error, corrtime

