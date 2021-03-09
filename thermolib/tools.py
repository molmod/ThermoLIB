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
        analysis. Include error estimation based on Bayesian analysis
        and the Gamma distribution.

        Upper and lower boundary of an nsigma-confidence interval
        will be returned.
    '''
    Ntot = len(data)
    ns, bin_edges = np.histogram(data, bins, density=False)
    cvs = 0.5*(bin_edges[:-1]+bin_edges[1:]) # bin centers
    ps = ns/Ntot
    fs = np.zeros(len(cvs), float)*np.nan
    fs[ps>0] = -boltzmann*temp*np.log(ps[ps>0])
    #estimate of upper and lower boundary of 95% confidence interval (i.e. 2 sigma)
    #for now this assumes Normal distribution (while in reality it is Gamma distributed)
    #TODO: use Gamma distribution for confidence interval
    perrors = np.sqrt(ns)/Ntot
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
