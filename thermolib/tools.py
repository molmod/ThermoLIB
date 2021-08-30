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
from molmod.constants import *
from molmod.io.xyz import XYZReader

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as pp
import sys


__all__ = [
    'integrate', 'integrate2d', 'format_scientific',
    'free_energy_from_histogram_with_error', 
    'trajectory_xyz_to_CV', 'blav', 'read_wham_input'
]

def integrate(xs, ys):
    '''
        A simple integration method using the trapezoid rule

        :param xs: array containing function argument values on grid
        :type xs: np.ndarray

        :param ys: array containing function values on grid
        :type ys: np.ndarray
    '''
    assert len(xs)==len(ys)
    result = 0.0
    for i in range(len(xs)-1):
        x = 0.5*xs[i]+0.5*xs[i+1]
        y = 0.5*ys[i]+0.5*ys[i+1]
        dx = xs[i+1]-xs[i]
        result += y*dx
    return result

def integrate2d(z,x=None,y=None,dx=1.,dy=1.):
    '''
        Integrates a regularly spaced 2D grid using the composite trapezium rule.

        :param z: 2 dimensional array containing the function values
        :type z: np.ndarray(flt)

        :param x: 1D array containing the first function argument values, is used to determine grid spacing of first function argument. If not given, optional argument dx is used to define the grid spacing. Defaults to None
        :type x: np.ndarray(flt), optional

        :param y: 1D array containing the second function argument values, is used to determine grid spacing of second function argument. If not given, optional argument dy is used to define the grid spacing. Defaults to None
        :type y: np.ndarray(flt), optional

        :param dx: grid spacing for first function argument. If not given, argument is used to determine grid spacing. Defaults to 1.
        :type dx: float, optional

        :param dy: grid spacing for second function argument. If not given, argument is used to determine grid spacing. Defaults to 1.
        :type dy: float, optional

        :return: integral value
        :rtype: float
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
        Construct probability and free energy profile from histogram analysis. Include error estimation based on the asymptotic normality of the maximum likelihood estimator. Upper and lower boundary of an n-sigma confidence interval will be returned.

        :param data: data array representing the cv values along a simulation trajectory
        :type data: np.ndarray

        :param bins: array representing the edges of the bins for which a histogram will be constructed
        :type bins: np.ndarray

        :param temp: the temperature at which the input data array was generated. This is required for transforming probability histogram to free energy profile.
        :type temp: float

        :param nsigma: define how large the error interval should be in terms of sigma, eg. nsigma of 2 means a 2-sigma error bar (corresponding to 95% confidence interval) will be returned. Defaults to 2
        :type nsigma: int, optional

        :return: cvs, ps, plower, pupper, fs, flower, fupper

            *  **cvs** (*np.ndarray*) -- array containing the cv grid points
            *  **ps** (*np.ndarray*) -- array containing the coresponding probability density profile
            *  **plower** (*np.ndarray*) -- array containing the lower limit of the error bar on the ps values
            *  **pupper** (*np.ndarray*) -- array containing the upper limit of the error bar on the ps values
            *  **fs** (*np.ndarray*) -- array containing the coresponding free energy profile 
            *  **flower** (*np.ndarray*) -- array containing the lower limit of the error bar on the fs values
            *  **fupper** (*np.ndarray*) -- array containing the upper limit of the error bar on the fs values
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
        Compute the CV along an XYZ trajectory. The XYZ trajectory is assumed to be composed out of a (list of subsequent) XYZ files.

        :param fns: (list of) names of XYZ trajectory file(s) containing the xyz coordinates of the system
        :type fns: str or list(str)

        :param CV: collective variable defining how to compute the collective variable along the trajectory
        :type CV: one from thermolib.thermodynamics.cv.__all__

        :return: array containing the CV value along the trajectory
        :rtype: np.ndarray(flt)
    '''
    cvs = []
    for fn in fns:
        xyz = XYZReader(fn)
        for title, coords in xyz:
            cv = CV.compute(coords, deriv=False)
            cvs.append(cv)
        del xyz
    return np.array(cvs)

def blav(data, blocksizes=None, fitrange=[0,-1], exponent=1, fn_plot=None, unit='au', plot_ac=False, ac_range=None, acft_plot_range=None):
    '''
        Routine to implement block averaging. This allows to estimate the sample error on correlated data by fitting a model function towards to the naive (i.e. as if uncorrelated) sample error on the block averages as function of the blocksize. This model function is based on the following model for the integrated correlation time :math:`\\tau` between the block averages:

        .. math::

            \\begin{aligned}
                \\tau = 1 + \\frac{t_0-1}{B^n}
            \\end{aligned}
            
        As a result, the model for the naive error estimate on the block averages becomes
            
        .. math::

            \\begin{aligned}
                error = TE\\cdot\\frac{B^n}{B^n+t_0-1}
            \\end{aligned}
        
        in which :math:`B` represents the block size, :math:`TE` the true error, :math:`t_0` the correlation time for the original time series (:math:`B=1`) and :math:`n` the exponential rate with which the block average integrated correlation time decreases as function of block size.

        :param data: 1D array representing the data to be analyzed
        :type data: np.ndarray
        
        :param blocksizes: array of block sizes, defaults to np.arange(1,len(data)+1,1)
        :type blocksizes: np.ndarray, optional

        :param fitrange: range of blocksizes to which fit will be performed, defaults to [0,-1]
        :type fitrange: list, optional

        :param exponent: the exponent of the blocksize in the models for the auto correlation time and correlated sample error, defaults to 1
        :type exponent: int, optional

        :param fn_plot: file name to which to write the plot. No plot is made if set to None. Defaults to None
        :type fn_plot: str, optional

        :param unit: unit in which to plot the data, defaults to 'au'
        :type unit: str, optional

        :param plot_ac: indicate whether or not to compute and plot the auto correlation function as well (might take some time), defaults to False
        :type plot_ac: bool, optional

        :param ac_range: the range for which to plot the auto correlation function, only relevant if ``plot_ac`` is set to True, defaults to np.arange(0,501,1)
        :type ac_range: np.ndarray, optional

        :param acft_plot_range: the range for which to plot the fourier transform of the auto correlation function, only relevant if ``plot_ac`` is set to True, defaults to entire freq range
        :type acft_plot_range: np.ndarray, optional

        :return: mean, error, corrtime

        *  **mean** (*float*) -- the sample mean
        *  **error** (*foat*) -- the error on the sample mean
        *  **corrtime** (*float*) -- the correlation time (in units of the timestep) of the original sample data 
    '''
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
    def function(blocksize, TE, t0):            
        n=exponent
        return TE*blocksize**n/(blocksize**n+t0-1)
    def fit(blocksizes, xs):
        sol, cov = curve_fit(function, blocksizes[fitrange[0]:fitrange[1]], xs[fitrange[0]:fitrange[1]])
        return sol
    error, corrtime = fit(blocksizes, errors)
    #make plot
    if fn_plot is not None:
        pp.clf()
        if plot_ac:
            fig, axs = pp.subplots(nrows=2,ncols=2, squeeze=False)
            nrows = 2
        else:
            fig, axs = pp.subplots(nrows=1,ncols=2, squeeze=False)
            nrows = 1
        axs[0,0].plot(data/parse_unit(unit), 'bo', markersize=1)
        axs[0,0].axhline(y=data.mean()/parse_unit(unit), color='b', linestyle='--', linewidth=1)
        axs[0,0].set_title('Samples', fontsize=12)
        axs[0,0].set_xlabel('Time [timestep]')
        axs[0,0].set_ylabel('Sample [%s]' %unit)
        
        if plot_ac:
            if ac_range is None:
                ac_range = np.arange(0,501,1)
            N = len(data)
            mu = data.mean()
            sigma = data.std(ddof=1)
            ac = np.zeros(len(ac_range))
            for k in ac_range:
                ac[k] = ((data[0:N-k]-mu)*(data[k:N]-mu)).sum()/((N-k)*sigma**2)
            axs[1,0].plot(ac)
            axs[1,0].axhline(y=0, color='k', linestyle='--')
            axs[1,0].set_ylim(-1.05,1.05)
            axs[1,0].set_title('Auto correlation function')
            axs[1,0].set_xlabel('Time [timestep]')
            axs[1,0].set_ylabel('Amplitude [-]')
            #fourier transform of ac
            acft = np.fft.rfft(ac).real
            if len(ac)%2==0:
                n = len(ac)/2+1
            else:
                n = len(ac+1)/2
            freqs = np.arange(0,n)*2*np.pi/len(ac)
            axs[1,1].plot(freqs, (acft)**2, color='b')
            axs[1,1].set_title('Power spectrum of auto correlation')
            axs[1,1].set_xlabel('Frequency [1/timestep]')
            axs[1,1].set_ylabel('Amplitude [-]')
            if acft_plot_range is not None:
                axs[1,1].set_xlim(acft_plot_range)
            
        axs[0,1].plot(blocksizes, errors/parse_unit(unit), color='b', linestyle='none', marker='o', markersize=1)
        axs[0,1].plot(blocksizes, function(blocksizes,error,corrtime)/parse_unit(unit), color='r', linestyle='-', linewidth=1)
        axs[0,1].axhline(y=error/parse_unit(unit), color='k', linestyle='--', linewidth=1)
        axs[0,1].set_title('Error of the estimate on the sample mean', fontsize=12)
        axs[0,1].set_xlabel('Block size [timestep]')
        axs[0,1].set_ylabel('Error [%s]' %unit)

        fig.set_size_inches([12,6*nrows])
        pp.show()
        pp.savefig(fn_plot, dpi=300)
    return blavs.mean(), error, corrtime

def bias_parabola(kappa, q0):
    '''
        Harmonic bias potential for use in WHAM reconstruction of the free energy profile from Umbrella Sampling simulations.

        :param kappa: the force constant of the parabola
        :type kappa: float

        :param q0: the center of the parabola
        :type q0: float
    '''
    def V(q):
        return 0.5*kappa*(q-q0)**2
    return V

def read_wham_input(fn, kappa_unit='kjmol', q0_unit='au', start=0, end=-1, stride=1, bias_potential='parabola', verbose=False):
    '''
        Read the input for a WHAM reconstruction of the free energy profile from a set of Umbrella Sampling simulations. The file specified by fn should have the following format:

        .. code-block:: python

            T = XXXK
            NAME1 Q01 KAPPA1
            NAME2 Q02 KAPPA2
            NAME3 Q03 KAPPA3
            ...

        where the first line specifies the temperature and the subsequent lines define the bias potential for each simulation as a parabola centered around ``Q0i`` and with force constant ``KAPPAi`` (the units used in this file can be specified the keyword arguments ``kappa_unit`` and ``q0_unit``). For each bias with name ``NAMEi`` defined in this file, there should be a trajectory file in the same directory with the name ``colvar_NAMEi.dat`` containing the trajectory of the relevant collective variable during the biased simulation. This trajectory file should in term be formatted as outputted by PLUMED:

        .. code-block:: python

            time_1 cv_value_1
            time_2 cv_value_2
            time_3 cv_value_3
            ...
        
        where the cv values again have a unit that can be specified by the keyword argument ``q0_unit``.

        :param fn: file name of the wham input file
        :type fn: str

        :param kappa_unit: unit used to express kappa in the wham input file, defaults to 'kjmol'
        :type kappa_unit: str, optional

        :param q0_unit: unit used to express q0 in the wham input file as well as the cv values in the trajectory files, defaults to 'au'
        :type q0_unit: str, optional
        
        :param stride: defines the sub sampling applied to the trajectory data to deal with correlations. For example a stride of 10 means only taking 1 in 10 samples and throw away 90% of the data. Defaults to 1 (i.e. no sub sampling).
        :type stride: int, optional

        :param start: defines the start point from which to take samples into account. This can be usefull for eliminating equilibration times as well as for taking various subsamples trajectories from the original data each starting at different timesteps. Defaults to 0.
        :type start: int, optional

        :param end: defines the end point to which to take samples into account. This can be usefull when it is desired to cut the original trajectory into blocks. Defaults to -1.
        :type end: int, optional

        :param bias_potential: mathematical form of the bias potential used, allowed values are

            * **parabola/harmonic** -- harmonic bias of the form 0.5*kappa*(q-q0)**2

        defaults to parabola

        :type bias_potantial: str, optional

        :param verbose: increases verbosity if set to True, defaults to False
        :type verbose: bool, optional
        
        :raises ValueError: when a line in the wham input file cannot be interpreted

        :return: temp, biasses, trajectories:

            * **temp** (float) -- temperature at which the simulations were performed
            * **biasses** (list of callables) -- list of bias potentials (defined as callable functions) for all Umbrella Sampling simulations
            * **trajectories** (list of np.ndarrays) -- list of trajectory data arrays containing the CV trajectory for all Umbrella Sampling simulations
    '''
    temp = None
    biasses = []
    trajectories = []
    root = '/'.join(fn.split('/')[:-1])
    with open(fn, 'r') as f:
        iline = -1
        for line in f.readlines():
            iline += 1
            line = line.rstrip('\n').rstrip('\x00')
            if line.startswith('T'):
                temp = float(line.split('=')[1].rstrip('K'))
                if verbose:
                    print('Temperature set at %f' %temp)
            elif line.startswith('U') and bias_potential.lower() in ['parabola', 'harmonic']:
                words = line.split()
                name = words[0]
                q0 = float(words[1])*parse_unit(q0_unit)
                kappa = float(words[2])*parse_unit(kappa_unit)
                biasses.append(bias_parabola(kappa,q0))
                if verbose:
                    print('Added bias potential nr. %i (Parabola with kappa = %.3f %s , q0 = %.3e %s)' %(len(biasses), kappa/parse_unit(kappa_unit), kappa_unit, q0/parse_unit(q0_unit), q0_unit))
                fn_traj = '%s/colvar_%s.dat' %(root,name)
                data = np.loadtxt(fn_traj)
                if end == -1:
                    trajectories.append(data[start::stride, 1])
                else:
                    trajectories.append(data[start:end:stride, 1])
                if verbose:
                    print('Read corresponding trajectory data from %s' %fn_traj)
            elif bias_potential.lower() not in ['parabola', 'harmonic']:
                    raise ValueError('Bias potential definition %s not supported, see documentation for allowed values.')
            elif len(line.split())>0:
                raise ValueError('Could not process line %i in %s: %s' %(iline, fn, line))
    return temp, biasses, trajectories

def extract_polynomial_bias_info(fn_plumed='plumed.dat'):
    #TODO make less error phrone. include check.
    with open(fn_plumed,'r') as plumed:
        for line in plumed:
            idx = line.find('COEFFICIENTS',0)
            if idx > 0:
                idx+=13 #13 len of the string
                idx_end = line.find('POWERS',0)
                short_line = line[idx:idx_end].rstrip(' ').rstrip(',')
                split_line = short_line.split(',')
                poly_coef = [float(i) for i in split_line] #remark that -float is needed to get the resulting fe.
                break

    return poly_coef

def bias_polynomial_with_parabola(taylor_coeffs, kappa, q0,poly_unit='kjmol',reflect_x=False):
    def Vpoly(taylor_coeffs,q):
        V_polynomial = np.polynomial.polynomial.Polynomial(taylor_coeffs)
        return V_polynomial(q)

    def Vh(q):
        return 0.5 * kappa * (q - q0) ** 2
    if reflect_x==True:
        sign = -1
    else:
        sign = 1
    sum = lambda q: Vpoly(taylor_coeffs,sign*q)*parse_unit(poly_unit) + Vh(sign*q)
    return sum


def read_wham_input_custom1(fn,temp,fn_plumed=None, kappa_unit='kjmol', q0_unit='au', start=0, end=-1, stride=1, bias_potential='poly_parabola',plumed_unit='kjmol',default_cv_directory='./',
                    verbose=False,reflect_x=False):
    '''
        Read the input for a WHAM reconstruction of the free energy profile from a set of Umbrella Sampling simulations. The file specified by fn should have the following format:

        .. code-block:: python

            FILENAME1 Q01 KAPPA1
            FILENAME2 Q02 KAPPA2
            FILENAME3 Q03 KAPPA3
            ...

        where the first line specifies the temperature and the subsequent lines define the bias potential for each simulation as a parabola centered around ``Q0i`` and with force constant ``KAPPAi`` (the units used in this file can be specified the keyword arguments ``kappa_unit`` and ``q0_unit``). For each bias with name ``NAMEi`` defined in this file, there should be a trajectory file in the same directory with the name ``colvar_NAMEi.dat`` containing the trajectory of the relevant collective variable during the biased simulation. This trajectory file should in term be formatted as outputted by PLUMED:

        .. code-block:: python

            time_1 cv_value_1
            time_2 cv_value_2
            time_3 cv_value_3
            ...

        where the cv values again have a unit that can be specified by the keyword argument ``q0_unit``.

        :param fn: file name of the wham input file
        :type fn: str

        :param kappa_unit: unit used to express kappa in the wham input file, defaults to 'kjmol'
        :type kappa_unit: str, optional

        :param q0_unit: unit used to express q0 in the wham input file as well as the cv values in the trajectory files, defaults to 'au'
        :type q0_unit: str, optional

        :param stride: defines the sub sampling applied to the trajectory data to deal with correlations. For example a stride of 10 means only taking 1 in 10 samples and throw away 90% of the data. Defaults to 1 (i.e. no sub sampling).
        :type stride: int, optional

        :param start: defines the start point from which to take samples into account. This can be usefull for eliminating equilibration times as well as for taking various subsamples trajectories from the original data each starting at different timesteps. Defaults to 0.
        :type start: int, optional

        :param end: defines the end point to which to take samples into account. This can be usefull when it is desired to cut the original trajectory into blocks. Defaults to -1.
        :type end: int, optional

        :param bias_potential: mathematical form of the bias potential used, allowed values are

            * **parabola/harmonic** -- harmonic bias of the form 0.5*kappa*(q-q0)**2

        defaults to parabola

        :type bias_potantial: str, optional

        :param verbose: increases verbosity if set to True, defaults to False
        :type verbose: bool, optional

        :raises ValueError: when a line in the wham input file cannot be interpreted

        :return: temp, biasses, trajectories:

            * **temp** (float) -- temperature at which the simulations were performed
            * **biasses** (list of callables) -- list of bias potentials (defined as callable functions) for all Umbrella Sampling simulations
            * **trajectories** (list of np.ndarrays) -- list of trajectory data arrays containing the CV trajectory for all Umbrella Sampling simulations

    '''
    temp = float(temp)
    biasses = []
    trajectories = []
    if bias_potential.lower() in ['parabola', 'harmonic']:
        bias = 'harm'
    elif bias_potential.lower() in ['poly_parabola']:
        bias = 'poly_harm'
        poly_coef = extract_polynomial_bias_info(fn_plumed)
    elif bias_potential.lower() not in ['parabola', 'harmonic','poly_parabola']:
        raise NotImplementedError('Bias potential definition %s not supported, see documentation for allowed values. One can define a custom bias function in the tools.py file.')

    with open(fn, 'r') as f:
        iline = -1
        for line in f.readlines():
            iline += 1
            line = line.rstrip('\n')
            words = line.split()
            fn_U = words[0]
            q0 = float(words[1]) * parse_unit(q0_unit)
            kappa = float(words[2]) * parse_unit(kappa_unit)
            if bias == 'harm':
                try:
                    biasses.append(bias_parabola(kappa, q0))
                except:
                    raise ValueError('Could not process line %i in %s: %s' % (iline, fn, line))
                if verbose:
                    print('Added bias potential nr. %i (Parabola with kappa = %.3f %s , q0 = %.3e %s)' % (
                    len(biasses), kappa / parse_unit(kappa_unit), kappa_unit, q0 / parse_unit(q0_unit), q0_unit))
            elif bias == 'poly_harm':
                try:
                    biasses.append(bias_polynomial_with_parabola(poly_coef,kappa, q0,poly_unit=plumed_unit,reflect_x=reflect_x))
                except:
                    raise ValueError('Could not process line %i in %s: %s' % (iline, fn, line))
                if verbose:
                    print('Added bias potential nr. %i (Parabola with kappa = %.3f %s , q0 = %.3e %s and polynomial with the coefficients specified in the provided plumed file)' % (
                        len(biasses), kappa / parse_unit(kappa_unit), kappa_unit, q0 / parse_unit(q0_unit), q0_unit))

            fn_traj = default_cv_directory+fn_U
            data = np.loadtxt(fn_traj)
            old: trajectories.append(data[start:end:stride, 1])
            if end == -1:
                trajectories.append(data[start::stride, 1])
            else:
                trajectories.append(data[start:end:stride, 1])
            if verbose:
                print('Read corresponding trajectory data from %s' % fn_traj)   
    return temp, biasses, trajectories
