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
from scipy import interpolate
import matplotlib.pyplot as pp
import sys, os


__all__ = [
    'integrate', 'integrate2d', 'format_scientific',
    'trajectory_xyz_to_CV', 'blav', 'read_wham_input',
    'read_wham_input_2D'
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

def read_wham_input(fn, path_template_colvar_fns='%s', colvar_cv_column_index=1, kappa_unit='kjmol', q0_unit='au', start=0, end=-1, stride=1, bias_potential='Parabola1D', additional_bias=None, verbose=False):
    '''
        Read the input for a WHAM reconstruction of the free energy profile from a set of Umbrella Sampling simulations. The file specified by fn should have the following format:

        .. code-block:: python

            T = XXXK
            NAME1 Q1 KAPPA1
            NAME2 Q2 KAPPA2
            NAME3 Q3 KAPPA3
            ...

        when a line starts with a T, it is supposed to specify the temperature. If no temperature line is found, a temperature of None will be returned. All other lines define the bias potential for each simulation as a parabola centered around ``Qi`` and with force constant ``KAPPAi`` (the units used in this file can be specified the keyword arguments ``kappa_unit`` and ``q0_unit``). For each bias with name ``NAMEi`` defined in this file, there should be a colvar trajectory file accessible through the path given by the string 'path_template_colvar_fns %(NAMEi)'. For example, if path_template_colvar_fns is defined as 'trajectories/%s/COLVAR' and the wham input file contains the following lines:

        .. code-block:: python

            T = 300K
            Window1/r1 1.40 1000.0
            Window2/r2 1.45 1000.0
            Window3/r1 1.50 1000.0
            ...
        
        Then the colvar trajectory file of the first potential can be found through the path (relative to the wham input file) 'Window1/r1/COLVAR' and so on. These colvar files contain the trajectory of the relevant collective variable during the biased simulation. Finally, these trajectory files should be formatted as outputted by PLUMED:

        .. code-block:: python

            time_1 cv_value_1
            time_2 cv_value_2
            time_3 cv_value_3
            ...
        
        where the cv values again have a unit that can be specified by the keyword argument ``q0_unit``.

        :param fn: file name of the wham input file
        :type fn: str

        :param path_template_colvar_fns: Template for defining the path (relative to the directory containing the wham input file given by argument fn) to the colvar trajectory file corresponding to each bias. See documentation above for more details. This argument should be string containing a single '%s' substring. 
        :type path_template_colvar_fns: str. Defaults to '%s'

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

        :param additional_bias: A single additional bias that is added for each simulation on top of the simulation-specific biases. Defaults to None
        :type additional_bias: BiasPotential1D, optional

        :param verbose: increases verbosity if set to True, defaults to False
        :type verbose: bool, optional

        :return: temp, biasses, trajectories:

            * **temp** (float) -- temperature at which the simulations were performed
            * **biasses** (list of callables) -- list of bias potentials (defined as callable functions) for all Umbrella Sampling simulations
            * **trajectories** (list of np.ndarrays) -- list of trajectory data arrays containing the CV trajectory for all Umbrella Sampling simulations
    '''
    from thermolib.thermodynamics.bias import BiasPotential1D, Parabola1D, MultipleBiasses1D
    temp = None
    biasses = []
    trajectories = []
    root = '/'.join(fn.split('/')[:-1])
    if additional_bias is not None:
        assert isinstance(additional_bias, BiasPotential1D), 'Given additional bias should be member/child of the class BiasPotential1D, got %s' %(additional_bias.__class__.__name__)
    with open(fn, 'r') as f:
        iline = -1
        for line in f.readlines():
            iline += 1
            line = line.rstrip('\n')
            words = line.split()
            if line.startswith('#'):
                continue
            elif line.startswith('T'):
                temp = float(line.split('=')[1].rstrip('K'))
                if verbose:
                    print('Temperature set at %f' %temp)
            elif bias_potential in ['Parabola1D'] and len(words)==3:
                name = words[0]
                q0 = float(words[1])*parse_unit(q0_unit)
                kappa = float(words[2])*parse_unit(kappa_unit)
                fn_traj = os.path.join(root, path_template_colvar_fns %name)
                if not os.path.isfile(fn_traj):
                    print("WARNING: could not read trajectory file for bias with name %s, skipping line in wham input." %name)
                    continue
                bias = Parabola1D(name, q0, kappa)
                if additional_bias is not None:
                    bias = MultipleBiasses1D([bias, additional_bias])
                data = np.loadtxt(fn_traj)
                biasses.append(bias)
                if end==-1:
                    trajectories.append(data[start::stride,colvar_cv_column_index])
                else:
                    trajectories.append(data[start:end:stride,colvar_cv_column_index])
                if verbose:
                    print('Added bias %s' %bias.print())
                    print('Read corresponding trajectory data from %s' %fn_traj)
            elif bias_potential not in ['Parabola1D']:
                    raise ValueError('Bias potential definition not supported (yet).')
            else:
                raise ValueError('Could not process line %i in %s: %s' %(iline, fn, line))
    if temp is None:
        print('WARNING: temperature could not be read from %s' %fn)
    return temp, biasses, trajectories

def read_wham_input_2D(fn, path_template_colvar_fns='%s', kappa1_unit='kjmol', kappa2_unit='kjmol', q01_unit='au', q02_unit='au', start=0, end=-1, stride=1, bias_potential='Parabola2D', verbose=False):
    '''
        Read the input for a WHAM reconstruction of the 2D free energy surface from a set of Umbrella Sampling simulations. The file specified by fn should have the following format:

        .. code-block:: python

            T = XXXK
            NAME_1 Q01_1 Q02_1 KAPPA1_1 KAPPA1_1
            NAME_2 Q01_2 Q02_2 KAPPA1_2 KAPPA1_2
            NAME_3 Q01_3 Q02_3 KAPPA1_3 KAPPA1_3
            ...

        when a line starts with a T, it is supposed to specify the temperature. If no temperature line is found, a temperature of None will be returned. All other lines define the bias potential for each simulation as a parabola centered around (``Q01_i``,``Q02_i``) and with force constants (``KAPPA1_i``,``KAPPA2_i``) (the units used in this file can be specified the keyword arguments ``kappa1_unit``, ``kappa2_unit``, ``q01_unit`` and ``q02_unit``). For each bias with name ``NAME_i`` defined in this file, there should be a colvar trajectory file accessible through the path given by the string 'path_template_colvar_fns %(NAME_i)'. For example, if path_template_colvar_fns is defined as 'trajectories/%s/COLVAR' and the wham input file contains the following lines:

        .. code-block:: python

            T = 300K
            Window1/r1 1.40 -0.2 1000.0 1000.0
            Window2/r2 1.45 -0.2 1000.0 1000.0
            Window3/r1 1.50 -0.2 1000.0 1000.0
            ...
        
        Then the colvar trajectory file of the first potential can be found through the path (relative to the wham input file) 'Window1/r1/COLVAR' and so on. These colvar files contain the trajectory of the relevant collective variable during the biased simulation. Finally, these trajectory files should be formatted as outputted by PLUMED:

        .. code-block:: python

            time_1 cv1_value_1 cv2_value_1
            time_2 cv1_value_2 cv2_value_2
            time_3 cv1_value_3 cv2_value_3
            ...
        
        where the cv1 and cv2 values again have a unit that can be specified by the keyword arguments ``q01_unit`` and ``q02_unit`` respectively.

        :param fn: file name of the wham input file
        :type fn: str

        :param path_template_colvar_fns: Template for defining the path (relative to the directory containing the wham input file given by argument fn) to the colvar trajectory file corresponding to each bias. See documentation above for more details. This argument should be string containing a single '%s' substring. 
        :type path_template_colvar_fns: str. Defaults to '%s'

        :param kappa1_unit: unit used to express the CV1 force constant kappa1 in the wham input file, defaults to 'kjmol'
        :type kappa1_unit: str, optional

        :param kappa2_unit: unit used to express the CV2 force constant kappa1 in the wham input file, defaults to 'kjmol'
        :type kappa2_unit: str, optional

        :param q01_unit: unit used to express q01 in the wham input file as well as the cv values in the trajectory files, defaults to 'au'
        :type q01_unit: str, optional

        :param q02_unit: unit used to express q02 in the wham input file as well as the cv values in the trajectory files, defaults to 'au'
        :type q02_unit: str, optional
        
        :param stride: defines the sub sampling applied to the trajectory data to deal with correlations. For example a stride of 10 means only taking 1 in 10 samples and throw away 90% of the data. Defaults to 1 (i.e. no sub sampling).
        :type stride: int, optional

        :param start: defines the start point from which to take samples into account. This can be usefull for eliminating equilibration times as well as for taking various subsamples trajectories from the original data each starting at different timesteps. Defaults to 0.
        :type start: int, optional

        :param end: defines the end point to which to take samples into account. This can be usefull when it is desired to cut the original trajectory into blocks. Defaults to -1.
        :type end: int, optional

        :param bias_potential: mathematical form of the bias potential used, allowed values are

            * **parabola2D/harmonic2D** -- harmonic bias of the form 0.5*kappa1*(q1-q01)**2 + 0.5*kappa2*(q2-q02)**2

        defaults to parabola2D

        :type bias_potantial: str, optional

        :param verbose: increases verbosity if set to True, defaults to False
        :type verbose: bool, optional
        
        :raises ValueError: when a line in the wham input file cannot be interpreted

        :return: temp, biasses, trajectories:

            * **temp** (float) -- temperature at which the simulations were performed
            * **biasses** (list of callables) -- list of bias potentials (defined as callable functions) for all Umbrella Sampling simulations
            * **trajectories** (list of np.ndarrays) -- list of trajectory data arrays containing the CV trajectory for all Umbrella Sampling simulations
    '''
    from thermolib.thermodynamics.bias import Parabola2D
    temp = None
    biasses = []
    trajectories = []
    root = '/'.join(fn.split('/')[:-1])
    with open(fn, 'r') as f:
        iline = -1
        for line in f.readlines():
            iline += 1
            line = line.rstrip('\n')
            words = line.split()
            if line.startswith('#'):
                continue
            elif line.startswith('T'):
                temp = float(line.split('=')[1].rstrip('K'))
                if verbose:
                    print('Temperature set at %f' %temp)
            elif bias_potential in ['Parabola2D'] and len(words)==5:
                name = words[0]
                if verbose:
                    print('Processing bias %s' %name)
                q01 = float(words[1])*parse_unit(q01_unit)
                q02 = float(words[2])*parse_unit(q02_unit)
                kappa1 = float(words[3])*parse_unit(kappa1_unit)
                kappa2 = float(words[4])*parse_unit(kappa2_unit)
                fn_traj = os.path.join(root, path_template_colvar_fns %name)
                if not os.path.isfile(fn_traj):
                    print("  WARNING: could not read trajectory file for bias with name %s, skipping line in wham input." %name)
                    print('')
                    continue
                bias = Parabola2D(name, q01, q02, kappa1, kappa2)
                data = np.loadtxt(fn_traj)
                biasses.append(bias)
                if end==-1:
                    trajectories.append(data[start::stride,1:3])#COLVAR format: CV1 is second column and CV2 is third column
                else:
                    trajectories.append(data[start:end:stride,1:3])#COLVAR format: CV1 is second column and CV2 is third column
                if verbose:
                    print('  added %s' %bias.print())
                    print('  trajectory read from %s' %fn_traj)
                    print('')
            elif bias_potential not in ['Parabola2D']:
                    raise ValueError('Bias potential definition not supported (yet).')
            else:
                raise ValueError('Could not process line %i in %s: %s' %(iline, fn, line))
    if temp is None:
        print('WARNING: temperature could not be read from %s' %fn)
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

def read_wham_input_custom1(fn,temp,fn_plumed=None, kappa_unit='kjmol', q0_unit='au', start=0, end=-1, stride=1, bias_potential='poly_parabola',plumed_unit='kjmol',default_cv_directory='./', verbose=False,reflect_x=False):
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
    from thermolib.thermodynamics.bias import BiasPotential1D, Parabola1D, Polynomial1D, AddMultiplePotentials1D
    temp = float(temp)
    biasses = []
    trajectories = []
    if bias_potential.lower() in ['parabola1d', 'harmonic']:
        bias = 'harm'
    elif bias_potential.lower() in ['poly_parabola']:
        bias = 'poly_harm'
        poly_coef = extract_polynomial_bias_info(fn_plumed)
    elif bias_potential.lower() not in ['parabola1d', 'harmonic','poly_parabola']:
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
                    biasses.append(Parabola1D(fn_U, kappa, q0))
                except:
                    raise ValueError('Could not process line %i in %s: %s' % (iline, fn, line))
                if verbose:
                    print('Added bias potential nr. %i (Parabola with kappa = %.3f %s , q0 = %.3e %s)' % (
                    len(biasses), kappa / parse_unit(kappa_unit), kappa_unit, q0 / parse_unit(q0_unit), q0_unit))
            elif bias == 'poly_harm':
                try:
                    parabola = Parabola1D('Parabola', kappa, q0, inverse_cv=reflect_x)
                    poly = Polynomial1D('Polynomial', poly_coef, unit=plumed_unit, inverse_cv=reflect_x)
                    poly_parabola = AddMultiplePotentials1D(fn_U, [parabola,poly])
                    biasses.append(poly_parabola)
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

