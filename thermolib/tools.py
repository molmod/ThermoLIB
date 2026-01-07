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


from re import I
from .units import *
from .constants import *

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import h5py as h5
from scipy.optimize import curve_fit
from inspect import signature
import matplotlib.pyplot as pp
import sys, os
import time

__all__ = [
    'format_scientific', 'h5_read_dataset',
    'integrate', 'integrate2d', 'interpolate_surface_2d', 'recollect_surface_2d', 'rolling_average',
    'read_wham_input', 'extract_polynomial_bias_info', 'plot_histograms_1d', 'plot_histograms_overlap_1d',
    'blav', 'corrtime_from_acf', 'decorrelate', 'multivariate_normal',
    'invert_fisher_to_covariance', 'fisher_matrix_mle_probdens',
]

#Miscellaneous utility routines

def format_scientific(x, prec=3, latex=True):
    """
        Format a numerical value in scientific notation with optional precision and LaTeX formatting.

        :param x: The numerical value to be formatted in scientific notation.
        :type x: float

        :param prec: (optional, default=3) The precision of the scientific notation, i.e., the number of decimal places to be displayed.
        :type prec: int

        :param latex: (optional, default=True) If set to ``True``, the function will return a LaTeX-formatted string; otherwise, it will return a regular string.
        :type latex: bool

        :returns: A formatted string representing the input value in scientific notation.
        :rtype: str
    """
    if np.isnan(x):
        return r'nan'
    if isinstance(x, np.ma.MaskedArray):
        x = x.item()
        
    a, b = ('{:.%iE}' %prec).format(x).split('E')
    if latex:
        return r'$%s\cdot 10^{%s}$' %(a,b)
    else:
        return '%s 10^%s' %(a, b)

def h5_read_dataset(fn, dset):
    """
        Read a dataset from an HDF5 file and return the data as a NumPy array.

        :param fn: The filename (including the path) of the HDF5 file.
        :type fn: str

        :param dset: The name of the dataset within the HDF5 file to be read.
        :type dset: str

        :returns: A NumPy array containing the data from the specified dataset.
        :rtype: numpy.ndarray
    """
    with h5.File(fn, mode = 'r') as f:
        data = np.array(f[dset])
    return data

#Routines related to integration, interpolation and recollection

def integrate(xs, ys, yerrs=None):
    '''
        Perform numerical integration of a dataset using the trapezoidal rule and return the integrated value. Optionally, analytically compute the error in the integration if uncertainties (`yerrs`) are provided (uncorrelated error bars are assumed).

        :param xs: A list or NumPy array representing the x-values of the dataset.
        :type xs: list or numpy.ndarray

        :param ys: A list or NumPy array representing the y-values of the dataset.
        :type ys: list or numpy.ndarray

        :param yerrs: (optional) A list or NumPy array representing the uncertainties in the y-values. If not provided (default), only the integration result is returned.
        :type yerrs: list or numpy.ndarray

        :returns: If `yerrs` is not provided, the integrated value of the dataset using the trapezoidal rule. If `yerrs` is provided, a tuple containing the integrated value and the error in the integration.
        :rtype: float or tuple
        '''
    assert len(xs)==len(ys)
    result = 0.0
    if yerrs is not None:
        error2 = 0.0
    for i in range(len(xs)-1):
        y = 0.5*ys[i]+0.5*ys[i+1]
        dx = xs[i+1]-xs[i]
        result += y*dx
        if yerrs is not None:
            tmp = (yerrs[i]**2+yerrs[i+1]**2)*(dx/2)**2
            if not np.isnan(tmp):
                error2 += tmp
    if yerrs is None:
        return result
    else:
        return result, np.sqrt(error2)

def integrate2d(z,x=None,y=None,dx=1.,dy=1.):
    '''
        Perform numerical integration of a regularly spaced two-dimensional dataset using the midpoint rule and return the integrated value. The function allows the specification of grid spacings (`dx` and `dy`) or infers them from the provided coordinate arrays (`x` and `y`).

        :param z: A two-dimensional array representing the values of the dataset.
        :type z: numpy.ndarray

        :param x: (optional) The x-coordinate array. If provided, the function will infer the grid spacing `dx`.
        :type x: numpy.ndarray or None

        :param y: (optional) The y-coordinate array. If provided, the function will infer the grid spacing `dy`.
        :type y: numpy.ndarray or None

        :param dx: (optional, default=1.0) The grid spacing in the x-direction. Ignored if `x` is provided.
        :type dx: float

        :param dy: (optional, default=1.0) The grid spacing in the y-direction. Ignored if `y` is provided.
        :type dy: float

        :returns: The integrated value of the two-dimensional dataset using the midpoint rule.
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

def rolling_average(ys, width, yerrs=None):
    """
        Compute a rolling average of a given dataset, optionally accompanied by error bars. The rolling average is calculated by dividing the dataset into windows of a specified width and averaging the values within each window. Optionally also compute the error bar on the rolling average if the input ``yerrs`` is provided.

        :param ys: The input dataset for which the rolling average is calculated.
        :type ys: list or numpy.ndarray

        :param width: The width of the rolling average window. Must be an integer, and at least 2.
        :type width: int

        :param yerrs: (optional) Error bars associated with the input dataset. If provided, the function will compute corresponding error bars for the rolling averages using standard error propagation.
        :type yerrs: list or numpy.ndarray or None

        :returns: If `yerrs` is not provided, a NumPy array containing the rolling averages of the input dataset. If `yerrs` is provided, a tuple containing a NumPy array with the rolling averages and a NumPy array with the corresponding error bars.
        :rtype: numpy.ndarray or tuple
    """
    assert isinstance(width, int), "Rolling average width needs to be an integer"
    assert width>=2, "Rolling average width needs to be at least 2"
    if yerrs is not None:
        assert len(ys)==len(yerrs), "ys and its corresponding error bars yerrs should be of same length"
    newN = int(np.ceil(len(ys)/width))
    new_ys = np.zeros(newN)
    if yerrs is not None:
        new_yerrs = np.zeros(newN)
    for i in range(newN-1):
        new_ys[i] = ys[i*width:(i+1)*width].sum()/width
        if yerrs is not None:
            new_yerrs[i] = np.sqrt((yerrs[i*width:(i+1)*width]**2).sum())/width 
    new_ys[-1] = ys[(newN-1)*width:].mean()
    if yerrs is not None:
        vals = yerrs[(newN-1)*width:]
        new_yerrs[-1] = np.sqrt((vals**2).sum())/len(vals)
        return new_ys, new_yerrs
    else:
        return new_ys

def interpolate_between_points_2D(datapoints, evalpoint, method='SVDLinear'):
    '''
        Routine to perform a 2D interpolation to estimate the z-value at the given evalpoint :math:`(x,y)` based on the given datapoints :math:`[(x_0,y_0,z_0),(x_1,y_1,z_1),...]`. Currently only one method of interpolation is supported: ``SVDLinear``, which is described in detail below:

        **SVDLinear**

        Solve the set of equations 
        
        .. math:
        
            z_0 &= a_0*x_0 + a_1*y_0 + a_2 \\\\
            z_1 &= a_0*x_1 + a_1*y_1 + a_2 \\\\
            z_2 &= a_0*x_2 + a_1*y_2 + a_2 \\\\
            \\ldots
        
        Herein, :math:`(x_i,y_i,z_i)` are the datapoints given in the input arguments. In matrix notation this becomes:

        .. math:

            \\boldsymbol Z = \\boldsymbol X \\cdot \\boldsymbol A
        
        with:
        
        .. math:

            Z &= \\left(\\begin{array}{c}
                z_0 \\\\
                z_1 \\\\
                z_2 \\\\
                \\vdots
            \\end{array}\\right) \\\\
            X &= \\left(\\begin{array}{ccc}
                x_0 & y_0 & 1 \\\\
                x_1 & y_1 & 1 \\\\
                x_2 & y_2 & 1 \\\\
                \\vdots & \\vdots & \\vdtos \\\\
            \\end{array}\\right)\\\\
            A &= \\left(\\begin{array}{c}
                a_0 \\\\
                a_1 \\\\
                a_2 \\\\
            \\end{array}\\right)

        The fitting is performed in a Least-Squares way using Singular Value Decomposition using the numpy.linalg.lstsq routine.

        :param datapoints: array containing the data points in the form [(x0,y0,z0),(x1,y1,z1),...]. If method is QuasiLinear, there should be 4 points (i.e. n=4).
        :type datapoints: np.ndarray[shape=(n,3)]

        :param evalpoint: (x,y) value of the point at which we want to interpolate the z-value
        :type evalpoint: np.ndarray[shape=(2,)]

        :param method: Specify which method to use for interpolation. Should be either 'QuasiLinear' or 'SVDLinear'. See documentation above for more details.
        :type method: str, optional, default='SVDLinear'
    '''
    if method.lower()=='quasilinear':
        #construct the matrixed D, X and Z
        #D = np.diag([1,1,1,0,0,0]) #not really required as it is easily encoded in F directly below
        X = np.ones([4,6], dtype=float)
        X[:,0] = datapoints[:,0]**2
        X[:,1] = datapoints[:,0]*datapoints[:,0]
        X[:,2] = datapoints[:,1]**2
        X[:,3] = datapoints[:,0]
        X[:,4] = datapoints[:,1]
        Z = datapoints[:,2]
        #construct block matrices E and F
        E = np.diag([1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        #E[:6,:6] = D #encoded directly in expression above
        E[:6,6:] = X.T
        E[6:,:6] = X
        F = np.zeros(10, dtype=float)
        F[6:] = Z
        #solve E*P=F for P
        assert abs(np.linalg.det(E))>1e-12, 'Could not interpolate, determinant of matrix E (=%.3e) is to small!' %(np.linalg.det(E))
        P = np.linalg.solve(E,F)
        #check if solution satifies the datapoints
        dev = Z-np.dot(X, P[:6])
        for i in range(4):
            assert abs(dev[i])<1e-12, 'Obtained solution did not satisfy datapoint %i: zi-f(xi,yi)=%.3e' %(i,dev[i])
        #define function to compute z value of interpolation at given evalpoint
        def interpol(x,y):
            return P[0]*x**2 + P[1]*x*y + P[2]*y**2 + P[3]*x + P[4]*y + P[5]
    elif method.lower()=='svdlinear':
        Z = datapoints[:,2]
        X = np.ones([len(datapoints),3], dtype=float)
        X[:,0:2] = datapoints[:,0:2]
        A, res, rank, svals = np.linalg.lstsq(X,Z, rcond=1e-6)
        #define function to compute z value of interpolation at given evalpoint
        def interpol(x,y):
            return A[0]*x+A[1]*y+A[2]
    else:
        raise ValueError("Interpolation method %s not supported, should be either 'QuasiLinear' or 'SVDLinear'." %method)
    return interpol(evalpoint[0],evalpoint[1])

def interpolate_surface_2d(cv1s, cv2s, fs, interpolation_depth=3, verbose=False):
    '''
        Routine to perform interpolation of F(CV1,CV2) defined by cv1s,cv2,fs grid. For all grid points where fs is np.nan, do interpolation using its right/left/up/down neighbors that are not np.nan. This routine will detect which grid points have fs=np.nan, search for neighbours valid for interpolation, collect there f value and parse to the :py:meth:`interpolate_between_points_2D <thermolib.tools.interpolate_between_points_2D>`` routine to do the actual interpolation.

        :param interpolation_depth: when interpolating at a certain bin from the neighbouring bins, go at max a number of neighbours far equal to interpolation_depth in each direction to try and find free energy data that is not np.nan
        :type interpolation_depth: int, optional, default=3
    '''
    #some init
    fs_new = fs.copy()
    interpolated = []
    not_found = []
    #find np.nans in fs and compute interpolated values
    for (k,l), f in np.ndenumerate(fs):
        if np.isnan(f):
            left = k
            while np.isnan(fs[left,l]):
                left -= 1
                if left<max(0,k-interpolation_depth):
                    left = None
                    break
            right = k
            while np.isnan(fs[right,l]):
                right += 1
                if min(fs.shape[0]-1,k+interpolation_depth)<right:
                    right = None
                    break
            down = l
            while np.isnan(fs[k,down]):
                down -= 1
                if down<max(0,l-interpolation_depth):
                    down = None
                    break
            up = l
            while np.isnan(fs[k,up]):
                up += 1
                if min(fs.shape[1]-1,l+interpolation_depth)<up:
                    up = None
                    break

            if not (left is None or right is None or up is None or down is None):
                datapoints = np.array([
                    [cv1s[left ], cv2s[l   ], fs[left ,l   ]], #left point
                    [cv1s[right], cv2s[l   ], fs[right,l   ]], #right point
                    [cv1s[k    ], cv2s[down], fs[k    ,down]], #down point
                    [cv1s[k    ], cv2s[  up], fs[k    ,  up]], #up point
                ])
                evalpoint = np.array([cv1s[k], cv2s[l]])
                f_int = interpolate_between_points_2D(datapoints, evalpoint)
                interpolated.append([k, l, f_int])
            else:
                if verbose:
                    print('WARNING: could not interpolate for (Q1[%i],Q2[%i]) as no left/right/up/down bound was found for interpolation' %(k,l))
                not_found.append((k,l))
    #set interpolated values in fs_new (needs to be done afterwards to avoid interpolated values affecting the interpolation of the next bins)
    for k, l, f_int in interpolated:
        fs_new[k,l] = f_int
    return fs_new

def recollect_surface_2d(cv1s_old, cv2s_old, fs_old, q1s_new, q2s_new):
    CV1s, CV2s = np.meshgrid(cv1s_old, cv2s_old, indexing='ij')
    fs_new = np.zeros([len(q1s_new),len(q2s_new)])*np.nan
    for k in range(len(q1s_new)-1):
        if k==0:
            lower1 = -np.inf
            upper1 = 0.5*(q1s_new[0]+q1s_new[1])
        elif k==(len(q1s_new)-1):
            lower1 = 0.5*(q1s_new[-2]+q1s_new[-1])
            upper1 = np.inf
        else:
            lower1 = 0.5*(q1s_new[k-1]+q1s_new[k  ])
            upper1 = 0.5*(q1s_new[k  ]+q1s_new[k+1])
        
        for l in range(len(q2s_new)-1):
            if l==0:
                lower2 = -np.inf
                upper2 = 0.5*(q2s_new[0]+q2s_new[1])
            elif l==(len(q2s_new)-1):
                lower2 = 0.5*(q2s_new[-2]+q2s_new[-1])
                upper2 = np.inf
            else:
                lower2 = 0.5*(q2s_new[l-1]+q2s_new[l  ])
                upper2 = 0.5*(q2s_new[l  ]+q2s_new[l+1])

            Is, Js = np.where((lower1<=CV1s)*(CV1s<upper1)*(lower2<=CV2s)*(CV2s<upper2))
            num, sum = 0, 0
            for i,j in zip(Is, Js):
                if not np.isnan(fs_old[i,j]):
                    sum += fs_old[i,j]
                    num += 1
            if num>0:
                fs_new[k,l] = sum/num
    return fs_new


#Routines for reading WHAM input

def read_wham_input(fn: str, trajectory_readers, trajectory_path_templates, bias_potential: str='None', q01_unit: str='au', kappa1_unit: str='au', q02_unit: str='au', kappa2_unit: str='au', inverse_cv1: bool=False, inverse_cv2: bool=False, additional_bias=None, additional_bias_dimension: str='cv1', skip_bias_names=[], verbose: bool=False):
    '''
        Read a WHAM input file (metadata file) to read the simulation temperature, trajectories (CV samples) and bias potentials.

        :param fn: The path to the WHAM input file, i.e. the metadata file. This file should be in the following format:

            .. code-block:: python

                T = XXXK
                NAME_1 Q0_1_1 KAPPA_1_1 [Q0_2_1 KAPPA_2_1 [...]]
                NAME_2 Q0_1_2 KAPPA_1_2 [Q0_2_2 KAPPA_2_2 [...]]
                NAME_3 Q0_1_3 KAPPA_1_3 [Q0_2_3 KAPPA_2_3 [...]]
                ...
            
            where NAME_i is a label for the i-th simulation that will be used to identify the location of its trajectory file (see parameter ``trajectory_path_templates`` below), (Q0_j_i, KAPPA_j_i) is the Q0 and KAPPA value of the j-th CV in the i-th bias potential (.i.e. of the i-th biased simulation). As a first example, a line ``W0 0.5 100`` would indicate a simulation labeled W0 with a bias applied along a single CV (1D bias) centered around CV=0.5 and with force constant 100 (the units of which can be further specified in the ``q01_unit`` and ``kappa1_unit`` arguments). As a second example, a line ``sim3 0.5 100 -1.0 150`` would indicate a simulation labeled sim3 with a bias applied along two CVs (2D bias) centered around (CV1,CV2)=(0.5,-1.0) and with force constant 100 along CV1 and 150 along CV2.

        :type fn: str

        :param trajectory_readers: implements how to read the CV values from the trajectory file of a simulation. If a list of readers is given, each reader should have its corresponding trajectory_path_template defined in the list of trajectory_path_templates. For information on how these trajectory files are determined, see description of the parameter ``trajectory_path_templates``.
        :type trajectory_readers: instance (or list of instances) of the :py:class:`CVComputer <thermolib.thermodynamics.trajectory.CVComputer>`, :py:class:`HDF5Reader <thermolib.thermodynamics.trajectory.HDF5Reader>`, :py:class:`ColVarReader <thermolib.thermodynamics.trajectory.ColVarReader>`, :py:class:`ASEExtendedXYZReader <thermolib.thermodynamics.trajectory.ASEExtendedXYZReader>` classes

        :param trajectory_path_templates: template (or list of templates) for defining the path (relative to the directory containing ``fn``) to the trajectory file corresponding to each simulation. Such template argument should be a string containing a single '%s' substring which gets replaced with the label of the simulation defined in fn (i.e. NAME_j in the code block above). For example, if trajectory_path_templates is given by ``trajectories/%s/colvar``, then the trajectory for the simulation labeled NAME_2 in fn is located at ``trajectories/NAME_2/colvar`` relative to the directory containing fn. If a list of templates is given, each template corresponds to a given trajectory reader defined in the ``trajectory_readers`` argument.
        :type trajectory_path_templates: str or list of strings, optional, default='%s'

        :param bias_potential: The type of bias potential, currently allowed values are 'Parabola1D', 'Parabola2D', or 'None'.

                * **Parabola1D** -- harmonic bias of the form 0.5*kappa1*(q1-q01)**2
                * **Parabola2D** -- harmonic bias of the form 0.5*kappa1*(q1-q01)**2 + 0.5*kappa2*(q2-q02)**2

        :type bias_potential: str, optional, allowed_values=['Parabola1D','Parabola2D','None'], default='None'

        :param q01_unit: The unit for the q01 value for each bias potential, q01 corresponds to the minimum in CV (or CV1 in case of a 2D bias) of the harmonic bias potential.
        :type q01_unit: str, optional, default='au'

        :param kappa1_unit: The unit for kappa1 value for each bias potential, kappa1 corresponds to the force constant along CV (or CV1 in case of a 2D bias) of the harmonic bias potential.
        :type kappa1_unit: str, optional, default='au'

        :param q02_unit: The unit for the q02 value for each bias potential, q02 corresponds to the minimum in CV2 of the SD harmonic bias potential. This argument is ignored in case of a 1D bias.
        :type q02_unit: str, optional, default='au'

        :param kappa2_unit: The unit for kappa2 value for each bias potential, kappa2 corresponds to the force constant along CV2 of the harmonic bias potential. This argument is ignored in case of a 1D bias.
        :type kappa2_unit: str, optional, default='au'

        :param inverse_cv1: If `True`, the CV1-axis will be inverted prior to bias evaluation. WARNING: the rest value parameter q01 of the potential will not be multiplied with -1!
        :type inverse_cv1: bool, optional, default=False

        :param inverse_cv2: If `True`, the CV2-axis will be inverted prior to bias evaluation. WARNING: the rest value parameter q01 of the potential will not be multiplied with -1! This argument is ignored in case of a 2D bias.
        :type inverse_cv2: bool, optional, default=False

        :param additional_bias: Additional bias potential to be added for each simulation on top of the  bias potential defined in fn
        :type additional_bias: object or None, optional, default=None

        :param additional_bias_dimension: The dimension in which the additional bias potential operates ('cv1'/'q1' or 'cv2'/'q2').
        :type additional_bias_dimension: str, optional, default='cv1'

        :param skip_bias_names: A list of bias names to be skipped during processing.
        :type skip_bias_names: list, optional, default=[]

        :param verbose: If `True`, print additional information during the reading process.
        :type verbose: bool, optional, default=False

        :returns: A tuple containing temperature, a list of bias potentials, and a list of trajectories.
        :rtype: tuple
    '''
    from thermolib.thermodynamics.bias import Parabola1D, Parabola2D, MultipleBiasses1D, MultipleBiasses2D
    #some argument dressing
    if not isinstance(trajectory_readers, list) and not isinstance(trajectory_path_templates, list):
        trajectory_readers = [trajectory_readers]
        trajectory_path_templates = [trajectory_path_templates]
    else:
        assert len(trajectory_readers)==len(trajectory_path_templates), 'Trajectory_readers and trajectory_path_templates need to be lists of matching length!'
    #initialize the three properties we need to extract
    temp = None
    biasses = []
    trajectories = []
    #determine number of CVS to be extracted
    ncvs = sum([reader.ncvs for reader in trajectory_readers])
    #iterate over lines in fn and extract temp, biasses and trajectories
    root = os.path.split(fn)[0]
    with open(fn, 'r') as f:
        iline = 0
        for line in f.readlines():
            line = line.rstrip('\n')
            words = line.split()
            if line.startswith('#'):
                continue
            elif skip_bias_names is not None and words[0] in skip_bias_names:
                if verbose:
                    print('Line %i (corresponding to bias %s) skipped in wham input file by user specification of skip_bias_names' %(iline, words[0]))
            elif line.startswith('T'):
                temp = float(line.split('=')[1].rstrip('K'))
                if verbose:
                    print('Temperature set at %f' %temp)
            elif len(words)==3: #1D bias
                name = words[0]
                q0 = float(words[1])*parse_unit(q01_unit)
                kappa = float(words[2])*parse_unit(kappa1_unit)
                #read trajectory cv data
                icv = 0
                nsamples = None
                for trajectory_reader, trajectory_path_template in zip(trajectory_readers, trajectory_path_templates):
                    fn_traj = os.path.join(root, trajectory_path_template %name)
                    if not os.path.isfile(fn_traj):
                        #TODO: check this part, if you only give a single trajectory reader and a certain umbrella does not have a valid trajectory (because e.g. simulation failed), it will now add the trajectory of the previous umbrella assuming the current umbrella is still valid (which is obviously WRONG)
                        print("WARNING: could not read trajectory file %s, SKIPPING!" %fn_traj)
                        continue
                    else:
                        data = trajectory_reader(fn_traj)
                        if nsamples is None:
                            nsamples = len(data)
                            trajdata = np.zeros([nsamples, ncvs])
                        else:
                            assert nsamples==len(data), 'Various readers do not have consistent number of samples: %i<==>%i' %(nsamples,len(data))
                        if trajectory_reader.ncvs==1:
                            trajdata[:,icv] = data
                        else:
                            trajdata[:,icv:icv+trajectory_reader.ncvs] = data
                        icv += trajectory_reader.ncvs
                if ncvs==1:
                    trajectories.append(trajdata[:,0])
                else:
                    trajectories.append(trajdata)
                #set bias
                if bias_potential.lower() in ['parabola1d', 'none']:
                    bias = Parabola1D(name, q0, kappa, inverse_cv=inverse_cv1)
                    if additional_bias is not None:
                        bias = MultipleBiasses1D([bias, additional_bias])
                    biasses.append(bias)
                    if verbose:
                        print('Added bias %s' %bias.print())
                elif bias_potential.lower() in ['parabola2d']:
                    #this is usefull in the case one want to perform 2D WHAM (for 2D FES) but when only 1D bias potentials were applied in terms of the first CV
                    bias = Parabola2D(name, q0, 0.0, kappa, 0.0, inverse_cv1=inverse_cv1)
                    if additional_bias is not None:
                        bias = MultipleBiasses2D([bias, additional_bias], additional_bias_dimension=additional_bias_dimension)
                    biasses.append(bias)
                    if verbose:
                        print('Added bias %s' %bias.print() + ' The 1D bias was redefined in 2D, the force constant in the second CV was set to zero.')
                else:
                    raise NotImplementedError('Bias potential of type %s not implemented.' %(bias_potential))
            elif len(words)==5: #2D bias
                name = words[0]
                q01 = float(words[1])*parse_unit(q01_unit)
                q02 = float(words[2])*parse_unit(q02_unit)
                kappa1 = float(words[3])*parse_unit(kappa1_unit)
                kappa2 = float(words[4])*parse_unit(kappa2_unit)
                #read trajectory cv data
                icv = 0
                nsamples = None
                for trajectory_reader, trajectory_path_template in zip(trajectory_readers, trajectory_path_templates):
                    fn_traj = os.path.join(root, trajectory_path_template %name)
                    if not os.path.isfile(fn_traj):
                        print("WARNING: could not read trajectory file %s, SKIPPING!" %fn_traj)
                        continue
                    else:
                        data = trajectory_reader(fn_traj)
                        if nsamples is None:
                            nsamples = len(data)
                            trajdata = np.zeros([nsamples, ncvs])
                        else:
                            assert nsamples==len(data), 'Various readers do not have consistent number of samples: %i<==>%i' %(nsamples,len(data))
                        if trajectory_reader.ncvs==1:
                            trajdata[:,icv] = data
                        else:
                            trajdata[:,icv:icv+trajectory_reader.ncvs] = data
                        icv += trajectory_reader.ncvs
                if ncvs==1:
                    trajectories.append(trajdata[:,0])
                else:
                    trajectories.append(trajdata)
                #set bias
                if bias_potential.lower() in ['parabola2d', 'none']:
                    bias = Parabola2D(name, q01, q02, kappa1, kappa2, inverse_cv1=inverse_cv1, inverse_cv2=inverse_cv2)
                    if additional_bias is not None:
                        bias = MultipleBiasses2D([bias, additional_bias], additional_bias_dimension=additional_bias_dimension)
                    biasses.append(bias)
                    if verbose:
                        print('Added bias %s' %bias.print())
                else:
                    raise NotImplementedError('Bias potential of type %s not implemented.' %(bias_potential))
            else:
                raise ValueError('Could not process line %i in %s: %s' %(iline, fn, line))
            iline += 1
    if temp is None:
        print('WARNING: temperature could not be read from %s' %fn)
    return temp, biasses, trajectories

def extract_polynomial_bias_info(fn_plumed: str='plumed.dat'):
    '''
        Extracts polynomial bias coefficients from a PLUMED input file.

        :param fn_plumed: The filename of the PLUMED input file.
        :type fn_plumed: str, optional, default='plumed.dat'

        :return: A list of polynomial coefficients for the bias potential.
        :rtype: list of float
    '''
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

def plot_histograms_1d(trajs, bins=200, width=None, cv_unit='au', alpha=0.8):
    '''
        Routine to plot 1d histograms of all trajectories to check overlap. This is usefull to check the choice of umbrellas in an umbrella simulation.

        :param trajs: List of all trajectories for which the histogram needs to be plotted.
        :type trajs: list(np.ndarray)

        :param bins: Specification of the bins for histograming. Can either be an integer specifying the number of bins over all CV range of all simulations or a numpy array explicitly defining all bins (in atomic units). Either bins or width needs to be specified, but not both.
        :type bins: int or np.ndarray, optional, default=200

        :param width: Specification of the width of the bins to be used in the histogramming (in atomic units). Either bins or width needs to be specified, but not both.
        :type width: float, optional, default=None

        :param cv_unit: unit of the CV to be used in the plot
        :type cv_unit: str, optional, default='au'

        :param alpha: transparancy value of each histogram to allow visualizing overlap
        :type alpha: float, optional, default=0.8
    '''
    #process bins argument
    assert not (bins is None and width is None), 'One out of the two arguments bins and width must be specified!'
    assert bins is None or width is None, 'Only one out of the two arguments bins and width can be specified!'
    if bins is not None:
        if isinstance(bins, int):
            min = np.array([traj.min() for traj in trajs]).min()
            max = np.array([traj.max() for traj in trajs]).max()
            bins = np.linspace(min, max, num=bins)
        else:
            assert isinstance(bins, np.ndarray), 'Invalid type for argument bins, should be integer (representing number of bins) or np.ndarray representing all bin edges'
        width = (bins[1:]-bins[:-1]).max()
    #construct and plot all histograms
    pp.clf()
    for traj in trajs:
        if bins is not None:
            current_bins = bins.copy()
        else:
            min_val = np.floor(traj.min()/width)*width
            max_val = np.ceil(traj.max()/width)*width
            current_bins = np.arange(min_val, max_val, width)
        hist, bin_edges = np.histogram(traj, bins=current_bins, density=True)
        bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:])
        unit_scale = parse_unit(cv_unit)
        pp.bar(bin_centers/unit_scale, hist, width=width/unit_scale, alpha=alpha)
    pp.xlabel('CV [%s]' %cv_unit, fontsize=16)
    pp.ylabel('Prob [a.u.]', fontsize=16)
    fig = pp.gcf()
    fig.set_size_inches([8,8])

def plot_histograms_overlap_1d(trajs, bins=200, sort=False, fn=None):
    '''
        Routine to compute and plot overlap between all pairs of given trajectories. The overlap metric used is the one suggested by Borgmans et al. [J. Chem. Theory Comput. 2023, 19, 9032-9048], which expressed overlap :math:`O_{ij}` between probability distribution :math:`p_i` and :math:`p_j` as:

        .. math::

            O_{ij} = \\int_{-\\infty}^{+\\infty} \\min(p_i(q),p_j(q))dq
        
        Three plots will be generated: 

            * **left**   -- CV mean for each trajectory. This should be monotonically increasing in order to adequately interpret the right pane. If this is not the case, set sort=True
            * **center** -- matrix of overlap between each pair of trajectories. In a well sorted set, this matrix should be dominant along the diagonal. All diagonal values are 1, as those elements represent overlap of a trajectory with itself.
            * **right** -- Overlap of each simulation with its right neighbor (as well as right second neighbor). In a good umbrella simulation set, this line should be consistingly high (e.g. above 0.33 as suggested by Borgmans et al. [J. Chem. Theory Comput. 2023, 19, 9032-9048]).


        :param trajs: List of all trajectories for which the histogram needs to be plotted.
        :type trajs: list(np.ndarray)

        :param bins: Specification of the bins in order to construct the probability distributions through histogramming. Can either be an integer specifying the number of bins over all CV range of all simulations or a numpy array explicitly defining all bins.
        :type bins: int or np.ndarray, optional, default=200

        :param sort: If set to True and it is detected that the trajectories are not sorted according to increasing mean, the routine will first do so. This is crucial in order to properly interpret the 'overlap with right neighbor' plot generated by this routine.
        :type sort: bool, optional, default=False

        :param fn: File name to write plot to. Plot will not be saved to file if fn=None
        :type fn: str, optional, default=None
    '''
    N = len(trajs)
    #process bins argument
    if isinstance(bins, int):
        min = np.array([traj.min() for traj in trajs]).min()
        max = np.array([traj.max() for traj in trajs]).max()
        bins = np.linspace(min, max, num=bins)
    else:
        assert isinstance(bins, np.ndarray), 'Invalid type for argument bins, should be integer (representing number of bins) or np.ndarray representing all bin edges'
    width = (bins[1:]-bins[:-1]).max()
    #construct and plot all histograms
    O = np.zeros([N,N], dtype=float)
    histograms = []
    means = []
    for traj in trajs:
        hist, bin_edges = np.histogram(traj, bins=bins, density=True)
        histograms.append(hist)
        bin_centers = 0.5*(bin_edges[0:-1]+bin_edges[1:])
        mean=(hist*bin_centers).sum()*width
        means.append(mean)
    #sort trajectories according to mean
    indices = sorted(range(N), key=lambda i: means[i])
    if not (np.array(indices)==np.arange(N)).all():
        if not sort:
            print('WARNING: Trajectories were not sorting according to increasing mean, to do so use sort=True')
        else:
            print('Trajectories were not sorting according to increasing mean, resorting as follows:')
            print(indices)
            histograms = [histograms[i] for i in indices]
            means      = [means[i]      for i in indices]
    #compte overlap
    for i, Hi in enumerate(histograms):
        for j, Hj in enumerate(histograms):
            O[i,j]=(np.minimum(Hi,Hj)).sum()*width
    Oright = np.array([O[i,i+1] for i in range(0,N-1)]+[np.nan])
    Oright2 = np.array([O[i,i+2] for i in range(0,N-2)]+[np.nan,np.nan])
    #make plot
    pp.clf()
    fig, axs = pp.subplots(1,3)
    axs[0].plot(means, 'bo-')
    axs[0].set_xlabel('Simulation', fontsize=16)
    axs[0].set_ylabel('CV mean [a.u.]', fontsize=16)
    axs[0].set_title('Trajectory means', fontsize=16)
    axs[1].imshow(O,cmap='Greys')
    axs[1].set_xlabel('Trajectory index', fontsize=16)
    axs[1].set_ylabel('Trajectory index', fontsize=16)
    axs[1].set_title('Overlap matrix', fontsize=16)
    axs[2].plot(Oright, 'r>-', label='with neighbor')
    axs[2].plot(Oright2, 'b>-', label='with second neighbor')
    #axs[2].axhline(0.5, color='k', linestyle='--', label='0.50 threshold')
    axs[2].axhline(0.33, color='k', linestyle=':', label='0.33 threshold')
    axs[2].legend(loc='best', fontsize=16, ncol=2)
    axs[2].set_ylim([0,1])
    axs[2].set_xlabel('Trajectory index', fontsize=16)
    axs[2].set_ylabel('Overlap', fontsize=16)
    axs[2].set_title('Overlap of each trajectory with right neighbor', fontsize=16)
    fig = pp.gcf()
    fig.set_size_inches([24,8])
    if fn is not None:
        pp.savefig(fn)


#Routines related to (de)correlation

def _next_pow_two(n):
    '''
        Utility routine required in _acf routine to find the clostest power of two that is larger than the argument n.
        (Taken from https://dfm.io/posts/autocorr/)

        :param n: lower bound for the sought after power of two
        :type n: float
    '''
    i = 1
    while i < n:
        i = i << 1
    return i

def _hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
        Utility routine required in acf routine to determine upper and lower envolopes.
        Taken from https://stackoverflow.com/questions/34235530/how-to-get-high-and-low-envelope-of-a-signal

        :param s: data signal from which to extract high and low envelopes
        :type s: 1D numpy array

        :param dmin: minimum size of chunks, use this if the size of the input signal is too big
        :type dmin: int, optional, default=1

        :param dmax: maximum size of chunks, use this if the size of the input signal is too big
        :type dmax:nt, optional, default=1

        :param split: if True, split the signal in half along its mean, might help to generate the envelope in some cases
        :type split: bool, optional, default=False

        :return: high/low envelope idx of input signal s
        :rtype: two lists of indices
    """
    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]
    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    return lmin,lmax

def _blav(data: np.ndarray, blocksizes=None, fitrange: list=[1, np.inf], model_function=None, **kwargs):
    '''
        Routine to implement block averaging in order to estimate the correlated error bar on the average of the data series as well as the corresponding integrated correlation time. This proceeds as follows:

        * Block the data in groups of a given blocksize B and compute the average for each block. These are denoted as the block averages.
        * Estimate the original data total average as the average of block averages, as well as the 'naive' error on this total average, i.e., assuming the block averages are uncorrelated.
        * This naive error bar is, in fact, not the true error because of correlations. However, upon increasing the block size, the correlations will diminish, and hence the naive error will converge towards the true error. Therefore, we vary the block size and fit a mathematical model on the naive error bars as a function of the block size. This model is defined by the argument ``model_function``.

        :param data: 1D array representing the data to be analyzed
        :type data: np.ndarray

        :param blocksizes: array of block sizes
        :type blocksizes: np.ndarray, optional, default=np.arange(1, len(data)+1, 1)

        :param fitrange: range of blocksizes to which the fit will be performed
        :type fitrange: list, optional, default=[1,np.inf]

        :param model_function: mathematical model for the naive error on block averages as a function of the block size. Should be a callable with as the first argument the block size and as the remaining arguments the model parameters to be fitted. The default model is given by:

        \[
        \Delta(B;\Delta_{\text{true}},\tau_{\text{int}}) = \Delta_{\text{true}} \cdot \sqrt{\frac{B}{B + \tau_{\text{int}} - 1}}
        \]

        :type model_function: callable, optional, default: see description

        :returns:
            - **errors** (*np.ndarray*) - The correlated error bars as a function of block sizes.
            - **true_error** (*float*) - The uncorrelated error on the sample mean.
            - **corrtime** (*float*) - The correlation time (in units of the timestep) of the original sample data.
            - **model_pars** (*list*) - Additional fitted parameters of the mathematical model for the naive error bars if any.
    '''
    #define model function for naive errors if not specified
    if model_function is None:
        def model_function(B, TE, tau):
            return TE*np.sqrt(B/(B+tau-1))
    #compute naive errors (=errors) on block averages
    errors = np.zeros(len(blocksizes))
    for i, blocksize in enumerate(blocksizes):
        nblocks = len(data)//blocksize
        blavs = np.zeros(nblocks)
        for iblock in range(nblocks):
            blavs[iblock] = data[iblock*blocksize:(iblock+1)*blocksize].mean()
        #the unbiased estimate on the variance of the block averages
        sigma = blavs.std(ddof=1)
        #the naive error on the mean of block averages 
        errors[i] = sigma/np.sqrt(nblocks)
    #fit the given mathematical model
    nparams = len(signature(model_function).parameters)-1
    lbounds, ubounds = [-np.inf,]*nparams, [np.inf,]*nparams
    lbounds[0] = 0.0 #lower bound of true error
    lbounds[1] = 1.0 #lower bound of correlation time
    mask = (fitrange[0]<=blocksizes) & (blocksizes<=fitrange[1])
    masked_errors = errors[mask]
    # scale the data to facilitate curve_fit convergence
    scale_factor = np.max(masked_errors)
    scaled_errors = masked_errors / scale_factor
    p0 = [scaled_errors[-1], 5.0]
    pars, pcov = curve_fit(model_function, blocksizes[mask], scaled_errors, bounds=(lbounds,ubounds), p0=p0)
    pars[0] *= scale_factor
    true_error, corrtime = pars[0], pars[1]
    model_fitted = lambda B: model_function(B, *list(pars))
    return errors, true_error, corrtime, model_fitted

def blav(data: np.ndarray, blocksizes=None, fitrange: list=[1, np.inf], model_function=None, plot: bool=False, fn_plot: str=None, plot_ylims: list=None, unit: str='au', **blav_kwargs):
    '''
        Wrapper routine around `_blav` to apply block averaging and estimate the correlated error bar as well as the corresponding integrated correlation time on the average of the given data series. For more details on the procedure as well as the meaning of the arguments `data`, `blocksizes`, `fitrange`, and `model_function`, see documentation of the routine :py:meth:`_wrap <thermolib.tools._blav>`.

        :param plot: If True, a plot of the samples and error estimates will be generated. Ignored if fn_plot is not None. Defaults to False.
        :type plot: bool, optional, default=False

        :param fn_plot: file name to which to write the plot. If None is given, either no plot is made (if plot=False) or the plot is not saved (if plot=True)
        :type fn_plot: str or None, optional, default=None

        :param plot_ylims: Limits for the y-axis in the error plot.
        :type plot_ylims: list or None, optional, default=None

        :param unit: unit in which to plot the data
        :type unit: str, optional, default='au'

        :returns: 
            - **mean** (*float*)- the sample mean.
            - **error** (*float*) - the error on the sample mean.
            - **corrtime** (*float*) - the correlation time (in units of the timestep) of the original sample data.
    '''
    #define blocksizes if not specified
    assert len(data)>100, 'I will not apply block averaging on data series with only 100 or less samples'
    if blocksizes is None:
        blocksizes = np.arange(1,int(len(data)/10)+1,1)
    errors, true_error, corrtime, model_fitted = _blav(data, blocksizes, fitrange, model_function=model_function, **blav_kwargs)
    #make plot
    if plot or fn_plot is not None:
        pp.clf()
        fig, axs = pp.subplots(nrows=1,ncols=2, squeeze=False)
        axs[0,0].plot(data/parse_unit(unit), 'bo', markersize=1)
        axs[0,0].axhline(y=data.mean()/parse_unit(unit), color='b', linestyle='--', linewidth=1)
        axs[0,0].set_title('Samples', fontsize=12)
        axs[0,0].set_xlabel('Time [timestep]')
        axs[0,0].set_ylabel('Sample [%s]' %unit)          
        axs[0,1].plot(blocksizes, errors/parse_unit(unit), color='b', linestyle='none', marker='o', markersize=1)
        axs[0,1].plot(blocksizes, model_fitted(blocksizes)/parse_unit(unit), color='r', linestyle='-', linewidth=1)
        #axs[0,1].axhline(y=true_error/parse_unit(unit), color='k', linestyle='--', linewidth=1)
        axs[0,1].set_title('Error of the estimate on the sample mean', fontsize=12)
        axs[0,1].set_xlabel('Block size [timestep]')
        axs[0,1].set_ylabel('Error [%s]' %unit)
        if plot_ylims is not None:
            axs[0,1].set_ylim(plot_ylims)
        fig.set_size_inches([12,6])
        if fn_plot is not None:
            pp.savefig(fn_plot, dpi=300)
        else:
            pp.show()
    return true_error, corrtime

def _acf(data, norm=True):
    '''
        Compute autocorrelation function (taken from https://dfm.io/posts/autocorr/)

        :param data: time series for which to construct autocorrelation function
        :type data: np.ndarray

        :param norm: if True, the acf will be normalized such that acf(0)=1
        :type norm: boolean, optional, default=True
    '''
    x = np.atleast_1d(data)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = _next_pow_two(len(x))
    #Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f*np.conjugate(f))[:len(x)].real
    acf /= 4*n
    #Normalize using the variance (i.e. autocorrelation with zero time lag)
    if norm:
        acf /= acf[0]
    return acf

def corrtime_from_acf(data: np.ndarray, nblocks=None, norm: bool=True, plot: bool=False, fn_plot=None, xlims=None, ylims: list=[-0.25, 1.05], n_nested_envelopes=1, legend=True, **curve_fit_kwargs):
    '''
        Routine to compute the integrated autocorrelation time as follows:

        - Compute the autocorrelation function (possibly after blocking data for noise suppression) using the routine :py:meth:`_acf <thermolib.tools._acf>`.
        - Optionally extract the upper envelope of the autocorrelation function to eliminate short time oscillations (when n_nested_envelopes>1).
        - Fit a single decaying exponential function of the form :math:`\\exp(-\\frac{t}{\\tau_exp})` to the acf (or its upper envelope) to extract the exponential correlation time.
        - Translate the exponential correlation time to the integrated correlation time as: :math:`\\tau_\\text{int} = 2\\tau_\\text{exp}`.

        :param data: 1D array representing the time series for which the correlation time needs to be computed
        :type data: np.ndarray

        :param nblocks: If not None, the data will be blocked (in a number of blocks given to nblocks) prior to constructing the acf. This is usefull for surpressing noise.
        :type nblocks: int or None, optional, default=None

        :param norm: If True, normalize the autocorrelation function such that acf(0)=1
        :type norm: bool, optional, default=True

        :param plot: If True, generate a plot of the acf and resulting fit. This parameter is ignored if fn_plot is not None.
        :type plot: bool, optional, default=False

        :param fn_plot: If not None, a plot will be made of the acf and resulting fit and stored to a file with the given file name
        :type fn_plot: str or None, optional, default=None

        :param xlims: Limits for the x-axis in the plot.
        :type xlims: list or None, optional, default=None

        :param ylims: Limits for the y-axis in the plot.
        :type ylims: list, optional, default=[-0.25, 1.05]

        :return: The integrated autocorrelation time
        :rtype: float
    '''
    acfs = None
    if nblocks is not None:
        bsize = int(len(data)/nblocks)
        acfs = np.zeros([nblocks, bsize])
        for iblock in range(nblocks):
            acfs[iblock,:]= _acf(data[iblock*bsize:(iblock+1)*bsize], norm=norm)[:bsize]
        acf = np.array(acfs).mean(axis=0)
    else:
        acf = _acf(data)
    #get envelope
    i_nested_envelope = 0
    upper_envelope_values = acf.copy()
    upper_envelope_indices = np.arange(len(acf))
    while i_nested_envelope<n_nested_envelopes:
        indices_lower, indices_upper = _hl_envelopes_idx(upper_envelope_values, dmin=1, dmax=1, split=False)
        if not indices_upper[0]==0: indices_upper = np.append([0],indices_upper)
        upper_envelope_values = upper_envelope_values[indices_upper]
        upper_envelope_indices = upper_envelope_indices[indices_upper]
        i_nested_envelope += 1
    #fit exponential to upper_envelope and extract integrated correlation time
    def function(t,tau):
        return np.exp(-t/tau)
    pars, pcovs = curve_fit(function, upper_envelope_indices, upper_envelope_values, **curve_fit_kwargs)
    def fitted_exp(t):
        return function(t,pars[0])
    corrtime = 2*pars[0] #pars[0] is the exp correlation time, corrtime is the integrated correlation time
    #Plot
    if plot or fn_plot is not None:
        #print('Fitted following function to ACF upper envoloppe: f(t)= shift + exp(-t/tau) with shift = %.3e and tau = %.3e' %(pars[1],pars[0]))
        fig = pp.gcf()
        if acfs is not None:
            for i, f in enumerate(acfs):
                pp.plot(f, label='iblock=%i'%i)
        pp.plot(acf, color='k',linewidth=1, label='acf')
        if n_nested_envelopes>0:
            pp.plot(upper_envelope_indices, upper_envelope_values, color='r',linewidth=2, label='acf - envelope')
        pp.plot(upper_envelope_indices, fitted_exp(upper_envelope_indices), color='r',linestyle='--',linewidth=2, label=r"Fit exp ($2\tau$=%.1f)" %corrtime)
        pp.title('Autocorrelation function', fontsize=14)
        pp.xlabel(r"Time delay t")
        pp.ylabel(r"acf(t)")
        if xlims is not None: pp.xlim(xlims)
        pp.ylim(ylims)
        if legend:
            pp.legend(loc='upper right')
        fig.set_size_inches([8,8])
        fig.tight_layout()
        if fn_plot is not None:
            pp.savefig(fn_plot)
        else:
            pp.show()
    return corrtime

def decorrelate(trajectories: list, method: str='acf', decorrelate_only=None, plot: bool=False, fn_plot=None, verbose: bool=False, return_decorrelated_trajectories: bool=False, **method_kwargs):
    '''
        Function to compute correlation times for a list of trajectories and optionally decorrelate the trajectories by averaging over a number of samples equal to the correlation time.

        :param trajectories: List of trajectories to be decorrelated.
        :type trajectories: list

        :param method: Method to compute correlation time, either 'acf' for fit to autocorrelation function or 'blav' for block averaging. Keyword arguments specific to the chosen method can be parsed to that method by specifyinng them at the end of the argument list. For example ``decorrelate(traj, method='acf', plot=True, n_nested_enveloppes=2, p0=[1000])`` will correctly parse the ``n_nested_enveloppes`` and ``p0`` parameters to the ``corrtime_from_acf`` routine.
        :type method: str, optional, default='acf'

        :param decorrelate_only: Index of the collective variable (CV) to be used for decorrelation when multiple CVs are present in the trajectories. If set to None, all CVs will be decorrelated simultaneously which generally leads to higher correlation times.
        :type decorrelate_only: int or None, optional, default=None

        :param plot: If True, generate a plot showing the correlation times for each trajectory and collective variable. Ignored if fn_plot is not None.
        :type plot: bool, optional, default=False

        :param fn_plot: File name to which to write the plot, defaults to None which means no plot is made (if plot=False) or the plot is not saved (if plot=True).
        :type fn_plot: str or None, optional

        :param verbose: If True, print additional information during computation.
        :type verbose: bool, optional, default=False

        :param return_decorrelated_trajectories: If True, also return the decorrelated trajectories which have been block averaged with a block size given by the correlation time
        :type return_decorrelated_trajectories: bool, optional, default=False

        :return: correlation time, optionally (if ``return_decorrelated_trajectories=True``) also the decorrelated trajectories are returned
        :rtype: np.ndarray or (if ``return_decorrelated_trajectories=True``) [np.ndarray, list]
    '''
    #determine the number of trajectories as well as cvs present in trajectory data
    ntrajs = len(trajectories)
    if len(trajectories[0].shape)==1:
        ncvs = 1
    else:
        ncvs = trajectories[0].shape[1]
    #set default function to compute correlation time to blav
    assert isinstance(method, str), 'Method argument should be string, either acf or blav'
    if method.lower()=='acf':
        def method_correlation_time(data):
            corrtime = corrtime_from_acf(data, **method_kwargs)
            if corrtime<1:
                print('Trajectory had corrtime smaller than 1, was set to 1!')
                return 1.0
            else:
                return corrtime
    elif method.lower()=='blav':
        def method_correlation_time(data):
            corrtime = blav(data, **method_kwargs)[1]
            if corrtime<1:
                print('Trajectory had corrtime smaller than 1, was set to 1!')
                return 1.0
            else:
                return corrtime
    else:
        raise ValueError('Method argument should be string, either acf or blav')
    #compute correlation times
    corrtimes_all = np.zeros([ntrajs,ncvs])
    corrtimes = np.zeros([ntrajs])
    for itraj, traj in enumerate(trajectories):
        if ncvs==1:
            corrtimes_all[itraj,0] = method_correlation_time(traj)
        else:
            for icv in range(ncvs):
                corrtimes_all[itraj,icv] = method_correlation_time(traj[:,icv])
        if decorrelate_only is None:
            corrtimes[itraj] = corrtimes_all[itraj,:].max()
        else:
            corrtimes[itraj] = corrtimes_all[itraj,decorrelate_only]
        if verbose:
            Nuncor = int(np.floor(len(traj)/max(1,corrtimes[itraj])))
            print('Processing trajectory %i/%i: %i samples, corr time = %.1f ==> %i uncorrelated samples' %(itraj+1,ntrajs,len(traj),corrtimes[itraj],Nuncor))
    #plot correlation times
    if plot or fn_plot is not None:
        pp.clf()
        fig, axs = pp.subplots(nrows=1, ncols=ncvs, squeeze=False)
        for icv in range(ncvs):
            axs[0,icv].plot(corrtimes_all[:,icv], marker='o', color='b')
            axs[0,icv].set_ylabel('Correlation time [-]', fontsize=14)
            axs[0,icv].set_xlabel('Trajectory number [-]', fontsize=14)
            axs[0,icv].set_title('Correlation time for %i-th CV' %icv)
        fig.tight_layout()
        fig.set_size_inches([8*ncvs,6])
        if fn_plot is not None:
            pp.savefig(fn_plot)
        else:
            pp.show()
    if return_decorrelated_trajectories:
        #decorrelate trajectory by averaging over a number of samples equal to the correlation time
        trajectories_decor = []
        for i, traj in enumerate(trajectories):
            bsize = int(np.ceil(corrtimes[i]))
            if bsize<=1:
                bsize = 1
                if verbose:
                    print('  estimated correlation time was smaller than 1, set blocksize to 1')
            nblocks = len(traj)//bsize
            if len(traj.shape)==1:
                new_traj = np.zeros(nblocks)
                for iblock in range(nblocks):
                    new_traj[iblock] = traj[iblock*bsize:(iblock+1)*bsize].mean()
                trajectories_decor.append(new_traj)
            else:
                new_traj = np.zeros([nblocks,traj.shape[1]])
                for index in range(traj.shape[1]):
                    for iblock in range(nblocks):
                        new_traj[iblock,index] = traj[iblock*bsize:(iblock+1)*bsize,index].mean()
                trajectories_decor.append(new_traj)
            if len(new_traj)==0:
                print('WARNING: trajectory of simulation nr %i has less then 1 uncorrelated simulation step!' %i)
        return corrtimes, trajectories_decor
    else:
        return corrtimes

def multivariate_normal(means: np.ndarray, covariance: np.ndarray, size=None):
    '''
        Wrapper around `numpy.random.multivariate_normal <https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html>`_ to handle `np.nan` columns in the covariance matrix. It also enforces symmetry on the covariance matrix to filter out non-symmetric noise. To generate the samples, Iit will first try the cholesky method in np.random.multivariate_normal as it is much faster. If that fails, the function tries again with the eigh setting and informs the user.

        :param means: Means of the multivariate normal distribution.
        :type means: np.ndarray

        :param covariance: Covariance matrix of the multivariate normal distribution.
        :type covariance: np.ndarray

        :param size: Number of samples to generate. If None, a single sample is generated
        :type size: int, list, np.ndarray or None, optional, default=None

        :return: the generated samples from the multivariate normal distribution
        :rtype: np.ndarray
    '''
    #wrapper around np.random.multivariate_normal to deal with np.nan columns in the covariance matrix
    mask = np.ones(len(means), dtype=bool)
    #loop over diagonal elements of cov matrix and if it is nan, check if entire row and column is nan and remove it
    for i, val in enumerate(covariance.diagonal()):
        if np.isnan(val): #the std_tol criterium is to assure that the covariance matrix will be positive definit (i.e. hass no zero eigenvalues)
            assert np.isnan(covariance[i,:]).all(), 'Upon filtering np.nans from covariance matrix, found diagonal element %i that is nan, but not the entire row' %i
            assert np.isnan(covariance[:,i]).all(), 'Upon filtering np.nans from covariance matrix, found diagonal element %i that is nan, but not the entire column' %i
            mask[i] = 0
    mus = means[mask]
    cov = covariance[np.outer(mask,mask)].reshape([mask.sum(),mask.sum()])
    #enforce symmetric covariance matrix (filters out non-symmetric noise)
    cov += cov.T
    cov *= 0.5
    try:
        samples_cropped = np.random.default_rng().multivariate_normal(mus, cov, size=size, method='cholesky')
    except np.linalg.LinAlgError:
        print('WARNING: multivariate normal sampling failed using Cholesky decomposition, switching to method eigh.')
        samples_cropped = np.random.default_rng().multivariate_normal(mus, cov, size=size, method='eigh')
    if size is None:
        samples = np.zeros(len(means))*np.nan
        samples[mask] = samples_cropped
    else:
        samples = np.zeros([size,len(means)])*np.nan
        samples[:,mask] = samples_cropped
    return samples

# Routines related to computing and inverting the fisher information matrix

def invert_fisher_to_covariance(F: np.ndarray, ps: np.ndarray, threshold: float=0.0, verbose: bool=False):
    '''
        Inverts the Fisher information matrix to obtain the covariance matrix, handling specified thresholds.

        :param F: Fisher information matrix.
        :type F: np.ndarray

        :param ps: Array of probabilities used to apply the threshold.
        :type ps: np.ndarray

        :param threshold: Threshold value for removing columns and rows corresponding with a probability lower than the threshold from the Fisher information matrix.
        :type threshold: float, optional, default=0.0

        :param verbose: If `True`, print information about removed columns and rows.
        :type verbose: bool, optional, default=False

        :returns: Covariance matrix obtained by inverting the Fisher information matrix.
        :rtype: np.ndarray
    '''
    mask = np.ones(F.shape, dtype=bool)
    for index in range(F.shape[0]-1):
        if np.isnan(ps[index]) or ps[index]<=threshold:
            if verbose: print('      ps[%i]=%.3e: removed column and row' %(index,ps[index]))
            mask[index,:] = 0
            mask[:,index] = 0
    N_mask2 = len(F[mask])
    assert abs(int(np.sqrt(N_mask2))-np.sqrt(N_mask2))==0 #consistency check, sqrt(N_mask2) should be integer valued
    N_mask = int(np.sqrt(N_mask2))
    F_mask = F[mask].reshape([N_mask,N_mask])
    cov = np.zeros(F.shape)*np.nan
    cov[mask] = np.linalg.inv(F_mask).reshape(N_mask2)
    return cov

def fisher_matrix_mle_probdens(ps: np.ndarray, method: str='mle_f', verbose: bool=False):
    '''
        Computes the Fisher information matrix for the maximum likelihood estimation (MLE) of probability distribution parameters given in argument ps.

        :param ps: Probability histogram values
        :type ps: np.ndarray

        :param method: Method used for computing the Fisher information matrix. Options include 'mle_p', 'mle_p_cov', 'mle_f', and 'mle_f_cov'.
        :type method: str, optional, default='mle_f'

        :param verbose: If `True`, prints additional information when no Fisher information is found.
        :type verbose: bool, optional, default=False

        :return: Fisher information matrix computed for the maximum likelihood estimation of probability distribution parameters.
        :rtype: np.ndarray
    '''
    F = np.zeros([len(ps)+1,len(ps)+1])
    if not np.isnan(ps).all():
        if method in ['mle_p', 'mle_p_cov']:
            for index, p in enumerate(ps):
                if p>0:
                    F[index,index] = 1.0/p
                    F[index, -1] = 1.0
                    F[-1, index] = 1.0
        elif method in ['mle_f', 'mle_f_cov']:
            for index, p in enumerate(ps):
                F[index,index] = p
                F[index, -1] = -p
                F[-1, index] = -p
        else:
            raise NotImplementedError('Error estimation method %s not supported to compute Fisher matrix of mle of probability distribution!')
    elif verbose:
        print('      No Fisher information found!')
    return F