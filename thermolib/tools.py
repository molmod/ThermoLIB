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


from re import I
from molmod.units import *
from molmod.constants import *
from molmod.io.xyz import XYZReader

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
    'integrate', 'integrate2d', 'rolling_average',
    'blav', 'corrtime_from_acf', 'decorrelate',
    'read_wham_input', 'multivariate_normal', 'invert_fisher_to_covariance', 'fisher_matrix_mle_probdens'
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

        :Example:

        .. code-block:: python

            import numpy as np

            # Example 1: Default precision and LaTeX formatting
            result1 = format_scientific(1.2345e-8)
            print(result1)  # Output: '$1.235\\cdot 10^{-8}$'

            # Example 2: Custom precision and regular string formatting
            result2 = format_scientific(3.14159e12, prec=2, latex=False)
            print(result2)  # Output: '3.14e+12'

        :Note:

        - If the input value is a `NaN` (Not a Number), the function returns the string 'nan'.
        - The scientific notation is represented in the format 'a * 10^b', where 'a' is the coefficient and 'b' is the exponent. The coefficient is formatted with the specified precision.
    """
    if np.isnan(x):
        return r'nan'
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

        :Example:

        .. code-block:: python

            # Example: Reading a dataset named 'temperature' from the file 'simulation_data.h5'
            data_array = h5_read_dataset('simulation_data.h5', 'temperature')

        :Dependencies:

        - This function relies on the `h5py` library, which is commonly used for working with HDF5 files in Python. Ensure that the library is installed before using this function.

        :Note:

        - The function uses the 'r' (read) mode when opening the HDF5 file (:code:`mode='r'`), indicating that the file will only be read and not modified.
        - It is assumed that the specified dataset (:code:`dset`) exists within the HDF5 file. If the dataset is not present, an error may occur.

    """
    with h5.File(fn, mode = 'r') as f:
        data = np.array(f[dset])
    return data

#Routines related to integration

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

        :Assertions:

        - The function includes an assertion to ensure that the lengths of `xs` and `ys` are equal.

        :Note:

        - The trapezoidal rule is used for numerical integration, assuming a piecewise linear interpolation between data points.
        - If uncertainties (`yerrs`) are provided, the error in the integration is computed (assuming uncorrelated `yerrs`) and returned along with the integrated value.

        :Caution:

        - The function assumes that the input dataset is well-behaved, and numerical issues may arise if the dataset has irregularities or extreme variations.
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

        :Note:

        - The midpoint rule is used for numerical integration, assuming a piecewise constant interpolation between data points.
        - If coordinate arrays (`x` and `y`) are provided, the function infers the grid spacings (`dx` and `dy`) based on the range of the coordinates.
        - The integrated value is computed using a weighted sum of the corner values (`s1`), edge values (`s2`), and interior values (`s3`).

        :Caution:

        - The function assumes that the input dataset is well-behaved, and numerical issues may arise if the dataset has irregularities or extreme variations.
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
        Compute a rolling average of a given dataset, optionally accompanied by error bars. The rolling average is calculated by dividing the dataset into windows of a specified width and averaging the values within each window.

        :param ys: The input dataset for which the rolling average is calculated.
        :type ys: list or numpy.ndarray

        :param width: The width of the rolling average window. Must be an integer, and at least 2.
        :type width: int

        :param yerrs: (optional) Error bars associated with the input dataset. If provided, the function will compute corresponding error bars for the rolling averages using standard error propagation.
        :type yerrs: list or numpy.ndarray or None

        :returns: If `yerrs` is not provided, a NumPy array containing the rolling averages of the input dataset. If `yerrs` is provided, a tuple containing a NumPy array with the rolling averages and a NumPy array with the corresponding error bars.
        :rtype: numpy.ndarray or tuple

        :Assertions:

        - The function includes assertions to ensure that the rolling average width is an integer and at least 2.
        - If error bars (`yerrs`) are provided, an assertion checks that the lengths of `ys` and `yerrs` are the same.
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

#Routines for reading WHAM input

def read_wham_input(fn: str, trajectory_readers, trajectory_path_templates, bias_potential: str='None', q01_unit: str='au', kappa1_unit: str='au', q02_unit: str='au', kappa2_unit: str='au', inverse_cv1: bool=False, inverse_cv2: bool=False, additional_bias=None, additional_bias_dimension: str='cv1', skip_bias_names=[], verbose: bool=False):
    '''
        Read a WHAM input file, extract relevant information, and initialize necessary objects for further analysis.

        :param fn: The path to the WHAM input file, i.e. the metadata file. This file should be in the following format:

            .. code-block:: python

                T = XXXK
                NAME_1 Q0_1_1 KAPPA_1_1 [Q0_2_1 KAPPA_2_1 [...]]
                NAME_2 Q0_1_2 KAPPA_1_2 [Q0_2_2 KAPPA_2_2 [...]]
                NAME_3 Q0_1_3 KAPPA_1_3 [Q0_2_3 KAPPA_2_3 [...]]
                ...
            
            where (Q0_i_j, KAPPA_i_j) is the Q0 and KAPPA value of the i-th CV in the j-th bias potential.

        :type fn: str

        :param trajectory_readers: an instance (or list of instances) of the TrajectoryReader class that implements how to read the CV values from the trajectory files. If a list of readers is given, each reader should have its corresponding trajectory_path_template defined in the list of trajectory_path_templates. For information on how these trajectory files are determined, see description of :param trajectory_path_templates:.
        :type trajectory_readers: TrajectoryReader or list of TrajectoryReader instances

        :param trajectory_path_templates: template (or list of templates) for defining the path (relative to the directory containing fn) to the trajectory file corresponding to each bias. Such template argument should be a string containing a single '%s' substring which gets replaced with the name of the biases defined in fn. For example, if trajectory_path_templates is given by 'trajectories/%s/colvar', then the trajectory for the simulation biased with the potential named NAME_2 in the file fn (see above) is given by the path 'trajectories/NAME_2/colvar' relative to the directory containing fn. Defaults to '%s'. If a list of templates is given, each template corresponds to a given trajectory reader defined in the trajectory_readers argument.
        :type trajectory_path_templates: str or list of strings, optional

        :param bias_potential: (optional, default='None') The type of bias potential, currently allowed values are 'Parabola1D', 'Parabola2D', or 'None'.

                * **Parabola1D** -- harmonic bias of the form 0.5*kappa1*(q1-q01)**2
                * **Parabola2D** -- harmonic bias of the form 0.5*kappa1*(q1-q01)**2 + 0.5*kappa2*(q2-q02)**2

        :type bias_potential: str

        :param q01_unit: (optional, default='au') The unit for the q01 value for each bias potential, q01 corresponds to the minimum in CV (or CV1 in case of a 2D bias) of the harmonic bias potential.
        :type q01_unit: str

        :param kappa1_unit: (optional, default='au') The unit for kappa1 value for each bias potential, kappa1 corresponds to the force constant along CV (or CV1 in case of a 2D bias) of the harmonic bias potential.
        :type kappa1_unit: str

        :param q02_unit: (optional, default='au') The unit for the q02 value for each bias potential, q02 corresponds to the minimum in CV2 of the SD harmonic bias potential. This argument is ignored in case of a 1D bias.
        :type q02_unit: str

        :param kappa2_unit: (optional, default='au') The unit for kappa2 value for each bias potential, kappa2 corresponds to the force constant along CV2 of the harmonic bias potential. This argument is ignored in case of a 1D bias.
        :type kappa2_unit: str

        :param inverse_cv1: (optional, default=False) If `True`, the CV1-axis will be inverted prior to bias evaluation. WARNING: the rest value parameter q01 of the potential will not be multiplied with -1!
        :type inverse_cv1: bool

        :param inverse_cv2: (optional, default=False) If `True`, the CV2-axis will be inverted prior to bias evaluation. WARNING: the rest value parameter q01 of the potential will not be multiplied with -1! This argument is ignored in case of a 2D bias.
        :type inverse_cv2: bool

        :param additional_bias: (optional, default=None) Additional bias potential to be added to the primary bias potential.
        :type additional_bias: object or None

        :param additional_bias_dimension: (optional, default='cv1') The dimension in which the additional bias potential operates ('cv1' or 'cv2').
        :type additional_bias_dimension: str

        :param skip_bias_names: (optional, default=[]) A list of bias names to be skipped during processing.
        :type skip_bias_names: list

        :param verbose: (optional, default=False) If `True`, print additional information during the reading process.
        :type verbose: bool

        :returns: A tuple containing temperature, a list of bias potentials, and a list of trajectories.
        :rtype: tuple

        :Note:

        - The function supports 1D and 2D bias potentials, as well as the option to add an additional bias potential.
        - The units for collective variables (`q01_unit`, `kappa1_unit`, `q02_unit`, `kappa2_unit`) can be specified to ensure unit consistency.
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

        :param fn_plumed: (optiona, default='plumed.dat') The filename of the PLUMED input file.
        :type fn_plumed: str

        :return: A list of polynomial coefficients for the bias potential.
        :rtype: list of float
    '''
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

#Routines related to (de)correlation

def _next_pow_two(n):
    '''
        Utility routine required in _acf routine to find the clostest power of two that is larger than the argument.
        (Taken from https://dfm.io/posts/autocorr/)
    '''
    i = 1
    while i < n:
        i = i << 1
    return i

def _hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Utility routine required in acf routine to determine upper and lower envolopes.
    Taken from https://stackoverflow.com/questions/34235530/how-to-get-high-and-low-envelope-of-a-signal

    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
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

def _blav(data: np.ndarray, blocksizes: np.ndarray|None=None, fitrange: list=[1, np.inf], model_function=None):
    '''
        Routine to implement block averaging in order to estimate the correlated error bar on the average of the data series as well as the corresponding integrated correlation time. This proceeds as follows:

        * Block the data in groups of a given blocksize B and compute the average for each block. These are denoted as the block averages.
        * Estimate the original data total average as the average of block averages, as well as the 'naive' error on this total average, i.e., assuming the block averages are uncorrelated.
        * This naive error bar is, in fact, not the true error because of correlations. However, upon increasing the block size, the correlations will diminish, and hence the naive error will converge towards the true error. Therefore, we vary the block size and fit a mathematical model on the naive error bars as a function of the block size. This model is defined by the argument `model_function`.

        Parameters
        ----------

        :data: 1D array representing the data to be analyzed
        :type data: np.ndarray

        :blocksizes: array of block sizes, defaults to np.arange(1, len(data)+1, 1)
        :type blocksizes: np.ndarray, optional

        :model_function: mathematical model for the naive error on block averages as a function of the block size. Should be a callable with as the first argument the block size and as the remaining arguments the model parameters to be fitted. The default model is given by:

        \[
        \Delta(B;\Delta_{\text{true}},\tau_{\text{int}}) = \Delta_{\text{true}} \cdot \sqrt{\frac{B}{B + \tau_{\text{int}} - 1}}
        \]

        :type model_function: callable, optional

        :fitrange: range of blocksizes to which the fit will be performed
        :type fitrange: list, optional

        Returns
        -------

        * **errors** (*np.ndarray*): The correlated error bars as a function of block sizes.
        * **true_error** (*float*): The uncorrelated error on the sample mean.
        * **corrtime** (*float*): The correlation time (in units of the timestep) of the original sample data.
        * **model_pars** (*list*): Additional fitted parameters of the mathematical model for the naive error bars if any.

        Note
        ----

        - The default mathematical model for the naive error bars is given by:

        \[
        \Delta(B; \Delta_{\text{true}}, \tau_{\text{int}}) = \Delta_{\text{true}} \cdot \sqrt{\frac{B}{B + \tau_{\text{int}} - 1}}
        \]

        - The function uses curve fitting to estimate the parameters of the mathematical model for the naive error bars.
        - The correlation time (`corrtime`) represents the characteristic time scale of the correlations in the original sample data.
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
    pars, pcov = curve_fit(model_function, blocksizes[mask], errors[mask], bounds=(lbounds,ubounds))
    true_error, corrtime = pars[0], pars[1]
    model_fitted = lambda B: model_function(B, *list(pars))
    return errors, true_error, corrtime, model_fitted

def blav(data: np.ndarray, blocksizes: np.ndarray|None=None, fitrange: list=[1, np.inf], model_function=None, plot: bool=False, fn_plot: str|None=None, plot_ylims: list|None=None, unit: str='au'):
    '''
        Wrapper routine around `_blav` to apply block averaging and estimate the correlated error bar as well as the corresponding integrated correlation time on the average of the given data series. For more details on the procedure as well as the meaning of the arguments `data`, `blocksizes`, `fitrange`, and `model_function`, see documentation of the routine `_wrap`.

        Parameters
        ----------

        :param data: 1D array representing the data to be analyzed
        :type data: np.ndarray

        :param blocksizes: array of block sizes, defaults to np.arange(1, len(data)//10+1, 1)
        :type blocksizes: np.ndarray or None, optional

        :param fitrange: range of block sizes to which the fit will be performed, defaults to [1, np.inf]
        :type fitrange: list, optional

        :param model_function: mathematical model for the naive error on block averages as a function of the block size. Should be a callable with as the first argument the block size and as the remaining arguments the model parameters to be fitted. The default model is a square root model.
        :type model_function: callable, optional

        :param plot: If True, a plot of the samples and error estimates will be generated. Ignored if fn_plot is not None. Defaults to False.
        :type plot: bool, optional

        :param fn_plot: file name to which to write the plot, defaults to None which means either no plot is made (if plot=False) or the plot is not saved (if plot=True)
        :type fn_plot: str or None, optional

        :param plot_ylims: Limits for the y-axis in the error plot. Defaults to None.
        :type plot_ylims: list or None, optional

        :param unit: unit in which to plot the data, defaults to 'au'
        :type unit: str, optional

        Returns
        -------

        * **mean** (*float*): The sample mean.
        * **error** (*float*): The error on the sample mean.
        * **corrtime** (*float*): The correlation time (in units of the timestep) of the original sample data.

        Note
        ----

        - The function provides an option to generate a plot showing the samples and error estimates as a function of block size.
        - The `unit` parameter allows for specifying the unit in which to plot the data.
        - The y-axis limits for the error plot can be adjusted using the `plot_ylims` parameter.
    '''
    #define blocksizes if not specified
    assert len(data)>100, 'I will not apply block averaging on data series with only 100 or less samples'
    if blocksizes is None:
        blocksizes = np.arange(1,int(len(data)/10)+1,1)
    errors, true_error, corrtime, model_fitted = _blav(data/parse_unit(unit), blocksizes, fitrange, model_function=model_function)
    #make plot
    if plot or fn_plot is not None:
        pp.clf()
        fig, axs = pp.subplots(nrows=1,ncols=2, squeeze=False)
        axs[0,0].plot(data/parse_unit(unit), 'bo', markersize=1)
        axs[0,0].axhline(y=data.mean()/parse_unit(unit), color='b', linestyle='--', linewidth=1)
        axs[0,0].set_title('Samples', fontsize=12)
        axs[0,0].set_xlabel('Time [timestep]')
        axs[0,0].set_ylabel('Sample [%s]' %unit)          
        axs[0,1].plot(blocksizes, errors, color='b', linestyle='none', marker='o', markersize=1)
        axs[0,1].plot(blocksizes, model_fitted(blocksizes), color='r', linestyle='-', linewidth=1)
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
    return true_error*parse_unit(unit), corrtime

def _acf(data, norm=True):
    'Compute autocorrelation function (taken from https://dfm.io/posts/autocorr/)'
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

def corrtime_from_acf(data: np.ndarray, nblocks: int|None=None, norm: bool=True, plot: bool=False, fn_plot: str|None=None, xlims: list|None=None, ylims: list=[-0.25, 1.05]):
    '''
        Routine to compute the integrated autocorrelation time as follows:

        - Compute the autocorrelation function (possibly after blocking data for noise suppression) using the routine `_acf`.
        - Extract the upper envelope of the autocorrelation function to eliminate short time oscillations.
        - Fit a single decaying exponential function to the upper envelope to extract the exponential correlation time.
        - Translate the exponential correlation time to the integrated correlation time as: :math:`\tau_\text{int} = 2\tau_\text{exp} - 1`.

        Parameters
        ----------

        :param data: 1D array representing the data to be analyzed
        :type data: np.ndarray

        :param nblocks: Number of blocks for blocking data. Defaults to None.
        :type nblocks: int or None, optional

        :param norm: If True, normalize the autocorrelation function. Defaults to True.
        :type norm: bool, optional

        :param plot: If True, generate a plot showing the autocorrelation function, its average, the upper envelope, and the fitted exponential function. Ignored if fn_plot is not None. Defaults to False.
        :type plot: bool, optional

        :param fn_plot: File name to which to write the plot, defaults to None which means no plot is made.
        :type fn_plot: str or None, optional

        :param xlims: Limits for the x-axis in the plot. Defaults to None.
        :type xlims: list or None, optional

        :param ylims: Limits for the y-axis in the plot. Defaults to [-0.25, 1.05].
        :type ylims: list, optional

        Returns
        -------

        * **corrtime** (*float*): The integrated autocorrelation time.

        Note
        ----

        - The function provides an option to generate a plot showing the autocorrelation function, its average, the upper envelope, and the fitted exponential function.
        - The `nblocks` parameter allows for blocking data to suppress noise in the autocorrelation function.
        - The `norm` parameter controls whether the autocorrelation function should be normalized.
        - The `xlims` parameter allows for adjusting the limits of the x-axis in the plot.
        - The `ylims` parameter allows for adjusting the limits of the y-axis in the plot.
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
    lower_envelope_indices, upper_envelope_indices = _hl_envelopes_idx(acf, dmin=1, dmax=1, split=False)
    upper_envelope = acf[upper_envelope_indices]
    #fit exponential to upper_envelope and extract integrated correlation time
    def function(t,tau):
        return np.exp(-t/tau)
    pars, pcovs = curve_fit(function, upper_envelope_indices, upper_envelope)
    def fitted_exp(t):
        return function(t,pars[0])
    corrtime = 2*pars[0]-1 #pars[0] is the exp correlation time, corrtime is the integrated correlation time
    #Plot
    if plot or fn_plot is not None:
        fig = pp.gcf()
        if acfs is not None:
            for i, f in enumerate(acfs):
                pp.plot(f, label='iblock=%i'%i)
        pp.plot(acf, color='k',linewidth=1, label='avg')
        pp.plot(upper_envelope_indices, upper_envelope, color='r',linewidth=2, label='avg - envelope')
        pp.plot(upper_envelope_indices, fitted_exp(upper_envelope_indices), color='r',linestyle='--',linewidth=2, label='avg - fitted')
        pp.title('Autocorrelation function', fontsize=14)
        pp.xlabel(r"$\tau$")
        pp.ylabel(r"$acf(\tau)$")
        if xlims is not None: pp.xlim(xlims)
        pp.ylim(ylims)
        fig.set_size_inches([8,8])
        fig.tight_layout()
        if fn_plot is not None:
            pp.savefig(fn_plot)
        else:
            pp.show()
    return corrtime

def decorrelate(trajectories: list, method: str='acf', acf_nblocks: int|None=None, blav_model_function=None, decorrelate_only:int|None=None, plot: bool=False, fn_plot: str|None=None, verbose: bool=False, return_decorrelated_trajectories: bool=False):
    '''
        Function to compute correlation times for a list of trajectories and decorrelate the trajectories by averaging over a number of samples equal to the correlation time.

        Parameters
        ----------

        :param trajectories: List of trajectories to be decorrelated.
        :type trajectories: list

        :param method: Method to compute correlation time. Either 'acf' for autocorrelation function or 'blav' for block averaging. Defaults to 'acf'.
        :type method: str, optional

        :param acf_nblocks: Number of blocks for blocking data when using autocorrelation function. Defaults to None.
        :type acf_nblocks: int or None, optional

        :param blav_model_function: Mathematical model for the naive error on block averages as a function of the block size when using block averaging. Defaults to the default set in the blav routine.
        :type blav_model_function: callable or None, optional

        :param decorrelate_only: Index of the collective variable (CV) to be used for decorrelation when multiple CVs are present in the trajectories. Defaults to None, meaning all CVs will be decorrelated simultaneously which generally leads to higher correlation times.
        :type decorrelate_only: int or None, optional

        :param plot: If True, generate a plot showing the correlation times for each trajectory and collective variable. Ignored if fn_plot is not None. Defaults to False.
        :type plot: bool, optional

        :param fn_plot: File name to which to write the plot, defaults to None which means no plot is made (if plot=False) or the plot is not saved (if plot=True).
        :type fn_plot: str or None, optional

        :param verbose: If True, print additional information during computation. Defaults to False.
        :type verbose: bool, optional

        :param return_decorrelated_trajectories: If True, return the decorrelated trajectories along with the correlation times. Defaults to False.
        :type return_decorrelated_trajectories: bool, optional

        Returns
        -------

        * **corrtimes** (*np.ndarray*): Array containing the computed correlation times for each trajectory.

        * **trajectories_decor** (*list*): List containing the decorrelated trajectories if `return_decorrelated_trajectories` is True.

        Note
        ----

        - The `method` parameter allows for choosing between 'acf' (autocorrelation function) and 'blav' (block averaging) to compute the correlation time.
        - The `acf_nblocks` parameter is used to specify the number of blocks when using the autocorrelation function method.
        - The `blav_model_function` parameter is a mathematical model for the naive error on block averages as a function of the block size when using block averaging.
        - The `decorrelate_only` parameter allows selecting a specific collective variable (CV) index for decorrelation when multiple CVs are present in the trajectories.
        - The `plot` parameter generates a plot showing the correlation times for each trajectory and collective variable if set to True.
        - The `return_decorrelated_trajectories` parameter, if set to True, returns the decorrelated trajectories along with the correlation times.
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
            return corrtime_from_acf(data, nblocks=acf_nblocks)
    elif method.lower()=='blav':
        def method_correlation_time(data):
            return blav(data, model_function=blav_model_function)
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
            axs[0,icv].set_title('Correlation time for CV%i' %icv)
        fig.tight_layout()
        fig.set_size_inches([8*ncvs,6])
        if fn_plot is not None:
            pp.savefig(fn_plot)
        else:
            pp.show()
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
    if return_decorrelated_trajectories:
        return corrtimes, trajectories_decor
    else:
        return corrtimes

def multivariate_normal(means: np.ndarray, covariance: np.ndarray, size: int|list|np.ndarray|None=None):
    '''
        Wrapper around `numpy.random.multivariate_normal` to handle `np.nan` columns in the covariance matrix.

        Parameters
        ----------

        :param means: Means of the multivariate normal distribution.
        :type means: np.ndarray

        :param covariance: Covariance matrix of the multivariate normal distribution.
        :type covariance: np.ndarray

        :param size: Number of samples to generate. Defaults to `None`.
        :type size: int, list, np.ndarray, or None, optional

        Returns
        -------

        * **samples** (*np.ndarray*): Array containing the generated samples from the multivariate normal distribution.

        Note
        ----

        - This function acts as a wrapper around `numpy.random.multivariate_normal` to handle `np.nan` columns in the covariance matrix.
        - It filters out `np.nan` columns from the covariance matrix, enforces a symmetric covariance matrix, and handles exceptions during the sampling process.
        - It will first try the cholesky method in np.random.multivariate_normal as it is much faster. If that fails, the function tries again with the eigh setting and informs the user.
        - The `size` parameter specifies the number of samples to generate. If `size` is `None`, a single sample is generated.
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

        Parameters
        ----------

        :param F: Fisher information matrix.
        :type F: np.ndarray

        :ps: Array of probabilities used to apply the threshold.
        :type ps: np.ndarray

        :threshold: Threshold value for removing columns and rows corresponding with a probability lower than the thresshold from the Fisher information matrix. Defaults to `0.0`.
        :type threshold: float, optional

        :verbose: If `True`, print information about removed columns and rows. Defaults to `False`.
        :type verbose: bool, optional

        Returns
        -------

        * **cov** (*np.ndarray*): Covariance matrix obtained by inverting the Fisher information matrix.
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

def fisher_matrix_mle_probdens(ps: np.ndarray, method: str='mle_p', verbose: bool=False):
    '''
        Computes the Fisher information matrix for the maximum likelihood estimation (MLE) of probability distribution parameters given in argument ps.

        Parameters
        ----------

        :param ps: Probability histogram values
        :type ps: np.ndarray

        :param method: Method used for computing the Fisher information matrix. Options include 'mle_p', 'mle_p_cov', 'mle_f', and 'mle_f_cov'. Defaults to 'mle_p'.
        :type method: str, optional

        :param verbose: If `True`, prints additional information when no Fisher information is found. Defaults to `False`.
        :type verbose: bool, optional

        Returns
        -------

        * **F** (*np.ndarray*): Fisher information matrix computed for the maximum likelihood estimation of probability distribution parameters.

        Note
        ----

        - This function computes the Fisher information matrix for the maximum likelihood estimation of probability distribution parameters.
        - The `method` parameter allows selecting different methods for computing the Fisher matrix.
        - If `verbose` is set to `True`, additional information is printed when no Fisher information is found.
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

#Set of old depricated routines that are only kept in ThermoLIB for backward compatibility. These routines will be removed in the near future, please use read_wham_input routine.

def trajectory_xyz_to_CV(fns, CV):
    '''
        ROUTINE IS DEPRICATED, included only for backward compatibility. Please use CVComputer class from thermolib.trajectory module.
        
        Compute the CV along an XYZ trajectory. The XYZ trajectory is assumed to be defined in a (list of subsequent) XYZ file(s).

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

def read_wham_input_old(fn, path_template_colvar_fns='%s', colvar_cv_column_index=1, kappa_unit='kjmol', q0_unit='au', start=0, end=-1, stride=1, bias_potential='Parabola1D', additional_bias=None, inverse_cv=False, verbose=False):
    ''' THIS ROUTINE IS DEPRECATED AND WILL BE DELETED IN THE FUTURE. It is still included for backward compatibility with the old read_wham_input routine.

        Read the input for a WHAM reconstruction of the free energy profile from a set of Umbrella Sampling simulations. The file specified by fn should have the following format:

        .. code-block:: python

            T = XXXK
            NAME_1 Q0_1 KAPPA_1
            NAME_2 Q0_2 KAPPA_2
            NAME_3 Q0_3 KAPPA_3
            ...

        when a line starts with a T, it is supposed to specify the temperature. If no temperature line is found, a temperature of None will be returned. All other lines define the bias potential for each simulation as a parabola centered around ``Q0_i`` and with force constants ``KAPPA_i``(the units used in this file can be specified the keyword arguments ``q0_unit`` and ``kappa_unit``). For each bias with name ``NAME_i`` defined in this file, there should be a colvar trajectory file accessible through the path given by the string:
        
        .. code-block:: python
        
            os.path.join(root, path_template_colvar_fns %name)
        
        For example, if path_template_colvar_fns is defined as 'trajectories/%s/COLVAR' and the wham input file contains a line 'Window1/r1 1.40 1000.0', then the trajectory file containing the CV values of the biased simulation with bias name 'Window1/r1' can be found in the directory trajectories/Window1/r1/COLVAR. Finally, these trajectory files should be formatted as outputted by PLUMED:

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
    from thermolib.thermodynamics.trajectory import ColVarReader
    trajectory_reader = ColVarReader([colvar_cv_column_index], units=['au'], start=start, stride=stride, end=end, verbose=verbose)
    return read_wham_input(
        fn, 
        trajectory_reader, trajectory_path_template=path_template_colvar_fns, 
        bias_potential=bias_potential, q01_unit=q0_unit, kappa1_unit=kappa_unit, inverse_cv1=inverse_cv, 
        additional_bias=additional_bias, 
        verbose=verbose
    )

def read_wham_input_h5_old(fn, h5_cv_path, path_template_h5_fns='%s', kappa_unit='kjmol', q0_unit='au', cv_unit='au', start=0, end=-1, stride=1, bias_potential='Parabola1D', additional_bias=None, inverse_cv=False, verbose=False):
    ''' THIS ROUTINE IS DEPRECATED AND WILL BE DELETED IN THE FUTURE. It is still included for backward compatibility with the old read_wham_input_h5 routine.

        Read the input for a WHAM reconstruction of the free energy profile from a set of Umbrella Sampling simulations. The file specified by fn should have the following format:

        .. code-block:: python

            T = XXXK
            NAME_1 Q0_1 KAPPA_1
            NAME_2 Q0_2 KAPPA_2
            NAME_3 Q0_3 KAPPA_3
            ...

        when a line starts with a T, it is supposed to specify the temperature. If no temperature line is found, a temperature of None will be returned. All other lines define the bias potential for each simulation as a parabola centered around ``Q0_i`` and with force constants ``KAPPA_i``(the units used in this file can be specified the keyword arguments ``q0_unit`` and ``kappa_unit``). For each bias with name ``NAME_i`` defined in this file, there should be a colvar trajectory file accessible through the path given by the string:
        
        .. code-block:: python
        
            os.path.join(root, path_template_h5_fns %name)
        
        For example, if path_template_h5_fns is defined as 'trajectories/%s/traj.h5' and the wham input file contains a line 'Window1/r1 1.40 1000.0', then the trajectory file with path trajectories/Window1/r1/traj.h5 contains a dataset given by ``h5_cv_path`` that represents the cv values of the biased simulation with bias name 'Window1/r1'. Finally, the cv values in these trajectory files have a unit that can be specified by the keyword argument ``cv_unit``.

        :param fn: file name of the wham input file
        :type fn: str
        
        :param h5_cv_path: the path of the dataset corresponding to the cv values in the h5 file.
        :type h5_cv_path: str

        :param path_template_h5_fns: Template for defining the path (relative to the directory containing the wham input file given by argument fn) to the HDF5 trajectory file corresponding to each biased simulation. See documentation above for more details. This argument should be string containing a single '%s' substring. 
        :type path_template_h5_fns: str. Defaults to '%s'

        :param kappa_unit: unit used to express kappa in the wham input file, defaults to 'kjmol'
        :type kappa_unit: str, optional

        :param q0_unit: unit used to express q0 in the wham input file, defaults to 'au'
        :type q0_unit: str, optional

        :param cv_unit: unit used to express the cv values in the trajectory files, defaults to 'au'
        :type cv_unit: str, optional
        
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
    from thermolib.thermodynamics.trajectory import HDF5Reader
    trajectory_reader = HDF5Reader([h5_cv_path], units=[cv_unit], start=start, stride=stride, end=end, verbose=verbose)
    return read_wham_input(
        fn, 
        trajectory_reader, trajectory_path_template=path_template_h5_fns, 
        bias_potential=bias_potential, q01_unit=q0_unit, kappa1_unit=kappa_unit, inverse_cv1=inverse_cv, 
        additional_bias=additional_bias, 
        verbose=verbose
    )

def read_wham_input_2D_old(fn, path_template_colvar_fns='%s', colvar_cv1_column_index=1, colvar_cv2_column_index=2, kappa1_unit='kjmol', kappa2_unit='kjmol', q01_unit='au', q02_unit='au', start=0, end=-1, stride=1, bias_potential='Parabola2D',additional_bias=None,additional_bias_dimension='q1',inverse_cv1=False,inverse_cv2=False, verbose=False):
    ''' THIS ROUTINE IS DEPRECATED AND WILL BE DELETED IN THE FUTURE. It is still included for backward compatibility with the old read_wham_input_2D routine.
    
        Read the input for a WHAM reconstruction of the 2D free energy surface from a set of Umbrella Sampling simulations. The wham input file specified by fn should have the following format:

        .. code-block:: python

            T = XXXK
            NAME_1 Q01_1 Q02_1 KAPPA1_1 KAPPA1_1
            NAME_2 Q01_2 Q02_2 KAPPA1_2 KAPPA1_2
            NAME_3 Q01_3 Q02_3 KAPPA1_3 KAPPA1_3
            ...

        when a line starts with a T, it is supposed to specify the temperature. If no temperature line is found, a temperature of None will be returned. All other lines define the bias potential for each simulation as a parabola centered around (``Q01_i``,``Q02_i``) and with force constants (``KAPPA1_i``,``KAPPA2_i``) (the units used in this file can be specified the keyword arguments ``kappa1_unit``, ``kappa2_unit``, ``q01_unit`` and ``q02_unit``). For each bias with name ``NAME_i`` defined in this file, there should be a colvar trajectory file accessible through the path given by the string:
        
        .. code-block:: python
        
            os.path.join(root, path_template_colvar_fns %name)
        
        For example, if path_template_colvar_fns is defined as 'trajectories/%s/COLVAR' and the wham input file contains a line 'Window1/r1 1.40 -0.2 1000.0 1000.0', then the trajectory file containing the CV values of the biased simulation with bias name 'Window1/r1' has the path trajectories/Window1/r1/COLVAR. Finally, these trajectory files should be formatted as outputted by PLUMED (if the desired collective variable columns are not the default second and third, these can be specified with colvar_cv1_column_index and colvar_cv2_column_index):

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

        :param additional_bias: A single additional bias that is added for each simulation on top of the simulation-specific biases. Defaults to None
        :type additional_bias: BiasPotential1D, optional

        :param inverse_q1: Define whether to invert q1 values. Defaults to False
        :type inverse_q1: boolean, optional

        :param inverse_q2: Define whether to invert q2 values. Defaults to False
        :type inverse_q2: boolean, optional

        :param verbose: increases verbosity if set to True, defaults to False
        :type verbose: bool, optional

        :raises ValueError: when a line in the wham input file cannot be interpreted

        :return: temp, biasses, trajectories:

            * **temp** (float) -- temperature at which the simulations were performed
            * **biasses** (list of callables) -- list of bias potentials (defined as callable functions) for all Umbrella Sampling simulations
            * **trajectories** (list of np.ndarrays) -- list of trajectory data arrays containing the CV trajectory for all Umbrella Sampling simulations
    '''
    from thermolib.thermodynamics.trajectory import ColVarReader
    trajectory_reader = ColVarReader([colvar_cv1_column_index, colvar_cv2_column_index], units=['au','au'], start=start, stride=stride, end=end, verbose=verbose)
    return read_wham_input(
        fn, 
        trajectory_reader, trajectory_path_template=path_template_colvar_fns, 
        bias_potential=bias_potential, q01_unit=q01_unit, q02_unit=q02_unit, kappa1_unit=kappa1_unit, kappa2_unit=kappa2_unit, 
        inverse_cv1=inverse_cv1, inverse_cv2=inverse_cv2,
        additional_bias=additional_bias, additional_bias_dimension=additional_bias_dimension,
        verbose=verbose
    )

def read_wham_input_2D_h5_old(fn, h5_cv1_path, h5_cv2_path, path_template_h5_fns='%s', kappa1_unit='kjmol', kappa2_unit='kjmol', q01_unit='au', q02_unit='au', cv1_unit='au', cv2_unit='au', start=0, end=-1, stride=1, bias_potential='Parabola2D', additional_bias=None, additional_bias_dimension='q1', inverse_cv1=False, inverse_cv2=False, verbose=False):
    ''' THIS ROUTINE IS DEPRECATED AND WILL BE DELETED IN THE FUTURE. It is still included for backward compatibility with the old read_wham_input_2D_h5 routine.
        
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
        
        when a line starts with a T, it is supposed to specify the temperature. If no temperature line is found, a temperature of None will be returned. All other lines define the bias potential for each simulation as a parabola centered around (``Q01_i``,``Q02_i``) and with force constants (``KAPPA1_i``,``KAPPA2_i``) (the units used in this file can be specified the keyword arguments ``kappa1_unit``, ``kappa2_unit``, ``q01_unit`` and ``q02_unit``). For each bias with name ``NAME_i`` defined in this file, there should be a colvar trajectory dataset accessible in the HDF5 file with path given by the string:
        
        .. code-block:: python
        
            os.path.join(root, path_template_h5_fns %name)
        
        For example, if path_template_h5_fns is defined as 'trajectories/%s/traj.h5' and the wham input file contains a line 'Window1/r1 1.40 -0.2 1000.0 1000.0', then the trajectory HDF5 file 'trajectories/Window1/r1/traj.h5' contains a dataset with path ``h5_cv1_path`` representing the CV1 values of the biased simulation with bias name 'Window1/r1' (and similar for the CV2 values). The units of the cv1 and cv2 values in these trajectories can be specified by the keyword arguments ``cv1_unit`` and ``cv2_unit`` respectively.

        :param fn: file name of the wham input file
        :type fn: str

        :param h5_cv1_path: the path of the dataset corresponding to the cv1 values in the h5 file.
        :type h5_cv1_path: str

        :param h5_cv2_path: the path of the dataset corresponding to the cv2 values in the h5 file.
        :type h5_cv2_path: str

        :param path_template_h5_fns: Template for defining the path (relative to the directory containing the wham input file given by argument fn) to the colvar trajectory file corresponding to each bias. See documentation above for more details. This argument should be string containing a single '%s' substring. 
        :type path_template_h5_fns: str. Defaults to '%s'

        :param kappa1_unit: unit used to express the CV1 force constant kappa1 in the wham input file, defaults to 'kjmol'
        :type kappa1_unit: str, optional

        :param kappa2_unit: unit used to express the CV2 force constant kappa1 in the wham input file, defaults to 'kjmol'
        :type kappa2_unit: str, optional

        :param q01_unit: unit used to express q01 in the wham input file as well as the cv values in the trajectory files, defaults to 'au'
        :type q01_unit: str, optional

        :param q02_unit: unit used to express q02 in the wham input file as well as the cv values in the trajectory files, defaults to 'au'
        :type q02_unit: str, optional
        
        :param cv1_unit: unit used to express the cv1 values in the trajectory files, defaults to 'au'
        :type cv1_unit: str, optional

        :param cv2_unit: unit used to express the cv2 values in the trajectory files, defaults to 'au'
        :type cv2_unit: str, optional

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
        
        :param additional_bias: A single additional bias that is added for each simulation on top of the simulation-specific biases. Defaults to None
        :type additional_bias: BiasPotential1D, optional

        :param inverse_q1: Define whether to invert q1 values. Defaults to False
        :type inverse_q1: boolean, optional
        
        :param inverse_q2: Define whether to invert q2 values. Defaults to False
        :type inverse_q2: boolean, optional
        
        :param verbose: increases verbosity if set to True, defaults to False
        :type verbose: bool, optional
        
        :raises ValueError: when a line in the wham input file cannot be interpreted

        :return: temp, biasses, trajectories:

            * **temp** (float) -- temperature at which the simulations were performed
            * **biasses** (list of callables) -- list of bias potentials (defined as callable functions) for all Umbrella Sampling simulations
            * **trajectories** (list of np.ndarrays) -- list of trajectory data arrays containing the CV trajectory for all Umbrella Sampling simulations
    '''
    from thermolib.thermodynamics.trajectory import HDF5Reader
    trajectory_reader = HDF5Reader([h5_cv1_path, h5_cv2_path], units=[cv1_unit,cv2_unit], start=start, stride=stride, end=end, verbose=verbose)
    return read_wham_input(
        fn, 
        trajectory_reader, trajectory_path_template=path_template_h5_fns, 
        bias_potential=bias_potential, q01_unit=q01_unit, q02_unit=q02_unit, kappa1_unit=kappa1_unit, kappa2_unit=kappa2_unit, 
        inverse_cv1=inverse_cv1, inverse_cv2=inverse_cv2,
        additional_bias=additional_bias, additional_bias_dimension=additional_bias_dimension,
        verbose=verbose
    )

