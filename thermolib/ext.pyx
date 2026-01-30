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
#
#cython: embedsignature=True

from .units import parse_unit

import matplotlib.pyplot as pp
import warnings, sys

import numpy as np
cimport numpy as np
np.seterr(divide='ignore', invalid='ignore')

from scipy.optimize import curve_fit

__all__ = [
    'wham1d_hs', 'wham1d_bias', 'wham1d_scf', 'wham1d_error',
    'wham2d_hs', 'wham2d_bias', 'wham2d_scf', 'wham2d_error',
    'fisher_matrix_mle_probdens',
]

### Routines for WHAM in 1D
def wham1d_hs(int Nsims, int Ngrid, np.ndarray[object] trajectories, np.ndarray[double] bins, np.ndarray[long] Nis):
    '''
        Internal WHAM routine to compute the 1D CV histogram of each (biased) simulation from the given CV trajectories.

        :param Nsims: number of simulations that were done and for which the histogram needs to be constructed
        :type Nsims: int

        :param Ngrid: number of points on the CV grid to be used for the histogram.
        :type Ngrid: int
        
        :param trajectories: array containing the simulation trajectory (i.e. CV samples) for each (biased) simulation
        :type trajectories: np.ndarray

        :param bins: bin edges for the CV histogram. Should have length one larger than Ngrid
        :type bins: np.ndarray[double, shape=(Ngrid+1,)]

        :param Nis: Nis[j] is the number of samples in the trajectory of simulation j
        :type Nis: np.ndarray[long, shape=(Nsims,)]

        :returns Hs: 2 dimensional array containing the CV histograms for each of the (biased) simulations. Hs[i,k] represents the histogram value of the k-th bin for the i-th simulation.
        :rtype: np.ndarray[long, shape=(Nsims, Ngrid)]
    '''
    cdef np.ndarray[long, ndim=2] Hs = np.zeros([Nsims, Ngrid], dtype=int)
    cdef np.ndarray[double] data, edges
    cdef int i, N
    assert len(trajectories)==Nsims, 'In wham1d_hs, length of argument trajectories not consistent with Nsims!'
    assert len(bins)-1==Ngrid, 'In wham1d_hs, length of argument bins not consistent with Ngrid!'
    for i, data in enumerate(trajectories):
        if len(data)==0:
            raise ValueError("The trajectory of simulation %i does not contain any data anymore. Are you sure you didn't remove too much data in post processing, e.g. to get rid of equilibration steps?" %i)
        Hs[i,:], edges = np.histogram(data, bins, density=False)
        assert (bins==edges).all()
        N = Hs[i,:].sum()
        if Nis[i]!=N:
            print("WARNING: the CV range you specified for the histogram does not cover the entire simulation range in trajectory %i. Simulation samples outside the given CV range were cropped out." %(i))
            Nis[i] = N
    return Hs


#
def wham1d_bias(int Nsims, int Ngrid, double beta, list biasses, double delta, int bias_subgrid_num, np.ndarray[double] bin_centers, double threshold=1e-3):
    '''
        Internal WHAM routine to compute the integrated boltzmann factors of the bias potentials W in each grid interval:

        .. math:: b_{ik} = \\frac{1}{\\delta}\\int_{Q_k-\\frac{\\delta}{2}}^{Q_k+\\frac{\\delta}{2}} e^{-\\beta W_i(q)}dq

        This routine implements a conservative algorithm which takes into account that for a given simulation i, only a limited number of grid points :math:`k` will give rise to a non-zero :math:`b_{ik}`. This is achieved by first using the bin-center approximation to the integral by computing the factor :math:`\\exp(-\\beta\\cdot W_i(Q_k))` on the CV grid (which is faster as there is no integral involved and which is also already a good approximation) and only performing the precise integral when the approximation exceeds the given threshold.

        :param Nsims: the total number of simulations performed
        :type Nsims: int

        :param Ngrid: number of points on the CV grid to be used for the histogram.
        :type Ngrid: int

        :param beta: 1/kT in atomic units with T the temperature at which the simulation was performed
        :type beta: float

        :param biasses: list of bias potentials that were applied for each of the simulations.
        :type biasses: list of instances of child classes of :py:class:`BiasPotential1D <thermolib.thermodynamics.bias.BiasPotential1D>`

        :param delta: width of CV-interval (:math:`\\delta` in equation above) over which the bias is integrated when the bin-center approximation is not sufficient (see doc above) 
        :type delta: float

        :param bias_subgrid_num: the number of grid points used by the :py:meth:`wham1d_bias <thermolib.ext.wham1d_bias>` routine for the sub-grid to compute the boltzmann-integrated bias factors in each CV bin.
		:type bias_subgrid_num: int

        :param bin_center: array containing the center of each bin in the histogram for application in the bin-center approximation (see doc above)
        :type bin_center: np.ndarray[double]

        :param threshold: see general documentation above
        :type thresholdm: float, optional, default=1e-3

        :returns: array containing the (integrated) bias on the CV-grid for each simulation, i.e. :math:`b_{ik}` array from equation above.
        :rtype: np.ndarray[double, shape=(Nsims, Ngrid)]
    '''
    from thermolib.tools import integrate
    cdef np.ndarray[double, ndim=2] bs = np.zeros([Nsims, Ngrid], dtype=float)
    cdef double subdelta = delta/bias_subgrid_num
    cdef np.ndarray[double] subgrid = np.arange(-delta/2, delta/2+subdelta, subdelta)
    cdef np.ndarray[double] Ws
    cdef int i, k
    for i, bias in enumerate(biasses):
        #first compute bin-center approximation
        bs[i,:] = np.exp(-beta*bias(bin_centers))
        #find large bias factors for exact computation
        ks = np.where(bs[i,:]>threshold)[0]
        for k in ks:
            Ws = np.exp(-beta*bias(bin_centers[k]+subgrid))
            bs[i,k] = integrate_c(bin_centers[k]+subgrid, Ws)/delta
    return bs


#
def wham1d_scf(np.ndarray[long] Nis, np.ndarray[long, ndim=2] Hs, np.ndarray[double, ndim=2] bs, int Nscf=1000, double convergence=1e-6, double overflow_threshold=1e-150, verbose=False):
    '''
        Internal WHAM routine to solve the WHAM equations for the unbiased probability distribution
        
        .. math::

            \\frac{1}{f_i} &= \\sum_k b_{ik} a_k \\\\
            a_k &= \\sum_i \\frac{H_{ik}}{\\sum_i N_i f_i b_{ik}}
        
        until self consistency is achieved.

        :param Nis: Nis[j] is the number of samples in the trajectory of simulation j
        :type Nis: np.ndarray[long, shape=(Nsims,)]

        :param Hs: 2 dimensional array containing the CV histograms for each of the (biased) simulations
        :type Hs: np.ndarray[long, shape=(Nsims, Ngrid)]

        :param bs: array containing the (integrated) bias on the CV-grid for each simulation, i.e. :math:`b_{ik}` array from equation above.
        :type bs: np.ndarray[double, shape=(Nsims, Ngrid)]

        :param Nscf: maximum number of SCF cycles to obtain self-consistency
        :type Nscf: int, optional, default=1000

        :param convergence: sets criterium determining self-consistency, SCF cycle will stop when the integrated absolute value of the difference between the unbiased probability distribution at the current step and the previous step is smaller then the given convergence parameter.
        :type convergence: double, optional, default=1e-6

        :param overflow_threshold: numerical threshold to avoid overflow errors when calculating the normalization factors and the denominator of the unbiased probability. This determines which simulations and which grid points to ignore. Decreasing it results in a FEP with a larger maximum free energy (lower unbiased probability). If it is too low, imaginary errors arise, so increase if necessary.
        :type overflow_threshold: double, optional, default=1e-150

        :returns: the unbiased probability distribution on the same grid as all the histograms in the Hs argument.
        :rtype: np.ndarray[double, shape=(Ngrid,)]
    '''
    cdef double integrated_diff, pmax
    cdef np.ndarray[double] as_old, as_new, inverse_fs, delta
    cdef np.ndarray[long] nominator
    cdef np.ndarray[np.uint8_t] sims_mask, new_sims_mask, grid_mask
    cdef int Ngrid, Nsims, iscf, i, k
    cdef int converged = 0

    #initialization
    Nsims = Hs.shape[0]
    Ngrid = Hs.shape[1]
    as_old = np.ones(Ngrid)/Ngrid #probability density (should sum to 1)
    nominator = Hs.sum(axis=0) #precomputable nominator in WHAM equations
    sims_mask = np.ones(Nsims,dtype=bool)
    grid_mask = np.ones((Ngrid),dtype=bool)
    for iscf in range(Nscf):
        #compute new normalization factors
        inverse_fs = np.einsum('ik,k->i', bs, as_old)

        # Calculate mask for simulations
        if np.any(inverse_fs<overflow_threshold):
            new_sims_mask = (inverse_fs>=overflow_threshold)
            if not np.array_equiv(new_sims_mask,sims_mask):
                sims_mask = new_sims_mask
                # Recalculate nominator when sims_mask changes
                nominator = Hs[sims_mask,:].sum(axis=0)

        #compute new probabilities
        denominator = np.einsum('i,i,ik->k', Nis[sims_mask], 1.0/inverse_fs[sims_mask], bs[sims_mask])

        # Calculate mask for grid
        if np.any(denominator<overflow_threshold):
            grid_mask = (denominator>=overflow_threshold)
            # check whether Hs is close to 0 for those points in the sims_mask that would be taken out by grid_mask, then we have as_new = 0/0
            if not np.isclose(np.sum(Hs[sims_mask][:,~grid_mask]),0.):
                warnings.warn('Grid indices are being masked that contain a total of at least 1 histogram count ({}/{} = {:2.4f}%).'.format(np.sum(Hs[sims_mask][:,~grid_mask]),np.sum(Hs[sims_mask,:]),(np.sum(Hs[sims_mask][:,~grid_mask])/np.sum(Hs[sims_mask,:]))*100))

        as_new = np.zeros(Ngrid) # if a is zero, it will be ignored in both fs and the error calculation
        as_new[grid_mask] = np.divide(nominator[grid_mask],denominator[grid_mask])
        as_new[grid_mask] /= np.sum(as_new) #enforce normalization

        #check convergence
        integrated_diff = np.abs(as_new-as_old).sum()
        if verbose:
            pmax = as_new.max()
            print('cycle %i/%i' %(iscf+1,Nscf))
            print('  norm prob. dens. [au] = %.3e' %as_new.sum())
            print('  max prob. dens.  [au] = %.3e' %pmax)
            print('  Integr. Diff.    [au] = %.3e' %integrated_diff)
            print('')
        if integrated_diff<convergence:
            converged = 1
            break
        as_old = as_new.copy()

    #compute final normalization factors
    fs = np.full(Nsims, np.nan)
    fs[sims_mask] = 1.0/np.einsum('ik,k->i', bs[sims_mask], as_new)

    return as_new, fs, converged


#
def wham1d_error(int Nsims, int Ngrid, np.ndarray[long] Nis, np.ndarray[double] ps, np.ndarray[double] fs, np.ndarray[double, ndim=2] bs, np.ndarray[double] corrtimes, method='mle_f_cov', p_threshold=0.0, verbosity='off'):
    '''
        Internal WHAM routine that allows to compute the error distribution on the unbiased probability distirbution that is estimated using the WHAM equations. The error estimation is based on the interpretation of the WHAM solutio as a Maximum Likelihood Estimater (MLE), which in term allows to estimate the error based on the Fisher information matrix.

        :param Nsims: the total number of simulations performed
        :type Nsims: int

        :param Ngrid: number of points on the CV grid to be used for the histogram.
        :type Ngrid: int

        :param Nis: Nis[j] is the number of samples in the trajectory of simulation j
        :type Nis: np.ndarray[long, shape=(Nsims,)]

        :param ps: the unbiased probability distribution as returned by the :py:meth:`wham1d_scf <thermolib.ext.wham1d_scf>` routine. Can contain nan values at specific grid locations (corresponding to disabled bins)
        :type ps: np.ndarray[double, shape=(Ngrid,)]

        :param fs: renormalisation factors fs figuring in the WHAM equations. These are computed as intermediate variables in the :py:meth:`wham1d_scf <thermolib.ext.wham1d_scf>` routine. Can contain nan values at specific sim indices (at disabled simulations).
        :type fs: np.ndarray[double, shape=(Nsims,)]

        :param bs: array containing the (integrated) bias on the CV-grid for each simulation, i.e. :math:`b_{ik}` array from equation above.
        :type bs: np.ndarray[double, shape=(Nsims,Ngrid)]

        :param corrtimes: array of (integrated) correlation times of the CV, one for each simulation. Such correlation times will be taken into account during the error estimation and hence make it more reliable. If set to None, the CV trajectories will be assumed to contain fully uncorrelated samples (which is not true when using trajectories representing each subsequent step from a molecular dynamics simulation). More information can be found in :ref:`the user guide <seclab_ug_errorestimation>`. This input can be generated using the :py:meth:`decorrelate <thermolib.tools.decorrelate>` routine.
        :type corrtimes: np.ndarray[double, shape=(Nsims,)]

        :param method: specification of the method on how to perform the estimation of the error distribution. One of following options is available:

				- **mle_p** - Estimating the error directly for the probability of each bin in the histogram. This method does not explicitly impose the positivity of the probability.

				- **mle_p_cov** - Estimate the full covariance matrix for the probability of all bins in the histogram. In other words, appart from the error on the probability/free energy of a bin itself, we now also account for the covariance between the probabilty/free energy of the bins. This method does not explicitly impose the positivity of the probability.

				- **mle_f** - Estimating the error for minus the logarithm of the probability, which is proportional to the free energy (hence f in mle_f). As the probability is expressed as :math:`\propto e^{-f}`, its positivity is explicitly accounted for.

				- **mle_f_cov** - Estimate the full covariance matrix for minus the logarithm of the probability of all bins in the histogram. In other words, appart from the error on the probabilty/free energy of a bin itself (including explicit positivity constraint), we now also account for the covariance between the probability/free energy of the bins.
        
        :type method: str or None, optional, default='mle_f_cov'

        :param p_threshold: only relevant when error estimation is enabled (see parameter ``error_estimate``). When ``error_p_threshold`` is set to x, bins in the histogram for which the probability resulting from the trajectory is smaller than x will be disabled for error estimation (i.e. its error will be set to np.nan). It is mainly usefull in the case of 2D histograms,as illustrated in :doc:`one of the tutorial notebooks <tut/advanced_projection>`.
        :type p_threshold: float, optional, default=0.0

        :param verbosity: specify the level of verbosity of the current routine
        :type verbosity: str, optional, default='off'

        :returns: the distribution of the error on the unbiased probability distribution
        :rtype: :py:class:`GaussianDistribution <thermolib.error.GaussianDistribution>` (if method='mle_p'), :py:class:`MultiGaussianDistribution <thermolib.error.MultiGaussianDistribution>` (if method='mle_p_cov'), :py:class:`LogGaussianDistribution <thermolib.error.LogGaussianDistribution>` (if method='mle_f'), :py:class:`MultiLogGaussianDistribution <thermolib.error.MultiLogGaussianDistribution>` (if method='mle_f_cov')
    '''
    from thermolib.error import GaussianDistribution, LogGaussianDistribution, MultiGaussianDistribution, MultiLogGaussianDistribution
    cdef np.ndarray[double, ndim=2] I, Ii, Imask, sigma
    cdef np.ndarray[double] err, logps
    cdef np.ndarray[np.uint8_t, ndim=2] mask
    cdef int i, k
    cdef long Nmask2, Nmask
    I = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1], dtype=float)
    for i in range(Nsims):
        Ii = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])
        for k in range(Ngrid):
            if method in ['mle_p', 'mle_p_cov']:
                #avoid singularity at ps[k]=0, this is an inherent isssue to mle_p not present in the mle_f error estimation
                if ps[k]>0:
                    Ii[k,k] = fs[i]*bs[i,k]/ps[k]
                Ii[k,Ngrid+i] = bs[i,k]
                Ii[Ngrid+i,k] = bs[i,k]
                Ii[k,Ngrid+Nsims+i] = fs[i]*bs[i,k]
                Ii[Ngrid+Nsims+i,k] = fs[i]*bs[i,k]
                Ii[k,-1] = 1
                Ii[-1,k] = 1
            elif method in ['mle_f', 'mle_f_cov']:
                #note that ps=exp(-gs) where gs are the degrees of freedom in the mle_f method (minus logarithm of the histogram counts)
                Ii[k,k] = ps[k]*fs[i]*bs[i,k]
                Ii[k,Ngrid+i] = -ps[k]*bs[i,k]
                Ii[Ngrid+i,k] = -ps[k]*bs[i,k]
                Ii[k,Ngrid+Nsims+i] = -fs[i]*ps[k]*bs[i,k]
                Ii[Ngrid+Nsims+i,k] = -fs[i]*ps[k]*bs[i,k]
                Ii[k,-1] = -ps[k]
                Ii[-1,k] = -ps[k]
            else:
                raise IOError('Recieved invalid argument for method, recieved %s. Check routine signiture for more information on allowed values.' %method)
        Ii[Ngrid+i,Ngrid+i] = 1/fs[i]**2
        Ii[Ngrid+i,Ngrid+Nsims+i] = 1/fs[i]
        Ii[Ngrid+Nsims+i,Ngrid+i] = 1/fs[i]
        I += Nis[i]/corrtimes[i]*Ii
    #the covariance matrix is the inverse of the total Fisher matrix (after filtering out unsampled bins and/or simulations with not enough contribution)
    sigma = wham1d_invert_fisher_to_covariance(I, ps, fs, bs, Nis/corrtimes, p_threshold=0.0, verbosity=verbosity)
    #the error bar on the probability of bin k is now simply the (k,k)-diagonal element of sigma
    err = np.sqrt(np.diagonal(sigma)[:Ngrid])
    if method in ['mle_p']:
        return GaussianDistribution(ps, err)
    elif method in ['mle_p_cov']:
        return MultiGaussianDistribution(ps, sigma[:Ngrid,:Ngrid])
    elif method in ['mle_f']:
        #if ps is LogNormal distributed, then the means argument of LogGaussianDistribution is log(ps). The error calculated by MLE-F represents the error on log(ps) so this can be fed directly to the LogGaussianDistribution err argument.
        logps = np.zeros(Ngrid)*np.nan
        logps[ps>0] = np.log(ps[ps>0])
        return LogGaussianDistribution(logps, err)
    elif method in ['mle_f_cov']:
        #if ps is LogNormal distributed, then the means argument of MultiLogGaussianDistribution is log(ps). The error calculated by MLE-F represents the error on log(ps) so this can be fed directly to the MultiLogGaussianDistribution err argument.
        logps = np.zeros(Ngrid)*np.nan
        logps[ps>0] = np.log(ps[ps>0])
        return MultiLogGaussianDistribution(logps, sigma[:Ngrid,:Ngrid])
    else:
        raise IOError('Recieved invalid argument for method, recieved %s. Check routine signiture for more information on allowed values.' %method)


#
def wham1d_invert_fisher_to_covariance(np.ndarray[double, ndim=2] F, np.ndarray[double] ps, np.ndarray[double] fs, np.ndarray[double, ndim=2] bs, np.ndarray[double] Nis, double p_threshold=0.0, verbosity='off'):
    cdef np.ndarray[double, ndim=2] F_mask
    cdef np.ndarray[double, ndim=2] cov = np.zeros([F.shape[0], F.shape[1]])*np.nan
    cdef np.ndarray[np.uint8_t, ndim=2, cast=True] mask
    cdef int k, i, N_mask2, N_mask, Ngrid, Nsims, num_deactivated_bins, num_deactivated_sims
    cdef double p, p_biased, Ni
    cdef np.uint8_t reached_end, faulty_simulations_present, sufficiently_sampled
    cdef np.ndarray[double] arr
    
    #before computing the inverse of the Fisher matrix, we define a mask to remove all rows/columns corresponding to:
    # - unpopulated bins (as there is no information and hence also no error bar on them)
    # - insufficiently sampled bins (determined by the biased probability)
    # - faulty simulations (corresponding fi is nan)
    
    if verbosity.lower() in ['medium', 'high']: print('  defining zero-mask ...')
    mask = np.ones([F.shape[0], F.shape[1]], dtype=bool)
    num_deactivated_bins = 0
    Nsims = len(Nis)
    Ngrid = len(ps)
    for k, p in enumerate(ps):
        if np.isnan(p) or p==0.0:
            mask[k,:] = 0
            mask[:,k] = 0
            num_deactivated_bins += 1
            if verbosity.lower() in ['high']: print('    deactivated bin %i because p=0,nan' %k)
        else:
            sufficiently_sampled = False
            for i in range(Nsims):
                p_biased = fs[i]*bs[i,k]*ps[k]
                if not np.isnan(fs[i]) and p_biased>p_threshold:
                    sufficiently_sampled = True
                    break
            if not sufficiently_sampled:
                mask[k,:] = 0
                mask[:,k] = 0
                num_deactivated_bins += 1
                if verbosity.lower() in ['high']: print('    deactivated bin %i because p_biased to low' %k)
    if verbosity.lower() in ['medium', 'high']: print('    deactivated %i bins' %num_deactivated_bins)

    num_deactivated_sims = 0
    for i in range(Nsims):
        if np.isnan(fs[i]): 
            mask[Ngrid+i,:] = 0
            mask[:, Ngrid+i] = 0
            mask[Ngrid+Nsims+i,:] = 0
            mask[:, Ngrid+Nsims+i] = 0
            num_deactivated_sims += 1
            if verbosity.lower() in ['high']:  print('    deactivated sim %i because f=nan' %i)
    if verbosity.lower() in ['medium', 'high']: print('    deactivated %i sims' %num_deactivated_sims)

    if verbosity.lower() in ['medium', 'high']: print('  applying zero-mask ...')
    N_mask2 = len(F[mask])
    assert abs(int(np.sqrt(N_mask2))-np.sqrt(N_mask2))==0 #consistency check, sqrt(N_mask2) should be integer valued
    N_mask = int(np.sqrt(N_mask2))
    F_mask = F[mask].reshape([N_mask,N_mask])
    
    if verbosity.lower() in ['medium', 'high']: print('  inverting Fisher matrix ...')
    try:
        cov[mask] = np.linalg.inv(F_mask).reshape(N_mask2)
    except np.linalg.LinAlgError as err:
        print('===================== ERROR during Fisher inversion ============================')
        print(' Could not invert the Fisher information matrix because it is singular! This    ')
        print(' could be due to inclusion of simulations without any steps (possibly after     ')
        print(' decorrelation) within the defined cv bin range. The number of (uncorrelated)   ')
        print(' simulation steps within the cv range in each subsequent simulation is:')
        reached_end = False
        i = 0
        while not reached_end:
            if i+10>=len(Nis):
                arr = Nis[i:]
                reached_end=True
            else:
                arr = Nis[i:i+10]
            print(' '+' '.join(['% 7i' %int(Ni) for Ni in arr]))
            i += 10
        print('')
        faulty_simulations_present = False
        for i, Ni in enumerate(Nis):
            if int(Ni)==0:
                faulty_simulations_present = True
                print('Simulation nr. %i has 0 (uncorrelated) steps!' %i)
        if faulty_simulations_present:
            print(' ==> Either exclude these simulations or increase the cv bin range')
        print('================================================================================')
        raise np.linalg.LinAlgError('Fisher information matrix is singular. See message above traceback for more details!') from err
    return cov

# 
### Routines for WHAM in 2D
def wham2d_hs(int Nsims, int Ngrid1, int Ngrid2, np.ndarray[object] trajectories, np.ndarray[double] bins1, np.ndarray[double] bins2, np.ndarray[long] Nis):
    '''
        Internal WHAM routine to compute the 2D CV histogram of each (biased) simulation from the given CV trajectories.

        :param Nsims: number of simulations that were done and for which the histogram needs to be constructed
        :type Nsims: int

        :param Ngrid1: number of points on the CV1 grid to be used for the histogram.
        :type Ngrid1: int

        :param Ngrid2: number of points on the CV2 grid to be used for the histogram.
        :type Ngrid2: int
        
        :param trajectories: array containing the simulation trajectory (i.e. [CV1,CV2] samples) for each (biased) simulation
        :type trajectories: np.ndarray

        :param bins1: bin edges for CV1 in the histogram. Should have length one larger than Ngrid1
        :type bins1: np.ndarray[double]

        :param bins2: bin edges for CV2 in the histogram. Should have length one larger than Ngrid2
        :type bins2: np.ndarray[double]

        :param Nis: Nis[j] is the number of samples in the trajectory of simulation j
        :type Nis: np.ndarray[long]

        :returns Hs: 3 dimensional array containing the (CV1,CV2) histograms for each of the (biased) simulations. Hs[i,k,l] represents the histogram value of bin (k,l) in (CV1,CV2)-space for the i-th simulation.
        :rtype: np.ndarray[long, ndim=3]
    '''
    cdef np.ndarray[long, ndim=3] Hs = np.zeros([Nsims, Ngrid1, Ngrid2], dtype=int)
    cdef np.ndarray[double, ndim=2] data
    cdef np.ndarray[double] edges
    cdef int i, N
    for i, data in enumerate(trajectories):
        if len(data)==0:
            raise ValueError("The trajectory of simulation %i does not contain any data anymore. Are you sure you didn't remove too much data in post processing, e.g. to get rid of equilibration steps?" %i)
        Hs[i,:,:], edges1, edges2 = np.histogram2d(data[:,0], data[:,1], [bins1, bins2], density=False)
        assert (bins1==edges1).all()
        assert (bins2==edges2).all()
        N = Hs[i,:,:].sum()
        if Nis[i]!=N:
            print('WARNING: Histogram of trajectory %i should have total count of %i (=number of simulation steps), but found %f (are you sure the CV range is sufficient?). Number of simulation steps adjusted to match total histogram count.' %(i,Nis[i],N))
            Nis[i] = N
    return Hs


#
def wham2d_bias(int Nsims, int Ngrid1, int Ngrid2, double beta, list biasses, double delta1, double delta2, int bias_subgrid_num1, int bias_subgrid_num2, np.ndarray[double] bin_centers1, np.ndarray[double] bin_centers2, double threshold=1e-3):
    '''
        Compute the integrated boltzmann factors of the biases in each grid interval:

        .. math:: b_ikl = \\frac{1}{\\delta1\\cdot\\delta2}\\int_{Q_{1,k}-\\frac{\\delta1}{2}}^{Q_{1,k}+\\frac{\\delta1}{2}}\\int_{Q_{2,l}-\\frac{\\delta2}{2}}^{Q_{2,l}+\\frac{\\delta2}{2}} e^{-\\beta W_i(q_1,q_2)}dq_1dq_2

        This routine implements a conservative algorithm which takes into account that for a given simulation i, only a limited number of grid points (k,l) will give rise to a non-zero b_ikl. This is achieved by first using the bin-center approximation to the integral by computing the factor :math:`\\exp(-\\beta\\cdot bias(Q_{1,k},Q_{2,l}))` on the CV1 and CV2 grid (which is faster as there is no integral involved and which is also already a good approximation for the :math:`b_{ikl}` array) and only performing the precise integral when the approximation exceeds a threshold.

        :param Nsims: the total number of simulations performed
        :type Nsims: int

        :param Ngrid1: number of points on the CV1 grid to be used for the histogram.
        :type Ngrid1: int

        :param Ngrid2: number of points on the CV2 grid to be used for the histogram.
        :type Ngrid2: int

        :param beta: 1/kT in atomic units with T the temperature at which the simulation was performed
        :type beta: float

        :param biasses: list of bias potentials that were applied for each of the simulations.
        :type biasses: list of instances of child classes of :py:class:`BiasPotential1D <thermolib.thermodynamics.bias.BiasPotential1D>`

        :param delta1: width of CV1-interval (:math:`\\delta_1` in equation above) over which the bias is integrated when the bin-center approximation is not sufficient (see doc above) 
        :type delta1: float

        :param delta2: width of CV2-interval (:math:`\\delta_2` in equation above) over which the bias is integrated when the bin-center approximation is not sufficient (see doc above) 
        :type delta2: float

        :param bias_subgrid_num1: the number of grid points along CV1 used by the :py:meth:`wham2d_bias <thermolib.ext.wham2d_bias>` routine for the sub-grid to compute the boltzmann-integrated bias factors in each (CV1,CV2) bin.
		:type bias_subgrid_num: int

        :param bias_subgrid_num2: the number of grid points along CV2 used by the :py:meth:`wham2d_bias <thermolib.ext.wham2d_bias>` routine for the sub-grid to compute the boltzmann-integrated bias factors in each (CV1,CV2) bin.
		:type bias_subgrid_num2: int

        :param bin_centers1: array containing the center (in CV1 direction) of each bin in the histogram for application in the bin-center approximation (see doc above)
        :type bin_centers1: np.ndarray[double]

        :param bin_centers2: array containing the center (in CV2 direction) of each bin in the histogram for application in the bin-center approximation (see doc above)
        :type bin_centers2: np.ndarray[double]

        :param threshold: see general documentation above
        :type thresholdm: float, optional, default=1e-3

        :returns: array containing the (integrated) bias on the CV-grid for each simulation, i.e. :math:`b_{ik}` array from equation above. bs[i,k,l] represents the bias value of bin (k,l) in (CV1,CV2)-space for the i-th simulation.
        :rtype: np.ndarray[double, shape=(Nsims,Ngrid1,Ngrid2)]
    '''
    cdef np.ndarray[double, ndim=3] bs = np.zeros([Nsims, Ngrid1, Ngrid2], dtype=float)
    cdef np.ndarray[double, ndim=2] CV1, CV2, Ws
    cdef int i, k, l
    cdef double subdelta1 = delta1/bias_subgrid_num1, subdelta2 = delta2/bias_subgrid_num2
    cdef np.ndarray[double] subgrid1 = np.arange(-delta1/2, delta1/2 + subdelta1, subdelta1), subgrid2 = np.arange(-delta2/2, delta2/2 + subdelta2, subdelta2)
    for i, bias in enumerate(biasses):
        #first compute bin-center approximation
        CV1, CV2 = np.meshgrid(bin_centers1, bin_centers2, indexing='ij')
        bs[i,:,:] = np.exp(-beta*bias(CV1, CV2))
        #find large bias factors for exact computation
        ks, ls = np.where(bs[i,:,:]>threshold)
        #perform exact computation
        for (k,l) in zip(ks,ls):
            CV1, CV2 = np.meshgrid(bin_centers1[k]+subgrid1, bin_centers2[l]+subgrid2, indexing='ij')
            Ws = np.exp(-beta*bias(CV1, CV2))
            bs[i,k,l] = integrate2d_c(Ws, subdelta1, subdelta2)/(delta1*delta2)
    return bs


#
def wham2d_scf(np.ndarray[long] Nis, np.ndarray[long, ndim=3] Hs, np.ndarray[double, ndim=3] bs, np.ndarray[double, ndim=2] pinit, int Nscf=1000, double convergence=1e-6, double overflow_threshold=1e-150, verbose=False):
    '''
        Internal WHAM routine to solve the 2D WHAM equations which can, after flattening the 2D unbiased probability histogram :math:`a_{kl}` to a 1D array :math:`a_{k}`, be written as

        .. math::

            \\frac{1}{f_i} &= \\sum_k b_{ik} a_k \\\\
            a_k &= \\sum_i \\frac{H_{ik}}{\\sum_i N_i f_i b_{ik}}

        which clearly requires an iterative solution. To avoid floating point errors in the calculation of the first equation, we check whether f_i > overflow_threshold.
        Similarly, we check whether the denominator of equation 2 > overflow_threshold. This corresponds to only keeping those simulations with relevant information.

        :param Nis: the number of simulation steps in each simulation
        :type Nis: np.ndarray[double, shape=(Nsims,)]

        :param Hs: the histogram counts for each simulation, for each grid point
        :type Nis: np.ndarray[double, shape=(Nsims, Ngrid1, Ngrid2)]

        :param bs: the biasses (for each simulation) precomputed on the 2D CV grid
        :type bs: np.ndarray[double, shape=(Nsims, Ngrid1, Ngrid2)]

        :param pinit: the initial unbiased probability density
        :type pinit: np.ndarray[double, shape=(Ngrid1, Ngrid2)]

        :param Nscf: maximum number of scf cycles, convergence should be reached before Nscf steps
        :type Nscf: int, optional, default=1000

        :param convergence: convergence criterion for scf cycle, if integrated difference (sum of the absolute element-wise difference between subsequent predictions of the unbiased probability a_k) is lower than this value, SCF is converged
        :type convergence: double, optional, default=1e-6

        :param overflow_threshold: numerical threshold to avoid overflow errors when calculating the normalization factors and the denominator of the unbiased probability a_k. This determines which simulations and which grid points to ignore. Decreasing it results in a FES with a larger maximum free energy (lower unbiased probability). If it is too low, imaginary errors (sigma^2) arise, so increase if necessary.
        :type overflow_threshold: double, optional, default=1e-150

        :returns: the unbiased probability distribution on the same grid as all the histograms in the Hs argument.
        :rtype: np.ndarray[double, shape=(Ngrid1,Ngrid2)]
    '''
    cdef double integrated_diff, pmax
    cdef np.ndarray[double, ndim=2] as_old, as_new
    cdef np.ndarray[double] fs
    cdef np.ndarray[long, ndim=2] nominator
    cdef np.ndarray[double, ndim=2] denominator
    cdef np.ndarray[np.uint8_t] sims_mask, new_sims_mask
    cdef np.ndarray[np.uint8_t, ndim=2] grid_mask
    cdef int Ngrid1, Ngrid2, Nsims, iscf
    cdef int converged = 0
    ##initialize
    Nsims = Hs.shape[0]
    Ngrid1 = Hs.shape[1]
    Ngrid2 = Hs.shape[2]
    as_old = pinit.copy()
    nominator = Hs.sum(axis=0) #precomputable factor in WHAM update equations
    sims_mask = np.ones(Nsims,dtype=bool)
    grid_mask = np.ones((Ngrid1, Ngrid2),dtype=bool)

    for iscf in range(Nscf):
        #compute new normalization factors
        inverse_fs = np.einsum('ikl,kl->i', bs, as_old)

        # Calculate mask for simulations
        if np.any(inverse_fs<overflow_threshold):
            new_sims_mask = (inverse_fs>=overflow_threshold)
            if not np.array_equiv(new_sims_mask,sims_mask):
                sims_mask = new_sims_mask.copy()
                # Recalculate nominator when sims_mask changes
                nominator = Hs[sims_mask,:,:].sum(axis=0)

        #compute new probabilities
        denominator = np.einsum('i,i,ikl->kl', Nis[sims_mask], 1.0/inverse_fs[sims_mask], bs[sims_mask,:,:])

        # Calculate mask for grid
        if np.any(denominator<overflow_threshold):
            grid_mask = (denominator>=overflow_threshold)
            # check whether Hs is close to 0 for those points in the sims_mask that would be taken out by grid_mask, then we have as_new = 0/0
            if not np.isclose(np.sum(Hs[sims_mask][:,~grid_mask]),0.):
                warnings.warn('Grid indices are being masked that contain a total of at least 1 histogram count ({}/{} = {:2.4f}%).'.format(np.sum(Hs[sims_mask][:,~grid_mask]),np.sum(Hs[sims_mask,:]),(np.sum(Hs[sims_mask][:,~grid_mask])/np.sum(Hs[sims_mask,:]))*100))

        as_new = np.zeros((Ngrid1, Ngrid2)) # if a is zero, it will be ignored in both fs and the error calculation
        as_new[grid_mask] = np.divide(nominator[grid_mask],denominator[grid_mask])
        as_new[grid_mask] /= np.sum(as_new) #enforce normalization

        #check convergence
        integrated_diff = np.sum(np.abs(as_new-as_old))
        if verbose:
            pmax = np.max(as_new)
            print('cycle %i/%i' %(iscf+1,Nscf))
            print('  norm prob. dens. = %.3e au' %np.sum(as_new))
            print('  max prob. dens.  = %.3e au' %pmax)
            print('  Integr. Diff.    = %.3e au' %integrated_diff)
            print('')
        if integrated_diff<convergence:
            converged = 1
            break
        else:
            if iscf==Nscf-1:
                print('WARNING: could not converge WHAM equations to convergence of %.3e in %i steps!' %(convergence, Nscf))
        as_old = as_new.copy()

    if verbose:
        print('WARNING: the following simulations were removed from the WHAM analysis because the bias was centered in a region with a prohibitively low unbiased probability: ' + ','.join([str(i) for i in np.where(~sims_mask)[0]]))
        print('WARNING: the following grid locations were ignored in the WHAM analysis due to extremely unlikely probability:')
        lines = []
        for i in range(Ngrid1):
            lines += [["|"] + ["."]*Ngrid2 + ["|"]]
        for i,j in np.argwhere(~grid_mask):
            lines[i][j+1]="x"
        for line in lines:
            print("".join([s for s in line]))

    #compute final normalization factors
    fs = np.full(Nsims, np.nan)
    fs[sims_mask] = 1.0/np.einsum('ikl,kl->i', bs[sims_mask], as_new)
    return as_new, fs, converged


#
def wham2d_error(np.ndarray[double, ndim=2] ps, np.ndarray[double] fs, np.ndarray[double, ndim=3] bs, np.ndarray[long] Nis, np.ndarray[double] corrtimes, method='mle_f', p_threshold=0.0, verbosity='off'):
    '''
        Internal WHAM routine that allows to compute the error distribution on the unbiased probability distirbution that is estimated using the WHAM equations. The error estimation is based on the interpretation of the WHAM solutio as a Maximum Likelihood Estimater (MLE), which in term allows to estimate the error based on the Fisher information matrix.

        :param ps: the unbiased probability density as returned by the :py:meth:`wham2d_scf <thermolib.ext.wham2d_scf>` routine. Can contain nan values at specific grid locations (corresponding to disabled bins)
        :type ps: np.ndarray[double, shape=(Ngrid1, Ngrid2)]

        :param fs: renormalisation factors fs figuring in the WHAM equations. These are computed as intermediate variables in the :py:meth:`wham2d_scf <thermolib.ext.wham2d_scf>` routine. Can contain nan values at specific sim indices (at disabled simulations).
        :type fs: np.ndarray[double, shape=(Nsims,)]
         
        :param bs: the biasses (for each simulation) precomputed on the 2D CV grid
        :type bs: np.ndarray[double, shape=(Nsim,Ngrid1,Ngrid2)]

        :param Nis: the number of simulation steps in each simulation
        :type Nis: np.ndarray[long, shape=(Nsim,)]

        :param corrtimes: array of (integrated) correlation times of the CV, one for each simulation. Such correlation times will be taken into account during the error estimation and hence make it more reliable. If set to None, the CV trajectories will be assumed to contain fully uncorrelated samples (which is not true when using trajectories representing each subsequent step from a molecular dynamics simulation). More information can be found in :ref:`the user guide <seclab_ug_errorestimation>`. This input can be generated using the :py:meth:`decorrelate <thermolib.tools.decorrelate>` routine.
        :type corrtimes: np.ndarray[double, shape=(Nsims,)]

        :param method: specification of the method on how to perform the estimation of the error distribution. One of following options is available:

				- **mle_p** - Estimating the error directly for the probability of each bin in the histogram. This method does not explicitly impose the positivity of the probability.

				- **mle_p_cov** - Estimate the full covariance matrix for the probability of all bins in the histogram. In other words, appart from the error on the probability/free energy of a bin itself, we now also account for the covariance between the probabilty/free energy of the bins. This method does not explicitly impose the positivity of the probability.

				- **mle_f** - Estimating the error for minus the logarithm of the probability, which is proportional to the free energy (hence f in mle_f). As the probability is expressed as :math:`\propto e^{-f}`, its positivity is explicitly accounted for.

				- **mle_f_cov** - Estimate the full covariance matrix for minus the logarithm of the probability of all bins in the histogram. In other words, appart from the error on the probabilty/free energy of a bin itself (including explicit positivity constraint), we now also account for the covariance between the probability/free energy of the bins.
        
        :param p_threshold: only relevant when error estimation is enabled (see parameter ``error_estimate``). When ``error_p_threshold`` is set to x, bins in the histogram for which the probability resulting from the trajectory is smaller than x will be disabled for error estimation (i.e. its error will be set to np.nan). It is mainly usefull in the case of 2D histograms,as illustrated in :doc:`one of the tutorial notebooks <tut/advanced_projection>`.
        :type p_threshold: float, optional, default=0.0

        :param verbosity: specify the level of verbosity of the current routine
        :type verbosity: str, optional, default='off'

        :returns: the distribution of the error on the unbiased probability distribution
        :rtype: :py:class:`GaussianDistribution <thermolib.error.GaussianDistribution>` (if method='mle_p'), :py:class:`MultiGaussianDistribution <thermolib.error.MultiGaussianDistribution>` (if method='mle_p_cov'), :py:class:`LogGaussianDistribution <thermolib.error.LogGaussianDistribution>` (if method='mle_f'), :py:class:`MultiLogGaussianDistribution <thermolib.error.MultiLogGaussianDistribution>` (if method='mle_f_cov')
    '''
    from thermolib.error import GaussianDistribution, LogGaussianDistribution, MultiGaussianDistribution, MultiLogGaussianDistribution
    from thermolib.flatten import Flattener
    if verbosity.lower() in ['mediume', 'high']: print('  initializing ...')
    #initialization
    cdef np.ndarray[double] ps_flattened, logps_flattened
    cdef np.ndarray[double, ndim=2] I, Ii, Imask, sigma
    cdef np.ndarray[double, ndim=2] perr, logps
    cdef np.ndarray[double, ndim=4] pcov
    cdef np.ndarray[np.uint8_t, ndim=2] mask
    cdef int Nsims = bs.shape[0]
    cdef int Ngrid1 = bs.shape[1]
    cdef int Ngrid2 = bs.shape[2]
    cdef long Ngrid = Ngrid1*Ngrid2
    cdef int i, k, l, K
    cdef long Nmask, Nmask2

    flattener = Flattener(Ngrid1, Ngrid2)

    if verbosity.lower() in ['mediume', 'high']: print('  computing extended Fisher matrix ...')
    #Compute the extended Fisher matrix
    I = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])
    for i in range(Nsims):
        # do not take simulations into account that correspond to prohibitively large unbiased probabilities
        if not np.isnan(fs[i]):
            Ii = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])
            for k in range(Ngrid1):
                for l in range(Ngrid2):
                    if not np.isnan(ps[k,l]):
                        K = flattener.flatten_index(k,l)
                        if method in ['mle_p', 'mle_p_cov']:
                            if ps[k,l]>0: #see below where we define mask to filter out rows/columns corresponding to histogram counts of zero
                                Ii[K,K] = fs[i]*bs[i,k,l]/ps[k,l]
                            Ii[K, Ngrid+i] = bs[i,k,l]
                            Ii[Ngrid+i, K] = bs[i,k,l]
                            Ii[K, Ngrid+Nsims+i] = fs[i]*bs[i,k,l]
                            Ii[Ngrid+Nsims+i, K] = fs[i]*bs[i,k,l]
                            Ii[K,-1] = 1
                            Ii[-1,K] = 1
                        elif method in ['mle_f', 'mle_f_cov']:
                            Ii[K,K] = ps[k,l]*fs[i]*bs[i,k,l]
                            Ii[K, Ngrid+i] = -ps[k,l]*bs[i,k,l]
                            Ii[Ngrid+i, K] = -ps[k,l]*bs[i,k,l]
                            Ii[K, Ngrid+Nsims+i] = -ps[k,l]*fs[i]*bs[i,k,l]
                            Ii[Ngrid+Nsims+i, K] = -ps[k,l]*fs[i]*bs[i,k,l]
                            Ii[K,-1] = -ps[k,l]
                            Ii[-1,K] = -ps[k,l]
                        else:
                            raise IOError('Received invalid argument for method, recieved %s. Check routine signature for more information on allowed values.' %method)
            Ii[Ngrid+i, Ngrid+i] = 1/fs[i]**2
            Ii[Ngrid+i, Ngrid+Nsims+i] = 1/fs[i]
            Ii[Ngrid+Nsims+i, Ngrid+i] = 1/fs[i]
            I += Nis[i]/corrtimes[i]*Ii
    
    #Compute the inverse of the masked Fisher information matrix. Rows or columns corresponding to a histogram count of zero will have a (co)variance set to nan
    sigma = wham2d_invert_fisher_to_covariance(I, ps, fs, bs, Nis/corrtimes, flattener, p_threshold=p_threshold, verbosity=verbosity)

    #the error bar on the probability of bin (k,l) is now simply the [K,K]-diagonal element of sigma (with K corresponding to the flattened (k,l))
    if verbosity.lower() in ['mediume', 'high']: print('  constructing error bars ...')
    if method in ['mle_p', 'mle_f']:
        #get deflattened error bars (diagonal elements in flattened sigma matrix)
        perr = np.zeros([Ngrid1,Ngrid2], dtype=float)
        var = np.array(sigma.diagonal())[:Ngrid]
        if np.any(var<0):
            warnings.warn('Negative sigma**2 values encountered. This is likely because of numerical noise. Increase the overflow threshold to fix this.')
            var[var<0] *= -1.0
        perr = flattener.unflatten_array(np.sqrt(var))
        if method in ['mle_p']:
            return GaussianDistribution(ps, perr)
        elif method in ['mle_f']:
            #if ps is LogNormal distributed, then the means argument of LogGaussianDistribution is log(ps). The error calculated by MLE-F represents the error on log(ps) so this can be fed directly to the LogGaussianDistribution err argument.
            logps = np.zeros([Ngrid1,Ngrid2])*np.nan
            logps[ps>0] = np.log(ps[ps>0])
            return LogGaussianDistribution(logps, perr)
        else:
            raise RuntimeError('Something went wrong') #it should never get here
    elif method in ['mle_p_cov', 'mle_f_cov']:
        #get flattened probability
        ps_flattened = flattener.flatten_array(ps)
        if method in ['mle_p_cov']:
            return MultiGaussianDistribution(ps_flattened, sigma[:Ngrid,:Ngrid], flattener=flattener)
        elif method in ['mle_f_cov']:
            #if ps is LogNormal distributed, then the means argument of MultiLogGaussianDistribution is log(ps). The error calculated by MLE-F represents the error on log(ps) so this can be fed directly to the MultiLogGaussianDistribution err argument.
            #get flattened log probability
            logps_flattened = np.zeros(Ngrid, dtype=float)*np.nan
            logps_flattened[ps_flattened>0] = np.log(ps_flattened[ps_flattened>0])
            return MultiLogGaussianDistribution(logps_flattened, sigma[:Ngrid,:Ngrid], flattener=flattener)
        else:
            raise RuntimeError('Something went wrong') #it should never get here
    else:
        raise IOError('Recieved invalid argument for method, recieved %s. Check routine signature for more information on allowed values.' %method)


#
def wham2d_invert_fisher_to_covariance(np.ndarray[double, ndim=2] F, np.ndarray[double, ndim=2] ps, np.ndarray[double] fs, np.ndarray[double, ndim=3] bs, np.ndarray[double] Nis, object flattener, double p_threshold=0.0, verbosity='off'):
    cdef np.ndarray[double, ndim=2] F_mask
    cdef np.ndarray[double, ndim=2] cov = np.zeros([F.shape[0], F.shape[1]])*np.nan
    cdef np.ndarray[np.uint8_t, ndim=2, cast=True] mask
    cdef int k, l, K, i, N_mask2, N_mask, Ngrid, Nsims, num_deactivated_bins, num_deactivated_sims
    cdef double p, p_biased, Ni
    cdef np.uint8_t reached_end, faulty_simulations_present, sufficiently_sampled
    cdef np.ndarray[double] arr
    
    #before computing the inverse of the Fisher matrix, we define a mask to remove all rows/columns corresponding to:
    # - unpopulated bins (as there is no information and hence also no error bar on them)
    # - insufficiently sampled bins (determined by the biased probability)
    # - faulty simulations (corresponding fi is nan)
    
    if verbosity.lower() in ['medium', 'high']: print('  defining zero-mask ...')
    mask = np.ones([F.shape[0], F.shape[1]], dtype=bool)
    num_deactivated_bins = 0
    Nsims = len(Nis)
    Ngrid = flattener.dim12
    for (k,l), p in np.ndenumerate(ps):
        K = flattener.flatten_index(k,l)
        if np.isnan(p) or p==0.0:
            mask[K,:] = 0
            mask[:,K] = 0
            num_deactivated_bins += 1
            if verbosity.lower() in ['high']: print('    deactivated bin (%i,%i) because p=0,nan' %(k,l))
        else:
            sufficiently_sampled = False
            for i in range(Nsims):
                p_biased = fs[i]*bs[i,k,l]*p
                if not np.isnan(fs[i]) and p_biased>p_threshold:
                    sufficiently_sampled = True
                    break
            if not sufficiently_sampled:
                mask[K,:] = 0
                mask[:,K] = 0
                num_deactivated_bins += 1
                if verbosity.lower() in ['high']: print('    deactivated bin (%i,%i) because p_biased to low' %(k,l))
    if verbosity.lower() in ['medium', 'high']: print('    deactivated %i bins' %num_deactivated_bins)

    num_deactivated_sims = 0
    for i in range(Nsims):
        if np.isnan(fs[i]): 
            mask[Ngrid+i,:] = 0
            mask[:, Ngrid+i] = 0
            mask[Ngrid+Nsims+i,:] = 0
            mask[:, Ngrid+Nsims+i] = 0
            num_deactivated_sims += 1
            if verbosity.lower() in ['high']:  print('    deactivated sim %i because f=nan' %i)
    if verbosity.lower() in ['medium', 'high']: print('    deactivated %i sims' %num_deactivated_sims)

    if verbosity.lower() in ['medium', 'high']: print('  applying zero-mask ...')
    N_mask2 = len(F[mask])
    assert abs(int(np.sqrt(N_mask2))-np.sqrt(N_mask2))==0 #consistency check, sqrt(N_mask2) should be integer valued
    N_mask = int(np.sqrt(N_mask2))
    F_mask = F[mask].reshape([N_mask,N_mask])
    
    if verbosity.lower() in ['medium', 'high']: print('  inverting Fisher matrix ...')
    try:
        cov[mask] = np.linalg.inv(F_mask).reshape(N_mask2)
    except np.linalg.LinAlgError as err:
        print('===================== ERROR during Fisher inversion ============================')
        print(' Could not invert the Fisher information matrix because it is singular! This    ')
        print(' could be due to inclusion of simulations without any steps (possibly after     ')
        print(' decorrelation) within the defined cv bin range. The number of (uncorrelated)   ')
        print(' simulation steps within the cv range in each subsequent simulation is:')
        reached_end = False
        i = 0
        while not reached_end:
            if i+10>=len(Nis):
                arr = Nis[i:]
                reached_end=True
            else:
                arr = Nis[i:i+10]
            print(' '+' '.join(['% 7i' %int(Ni) for Ni in arr]))
            i += 10
        print('')
        faulty_simulations_present = False
        for i, Ni in enumerate(Nis):
            if int(Ni)==0:
                faulty_simulations_present = True
                print('Simulation nr. %i has 0 (uncorrelated) steps!' %i)
        if faulty_simulations_present:
            print(' ==> Either exclude these simulations or increase the cv bin range')
        print('================================================================================')
        raise np.linalg.LinAlgError('Fisher information matrix is singular. See message above traceback for more details!') from err
    return cov


#
### Misc routines
def integrate_c(np.ndarray[double] xs, np.ndarray[double] ys):
    '''
        A simple integration method using the trapezoid rule

        :param xs: array containing function argument values on grid
        :type xs: np.ndarray

        :param ys: array containing function values on grid
        :type ys: np.ndarray
    '''
    assert len(xs)==len(ys)
    return 0.5*np.dot(ys[1:]+ys[:-1],xs[1:]-xs[:-1])

#
def integrate2d_c(np.ndarray[double, ndim=2] z, double dx, double dy):
    '''
        Integrates a regularly spaced 2D grid using the composite trapezium rule.

        :param z: 2 dimensional array containing the function values
        :type z: np.ndarray(flt)

        :param dx: grid spacing for first function argument. If not given, argument is used to determine grid spacing. Defaults to 1.
        :type dx: float

        :param dy: grid spacing for second function argument. If not given, argument is used to determine grid spacing. Defaults to 1.
        :type dy: float

        :return: integral value
        :rtype: float
    '''
    cdef double s1, s2, s3
    s1 = z[0,0] + z[-1,0] + z[0,-1] + z[-1,-1]
    s2 = np.sum(z[1:-1,0]) + np.sum(z[1:-1,-1]) + np.sum(z[0,1:-1]) + np.sum(z[-1,1:-1])
    s3 = np.sum(z[1:-1,1:-1])
    return 0.25*dx*dy*(s1 + 2*s2 + 4*s3)



