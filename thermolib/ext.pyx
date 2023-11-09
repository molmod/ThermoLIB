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
#
#cython: embedsignature=True

from molmod.units import parse_unit

import matplotlib.pyplot as pp
import warnings

import numpy as np
cimport numpy as np

import scipy

__all__ = [
    'wham1d_hs', 'wham1d_bias', 'wham1d_scf', 'wham1d_error',
    'wham2d_hs', 'wham2d_bias', 'wham2d_scf', 'wham2d_error',
]


def wham1d_hs(int Nsims, int Ngrid, np.ndarray[object] trajectories, np.ndarray[double] bins, np.ndarray[long] Nis):
    cdef np.ndarray[long, ndim=2] Hs = np.zeros([Nsims, Ngrid], dtype=int)
    cdef np.ndarray[double] data, edges
    cdef int i, N
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


def wham1d_bias(int Nsims, int Ngrid, double beta, list biasses, double delta, int bias_subgrid_num, np.ndarray[double] bin_centers, double thresshold=1e-3):
    '''
        Compute the integrated boltzmann factors of the biases in each grid interval:

        .. math:: b_ik = \\frac{1}{\\delta}\\int_{Q_k-\\frac{\\delta}{2}}^{Q_k+\\frac{\\delta}{2}} e^{-\\beta W_i(q)}dq

        This routine implements a conservative algorithm which takes into account that for a given simulation i, only a limited number of grid points k will give rise to a non-zero b_ik. This is achieved by first using the bin-center approximation to the integral by computing the factor exp(-beta*bias(Qk)) on the CV grid (which is faster as there is no integral involved and which is also already a good approximation for the b_ik array) and only performing the precise integral when the approximation exceeds a thresshold.
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
        ks = np.where(bs[i,:]>thresshold)[0]
        for k in ks:
            Ws = np.exp(-beta*bias(bin_centers[k]+subgrid))
            bs[i,k] = integrate_c(bin_centers[k]+subgrid, Ws)/delta
    return bs


def wham1d_scf(np.ndarray[long] Nis, np.ndarray[long, ndim=2] Hs, np.ndarray[double, ndim=2] bs, int Nscf=1000, double convergence=1e-6, double overflow_threshold=1e-150, verbose=False):
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
        
        #fs = np.zeros(Nsims)
        #for i in range(Nsims):
        #    fs[i] = 1.0/np.dot(bs[i,:],as_old)

        # Calculate mask for simulations
        if np.any(inverse_fs<overflow_threshold):
            new_sims_mask = (inverse_fs>=overflow_threshold)
            if not np.array_equiv(new_sims_mask,sims_mask):
                sims_mask = new_sims_mask
                # Recalculate nominator when sims_mask changes
                nominator = Hs[sims_mask,:].sum(axis=0)

        #compute new probabilities
        denominator = np.einsum('i,i,ik->k', Nis[sims_mask], 1.0/inverse_fs[sims_mask], bs[sims_mask])
        #denominator = sum([Nis[i]*fs[i]*bs[i,k] for i in range(Nsims)])

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


    """# calculate consistency error
    delta = np.zeros(Nsims)
    # delta_i = 1/M sum_k (P_b_ik - f_i b_ik P_k)**2
    # with P_b the biased probability density calculated from the histograms
    as_biased_sampled = ((Hs.T)/Hs.sum(axis=-1)).T # normalize per simulation (Nsims,Ngrid)

    # Jensen-Shannon
    for i in range(Nsims):
        delta[i] = scipy.spatial.distance.jensenshannon(as_biased_sampled[i,:],np.round(fs[i]*bs[i,:]*as_new,15))**2

    print(delta)"""

    """# MSE
    for i in range(Nsims):
        for k in range(Ngrid):
            delta[i] += (as_biased_sampled[i,k]-fs[i]*bs[i,k]*as_new[k])**2
        delta[i] /= Ngrid
    print(delta)"""

    return as_new, fs, converged


def wham1d_error(int Nsims, int Ngrid, np.ndarray[long] Nis, np.ndarray[double] ps, np.ndarray[double] fs, np.ndarray[double, ndim=2] bs, method='mle_f', int nsigma=2):
    cdef np.ndarray[double, ndim=2] I, Ii, Imask, sigma
    cdef np.ndarray[double] perr, plower, pupper
    cdef np.ndarray[np.uint8_t, ndim=2] mask
    cdef int i, k
    cdef long Nmask2, Nmask
    I = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1], dtype=float)
    for i in range(Nsims):
        Ii = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])
        for k in range(Ngrid):
            if method in ['mle_p']:
                #avoid singularity at ps[k]=0, this is an inherent isssue to mle_p not present in the mle_f error estimation
                if ps[k]>0:
                    Ii[k,k] = fs[i]*bs[i,k]/ps[k]
                Ii[k,Ngrid+i] = bs[i,k]
                Ii[Ngrid+i,k] = bs[i,k]
                Ii[k,Ngrid+Nsims+i] = fs[i]*bs[i,k]
                Ii[Ngrid+Nsims+i,k] = fs[i]*bs[i,k]
                Ii[k,-1] = 1
                Ii[-1,k] = 1
            elif method in ['mle_f']:
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
        I += Nis[i]*Ii
    #the covariance matrix is now simply the inverse of the total Fisher matrix. However, for histogram bins with count 0 (and hence probability 0), we cannot estimate the corresponding error. This can be seen above as ps[k]=0 gives an divergence. Therefore, we define a mask corresponding to only non-zero probabilities and define and inverted the masked information matrix
    mask = np.ones([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1], dtype=bool)
    for k in range(Ngrid):
        if ps[k]==0:
            mask[k,:] = np.zeros(Ngrid+2*Nsims+1, dtype=bool)
            mask[:,k] = np.zeros(Ngrid+2*Nsims+1, dtype=bool)
    Nmask2 = len(I[mask])
    assert abs(int(np.sqrt(Nmask2))-np.sqrt(Nmask2))==0
    Nmask = int(np.sqrt(Nmask2))
    Imask = I[mask].reshape([Nmask,Nmask])
    sigma = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])*np.nan
    sigma[mask] = np.linalg.inv(Imask).reshape([Nmask2])
    #the error bar on the probability of bin k is now simply the (k,k)-diagonal element of sigma
    perr = np.sqrt(np.diagonal(sigma)[:Ngrid])
    if method in ['mle_p']:
        plower = ps - nsigma*perr
        plower[plower<0] = 0.0
        pupper = ps + nsigma*perr
    elif method in ['mle_f']:
        plower = ps*np.exp(-nsigma*perr)
        pupper = ps*np.exp(nsigma*perr)
    else:
        raise IOError('Recieved invalid argument for method, recieved %s. Check routine signiture for more information on allowed values.' %method)
    return plower, pupper


def wham2d_hs(int Nsims, int Ngrid1, int Ngrid2, np.ndarray[object] trajectories, np.ndarray[double] bins1, np.ndarray[double] bins2, np.ndarray[long] Nis):
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


def wham2d_bias(int Nsims, int Ngrid1, int Ngrid2, double beta, list biasses, double delta1, double delta2, int bias_subgrid_num1, int bias_subgrid_num2, np.ndarray[double] bin_centers1, np.ndarray[double] bin_centers2, double thresshold=1e-3):
    '''
        Compute the integrated boltzmann factors of the biases in each grid interval:

        .. math:: b_ikl = \\frac{1}{\\delta1\\cdot\\delta2}\\int_{Q_{1,k}-\\frac{\\delta1}{2}}^{Q_{1,k}+\\frac{\\delta1}{2}}\\int_{Q_{2,l}-\\frac{\\delta2}{2}}^{Q_{2,l}+\\frac{\\delta2}{2}} e^{-\\beta W_i(q_1,q_2)}dq_1dq_2

        This routine implements a conservative algorithm which takes into account that for a given simulation i, only a limited number of grid points (k,l) will give rise to a non-zero b_ikl. This is achieved by first using the bin-center approximation to the integral by computing the factor exp(-beta*bias(Q1,k,Q2,l)) on the CV1 and CV2 grid (which is faster as there is no integral involved and which is also already a good approximation for the b_ikl array) and only performing the precise integral when the approximation exceeds a thresshold.
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
        ks, ls = np.where(bs[i,:,:]>thresshold)
        #perform exact computation
        for (k,l) in zip(ks,ls):
            CV1, CV2 = np.meshgrid(bin_centers1[k]+subgrid1, bin_centers2[l]+subgrid2, indexing='ij')
            Ws = np.exp(-beta*bias(CV1, CV2))
            bs[i,k,l] = integrate2d_c(Ws, subdelta1, subdelta2)/(delta1*delta2)
    return bs


def wham2d_scf(np.ndarray[long] Nis, np.ndarray[long, ndim=3] Hs, np.ndarray[double, ndim=3] bs, np.ndarray[double, ndim=2] pinit, int Nscf=1000, double convergence=1e-6, double overflow_threshold=1e-150, verbose=False):
    '''
        Internal routine to solve the 2D WHAM equations,

        * 1/f_i = sum_k b_ik a_k
        * a_k = sum_i H_ik / sum_i N_i f_i b_ik

        which clearly requires an iterative solution. To avoid floating point errors in the calculation of the first equation, we check whether f_i > overflow_threshold.
        Similarly, we check whether the denominator of eqution 2 > overflow_threshold. This corresponds to only keeping those simulations with relevant information.

        :param Nis: the number of simulation steps in each simulation
        :type Nis: np.ndarray(Nsims)

        :param Hs: the histogram counts for each simulation, for each grid point
        :type Nis: np.ndarray(Nsims, Ngrid1, Ngrid2)

        :param bs: the biasses (for each simulation) precomputed on the 2D CV grid
        :type bs: np.ndarray(Nsims,Ngrid1,Ngrid2)

        :param pinit: the initial unbiased probability density
        :type pinit: np.ndarray(Ngrid1, Ngrid2)

        :param Nscf: maximum number of scf cycles, convergence should be reached before Nscf steps
        :type Nscf: int, optional, default=1000

        :param convergence: convergence criterion for scf cycle, if integrated difference (sum of the absolute element-wise difference between subsequent predictions of the unbiased probability a_k) is lower than this value, SCF is converged
        :type convergence: double, optional, default=1e-6

        :param overflow_threshold: numerical threshold to avoid overflow errors when calculating the normalization factors and the denominator of the unbiased probability a_k
        :type overflow_threshold: double, optional, default=1e-150
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
                sims_mask = new_sims_mask
                # Recalculate nominator when sims_mask changes
                nominator = Hs[sims_mask,:,:].sum(axis=0)

        #compute new probabilities
        denominator = np.einsum('i,i,ikl->kl', Nis[sims_mask], 1.0/inverse_fs[sims_mask], bs[sims_mask])

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
            lines[i][j+1]=" "
        for line in lines:
            print("".join([s for s in line]))

    #compute final normalization factors
    fs = np.full(Nsims, np.nan)
    fs[sims_mask] = 1.0/np.einsum('ikl,kl->i', bs[sims_mask], as_new)
    return as_new, fs, converged


def wham2d_error(np.ndarray[double, ndim=2] ps, np.ndarray[double] fs, np.ndarray[double, ndim=3] bs, np.ndarray[long] Nis,  method='mle_f', int nsigma=2, verbose=False):
    '''
        Internal routine to compute the error associated with solving the 2D WHAM equations using the Fisher information coming from the Maximum Likelihood Estimator. The procedure is as follows:

        * construct the extended Fisher information matrix by taking the weighted sum of the Fisher information matrix of each simulation. This is very similar as in the 1D case. However, we first have to flatten the CV1,CV2 2D grid to a 1D CV12 grid. This is achieved with the flatten function (which flattens a 2D index to a 1D index). Deflattening of the CV12 grid to a 2D CV1,CV2 grid is achieved with the deflatten function (which deflattens a 1D index to a 2D index). Using the flatten function, the ps array is flattened and a conventional Fisher matrix can be constructed and inverted. Afterwards the covariance matrix is deflattened using the deflatten function to arrive to a multidimensional matrix giving (co)variances on the 2D probability array.
        * filter out the zero-rows and columns corresponding to absent histogram counts using the masking procedure
        * invert the masked extended Fisher matrix and use the square root of its diagonal elements to compute errors

        :param ps: the final unbiased probability density as computed by solving the WHAM equations.
        :type ps: np.ndarray(Ngrid1, Ngrid2)
        :note: contains nan values at specific grid locations

        :param fs: the final normalization factor for the biased probability density of each simulation as computed by solving the WHAM equations
        :type fs: np.ndarray(Nsim)
        :note: contains nan values at sim indices

        :param bs: the biasses (for each simulation) precomputed on the 2D CV grid
        :type bs: np.ndarray(Nsim,Ngrid1,Ngrid2)

        :param Nis: the number of simulation steps in each simulation
        :type Nis: np.ndarray(Nsims)

        :param method: Define the method for computing the error:

            * *mle_p*: the error is computed on the probability density directly. This method corresponds to ignoring the positivity constraints of the histogram parameters.
            * *mle_f*: the error is first computed on minus of the logarithm of the probability density (corresponding to the scaled free energy) and afterwards propagated to the probability density. This method corresponds to taking the positivity constraints of the histogram parameters explicitly into account.

        :type method: str, optional, default='mle_f'

        :param nsigma: specify the length of the error bar in terms of the number of sigmas. For example, a 2-sigma error bar (i.e. nsigma=2) would correspond to a 95% confidence interval.
        :type nsimga: float, optional, default=2

        :return: pupper, plower corresponding to the upper and lower value of the nsigma error bar
        :rtype: np.ndarray, np.ndarray
    '''
    if verbose:
        print('  initializing ...')
    #initialization
    cdef np.ndarray[double, ndim=2] I, Ii, Imask, sigma
    cdef np.ndarray[double, ndim=2] perr, plower, pupper
    cdef np.ndarray[np.uint8_t, ndim=2] mask
    cdef int Nsims = bs.shape[0]
    cdef int Ngrid1 = bs.shape[1]
    cdef int Ngrid2 = bs.shape[2]
    cdef long Ngrid = Ngrid1*Ngrid2
    cdef int i, k, l, K
    cdef long Nmask, Nmask2

    def flatten(int kk, int ll):
        return Ngrid2*kk+ll
    def deflatten(int KK):
        cdef int kk = int(K/Ngrid2)
        cdef int ll = KK - Ngrid2*kk
        return kk,ll

    if verbose:
        print('  computing extended Fisher matrix ...')
    #Compute the extended Fisher matrix
    I = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])
    for i in range(Nsims):
        # do not take simulations into account that correspond to prohibitively large unbiased probabilities
        if not np.isnan(fs[i]):
            Ii = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])
            for k in range(Ngrid1):
                for l in range(Ngrid2):
                    if not np.isnan(ps[k,l]):
                        K = flatten(k,l)
                        if method in ['mle_p']:
                            if ps[k,l]>0: #see below where we define mask to filter out rows/columns corresponding to histogram counts of zero
                                Ii[K,K] = fs[i]*bs[i,k,l]/ps[k,l]
                            Ii[K, Ngrid+i] = bs[i,k,l]
                            Ii[Ngrid+i, K] = bs[i,k,l]
                            Ii[K, Ngrid+Nsims+i] = fs[i]*bs[i,k,l]
                            Ii[Ngrid+Nsims+i, K] = fs[i]*bs[i,k,l]
                            Ii[K,-1] = 1
                            Ii[-1,K] = 1
                        elif method in ['mle_f']:
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
            I += Nis[i]*Ii


    #Define and apply mask to filter out zero counts in histogram (as no error can be computed on them)
    if verbose:
        print('  defining and applying zero-mask ...')
    mask = np.ones([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1], dtype=bool)

    for i in range(Nsims):
        if np.isnan(fs[i]): # acount for faulty simulations
            mask[Ngrid+i,:] = np.zeros(Ngrid+2*Nsims+1, dtype=bool)
            mask[:, Ngrid+i] = np.zeros(Ngrid+2*Nsims+1, dtype=bool)
            mask[Ngrid+Nsims+i,:] = np.zeros(Ngrid+2*Nsims+1, dtype=bool)
            mask[:, Ngrid+Nsims+i] = np.zeros(Ngrid+2*Nsims+1, dtype=bool)

    for k in range(Ngrid1):
        for l in range(Ngrid2):
            K = flatten(k,l)
            if np.isnan(ps[k,l]) or ps[k,l]==0: # account for not-sampled grid locations
                mask[K,:] = np.zeros(Ngrid+2*Nsims+1, dtype=bool)
                mask[:,K] = np.zeros(Ngrid+2*Nsims+1, dtype=bool)

    Nmask2 = len(I[mask])
    assert abs(int(np.sqrt(Nmask2))-np.sqrt(Nmask2))==0 #consistency check, sqrt(Nmask2) should be integer valued
    Nmask = int(np.sqrt(Nmask2))
    Imask = I[mask].reshape([Nmask,Nmask])

    #Compute the inverse of the masked Fisher information matrix. Rows or columns corresponding to a histogram count of zero will have a (co)variance set to nan
    if verbose:
        print('  computing inverse of masked flattened Fisher matrix ...')
    sigma = np.zeros([Ngrid+2*Nsims+1, Ngrid+2*Nsims+1])*np.nan
    sigma[mask] = np.linalg.inv(Imask).reshape([Nmask2])

    #the error bar on the probability of bin (k,l) is now simply the [K,K]-diagonal element of sigma (with K corresponding to the flattened (k,l))
    if verbose:
        print('  constructing error bars ...')
    perr = np.zeros([Ngrid1,Ngrid2], dtype=float)
    for K in range(Ngrid):
        k, l = deflatten(K)
        try:
            perr[k,l] = np.sqrt(sigma[K,K])
        except FloatingPointError: #negative sigma
            warnings.warn('Negative sigma**2 values encountered. This is likely because of numerical noise. Increase the overflow threshold to fix this.')
            if verbose:
                print('Got a negative sigma for grid point: ', k,l, sigma[K,K], ps[k,l])
            perr[k,l] = np.sqrt(-sigma[K,K])

    if method in ['mle_p']:
        plower = ps - nsigma*perr
        plower[plower<0] = 0.0
        pupper = ps + nsigma*perr
    elif method in ['mle_f']:
        plower = ps*np.exp(-nsigma*perr)
        pupper = ps*np.exp(nsigma*perr)
    else:
        raise IOError('Recieved invalid argument for method, recieved %s. Check routine signature for more information on allowed values.' %method)

    return pupper, plower

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
