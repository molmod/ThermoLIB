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
#
#cython: embedsignature=True

import numpy as np
cimport numpy as np

__all__ = [
    'wham1d_hs', 'wham1d_bias', 'wham1d_scf', 'wham1d_error',
    'wham2d_hs', 'wham2d_bias', 'wham2d_scf', 'wham2d_error',
]

def wham1d_hs(int Nsims, int Ngrid, np.ndarray[double, ndim=2] trajectories, np.ndarray[double] bins, np.ndarray[long] Nis):
    cdef np.ndarray[long, ndim=2] Hs = np.zeros([Nsims, Ngrid], dtype=int)
    cdef np.ndarray[double] data, edges
    cdef int i, N
    for i, data in enumerate(trajectories):
        Hs[i,:], edges = np.histogram(data, bins, density=False)
        assert (bins==edges).all()
        N = Hs[i,:].sum()
        if Nis[i]!=N:
            print("WARNING: the CV range you specified for the histogram does not cover the entire simulation range in trajectory %i. Simulation samples outside the given CV range were cropped out." %(i))
            Nis[i] = N
    return Hs


def wham1d_bias(int Nsims, int Ngrid, double beta, list biasses, double delta, int bias_subgrid_num, np.ndarray[double] bin_centers):
    '''
        Compute the integrated boltzmann factors of the biases in each grid interval:

        .. math:: b_ik = \\frac{1}{\\delta}\\int_{Q_k-\\frac{\\delta}{2}}^{Q_k+\\frac{\\delta}{2}} e^{-\\beta W_i(q)}dq
    '''
    from thermolib.tools import integrate
    cdef np.ndarray[double, ndim=2] bs = np.zeros([Nsims, Ngrid], dtype=float)
    cdef int i, k
    cdef double center, subdelta
    cdef np.ndarray[double] subgrid
    for i, bias in enumerate(biasses):
        for k, center in enumerate(bin_centers):
            subdelta = delta/bias_subgrid_num
            subgrid = np.arange(center-delta/2, center+delta/2 + subdelta, subdelta)
            bs[i,k] = integrate(subgrid, np.exp(-beta*bias(subgrid)))/delta
    return bs


def wham1d_scf(np.ndarray[long] Nis, np.ndarray[long, ndim=2] Hs, np.ndarray[double, ndim=2] bs, int Nscf=1000, double convergence=1e-6, verbose=False):
    cdef double integrated_diff, pmax
    cdef np.ndarray[double] as_old, as_new, fs
    cdef np.ndarray[long] nominator
    cdef int Ngrid, Nsims, iscf, i, k
    #initialization
    Nsims = Hs.shape[0]
    Ngrid = Hs.shape[1]
    as_old = np.ones(Ngrid)/Ngrid #probability density (should sum to 1)
    nominator = Hs.sum(axis=0) #precomputable nominator in WHAM equations
    for iscf in range(Nscf):
        #compute new normalization factors
        fs = np.zeros(Nsims)
        for i in range(Nsims):
            fs[i] = 1.0/np.dot(bs[i,:],as_old)
        #compute new probabilities
        as_new = as_old.copy()
        for k in range(Ngrid):
            denominator = sum([Nis[i]*fs[i]*bs[i,k] for i in range(Nsims)])
            as_new[k] = nominator[k]/denominator			
        as_new /= as_new.sum() #enforce normalization
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
            print('WHAM SCF Converged!')
            break
        else:
            if iscf==Nscf-1:
                print('WARNING: could not converge WHAM equations to convergence of %.3e in %i steps!' %(convergence, Nscf))
        as_old = as_new.copy()
    #compute final normalization factors
    fs = 1.0/np.einsum('ik,k->i', bs, as_new)
    return as_new, fs


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
                #avoid singularity at ps[k]=0, this is an inherint isssue to mle_p not present in the mle_f error estimation
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
    plower = ps - nsigma*perr
    plower[plower<0] = 0.0
    pupper = ps + nsigma*perr
    return plower, pupper


def wham2d_hs(int Nsims, int Ngrid1, int Ngrid2, np.ndarray[object] trajectories, np.ndarray[double] bins1, np.ndarray[double] bins2, np.ndarray[long] Nis):
    cdef np.ndarray[long, ndim=3] Hs = np.zeros([Nsims, Ngrid1, Ngrid2], dtype=int)
    cdef np.ndarray[double, ndim=2] data
    cdef np.ndarray[double] edges
    cdef int i, N
    for i, data in enumerate(trajectories):
        Hs[i,:,:], edges1, edges2 = np.histogram2d(data[:,0], data[:,1], [bins1, bins2], density=False)
        assert (bins1==edges1).all()
        assert (bins2==edges2).all()
        N = Hs[i,:,:].sum()
        if Nis[i]!=N:
            print('WARNING: Histogram of trajectory %i should have total count of %i (=number of simulation steps), but found %f (are you sure the CV range is sufficient?). Number of simulation steps adjusted to match total histogram count.' %(i,Nis[i],N))
            Nis[i] = N
    return Hs


def wham2d_bias(int Nsims, int Ngrid1, int Ngrid2, double beta, list biasses, double delta1, double delta2, int bias_subgrid_num1, int bias_subgrid_num2, np.ndarray[double] bin_centers1, np.ndarray[double] bin_centers2):
    '''
        Compute the integrated boltzmann factors of the biases in each grid interval:

        .. math:: b_ikl = \\frac{1}{\\delta1\\cdot\\delta2}\\int_{Q_{1,k}-\\frac{\\delta1}{2}}^{Q_{1,k}+\\frac{\\delta1}{2}}\\int_{Q_{2,l}-\\frac{\\delta2}{2}}^{Q_{2,l}+\\frac{\\delta2}{2}} e^{-\\beta W_i(q_1,q_2)}dq_1dq_2

    '''
    from thermolib.tools import integrate2d
    cdef np.ndarray[double, ndim=3] bs = np.zeros([Nsims, Ngrid1, Ngrid2], dtype=float)
    cdef int i, k, l
    cdef double center1, center2, subdelta1, subdelta2
    cdef np.ndarray[double] subgrid1, subgrid2
    for i, bias in enumerate(biasses):
        for k, center1 in enumerate(bin_centers1):
            subdelta1 = delta1/bias_subgrid_num1
            subgrid1 = np.arange(center1-delta1/2, center1+delta1/2 + subdelta1, subdelta1)
            for l, center2 in enumerate(bin_centers2):
                subdelta2 = delta2/bias_subgrid_num2
                subgrid2 = np.arange(center2-delta2/2, center2+delta2/2 + subdelta2, subdelta2)
                CV1, CV2 = np.meshgrid(subgrid1, subgrid2, indexing='ij')
                Ws = np.exp(-beta*bias(CV1,CV2))
                bs[i,k,l] = integrate2d(Ws, dx=subdelta1, dy=subdelta2)/(delta1*delta2)
    return bs


def wham2d_scf(np.ndarray[long] Nis, np.ndarray[long, ndim=3] Hs, np.ndarray[double, ndim=3] bs, np.ndarray[double, ndim=2] pinit, int Nscf=1000, double convergence=1e-6, verbose=False):
    cdef double integrated_diff, pmax
    cdef np.ndarray[double, ndim=2] as_old, as_new
    cdef np.ndarray[double] fs
    cdef np.ndarray[long, ndim=2] nominator
    cdef np.ndarray[double, ndim=2] denominator
    cdef int Ngrid1, Ngrid2, Nsims, iscf
    ##initialize
    Nsims = Hs.shape[0]
    Ngrid1 = Hs.shape[1]
    Ngrid2 = Hs.shape[2]
    as_old = pinit #(transpose because of difference in ij and xy indexing)
    nominator = Hs.sum(axis=0) #precomputable factor in WHAM update equations
    for iscf in range(Nscf):
        #compute new normalization factors
        fs = 1.0/np.einsum('ikl,kl->i', bs, as_old)
        #compute new probabilities
        as_new = np.zeros([as_old.shape[0], as_old.shape[1]], dtype=float)
        denominator = np.einsum('i,i,ikl->kl', Nis, fs, bs)
        as_new = np.divide(nominator,denominator)
        as_new /= as_new.sum() #enforce normalization
        #check convergence
        integrated_diff = np.abs(as_new-as_old).sum()
        if verbose:
            pmax = as_new.max()
            print('cycle %i/%i' %(iscf+1,Nscf))
            print('  norm prob. dens. = %.3e au' %as_new.sum())
            print('  max prob. dens.  = %.3e au' %pmax)
            print('  Integr. Diff.    = %.3e au' %integrated_diff)
            print('')
        if integrated_diff<convergence:
            print('WHAM SCF Converged!')
            break
        else:
            if iscf==Nscf-1:
                print('WARNING: could not converge WHAM equations to convergence of %.3e in %i steps!' %(convergence, Nscf))
        as_old = as_new.copy()
    #compute final normalization factors
    fs = 1.0/np.einsum('ikl,kl->i', bs, as_new)
    return as_new, fs