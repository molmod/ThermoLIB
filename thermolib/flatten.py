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

from __future__ import annotations

import numpy as np

__all__ =['DummyFlattener', 'Flattener']


class DummyFlattener(object):
    '''
        A dummy class for when no flattening is required but it is mandatory to parse a flattener instance for which the (un)flatten_array and (un)flatten_matrix routines can be called.
    '''
    def __init__(self):
        self.dim1 = None
        self.dim2 = None
        self.dim12 = None

    def flatten_index(self, k, l):
        raise NotImplementedError
    
    def unflatten_index(self, K):
        raise NotImplementedError

    def flatten_array(self, arr):
        return arr

    def unflatten_array(self, arr):
        return arr
    
    def flatten_matrix(self, mat):
        return mat
    
    def unflatten_matrix(self, mat):
        return mat



class Flattener(object):
    "Class to flatten a 2D grid onto a 1D grid. This is used in order to deal with the covariance matrix of a 2D free energy surface. The class contains routines to convert a 2D index (k,l) on to a 1D index K, to flatten a 2D array to a 1D array as well as to flatten a 4D matrix to a 2D matrix (as well as all the inverse routines.)"
    def __init__(self, dim1, dim2):
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim12 = dim1*dim2

    def flatten_index(self, k, l):
        assert k<self.dim1, 'k(=%i) should be smaller than dim1(=%i)' %(l,self.dim1)
        assert l<self.dim2, 'l(=%i) should be smaller than dim2(=%i)' %(l,self.dim2)
        return self.dim2*k + l

    def unflatten_index(self, K):
        assert K<self.dim12, 'K(=%i) should be smaller than dim12(=%i)' %(K,self.dim12)
        k = int(K/self.dim2)
        l = K - self.dim2*k
        return k, l

    def flatten_array(self, arr):
        assert len(arr.shape)==2, 'array should be two dimensional, but got %i dimensional input' %(len(arr.shape))
        assert arr.shape[0]==self.dim1, 'first dimension of array should be of length %i, but got %i' %(self.dim1,arr.shape[0])
        assert arr.shape[1]==self.dim2, 'second dimension of array should be of length %i, but got %i' %(self.dim2,arr.shape[1])
        arr_flattened = np.zeros(self.dim12, dtype=float)
        for (k,l), val in np.ndenumerate(arr):
            K = self.flatten_index(k,l)
            arr_flattened[K] = val
        return arr_flattened

    def unflatten_array(self, arr):
        assert len(arr.shape)==1, 'array should be one dimensional, but got %i dimensional input' %(len(arr.shape))
        assert len(arr)==self.dim12, 'array should be of length %i, but got %i' %(self.dim12,len(arr))
        arr_deflattened = np.zeros([self.dim1, self.dim2], dtype=float)
        for K, val in enumerate(arr):
            k, l = self.unflatten_index(K)
            arr_deflattened[k,l] = val
        return arr_deflattened
            
    def flatten_matrix(self, mat):
        mat_flattened = np.zeros([self.dim12, self.dim12], dtype=float)
        for (k,l,m,n), val in np.ndenumerate(mat):
            K = self.flatten_index(k,l)
            M = self.flatten_index(m,n)
            mat_flattened[K,M] = val
        return mat_flattened

    def unflatten_matrix(self, mat):
        mat_unflattened = np.zeros([self.dim1, self.dim2, self.dim1, self.dim2], dtype=float)
        for (K, M), val in np.ndenumerate(mat):
            k, l = self.unflatten_index(K)
            m, n = self.unflatten_index(M)
            mat_unflattened[k,l,m,n] = val
        return mat_unflattened