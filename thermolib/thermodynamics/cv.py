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
from molmod.periodic import periodic as pt
from molmod.io.xyz import XYZReader, XYZFile
from molmod.minimizer import check_delta
from molmod.unit_cells import UnitCell

import numpy as np
import sys, os
import matplotlib.pyplot as pp

__all__ = [
    'Distance', 'DistanceCOP', 'CoordinationNumber', 'OrthogonalDistanceToPore',
    'Average', 'Difference', 'Minimum',
]

class Distance(object):
    '''
        Class to implement a collective variable representing the distance
        between two atoms given by index1 and index2.
    '''
    def __init__(self, index1, index2, name=None, unit_cell_pars=None):
        self.i1 = index1
        self.i2 = index2
        if unit_cell_pars is not None:
            lengths, angles = unit_cell_pars
            self.unit_cell = UnitCell.from_parameters3(lengths, angles)
        else:
            self.unit_cell = None
        if name is None:
            self.name = 'Distance(%i,%i)' %(index1, index2)
        else:
            self.name = name

    def compute(self, coords, deriv=True):
        '''
            Compute the value of the collective variable given the coordinates.
            If deriv is set to True, also compute and return the analytical
            derivative of Q towards the cartesian coordinates.
        '''
        #computation of value
        r1 = coords[self.i1,:]
        r2 = coords[self.i2,:]
        r = r2-r1
        if self.unit_cell is not None:
            r = self.unit_cell.shortest_vector(r)
        value = np.linalg.norm(r)
        if not deriv:
            return value
        #computation of deriv
        grad = np.zeros(coords.shape, float)
        grad[self.i1,:] += -r/value
        grad[self.i2,:] += r/value
        return value, grad


class DistanceCOP(object):
    '''
        Class to implement a collective variable representing the distance
        between an atom (index1) and the center of position (i.e. the geometric
        center) of two other atoms (index2a and index2b).
    '''
    def __init__(self, index1, index2a, index2b, name=None, unit_cell_pars=None):
        self.i1  = index1
        self.i2a = index2a
        self.i2b = index2b
        if unit_cell_pars is not None:
            lengths, angles = unit_cell_pars
            self.unit_cell = UnitCell.from_parameters3(lengths, angles)
        else:
            self.unit_cell = None
        if name is None:
            self.name = 'DistanceCOP(%i,%i,%i)' %(index1, index2a, index2b)
        else:
            self.name = name

    def compute(self, coords, deriv=True):
        #computation of value
        r1 = coords[self.i1,:]
        com = 0.5*(coords[self.i2a,:]+coords[self.i2b,:])
        r = r1-com
        if self.unit_cell is not None:
            r = self.unit_cell.shortest_vector(r)
        value = np.linalg.norm(r)
        if not deriv:
            return value
        #computation of deriv
        grad = np.zeros(coords.shape, float)
        grad[self.i1 ,:] += r/value
        grad[self.i2a,:] += -0.5*r/value
        grad[self.i2b,:] += -0.5*r/value
        return value, grad

class CoordinationNumber(object):
    '''
        Class to implement a collective variable representing the coordination
        number of a certain atom pair or a set of atom pairs. If n atom pairs
        are defined, the coordination number should be a number between 0 and n.

            CN = sum( (1-x_ij^nn)/(1-x_ij^nd), ij in pairs)

        with

            x_ij = r_ij/r0

        in which r_ij is the distance between the atoms of pair ij, r0 is a
        reference distance set to 2 angstrom by default but can ge defined by
        the user and nn and nd are integers that are set to 6,12 by default but
        can also be defined by the user.
    '''
    def __init__(self, pairs, r0=2.0*angstrom, nn=6, nd=12, name=None, unit_cell_pars=None):
        self.pairs = pairs
        self.r0 = r0
        self.nn = nn
        self.nd = nd
        if unit_cell_pars is not None:
            lengths, angles = unit_cell_pars
            self.unit_cell = UnitCell.from_parameters3(lengths, angles)
        else:
            self.unit_cell = None
        if name is None:
            self.name = 'CoordinationNumber([' + ' , '.join(['(%i,%i)' %(i,j) for i,j in pairs]) + '])'
        else:
            self.name = name

    def compute(self, coords, deriv=True):
        value = 0.0
        grad = np.zeros(coords.shape, float)
        for i,j in self.pairs:
            ri = coords[i,:]
            rj = coords[j,:]
            r = ri-rj
            if self.unit_cell is not None:
                r = self.unit_cell.shortest_vector(r)
            rij = np.linalg.norm(r)
            xij = rij/self.r0
            cij = (1-xij**self.nn)/(1-xij**self.nd)
            value += cij
            if deriv:
                T = (self.nn-self.nd)*xij**(self.nn+self.nd-1)+self.nd*xij**(self.nd-1)-self.nn*xij**(self.nn-1)
                N = (1-xij**self.nd)**2
                dcij_drij = 1./self.r0*T/N
                grad[i,:] += (ri-rj)/rij*dcij_drij
                grad[j,:] += (rj-ri)/rij*dcij_drij
        if deriv:
            return value, grad
        else:
            return value


class OrthogonalDistanceToPore(object):
    '''
        Class to implement a collective variable that represents the orthogonal
        distance between the center of mass of a guest molecule defined by its
        atom indices (guest_indices) on the one hand, and a pore ring defined
        by the atom indices of its constituting atoms (ring_indices) on the
        other hand.
    '''
    def __init__(self, ring_indices, guest_indices, masses, unit_cell_pars=None, name=None):
        self.ring_indices = np.array(ring_indices)
        self.guest_indices = np.array(guest_indices)
        if unit_cell_pars is not None:
            lengths, angles = unit_cell_pars
            self.unit_cell = UnitCell.from_parameters3(lengths, angles)
        else:
            self.unit_cell = None
        self.masses = masses
        if name is None:
            self.name = 'OrthogonalDistanceToPore(ring=[%s],guest=[%s])' %(','.join([str(i) for i in ring_indices]), ','.join([str(i) for i in guest_indices]))
        else:
            self.name = name

    def _unwrap(self, rs):
        unwrapped = np.zeros(rs.shape, float)
        ref = rs[0,:]
        unwrapped[0,:] = rs[0,:].copy()
        for i, r in enumerate(rs):
            if i>0:
                unwrapped[i,:] = ref + self.unit_cell.shortest_vector(r-ref)
        return unwrapped

    def _guest_com(self, coords, deriv=True):
        #unwrap guest coords with periodic boundary conditions if unit_cell is specified
        if self.unit_cell is not None:
            coords_guest = self._unwrap(coords[self.guest_indices,:])
        else:
            coords_guest = coords[self.guest_indices,:].copy()
        #Compute center of mass
        mass = 0.0
        com = np.zeros(3, float)
        if deriv:
            grad = np.zeros([3, len(self.masses), 3], float)
        for index, r in zip(self.guest_indices, coords_guest):
            mass += self.masses[index]
            com += self.masses[index]*r
            if deriv:
                grad[:,index,:] = np.identity(3)*self.masses[index]
        com /= mass
        if not deriv:
            return com, None
        else:
            grad /= mass
            return com, grad

    def _ring_center(self, coords, deriv=True):
        #unwrap ring coords with periodic boundary conditions if unit_cell is specified
        if self.unit_cell is not None:
            coords_ring = self._unwrap(coords[self.ring_indices,:])
        else:
            coords_ring = coords[self.ring_indices,:].copy()
        cop = coords_ring.mean(axis=0)
        if not deriv:
            return cop, None
        else:
            grad = np.zeros([3, len(self.masses), 3], float)
            grad[0, self.ring_indices, 0] = 1/len(self.ring_indices)
            grad[1, self.ring_indices, 1] = 1/len(self.ring_indices)
            grad[2, self.ring_indices, 2] = 1/len(self.ring_indices)
            return cop, grad

    def _ring_axis(self, coords, deriv=True):
        #this routine to compute the normal to the ring plane assumes that the n atoms that constitute
        #the ring are ordered and at the cornerpoints of a regular n-fold polygon.
        theta = 2*np.pi/len(self.ring_indices)
        #unwrap ring coords with periodic boundary conditions if unit_cell is specified
        if self.unit_cell is not None:
            coords_ring = self._unwrap(coords[self.ring_indices,:])
        else:
            coords_ring = coords[self.ring_indices,:].copy()
        R1 = np.zeros(3, float)
        R2 = np.zeros(3, float)
        for i, r in enumerate(coords_ring):
            R1 += np.cos((i+1)*theta)*r
            R2 += np.sin((i+1)*theta)*r
        vec = np.cross(R1, R2)
        v = np.linalg.norm(vec)
        normal = vec/v
        if not deriv:
            return normal, None
        else:
            grad = np.zeros([3, len(self.masses), 3], float)
            tensor = (np.identity(3)-np.outer(vec, vec)/v**2)/v
            for i, index in enumerate(self.ring_indices):
                cosi = np.cos((i+1)*theta)
                sini = np.sin((i+1)*theta)
                for alpha in [0,1,2]:
                    e_alpha = np.zeros(3, float)
                    e_alpha[alpha] = 1.0
                    grad[:, index, alpha] = np.dot(tensor, cosi*np.cross(e_alpha,R2)+sini*np.cross(R1,e_alpha))
            return normal, grad

    def _ring_axis2(self, coords, deriv=True):
        assert not deriv, 'Routine _ring_axis2 does not support analytic derivatives'
        #this routine computes the normal to the ring plane by means of singular value decomposition.
        points = coords[ring_indices, :] - self._ring_center(coords)
        u, s, vh = np.linalg.svd(points)
        return vh[-1,:], None

    def compute(self, coords, deriv=True):
        com   , grad_com    = self._guest_com(coords, deriv=deriv)
        center, grad_center = self._ring_center(coords, deriv=deriv)
        normal, grad_normal = self._ring_axis(coords, deriv=deriv)
        #compute cv
        if self.unit_cell is not None:
            cv = np.dot(self.unit_cell.shortest_vector(com-center), normal)
        else:
            cv = np.dot(com-center, normal)
        #compute derivative
        if not deriv:
            return cv
        else:
            grad = np.einsum('bia,b->ia', grad_com-grad_center, normal) + np.einsum('b,bia->ia', com-center, grad_normal)
            return cv, grad

class Average(object):
    '''
        Class to implement a collective variable representing the average of two
        other collective variables:

            CV = 0.5*(CV1 + CV2)
    '''
    def __init__(self, cv1, cv2, name=None):
        self.cv1 = cv1
        self.cv2 = cv2
        if name is None:
            self.name = '0.5*(%s+%s)' %(cv1.name, cv2.name)
        else:
            self.name

    def compute(self, coords, deriv=True):
        #computation of value
        if not deriv:
            cv1 = self.cv1.compute(coords, deriv=False)
            cv2 = self.cv2.compute(coords, deriv=False)
            return 0.5*(cv1+cv2)
        else:
            cv1, grad1 = self.cv1.compute(coords, deriv=True)
            cv2, grad2 = self.cv2.compute(coords, deriv=True)
            value = 0.5*(cv1+cv2)
            grad = 0.5*(grad1 + grad2)
        return value, grad


class Difference(object):
    '''
        Class to implement a collective variable representing the difference
        between two other collective variables:

            CV = CV2 - CV1
    '''
    def __init__(self, cv1, cv2, name=None):
        self.cv1 = cv1
        self.cv2 = cv2
        if name is None:
            self.name = '%s-%s' %(cv2.name, cv1.name)
        else:
            self.name = name

    def compute(self, coords, deriv=True):
        #computation of value
        if not deriv:
            cv1 = self.cv1.compute(coords, deriv=False)
            cv2 = self.cv2.compute(coords, deriv=False)
            return cv2-cv1
        else:
            cv1, grad1 = self.cv1.compute(coords, deriv=True)
            cv2, grad2 = self.cv2.compute(coords, deriv=True)
            value = cv2-cv1
            grad = grad2 - grad1
        return value, grad


class Minimum(object):
    '''
        Class to implement a collective variable representing the minimum of two
        other collective variables:

            CV = min(CV1,CV2)
    '''
    def __init__(self, cv1, cv2, name=None):
        self.cv1 = cv1
        self.cv2 = cv2
        if name is None:
            self.name = 'min(%s,%s)' %(cv1.name, cv2.name)
        else:
            self.name = name

    def compute(self, coords, deriv=True):
        #computation of value
        if not deriv:
            cv1 = self.cv1.compute(coords, deriv=False)
            cv2 = self.cv2.compute(coords, deriv=False)
            return min(cv1,cv2)
        else:
            cv1, grad1 = self.cv1.compute(coords, deriv=True)
            cv2, grad2 = self.cv2.compute(coords, deriv=True)
            value = min(cv1,cv2)
            if cv1<cv2:
                grad = grad1
            else:
                grad = grad2
        return value, grad
    
class LinearCombination(object):
    '''
        Class to implement a collective variable that is the linear combination of three collective variables:

            CV = aCV1 + bCV2 + cCV3
    '''
    def __init__(self, a, cv1, b, cv2, c, cv3, name=None):
        self.a = a
        self.cv1 = cv1
        self.b = b
        self.cv2 = cv2
        self.c = c
        self.cv3 = cv3
        if name is None:
            self.name = '%+.2f%s%+.2f%s%+.2f%s' %(a, cv1.name, b, cv2.name, c, cv3.name)
        else:
            self.name = name

    def compute(self, coords, deriv=True):
        #computation of value
        if not deriv:
            cv1 = self.cv1.compute(coords, deriv=False)
            cv2 = self.cv2.compute(coords, deriv=False)
            cv3 = self.cv3.compute(coords, deriv=False)
            return self.a*cv1+self.b*cv2+self.c*cv3
        else:
            cv1, grad1 = self.cv1.compute(coords, deriv=True)
            cv2, grad2 = self.cv2.compute(coords, deriv=True)
            cv3, grad3 = self.cv3.compute(coords, deriv=True)
            value = self.a*cv1+self.b*cv2+self.c*cv3
            grad = self.a*grad1+self.b*grad2+self.c*grad3
        return value, grad

def test_CV_implementations(fn, cvs, dx=0.001*angstrom, maxframes=100):
    xyz = XYZReader(fn)
    for cv in cvs:
        print('Testing consistency between value and gradient of %s' %cv.name)
        for i, (title, coords) in enumerate(xyz):
            if i>=maxframes: break
            value, grad = cv.compute(coords)
            for aindex in range(len(coords)):
                for carindex in range(3):
                    delta = np.zeros(coords.shape, float)
                    delta[aindex,carindex] = 1
                    v1 = cv.compute(coords-delta*dx, deriv=False)
                    v2 = cv.compute(coords+delta*dx, deriv=False)
                    numerical = (v2-v1)/(2*dx)
                    assert abs(numerical-grad[aindex,carindex])<1e-9, 'Analytical derivative check failed! grad[%i,%i]=%20.15e    numerical=%20.15e' %(aindex,carindex,grad[aindex,carindex],numerical)
    del xyz
