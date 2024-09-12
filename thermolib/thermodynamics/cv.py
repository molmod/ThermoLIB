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

from molmod.units import *
from molmod.constants import *
from molmod.io.xyz import XYZReader
from molmod.unit_cells import UnitCell

import numpy as np

__all__ = [
    'CenterOfMass', 'CenterOfPosition', 'NormalizedAxis', 'NormalToPlane', 'Distance', 'DistanceCOP', 'CoordinationNumber', 'OrthogonalDistanceToPore',
    'Average', 'Difference', 'Minimum', 'LinearCombination', 'DotProduct', 'DistOrthProjOrig'
]

class CollectiveVariable(object):

    type = None

    def __init__(self, name=None, unit_cell_pars=None):
        if name is None:
            name = self._default_name()
        self.name = name
        if unit_cell_pars is not None:
            self.unit_cell = UnitCell.from_parameters3(*unit_cell_pars)
        else:
            self.unit_cell = None

    def _default_name(self):
        return 'CV'
    
    def _unwrap(self, coords, ref=None):
        'This routine will unwrap the PBC conditions around the given ref, i.e. make sure that for each r in coords, r-ref is the shortest vec'
        if self.unit_cell is None:
            return coords.copy()
        else:
            if ref is None:
                ref = coords[0,:].copy()
            unwrapped = np.zeros(coords.shape, float)
            unwrapped[0,:] = coords[0,:].copy()
            for i, r in enumerate(coords):
                if i>0:
                    unwrapped[i,:] = ref + self.unit_cell.shortest_vector(r-ref)
            return unwrapped
    
    def compute(self, coords, deriv=True):
        raise NotImplementedError

class CenterOfMass(CollectiveVariable):

    type = 'vector'

    def __init__(self, indices, masses, name=None, unit_cell_pars=None):
        self.indices = indices
        self.masses = masses
        CollectiveVariable.__init__(self, name=name, unit_cell_pars=unit_cell_pars)

    def _default_name(self):
        return 'COM(%s)' %('-'.join([str(i) for i in self.indices]))
    
    def compute(self, coords, deriv=True):
        #unwrap coords of given indices with periodic boundary conditions if unit_cell is specified
        rs = self._unwrap(coords[self.indices])
        #Compute center of mass
        mass = 0.0
        com = np.zeros(3, float)
        if deriv:
            grad = np.zeros([3, len(coords), 3], float)
        for index, r in zip(self.indices, rs):
            mass += self.masses[index]
            com += self.masses[index]*r
            if deriv:
                grad[:,index,:] = np.identity(3)*self.masses[index]
        com /= mass
        if not deriv:
            return com
        else:
            grad /= mass
            return com, grad

class CenterOfPosition(CollectiveVariable):

    type = 'vector'

    def __init__(self, indices, name=None, unit_cell_pars=None):
        self.indices = indices
        CollectiveVariable.__init__(self, name=name, unit_cell_pars=unit_cell_pars)
    
    def _default_name(self):
        return 'COP(%s)' %('-'.join([str(i) for i in self.indices]))
    
    def compute(self, coords, deriv=True):
        #unwrap coords of given indices with periodic boundary conditions if unit_cell is specified
        rs = self._unwrap(coords[self.indices,:])
        cop = rs.mean(axis=0)
        if not deriv:
            return cop
        else:
            grad = np.zeros([3, len(coords), 3], float)
            grad[0, self.indices, 0] = 1/len(self.indices)
            grad[1, self.indices, 1] = 1/len(self.indices)
            grad[2, self.indices, 2] = 1/len(self.indices)
            return cop, grad

class NormalizedAxis(CollectiveVariable):
    '''Class to implement a vectorial CV that defines a normalized axis between two points, i.e. as the normalized difference of two input vectors.
    '''

    type = 'vector'

    def __init__(self, vec1, vec2, name=None, unit_cell_pars=None):
        self.vec1 = vec1
        self.vec2 = vec2
        CollectiveVariable.__init__(self, name=name, unit_cell_pars=unit_cell_pars)

    def _default_name(self):
        return 'NormalAxis(%s,%s)' %(self.vec1.name, self.vec2.name)
    
    def compute(self, coords, deriv=True):
        if not deriv:
            v1 = self.vec1.compute(coords, deriv=False)
            v2 = self.vec2.compute(self._unwrap(coords, ref=v1), deriv=False)
            norm = np.linalg.norm(v2-v1)
            return (v2-v1)/norm
        if deriv:
            v1, grad1 = self.vec1.compute(coords, deriv=True)
            v2, grad2 = self.vec2.compute(self._unwrap(coords, ref=v1), deriv=True)
            norm = np.linalg.norm(v2-v1)
            tmp = grad2 - grad1 - np.einsum('a,bic,b->aic',v2-v1, grad2-grad1,v2-v1)/norm**2
            return (v2-v1)/norm, tmp/norm

class NormalToPlane(CollectiveVariable):
    
    type = 'vector'
    
    def __init__(self, indices, name=None, unit_cell_pars=None):
        self.indices = indices
        CollectiveVariable.__init__(self, name=name, unit_cell_pars=unit_cell_pars)
    
    def _default_name(self):
        return 'NormalToPlane(%s)' %('-'.join([str(i) for i in self.indices]))

    def compute(self, coords, deriv=True):
        #this routine to compute the normal to the ring plane assumes that the n atoms that constitute
        #the ring are ordered and at the cornerpoints of a regular n-fold polygon.
        theta = 2*np.pi/len(self.indices)
        #unwrap ring coords with periodic boundary conditions if unit_cell is specified
        rs = self._unwrap(coords[self.indices,:])
        R1 = np.zeros(3, float)
        R2 = np.zeros(3, float)
        for i, r in enumerate(rs):
            R1 += np.cos((i+1)*theta)*r
            R2 += np.sin((i+1)*theta)*r
        vec = np.cross(R1, R2)
        v = np.linalg.norm(vec)
        normal = vec/v
        if not deriv:
            return normal, None
        else:
            grad = np.zeros([3, len(coords), 3], float)
            tensor = (np.identity(3)-np.outer(vec, vec)/v**2)/v
            for i, index in enumerate(self.indices):
                cosi = np.cos((i+1)*theta)
                sini = np.sin((i+1)*theta)
                for alpha in [0,1,2]:
                    e_alpha = np.zeros(3, float)
                    e_alpha[alpha] = 1.0
                    grad[:, index, alpha] = np.dot(tensor, cosi*np.cross(e_alpha,R2)+sini*np.cross(R1,e_alpha))
            return normal, grad

class DotProduct(CollectiveVariable):
    '''
    Class to implement a collective variable that is the dot product of two given vectors
    '''

    type = 'scalar'

    def __init__(self, vec1, vec2, name=None, unit_cell_pars=None):
        assert isinstance(vec1, CollectiveVariable) and vec1.type=='vector'
        assert isinstance(vec2, CollectiveVariable) and vec2.type=='vector'
        self.vec1 = vec1
        self.vec2 = vec2
        CollectiveVariable.__init__(self, name=name, unit_cell_pars=unit_cell_pars)
    
    def _default_name(self):
        return 'Dot(%s,%s)' %(self.vec1.name, self.vec2.name)
    
    def compute(self, coords, deriv=True):
        if not deriv:
            v1 = self.vec1.compute(coords, deriv=False)
            v2 = self.vec2.compute(coords, deriv=False)
            return np.dot(v1, v2)
        else:
            v1, grad1 = self.vec1.compute(coords, deriv=True)
            v2, grad2 = self.vec2.compute(coords, deriv=True)
            cv = np.dot(v1,v2)
            grad = np.einsum('ikl,i->kl', grad1, v2) + np.einsum('i,ikl->kl', v1, grad2)
            return cv, grad

class Distance(CollectiveVariable):
    '''
        Class to implement a collective variable representing the distance
        between two atoms given by index1 and index2.
    '''
    
    type = 'scalar'
    
    def __init__(self, index1, index2, name=None, unit_cell_pars=None):
        self.i1 = index1
        self.i2 = index2
        CollectiveVariable.__init__(self, name=name, unit_cell_pars=unit_cell_pars)

    def _default_name(self):
        return 'Distance(%i,%i)' %(self.i1, self.i2)

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

class DistanceCOP(CollectiveVariable):
    '''
        Class to implement a collective variable representing the distance
        between an atom (index1) and the center of position (i.e. the geometric
        center) of two other atoms (index2a and index2b).
    '''
    
    type = 'scalar'
    
    def __init__(self, index1, index2a, index2b, name=None, unit_cell_pars=None):
        self.i1  = index1
        self.i2a = index2a
        self.i2b = index2b
        CollectiveVariable.__init__(self, name=name, unit_cell_pars=unit_cell_pars)

    def _default_name(self):
        return 'DistanceCOP(%i,%i,%i)' %(self.i1, self.i2a, self.i2b)

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

class CoordinationNumber(CollectiveVariable):
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
    
    type = 'scalar'
    
    def __init__(self, pairs, r0=2.0*angstrom, nn=6, nd=12, name=None, unit_cell_pars=None):
        self.pairs = pairs
        self.r0 = r0
        self.nn = nn
        self.nd = nd
        CollectiveVariable.__init__(self, name=name, unit_cell_pars=unit_cell_pars)

    def _default_name(self):
        return 'CoordinationNumber([' + ' , '.join(['(%i,%i)' %(i,j) for i,j in self.pairs]) + '])'

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

class OrthogonalDistanceToPore(DotProduct):
    '''
        Class to implement a collective variable that represents the orthogonal
        distance between the center of mass of a guest molecule defined by its
        atom indices (guest_indices) on the one hand, and a pore ring defined
        by the atom indices of its constituting atoms (ring_indices) on the
        other hand.
    '''
    
    type = 'scalar'
    
    def __init__(self, ring_indices, guest_indices, masses, unit_cell_pars=None, name=None):
        self.guest_indices = guest_indices
        self.ring_indices = ring_indices
        com  = CenterOfMass(guest_indices, masses, unit_cell_pars=unit_cell_pars)
        cop  = CenterOfPosition(ring_indices, unit_cell_pars=unit_cell_pars)
        vec1 = Difference(cop, com)
        vec2 = NormalToPlane(ring_indices, unit_cell_pars=unit_cell_pars)
        DotProduct.__init__(self, vec1, vec2, name=name)
    
    def _default_name(self):
        return 'OrthogonalDistanceToPore(ring=[%s],guest=[%s])' %(
            ','.join([str(i) for i in self.ring_indices]), 
            ','.join([str(i) for i in self.guest_indices])
        )

class OrthogonalDistanceToPore_depricated(CollectiveVariable):
    '''
        Old implementation of OrthogonalDistanceToPore, is depricated and will be removed soon
    '''
    
    type = 'scalar'
    
    def __init__(self, ring_indices, guest_indices, masses, unit_cell_pars=None, name=None):
        self.com  = CenterOfMass(guest_indices, masses, unit_cell_pars=unit_cell_pars)
        self.cop  = CenterOfPosition(ring_indices, unit_cell_pars=unit_cell_pars)
        self.norm = NormalToPlane(ring_indices, unit_cell_pars=unit_cell_pars)
        CollectiveVariable.__init__(self, name=name, unit_cell_pars=unit_cell_pars)
    
    def _default_name(self):
        return 'OrthogonalDistanceToPore(ring=[%s],guest=[%s])' %(
            ','.join([str(i) for i in self.cop.indices]), 
            ','.join([str(i) for i in self.com.indices])
        )

    def compute(self, coords, deriv=True):
        if deriv:
            com , grad_com  = self.com.compute(coords, deriv=True)
            cop , grad_cop  = self.cop.compute(coords, deriv=True)
            norm, grad_norm = self.norm.compute(coords, deriv=True)
        else:
            com  = self.com.compute(coords, deriv=False)
            cop  = self.cop.compute(coords, deriv=False)
            norm = self.norm.compute(coords, deriv=False)
        #compute cv
        if self.unit_cell is not None:
            cv = np.dot(self.unit_cell.shortest_vector(com-cop), norm)
        else:
            cv = np.dot(com-cop, norm)
        #compute derivative
        if not deriv:
            return cv
        else:
            grad = np.einsum('bia,b->ia', grad_com-grad_cop, norm) + np.einsum('b,bia->ia', com-cop, grad_norm)
            return cv, grad

class Average(CollectiveVariable):
    '''
        Class to implement a collective variable representing the average of two
        other collective variables:

            CV = 0.5*(CV1 + CV2)
    '''
    
    type = None #depends on type of argument cvs
    
    def __init__(self, cv1, cv2, name=None, unit_cell_pars=None):
        assert cv1.type==cv2.type
        self.type = cv1.type
        self.cv1 = cv1
        self.cv2 = cv2
        CollectiveVariable.__init__(self, name=name, unit_cell_pars=unit_cell_pars)

    def _default_name(self):
        return '0.5*(%s+%s)' %(self.cv1.name, self.cv2.name)

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

class Difference(CollectiveVariable):
    '''
        Class to implement a collective variable representing the difference
        between two other collective variables:

            CV = CV2 - CV1
    '''
    
    type = None #depends on type of argument cvs
    
    def __init__(self, cv1, cv2, name=None, unit_cell_pars=None):
        assert cv1.type==cv2.type
        self.type = cv1.type
        self.cv1 = cv1
        self.cv2 = cv2
        CollectiveVariable.__init__(self, name=name, unit_cell_pars=unit_cell_pars)

    def _default_name(self):
        return 'Diff(%s,%s)' %(self.cv2.name, self.cv1.name)

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

class Minimum(CollectiveVariable):
    '''
        Class to implement a collective variable representing the minimum of two
        other collective variables:

            CV = min(CV1,CV2)
    '''
    
    type = 'scalar'
    
    def __init__(self, cv1, cv2, name=None):
        self.cv1 = cv1
        self.cv2 = cv2
        CollectiveVariable.__init__(self, name=name)

    def _default_name(self):
        return 'Min(%s,%s)' %(self.cv1.name, self.cv2.name)

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
    
class LinearCombination(CollectiveVariable):
    '''
        Class to implement a collective variable that is the linear combination of other collective variables

            CV = sum(coeffs[i]*cvs[i], i=1..N)
        
        in which cvs is the list of involved collective variables and coeffs is list of equal length with the corresponing coefficients
    '''
    
    type = None #depends on type of argument cvs
    
    def __init__(self, cvs, coeffs, name=None):
        assert len(cvs)==len(coeffs), "List of cvs and list of coefficients should be of equal length"
        for i in range(1,len(cvs)):
            assert cvs[i].type==cvs[0].type, 'cvs[%i] and cvs[0] are not both scalar CVS or vector CVS' %(i)
        self.type = cvs[0].type
        self.cvs = cvs
        self.coeffs = coeffs
        CollectiveVariable.__init__(self, name=name)

    def _default_name(self):
        return ''.join(['%+.2f%s' %(coeff,cv.name) for (coeff,cv) in zip(self.coeffs, self.cvs)])

    def compute(self, coords, deriv=True):
        #computation of value
        if not deriv:
            value = 0.0
            for coeff, cv in zip(self.coeffs, self.cvs):
                value += coeff*cv.compute(coords, deriv=False)
            return value
        else:
            value = 0.0
            grad = np.zeros(coords.shape)
            for coeff, cv in zip(self.coeffs, self.cvs):
                v,g = cv.compute(coords, deriv=True)
                value += coeff*v
                grad += coeff*g
            return value, grad

class DistOrthProjOrig(DotProduct):
    '''
    Class to implement a collective variable that represents distance between (1) the orthogonal projection of a position (defined by a vector CV denoted as pos) on the axis (defined by a normalized vector CV denoted as axis) through an origin (defined by a vector CV denoted as orig) and (2) the origin.
    '''

    type = 'scalar'

    def __init__(self, pos, axis, orig, name=None):
        self.pos = pos
        self.axis = axis
        self.orig = orig
        DotProduct.__init__(self, Difference(orig,pos), axis, name=name)

    def _default_name(self):
        return 'DistOrthProjOrig(orig=%s,pos=%s,axis=%s)' %(self.orig.name, self.pos.name, self.axis.name)

class FunctionCV(CollectiveVariable):
    def __init__(self, CV, function, derivative, name=None, unit_cell_pars=None):
        self.CV = CV
        self.function = function
        self.derivative = derivative
        CollectiveVariable.__init__(self, name=name, unit_cell_pars=unit_cell_pars)

    def _default_name(self):
        return 'fun(%s)' %(self.CV.name)
    
    def compute(self, coords, deriv=True):
        if deriv:
            cv, grad = self.CV.compute(coords, deriv=True)
            q = self.function(cv)
            qgrad = self.derivative(cv)*grad
            return q, qgrad
        else:
            cv = self.CV.compute(coords, deriv=False)
            return self.function(cv)

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
