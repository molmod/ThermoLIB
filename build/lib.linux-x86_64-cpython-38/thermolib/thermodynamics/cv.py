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

from ..units import *
from ..constants import *

from ase.io import read
from ase import Atom

import numpy as np

__all__ = [
    'CenterOfMass', 'CenterOfPosition', 'NormalizedAxis', 'NormalToPlane', 'Distance', 'DistanceCOP', 'CoordinationNumber', 'OrthogonalDistanceToPore',
    'Average', 'Difference', 'Minimum', 'LinearCombination', 'DotProduct', 'DistOrthProjOrig'
]

class CollectiveVariable(object):

    '''
        Abstract class for definition of Collective Variable child classes that allow to compute a CV value from the coordinates of the molecular system.
    '''

    type = None

    def __init__(self, name=None):
        '''
            :param name: Name of CV for printing/logging purposes. If None, the default implemented in the ``_default_name`` routine will be used.
            :type name: str | None, optional, default=None
        '''

        if name is None:
            name = self._default_name()
        self.name = name

    def _default_name(self):
        return 'CV'
    
    def compute(self, atoms, deriv=True):
        '''
            This routine needs to be implemented in each child class
        '''
        raise NotImplementedError


class CenterOfMass(CollectiveVariable):
    '''
        Class the implement the computation of the center of mass (COM) of a set of atoms defined by their atomic indices.
    '''

    type = 'vector'

    def __init__(self, indices, masses=None, name=None):
        '''
            :param indices: indices of the atoms of which the COM needs to be computed
            :type indices: list of integers

            :param masses: masses of of the atoms listed in indices. If None, the masses are retrieved based on the chemical element.
            :type masses: np.ndarray

            :param name: Name of CV for printing/logging purposes. If None, the default implemented in the ``_default_name`` routine will be used.
            :type name: str | None, optional, default=None
        '''
        self.indices = indices
        self.masses = masses
        CollectiveVariable.__init__(self, name=name)

    def _default_name(self):
        return 'COM(%s)' %('-'.join([str(i) for i in self.indices]))
    
    def compute(self, atoms, deriv=True):
        '''
            Compute CV value (and optionally its gradient) for given coordinates of the molecular system.

            :param atoms: ASE Atoms object on which the CV needs to be computed
            :type atoms: ase.Atoms

            :param deriv: if True, also compute and return the gradient of the CV towards all atomic coordinates
            :type deriv: bool, optional, default=True

            :return: CV value and potentially the gradient
            :rtype: np.ndarray(3) or float,np.ndarray([3,Natoms,3])
        '''
        #Compute center of mass
        com = np.zeros(3, float)
        if deriv:
            grad = np.zeros([3, len(atoms), 3], float)
        
        if self.masses is None:
            self.masses = atoms.get_masses()[self.indices]
        else:
            assert len(self.masses)==len(self.indices), 'Number of masses should be equal to the number of indices'
        
        if atoms.get_pbc().any():
            atoms.wrap(center=atoms.get_positions()[0]) # wrap atoms to the first atom
        
        positions = atoms.get_positions() * angstrom # temporary fix to unify ase with the old molmod units
        com = self.masses @ positions[self.indices] / self.masses.sum()
            
        if deriv:
            grad[:, self.indices, :] = np.identity(3)[:, None, :] * self.masses[:, None]
        
        if not deriv:
            return com
        else:
            grad /= self.masses.sum()
            return com, grad


class CenterOfPosition(CollectiveVariable):

    '''
        Class the implement the computation of the center of positions of a set of atoms defined by their atomic indices.
    '''

    type = 'vector'

    def __init__(self, indices, name=None):
        '''
            :param indices: indices of the atoms of which the COP needs to be computed
            :type indices: list of integers

            :param name: Name of CV for printing/logging purposes. If None, the default implemented in the ``_default_name`` routine will be used.
            :type name: str | None, optional, default=None
        '''
        self.indices = indices
        CollectiveVariable.__init__(self, name=name)
    
    def _default_name(self):
        return 'COP(%s)' %('-'.join([str(i) for i in self.indices]))
    
    def compute(self, atoms, deriv=True):
        '''
            Compute CV value (and optionally its gradient) for given coordinates of the molecular system.

            :param atoms: ASE Atoms object on which the CV needs to be computed
            :type atoms: ase.Atoms

            :param deriv: if True, also compute and return the gradient of the CV towards all atomic coordinates
            :type deriv: bool, optional, default=True

            :return: CV value and potentially the gradient
            :rtype: np.ndarray(3) or float,np.ndarray([3,Natoms,3])
        '''
        #unwrap coords of given indices with periodic boundary conditions if unit_cell is specified
        if atoms.get_pbc().any():
            atoms.wrap(center=atoms.get_positions()[0]) # wrap atoms to the first atom

        positions = atoms.get_positions() * angstrom 
        cop = positions[self.indices].mean(axis=0)
        if not deriv:
            return cop
        else:
            grad = np.zeros([3, len(positions), 3], float)
            grad[0, self.indices, 0] = 1/len(self.indices)
            grad[1, self.indices, 1] = 1/len(self.indices)
            grad[2, self.indices, 2] = 1/len(self.indices)
            return cop, grad


class NormalizedAxis(CollectiveVariable):
    '''
        Class the implement the computation of the normalized axis between two points, i.e. as the normalized difference of two input vectors.
    '''

    type = 'vector'

    def __init__(self, vec1, vec2, name=None):
        '''
            :param vec1: first of two vectors defining the plane for which the normal needs to be computed
            :type vec1: instance of child class of CollectiveVariable with type='vector'

            :param vec2: second of two vectors defining the plane for which the normal needs to be computed
            :type vec2: instance of child class of CollectiveVariable with type='vector'

            :param name: Name of CV for printing/logging purposes. If None, the default implemented in the ``_default_name`` routine will be used.
            :type name: str | None, optional, default=None

            :raises AssertionError: if vec1 is not instance of (child of) CollectiveVariable
            :raises AssertionError: if vec1.type is not 'vector'
            :raises AssertionError: if vec2 is not instance of (child of) CollectiveVariable
            :raises AssertionError: if vec2.type is not 'vector'
        '''
        assert isinstance(vec1, CollectiveVariable), 'input argument vec1 should be an instance of CollectiveVariable'
        assert vec1.type=='vector', 'input argument vec1 should be an instance of CollectiveVariable with type vector'
        assert isinstance(vec2, CollectiveVariable), 'input argument vec2 should be an instance of CollectiveVariable'
        assert vec2.type=='vector', 'input argument vec2 should be an instance of CollectiveVariable with type vector'
        self.vec1 = vec1
        self.vec2 = vec2
        CollectiveVariable.__init__(self, name=name)

    def _default_name(self):
        return 'NormalAxis(%s,%s)' %(self.vec1.name, self.vec2.name)
    
    def compute(self, atoms, deriv=True):
        '''
            Compute CV value (and optionally its gradient) for given coordinates of the molecular system.

            :param atoms: ASE Atoms object on which the CV needs to be computed
            :type atoms: ase.Atoms

            :param deriv: if True, also compute and return the gradient of the CV towards all atomic coordinates
            :type deriv: bool, optional, default=True

            :return: CV value and potentially the gradient
            :rtype: np.ndarray(3) or float,np.ndarray([3,Natoms,3])
        '''
        if not deriv:
            v1 = self.vec1.compute(atoms, deriv=False)
            v2 = self.vec2.compute(atoms, deriv=False)
            norm = np.linalg.norm(v2-v1)
            return (v2-v1) / norm
        if deriv:
            v1, grad1 = self.vec1.compute(atoms, deriv=True)
            v2, grad2 = self.vec2.compute(atoms, deriv=True)
            norm = np.linalg.norm(v2-v1)
            tmp = grad2 - grad1 - np.einsum('a,bic,b->aic',v2-v1, grad2-grad1, v2-v1) / norm**2
            return (v2-v1) / norm, tmp / norm


class NormalToPlane(CollectiveVariable):
    
    '''
        Class the implement the computation of the normal to a plane defined by a set of atoms that are assumed to be orderd at the cornerpoints of a regular n-fold polygon
    '''

    type = 'vector'
    
    def __init__(self, indices, name=None):
        '''
            :param indices: indices of the atoms in the plane for which the normal needs to be computed
            :type indices: list of integers

            :param name: Name of CV for printing/logging purposes. If None, the default implemented in the ``_default_name`` routine will be used.
            :type name: str or None, optional, default=None
        '''
        self.indices = indices
        CollectiveVariable.__init__(self, name=name)
    
    def _default_name(self):
        return 'NormalToPlane(%s)' %('-'.join([str(i) for i in self.indices]))

    def compute(self, atoms, deriv=True):
        '''
            Compute the normal to the ring plane (and optionally the gradient) for the given atomic coordinates. Calculations assumes that the n atoms that constitute the ring are orderd at the cornerpoints of a regular n-fold polygon

            :param atoms: ASE Atoms object on which the CV needs to be computed
            :type atoms: ase.Atoms

            :param deriv: if True, also compute and return the gradient of the CV towards all atomic coordinates
            :type deriv: bool, optional, default=True

            :return: CV value and potentially the gradient
            :rtype: np.ndarray(3) or float,np.ndarray([3,Natoms,3])
        '''
        theta = 2*np.pi/len(self.indices)
        if atoms.get_pbc().any():
            atoms.wrap(center=atoms.get_positions()[0]) # wrap atoms to the first atom
        positions = atoms.get_positions()[self.indices] * angstrom
        R1 = np.zeros(3, float)
        R2 = np.zeros(3, float)
        for i, position in enumerate(positions):
            R1 += np.cos((i+1)*theta)*position
            R2 += np.sin((i+1)*theta)*position
        vec = np.cross(R1, R2)
        v = np.linalg.norm(vec)
        normal = vec/v
        if not deriv:
            return normal, None
        else:
            grad = np.zeros([3, len(positions), 3], float)
            tensor = (np.identity(3)-np.outer(vec, vec)/v**2)/v
            for i, index in enumerate(self.indices):
                cosi = np.cos((i+1)*theta)
                sini = np.sin((i+1)*theta)
                for alpha in [0,1,2]:
                    e_alpha = np.zeros(3, float)
                    e_alpha[alpha] = 1.0
                    grad[:, i, alpha] = np.dot(tensor, cosi*np.cross(e_alpha,R2)+sini*np.cross(R1,e_alpha))
            return normal, grad


class DotProduct(CollectiveVariable):
    '''
        Class to implement a collective variable that is the dot product of two given vectors
    '''

    type = 'scalar'

    def __init__(self, vec1, vec2, name=None):
        '''
            :param vec1: first of two vectors defining the plane for which the normal needs to be computed
            :type vec1: instance of child class of CollectiveVariable with type='vector'

            :param vec2: second of two vectors defining the plane for which the normal needs to be computed
            :type vec2: instance of child class of CollectiveVariable with type='vector'

            :param name: Name of CV for printing/logging purposes. If None, the default implemented in the ``_default_name`` routine will be used.
            :type name: str or None, optional, default=None

            :raises AssertionError: if vec1 is not instance of (child of) CollectiveVariable
            :raises AssertionError: if vec1.type is not 'vector'
            :raises AssertionError: if vec2 is not instance of (child of) CollectiveVariable
            :raises AssertionError: if vec2.type is not 'vector'
        '''
        assert isinstance(vec1, CollectiveVariable) and vec1.type=='vector'
        assert isinstance(vec2, CollectiveVariable) and vec2.type=='vector'
        self.vec1 = vec1
        self.vec2 = vec2
        CollectiveVariable.__init__(self, name=name)
    
    def _default_name(self):
        return 'Dot(%s,%s)' %(self.vec1.name, self.vec2.name)
    
    def compute(self, atoms, deriv=True):
        '''
            Compute the dot product (and optionally gradient) of the two vectors for the given atomic coordinates

            :param atoms: ASE Atoms object on which the CV needs to be computed
            :type atoms: ase.Atoms

            :param deriv: if True, also compute and return the gradient of the CV towards all atomic coordinates
            :type deriv: bool, optional, default=True

            :return: CV value and potentially the gradient
            :rtype: np.ndarray(3) or float,np.ndarray([3,Natoms,3])
        '''
        if not deriv:
            v1 = self.vec1.compute(atoms, deriv=False)
            v2 = self.vec2.compute(atoms, deriv=False)
            return np.dot(v1, v2)
        else:
            v1, grad1 = self.vec1.compute(atoms, deriv=True)
            v2, grad2 = self.vec2.compute(atoms, deriv=True)
            cv = np.dot(v1,v2)
            grad = np.einsum('ikl,i->kl', grad1, v2) + np.einsum('i,ikl->kl', v1, grad2)
            return cv, grad


class Distance(CollectiveVariable):
    '''
        Class to implement a collective variable representing the distance between two atoms given by index1 and index2.
    '''
    
    type = 'scalar'
    
    def __init__(self, index1, index2, name=None):
        '''
            :param index1: index of the first atom
            :type indices: int

            :param index2: index of the second atom
            :type indices: int

            :param name: Name of CV for printing/logging purposes. If None, the default implemented in the ``_default_name`` routine will be used.
            :type name: str or None, optional, default=None
        '''
        self.i1 = index1
        self.i2 = index2
        CollectiveVariable.__init__(self, name=name)

    def _default_name(self):
        return 'Distance(%i,%i)' %(self.i1, self.i2)

    def compute(self, atoms, deriv=True):
        '''
            Compute the distance (and optionally gradient) between the two atoms for the given atomic coordinates

            :param atoms: ASE Atoms object on which the CV needs to be computed
            :type atoms: ase.Atoms

            :param deriv: if True, also compute and return the gradient of the CV towards all atomic coordinates
            :type deriv: bool, optional, default=True

            :return: CV value and potentially the gradient
            :rtype: np.ndarray(3) or float,np.ndarray([3,Natoms,3])
        '''
        #computation of value
        r = atoms.get_distance(self.i1, self.i2, mic=True, vector=True) * angstrom
        value = np.linalg.norm(r)
        if not deriv:
            return value
        #computation of deriv
        grad = np.zeros(atoms.get_positions().shape, float)
        grad[self.i1,:] += -r/value
        grad[self.i2,:] += r/value
        return value, grad


class DistanceCOP(CollectiveVariable):
    '''
        Class to implement a collective variable representing the distance between a first atom (index1) and the center of position (i.e. the geometric center) of two other atoms (index2a and index2b).
    '''
    
    type = 'scalar'
    
    def __init__(self, index1, index2a, index2b, name=None):
        '''
            :param index1: index of the first atom
            :type indices: int

            :param index2a: index of the other atom a
            :type indices: int

            :param index2b: index of the other atom b
            :type indices: int

            :param name: Name of CV for printing/logging purposes. If None, the default implemented in the ``_default_name`` routine will be used.
            :type name: str or None, optional, default=None
        '''
        self.i1  = index1
        self.i2a = index2a
        self.i2b = index2b
        CollectiveVariable.__init__(self, name=name)

    def _default_name(self):
        return 'DistanceCOP(%i,%i,%i)' %(self.i1, self.i2a, self.i2b)

    def compute(self, atoms, deriv=True):
        '''
            Compute the COP distance (and optionally gradient) between the atoms for the given atomic coordinates

            :param atoms: ASE Atoms object on which the CV needs to be computed
            :type atoms: ase.Atoms

            :param deriv: if True, also compute and return the gradient of the CV towards all atomic coordinates
            :type deriv: bool, optional, default=True

            :return: CV value and potentially the gradient
            :rtype: np.ndarray(3) or float,np.ndarray([3,Natoms,3])
        '''
        #computation of value
        positions = atoms.get_positions() * angstrom
        cop = 0.5*(positions[self.i2a,:]+positions[self.i2b,:])
        atoms.append(Atom('X', cop / angstrom))  # Append cop as a dummy atom

        r = atoms.get_distance(self.i1, len(atoms)-1, mic=True, vector=True) * angstrom
        value = np.linalg.norm(r)
        if not deriv:
            return value
        #computation of deriv
        grad = np.zeros(positions.shape, float)
        grad[self.i1 ,:] += - r / value
        grad[self.i2a,:] += 0.5 * r / value
        grad[self.i2b,:] += 0.5 * r / value
        return value, grad


class CoordinationNumber(CollectiveVariable):
    '''
        Class to implement a collective variable representing the coordination number of a certain atom pair or a set of atom pairs. If :math:`n` atom pairs are defined, the coordination number should be a number between :math:`0` and :math:`n`.

        .. math::
        
            CN &= \\sum_{ij \\in pairs} \\frac{1-\\left(\\frac{r_{ij}}{r_0}\\right)^{nn}}{1-\\left(\\frac{r_{ij}}{r_0}\\right)^{nd}}

        in which 
        
        - :math:`r_{ij}` is the distance between atom i an atom j as defined in pair (i,j) present in pairs
        - :math:`r_0` is a reference distance for the bond between two atoms set to 2 angstrom by default but can be defined in the keyword arguments by the user
        - :math:`nn` and :math:`nd` are integers that are set to 6 and 12 respectively by default but can also be defined in the keyword arguments by the user
    '''
    
    type = 'scalar'
    
    def __init__(self, pairs, r0=2.0*angstrom, nn=6, nd=12, name=None):
        '''
            :param pairs: pairs of atoms between which the coordination number needs to be computed
            :type pairs: list tuples

            :param r0: reference index for a chemical bond used in the definition of the coordination number
            :type r0: float, optional, default=2*angstrom

            :param nn: power coefficient used in the nominator in the definition of the coordination number
            :type nn: int, optional, default=6

            :param nd: power coefficient used in the denominator in the definition of the coordination number
            :type nd: int, optional, default=12

            :param name: Name of CV for printing/logging purposes. If None, the default implemented in the ``_default_name`` routine will be used.
            :type name: str or None, optional, default=None
        '''
        self.pairs = pairs
        self.r0 = r0
        self.nn = nn
        self.nd = nd
        CollectiveVariable.__init__(self, name=name)

    def _default_name(self):
        return 'CoordinationNumber([' + ' , '.join(['(%i,%i)' %(i,j) for i,j in self.pairs]) + '])'

    def compute(self, atoms, deriv=True):
        '''
            Compute the coordination number (and optionally gradient) between the atom pairs for the given atomic coordinates

            :param atoms: ASE Atoms object on which the CV needs to be computed
            :type atoms: ase.Atoms

            :param deriv: if True, also compute and return the gradient of the CV towards all atomic coordinates
            :type deriv: bool, optional, default=True

            :return: CV value and potentially the gradient
            :rtype: np.ndarray(3) or float,np.ndarray([3,Natoms,3])
        '''
        value = 0.0
        grad = np.zeros(atoms.get_positions().shape, float)
        for i,j in self.pairs:
            rij = atoms.get_distance(j, i, mic=True, vector=True) * angstrom
            r = np.linalg.norm(rij)
            value += (1 - (r / self.r0)**self.nn) / (1 - (r / self.r0)**self.nd)
            if deriv:
                T = (self.nn - self.nd) * (r /  self.r0)**(self.nn +self.nd -1) + self.nd * (r / self.r0)**(self.nd - 1) - self.nn * (r / self.r0)**(self.nn - 1)
                N = (1 - (r / self.r0)**self.nd)**2
                dcij_drij = (1. / self.r0) * T / N
                grad[i,:] +=  rij / r * dcij_drij
                grad[j,:] += -rij / r * dcij_drij
        if deriv:
            return value, grad
        else:
            return value


class OrthogonalDistanceToPore(DotProduct):
    '''
        Class to implement a collective variable that represents the orthogonal distance between the center of mass of a guest molecule defined by its atom indices (guest_indices) on the one hand, and a pore ring defined by the atom indices of its constituting atoms (ring_indices) on the other hand.
    '''
    
    type = 'scalar'
    
    def __init__(self, ring_indices, guest_indices, masses=None, name=None):
        '''
            :param ring_indices: atomic indices of the ring
            :type ring_indices: list of integers

            :param guest_indices: atomic indices of the guest
            :type guest_indices: list of integers

            :param masses: masses of the atoms specified in indices. If none, default atomic masses are used.
            :type masses: np.ndarray

            :param name: Name of CV for printing/logging purposes. If None, the default implemented in the ``_default_name`` routine will be used.
            :type name: str | None, optional, default=None
        '''
        self.guest_indices = guest_indices
        self.ring_indices = ring_indices
        com  = CenterOfMass(guest_indices, masses)
        cop  = CenterOfPosition(ring_indices)
        vec1 = Difference(cop, com)
        vec2 = NormalToPlane(ring_indices)
        DotProduct.__init__(self, vec1, vec2, name=name)
    
    def _default_name(self):
        return 'OrthogonalDistanceToPore(ring=[%s],guest=[%s])' %(
            ','.join([str(i) for i in self.ring_indices]),
            ','.join([str(i) for i in self.guest_indices])
        )


class Average(CollectiveVariable):
    '''
        Class to implement a collective variable representing the average of two other collective variables as

        .. math:: CV &= \\frac{CV_1 + CV_2}{2}
    '''
    
    type = None #depends on type of argument cvs
    
    def __init__(self, cv1, cv2, name=None):
        '''
            :param cv1: first collective variable in the average
            :type cv1: any child class of :py:class:`CollectiveVariable <thermolib.thermodynamics.cv.CollectiveVariable>`

            :param cv2: second collective variable in the average
            :type cv2: any child class of :py:class:`CollectiveVariable <thermolib.thermodynamics.cv.CollectiveVariable>`

            :param name: Name of CV for printing/logging purposes. If None, the default implemented in the ``_default_name`` routine will be used.
            :type name: str | None, optional, default=None
        '''
        assert cv1.type==cv2.type
        self.type = cv1.type
        self.cv1 = cv1
        self.cv2 = cv2
        CollectiveVariable.__init__(self, name=name)

    def _default_name(self):
        return '0.5*(%s+%s)' %(self.cv1.name, self.cv2.name)

    def compute(self, atoms, deriv=True):
        '''
            Compute the average (and optionally gradient) of the two CVs for the given atomic coordinates

            :param atoms: ASE Atoms object on which the CV needs to be computed
            :type atoms: ase.Atoms

            :param deriv: if True, also compute and return the gradient of the CV towards all atomic coordinates
            :type deriv: bool, optional, default=True

            :return: CV value and potentially the gradient
            :rtype: np.ndarray(3) or float,np.ndarray([3,Natoms,3])
        '''
        if not deriv:
            cv1 = self.cv1.compute(atoms, deriv=False)
            cv2 = self.cv2.compute(atoms, deriv=False)
            return 0.5*(cv1+cv2)
        else:
            cv1, grad1 = self.cv1.compute(atoms, deriv=True)
            cv2, grad2 = self.cv2.compute(atoms, deriv=True)
            value = 0.5*(cv1+cv2)
            grad = 0.5*(grad1 + grad2)
        return value, grad


class Difference(CollectiveVariable):
    '''
        Class to implement a collective variable representing the difference between two other collective variables:

        .. math:: CV &= CV_2 - CV_1
    '''
    
    type = None #depends on type of argument cvs
    
    def __init__(self, cv1, cv2, name=None):
        '''
            :param cv1: first collective variable in the difference
            :type cv1: any child class of :py:class:`CollectiveVariable <thermolib.thermodynamics.cv.CollectiveVariable>`

            :param cv2: second collective variable in the difference
            :type cv2: any child class of :py:class:`CollectiveVariable <thermolib.thermodynamics.cv.CollectiveVariable>`

            :param name: Name of CV for printing/logging purposes. If None, the default implemented in the ``_default_name`` routine will be used.
            :type name: str | None, optional, default=None
        '''
        assert cv1.type==cv2.type
        self.type = cv1.type
        self.cv1 = cv1
        self.cv2 = cv2
        CollectiveVariable.__init__(self, name=name)

    def _default_name(self):
        return 'Diff(%s,%s)' %(self.cv2.name, self.cv1.name)

    def compute(self, atoms, deriv=True):
        '''
            Compute the difference (and optionally gradient) between the two CVs for the given atomic coordinates

            :param atoms: ASE Atoms object on which the CV needs to be computed
            :type atoms: ase.Atoms

            :param deriv: if True, also compute and return the gradient of the CV towards all atomic coordinates
            :type deriv: bool, optional, default=True

            :return: CV value and potentially the gradient
            :rtype: np.ndarray(3) or float,np.ndarray([3,Natoms,3])
        '''
        if not deriv:
            cv1 = self.cv1.compute(atoms, deriv=False)
            cv2 = self.cv2.compute(atoms, deriv=False)
            return cv2-cv1
        else:
            cv1, grad1 = self.cv1.compute(atoms, deriv=True)
            cv2, grad2 = self.cv2.compute(atoms, deriv=True)
            value = cv2-cv1
            grad = grad2 - grad1
        return value, grad


class Minimum(CollectiveVariable):
    '''
        Class to implement a collective variable representing the minimum of two other collective variables:

        .. math:: CV &= \\min\\left(CV_1,CV_2\\right)
    '''
    
    type = 'scalar'
    
    def __init__(self, cv1, cv2, name=None):
        '''
            :param cv1: first collective variable in the minimum
            :type cv1: any child class of :py:class:`CollectiveVariable <thermolib.thermodynamics.cv.CollectiveVariable>`

            :param cv2: second collective variable in the minimum
            :type cv2: any child class of :py:class:`CollectiveVariable <thermolib.thermodynamics.cv.CollectiveVariable>`

            :param name: Name of CV for printing/logging purposes. If None, the default implemented in the ``_default_name`` routine will be used.
            :type name: str | None, optional, default=None
        '''
        self.cv1 = cv1
        self.cv2 = cv2
        CollectiveVariable.__init__(self, name=name)

    def _default_name(self):
        return 'Min(%s,%s)' %(self.cv1.name, self.cv2.name)

    def compute(self, atoms, deriv=True):
        '''
            Compute the minimum (and optionally gradient) of the two CVs for the given atomic coordinates

            :param atoms: ASE Atoms object on which the CV needs to be computed
            :type atoms: ase.Atoms

            :param deriv: if True, also compute and return the gradient of the CV towards all atomic coordinates
            :type deriv: bool, optional, default=True

            :return: CV value and potentially the gradient
            :rtype: np.ndarray(3) or float,np.ndarray([3,Natoms,3])
        '''
        if not deriv:
            cv1 = self.cv1.compute(atoms, deriv=False)
            cv2 = self.cv2.compute(atoms, deriv=False)
            return min(cv1,cv2)
        else:
            cv1, grad1 = self.cv1.compute(atoms, deriv=True)
            cv2, grad2 = self.cv2.compute(atoms, deriv=True)
            value = min(cv1,cv2)
            if cv1<cv2: # note that right not gradient is discontinuous, can be fixed using plumed MIN CV
                grad = grad1
            else:
                grad = grad2
        return value, grad


class LinearCombination(CollectiveVariable):
    '''
        Class to implement a collective variable that is the linear combination of other collective variables

        .. math:: CV &= \\sum_{i=1}^N c_i\\cdot CV_i
        
        in which cvs is the list of involved collective variables and coeffs is list of equal length with the corresponing coefficients
    '''
    
    type = None #depends on type of argument cvs
    
    def __init__(self, cvs, coeffs, name=None):
        '''
            :param cvs: list of CVs in the linear combination
            :type cvs: list of instances of child classes of :py:class:`CollectiveVariable <thermolib.thermodynamics.cv.CollectiveVariable>`

            :param coeffs: coefficients corresponding to the weight of each CV in the linear combination
            :type coeffs: list or array

            :param name: Name of CV for printing/logging purposes. If None, the default implemented in the ``_default_name`` routine will be used.
            :type name: str | None, optional, default=None
        '''
        assert len(cvs)==len(coeffs), "List of cvs and list of coefficients should be of equal length"
        for i in range(1,len(cvs)):
            assert cvs[i].type==cvs[0].type, 'cvs[%i] and cvs[0] are not both scalar CVS or vector CVS' %(i)
        self.type = cvs[0].type
        self.cvs = cvs
        self.coeffs = coeffs
        CollectiveVariable.__init__(self, name=name)

    def _default_name(self):
        return ''.join(['%+.2f%s' %(coeff,cv.name) for (coeff,cv) in zip(self.coeffs, self.cvs)])

    def compute(self, atoms, deriv=True):
        '''
            Compute the linear combination (and optionally gradient) of the CVs for the given atomic coordinates

            :param atoms: ASE Atoms object on which the CV needs to be computed
            :type atoms: ase.Atoms

            :param deriv: if True, also compute and return the gradient of the CV towards all atomic coordinates
            :type deriv: bool, optional, default=True

            :return: CV value and potentially the gradient
            :rtype: np.ndarray(3) or float,np.ndarray([3,Natoms,3])
        '''
        if not deriv:
            value = 0.0
            for coeff, cv in zip(self.coeffs, self.cvs):
                value += coeff*cv.compute(atoms, deriv=False)
            return value
        else:
            value = 0.0
            grad = np.zeros(atoms.get_positions().shape)
            for coeff, cv in zip(self.coeffs, self.cvs):
                v, g = cv.compute(atoms, deriv=True)
                value += coeff*v
                grad += coeff*g
            return value, grad


class DistOrthProjOrig(DotProduct):
    '''
        Class to implement a collective variable that represents the distance between (1) the orthogonal projection of a position (defined by a CV denoted as pos) on an axis (defined by a CV denoted as axis) through an origin (defined by a CV denoted as orig) and (2) the origin.
    '''

    type = 'scalar'

    def __init__(self, pos, axis, orig, name=None):
        '''
            :param pos: the CV corresponding to the position that needs to be projected on the axis
            :type pos: child instance of :py:class:`CollectiveVariable <thermolib.thermodynamics.cv.CollectiveVariable>` with type='vector'

            :param axis: the CV corresponding to the axis on which the position needs to be projected
            :type axis: :py:class:`NormalizedAxis <thermolib.thermodynamics.cv.NormalizedAxis>`

            :param orig: the CV corresponding to the origin through which the axis needs to go and with which the difference of the projected pos will be computed
            :type orig: child instance of :py:class:`CollectiveVariable <thermolib.thermodynamics.cv.CollectiveVariable>` with type='vector'

            :param name: Name of CV for printing/logging purposes. If None, the default implemented in the ``_default_name`` routine will be used.
            :type name: str | None, optional, default=None

            :raises AssertionError: if pos is not a (child) instance of :py:class:`CollectiveVariable <thermolib.thermodynamics.cv.CollectiveVariable>` with pos.type='vector'
            :raises AssertionError: if axis is not an instance of :py:class:`NormalizedAxis <thermolib.thermodynamics.cv.NormalizedAxis>`
            :raises AssertionError: if orig is not a (child) instance of :py:class:`CollectiveVariable <thermolib.thermodynamics.cv.CollectiveVariable>` with orig.type='vector'
        '''
        assert isinstance(pos, CollectiveVariable) and pos.type=='vector'
        assert isinstance(axis, NormalizedAxis)
        assert isinstance(pos, CollectiveVariable) and pos.type=='vector'
        self.pos = pos
        self.axis = axis
        self.orig = orig 
        DotProduct.__init__(self, Difference(orig,pos), axis, name=name)

    def _default_name(self):
        return 'DistOrthProjOrig(orig=%s,pos=%s,axis=%s)' %(self.orig.name, self.pos.name, self.axis.name)


class FunctionCV(CollectiveVariable):
    '''
        Class to implement a collective variable that represents the function of a given CV
    '''
    def __init__(self, CV, function, derivative, name=None):
        '''
            :param CV: the CV of which the function will be computed
            :type CV: child instance of :py:class:`CollectiveVariable <thermolib.thermodynamics.cv.CollectiveVariable>`

            :param function: function to be evaluated with the CV
            :type function: callable

            :param derivative: derivative of the function to be evaluated with the CV
            :type derivative: callable

            :param name: Name of new CV for printing/logging purposes. If None, the default implemented in the ``_default_name`` routine will be used.
            :type name: str | None, optional, default=None

            :raises AssertionError: if pos is not a (child) instance of :py:class:`CollectiveVariable <thermolib.thermodynamics.cv.CollectiveVariable>`
        '''
        assert isinstance(CV, CollectiveVariable)
        self.CV = CV
        self.function = function
        self.derivative = derivative
        CollectiveVariable.__init__(self, name=name)

    def _default_name(self):
        return 'fun(%s)' %(self.CV.name)
    
    def compute(self, atoms, deriv=True):
        '''
            Compute the function (and optionally gradient) of the CV for the given atomic coordinates

            :param atoms: ASE Atoms object on which the CV needs to be computed
            :type atoms: ase.Atoms

            :param deriv: if True, also compute and return the gradient of the CV towards all atomic coordinates
            :type deriv: bool, optional, default=True

            :return: CV value and potentially the gradient
            :rtype: np.ndarray(3) or float,np.ndarray([3,Natoms,3])
        '''
        if deriv:
            cv, grad = self.CV.compute(atoms, deriv=True)
            q = self.function(cv)
            qgrad = self.derivative(cv)*grad
            return q, qgrad
        else:
            cv = self.CV.compute(atoms, deriv=False)
            return self.function(cv)


def test_CV_implementations(fn, cvs, dx=0.001*angstrom, maxframes=100, tol=1E-9):
    '''
        Routine to test the implementation of the derivative in the compute methods of child class of :py:class:`CollectiveVariable <thermolib.thermodynamics.cv.CollectiveVariable>` by comparing it with a numerical derivative. This routine serves as a tool to test new CV implementations. In the future, this routine will be moved to a new dedicated test module that should be run upon installation of ThermoLIB.

        :param fn: names of trajectory files that contain XYZ coordinates to be used in the testing
        :type fn: str

        :param cvs: list of CVs to be tested
        :type cvs: list of instances of child classes of :py:class:`CollectiveVariable <thermolib.thermodynamics.cv.CollectiveVariable>`

        :param dx: approximation of infinitesimal small displacement to be used in the numerical derivative
        :type dx: float, optional, default=0.001*angstrom

        :param maxframes: the first maxframes from the given XYZ trajectory will be used in the testing
        :type maxframes: int, optional, default=100

        :param tol: the tolerance between the numerical and analytical CV value.
        :type tol: float, optional, default 1E-9

        :raises AssertionError: if a CV failes the test
    '''
    trajectory = read(fn, index=f':{maxframes}')
    for cv in cvs:
        print('Testing consistency between value and gradient of %s' % cv.name)
        for i, atoms in enumerate(trajectory):
            value, grad = cv.compute(atoms)
            coords = atoms.get_positions() * angstrom
            grad_shape = grad.shape
            # Handle (3, N, 3) shape (vector-valued CVs)
            if len(grad_shape) == 3 and grad_shape[0] == 3 and grad_shape[2] == 3:
                n_atoms = grad_shape[1]
                for comp in range(3):
                    for aindex in range(n_atoms):
                        for carindex in range(3):
                            delta = np.zeros(coords.shape, float)
                            # Only update if aindex is within coords shape
                            if aindex < coords.shape[0]:
                                delta[aindex, carindex] = 1
                            _atoms = atoms.copy()
                            _atoms.set_positions((coords - delta * dx) / angstrom)
                            v1 = cv.compute(_atoms, deriv=False)
                            _atoms = atoms.copy()
                            _atoms.set_positions((coords + delta * dx) / angstrom)
                            v2 = cv.compute(_atoms, deriv=False)
                            numerical = (v2[comp] - v1[comp]) / (2 * dx)
                            diff = np.abs(numerical - grad[comp, aindex, carindex])
                            assert diff < tol, (
                                'Analytical derivative check failed! grad[%i,%i,%i]=%20.15e numerical=%20.15e (difference=%20.15e)'
                                % (comp, aindex, carindex, grad[comp, aindex, carindex], numerical, diff)
                            )
            # Handle (N, 3) shape (scalar-valued CVs)
            elif len(grad_shape) == 2 and grad_shape[1] == 3:
                n_atoms = grad_shape[0]
                for aindex in range(n_atoms):
                    for carindex in range(3):
                        delta = np.zeros(coords.shape, float)
                        if aindex < coords.shape[0]:
                            delta[aindex, carindex] = 1
                        _atoms = atoms.copy()
                        _atoms.set_positions((coords - delta * dx) / angstrom)
                        v1 = cv.compute(_atoms, deriv=False)
                        _atoms = atoms.copy()
                        _atoms.set_positions((coords + delta * dx) / angstrom)
                        v2 = cv.compute(_atoms, deriv=False)
                        numerical = (v2 - v1) / (2 * dx)
                        diff = np.abs(numerical - grad[aindex, carindex])
                        assert diff < tol, (
                            'Analytical derivative check failed! grad[%i,%i]=%20.15e numerical=%20.15e (difference=%20.15e)'
                            % (aindex, carindex, grad[aindex, carindex], numerical, diff)
                        )
            else:
                raise ValueError(f"Unexpected gradient shape: {grad_shape}")
        print("Test for %s successful!" % cv._default_name())
    del trajectory
