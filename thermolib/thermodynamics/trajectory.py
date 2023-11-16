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

from molmod.units import *
from molmod.io.xyz import XYZReader as mmXYZReader

import numpy as np, h5py as h5

__all__ = ['XYZReader', 'HDF5Reader', 'ColVarReader']

class TrajectoryReader(object):
    '''Abstract class for reading trajectory files and extracting CV values. Child classes will implement specific readers depending of the format of the trajectory files (COLVAR, HDF5, XYZ, ...)
    '''

    def __init__(self, units: list=[], start: int=0, stride: int=1, end: int=-1, verbose: bool=False):
        '''
        :param units: List of units for each CV that needs to be read. Defaults to 'au' for each CV., defaults to []
        :type units: list, optional

        :param start: index of starting point in trajectory to extract CV values, can be used to crop out initial equilibration time. Defaults to 0
        :type start: int, optional

        :param stride: integer defining how to subsample the trajectory CV data. If set to a value larger than 1, only take sub samples every 'stride' steps. If set to 1, sample all trajectory steps. Can be used to decrease correlations between subsequent samples. Defaults to 1
        :type stride: int, optional

        :param end: index of last trajectory step to be sampled. Can be used to crop out the final part of a simulation trajectory. If set to -1, take all trajectory samples, i.e. no cropping. Defaults to -1
        :type end: int, optional

        :param verbose: Switch on the routine verbosity and print more logging, defaults to False
        :type verbose: bool, optional
        '''
        self.units = units
        self.start = start
        self.stride = stride
        self.end = end
        self.verbose = verbose

    def print(self, line):
        if self.verbose:
            print(line)

    def _slice(self, data):
        '''Slices the input data to crop between self.start and self.end and take sub sample according to self.stride
        '''
        if len(data.shape)==2 and data.shape[1]==1:
            data = data[:,0]
        if self.end==-1:
            return data[self.start::self.stride]
        else:
            return data[self.start:self.end:self.stride]

    def _read(self, fn):
        raise NotImplementedError('Read routine must be implemented in child class')

    def __call__(self, fn):
        data = self._read(fn)
        if data is None:
            raise ValueError("Could not read trajectory from %s" %fn)
        self.print("Read %i trajectory samples for %i CVs from %s" %(data.shape[0], data.shape[1], fn))
        if data.shape[1]==1:
            return data[:,0]
        else:
            return data


class XYZReader(TrajectoryReader):
    '''Compute the CV along an XYZ trajectory. The XYZ trajectory is assumed to be defined in a (list of subsequent) XYZ file(s).
    '''
    def __init__(self, CVs: list, start: int=0, stride: int=1, end: int=-1, verbose: bool=False):
        '''
        :param CVq: a list of collective variable defining how to compute the collective variable along the trajectory
        :type CVs: list of instances from thermolib.thermodynamics.cv module
        '''
        self.CVs = CVs
        TrajectoryReader.__init__(self, start=start, stride=stride, end=end, verbose=verbose)

    def _read(self, fn):
        cvdata = None
        xyz = mmXYZReader(fn)
        for i, CV in enumerate(self.CVs):
            data = []
            for title, coords in xyz:
                data.append(CV.compute(coords, deriv=False))
            col = self._slice(np.array(data))
            if len(col)==0:
                raise ValueError('No data for CV(%s) could be read from trajectory %s. Are you sure you did not choose start:end:stride to restrictive?' %(CV.name,fn))
            if cvdata is None:
                assert i==0
                cvdata = np.zeros([len(col), len(self.CVs)])
                cvdata[:,0] = col
            else:
                assert len(col)==len(cvdata[:,0]), 'Trajectory for CV(%s) has %i time steps, while first CV has %i. They should both have an equal number.' %(CV.name, len(col), len(cvdata[:,0]))
                cvdata[:,i] = col
        return cvdata


class HDF5Reader(TrajectoryReader):
    def __init__(self, keys: list, units: list=[], start: int=0, stride: int=1, end: int=-1, verbose: bool=False):
        '''
        :param keys: Represent the keys of datasets in HDF5 from which to read CV values.
        :type keys: list
        '''
        if len(keys)==0:
            raise ValueError('No dataset keys defined, received empty list for argument keys.')
        if len(units)==0:
            self.print('No units defined, set all units to au')
            units = ['au',]*len(keys)
        assert len(keys)==len(units), 'keys and units should be list of same length'
        self.keys = keys
        TrajectoryReader.__init__(self, units=units, start=start, stride=stride, end=end, verbose=verbose)

    def _read(self, fn):
        f = h5.File(fn, mode = 'r')
        for key in self.keys:
            assert key in f.keys(), 'Specified dataset key %s was not found in HDF5 file %s. Possible keys are: %s' %(key, fn, ','.join(f.keys()))
        cvdata = None
        for i, (key, unit) in enumerate(zip(self.keys,self.units)):
            col = self._slice(f[key][:])*parse_unit(unit)
            if len(col)==0:
                raise ValueError('No data for CV(%s) could be read from trajectory %s. Are you sure you did not choose start:end:stride to restrictive?' %(key,fn))
            if cvdata is None:
                assert i==0
                cvdata = np.zeros([len(col), len(self.keys)])
                cvdata[:,0] = col
            else:
                assert len(col)==len(cvdata[:,0]), 'Trajectory for CV(%s) has %i time steps, while first CV has %i. They should both have an equal number.' %(key, len(col), len(cvdata[:,0]))
                cvdata[:,i] = col
        return cvdata


class ColVarReader(TrajectoryReader):
    def __init__(self, indices: list, units: list=[], start: int=0, stride: int=1, end: int=-1, verbose: bool=False):
        '''
        :param indices: Represents the indices of the columns in COLVAR file from which to read the CV values.
        :type indices: list
        '''
        
        if len(indices)==0:
            raise ValueError('No column indices defined, received empty list for argument indices.')
        if len(units)==0:
            self.print('No units defined, set all units to au')
            units = ['au',]*len(indices)
        assert len(indices)==len(units), 'indices and units should be list of same length'
        self.indices = indices
        TrajectoryReader.__init__(self, units=units, start=start, stride=stride, end=end, verbose=verbose)
    
    def _read(self, fn):
        data = np.loadtxt(fn)
        cvdata = None
        for i, (index, unit) in enumerate(zip(self.indices,self.units)):
            col = self._slice(data[:,index])*parse_unit(unit)
            if len(col)==0:
                raise ValueError('No data for CV(%i) could be read from trajectory %s. Are you sure you did not choose start:end:stride to restrictive?' %(index,fn))
            if cvdata is None:
                assert i==0
                cvdata = np.zeros([len(col), len(self.indices)])
                cvdata[:,0] = col
            else:
                assert len(col)==len(cvdata[:,0]), 'Trajectory for CV(%i) has %i time steps, while CV0 has %i. They should both have an equal number.' %(index, len(col), len(cvdata[:,0]))
                cvdata[:,i] = col
        return np.array(cvdata)