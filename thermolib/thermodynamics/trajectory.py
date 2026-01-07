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
from .cv import CollectiveVariable

from ase import Atoms
from ase.io import read

import numpy as np, h5py as h5, os

__all__ = ['CVComputer', 'HDF5Reader', 'ColVarReader', 'ASEExtendedXYZReader']

class TrajectoryReader(object):
    '''
        Abstract class for reading trajectory files and extracting CV values. Child classes will implement specific readers depending of the format of the trajectory files (COLVAR, HDF5, XYZ, ...)
    '''

    def __init__(self, units: list=[], name: str=None, start: int=0, stride: int=1, end: int=-1, reload: bool=False, verbose: bool=False):
        '''
            :param units: List of units for each CV that needs to be read.
            :type units: list, optional, default=[]

            :param name: a name for printing/logging purposes
            :type name: str, optional, default=None

            :param start: index of starting point in trajectory to extract CV values, can be used to crop out initial equilibration time.
            :type start: int, optional, default=0

            :param stride: integer defining how to subsample the trajectory CV data. If set to a value larger than 1, only take sub samples every 'stride' steps. If set to 1, sample all trajectory steps. Can be used to decrease correlations between subsequent samples.
            :type stride: int, optional, default=1

            :param end: index of last trajectory step to be sampled. Can be used to crop out the final part of a simulation trajectory. If set to -1, take all trajectory samples, i.e. no cropping.
            :type end: int, optional, default=-1

            :param reload: If True, the first time the __call__ routine is called, the data that is read/computed from the trajectory will be written in numpy format to a file (with name equal to original trajectory file name appended with '.reload' at the end). The next time the __call__ routine is called again afterwards, the trajectory data will read directly from the .reload file. This results in a considerable speedup in the case the CV is computed from a XYZ trajectory. If False, the data will be read from the trajectory file with every call to the __call__ routine.
            :type reload: bool, optional, default=False

            :param verbose: If True, switch on the routine verbosity and print more logging
            :type verbose: bool, optional, default=False
        '''
        self.units = units
        self.name = name
        self.start = start
        self.stride = stride
        self.end = end
        self.verbose = verbose
        self.reload = reload

    def print(self, line):
        '''
            Small wrapper around the python built-in print statement to allow for verbosity control.

            :param line: Line/data to be printed
            :type line: any python printable data type
        '''
        if self.verbose:
            print(line)

    def _slice(self, data):
        '''
            Slices the input data to crop between self.start and self.end and take sub sample according to self.stride

            :param data: the data to be sliced
            :type data: np.ndarray
        '''
        if len(data.shape)==2 and data.shape[1]==1:
            data = data[:,0]
        if len(data.shape)==3 and data.shape[1]==1 and data.shape[2]==1:
            data = data[:,0,0]
        if self.end==-1:
            return data[self.start::self.stride]
        else:
            return data[self.start:self.end:self.stride]

    def _read(self, fn):
        '''
            Routine to handle how a given file should be read, will be called by the __call__ routine. Needs to be implemented in each inheriting child class.
        '''
        raise NotImplementedError('Read routine must be implemented in child class')

    def __call__(self, fn):
        '''
            Call routine to read the required data from the trajectory file fn using the _read routine §implemented in each child class) and return the corresponding CV samples along the trajectory.

            :param fn: file name of the trajectory file
            :type fn: str

            :raises ValueError: if no data could be read from the file using the _read routine.
            _
            :return: trajectory samples
            :rtype: np.ndarray
        '''
        fn_reload = fn+'.reload'
        if self.reload and os.path.isfile(fn_reload):
            return np.loadtxt(fn_reload)
        else:
            data = self._read(fn)
            if data is None:
                raise ValueError("Could not read trajectory from %s" %fn)
            self.print("Read %i trajectory samples for %i CVs from %s" %(data.shape[0], data.shape[1], fn))
            if self.reload:
                np.savetxt(fn_reload, data)
            if data.shape[1]==1:
                return data[:,0]
            else:
                return data



class CVComputer(TrajectoryReader):
    '''
        Child class of TrajectoryReader to compute the specified CVs along a given trajectory stored in XYZ or HDF5 files.
    '''
    def __init__(self, CVs: list, coords_key: str=None, name: str=None, start: int=0, stride: int=1, end: int=-1, reload: bool=False, verbose: bool=False):
        '''
            :param CVs: a list of collective variable defining how to compute each collective variable along the trajectory
            :type CVs: list of instances from :py:mod:`cv <thermolib.thermodynamics.cv>` module

            :param coords_key: This parameter is only relevant when trajectory files require data set names to find the coordinates (i.e. in HDF5 files) and will be ignored otherwise (i.e. if trajectory files are XYZ files). This parameter then defines the location of the coordinates in the trajectory file (e.g. name of coordinates data set in HDF5 file)
            :type coords_key: str, optional, default=None

            :param name: a name for printing/logging purposes. If None, set to '/'-seperated list of the name attributes of the CVs
            :type name: str, optional, default=None

            :param start: index of starting point in trajectory to extract CV values, can be used to crop out initial equilibration time.
            :type start: int, optional, default=0

            :param stride: integer defining how to subsample the trajectory CV data. If set to a value larger than 1, only take sub samples every 'stride' steps. If set to 1, sample all trajectory steps. Can be used to decrease correlations between subsequent samples.
            :type stride: int, optional, default=1

            :param end: index of last trajectory step to be sampled. Can be used to crop out the final part of a simulation trajectory. If set to -1, take all trajectory samples, i.e. no cropping.
            :type end: int, optional, default=-1

            :param reload: If True, the first time the __call__ routine is called, the data that is read/computed from the trajectory will be written in numpy format to a file (with name equal to original trajectory file name appended with '.reload' at the end). The next time the __call__ routine is called again afterwards, the trajectory data will read directly from the .reload file. This results in a considerable speedup in the case the CV is computed from a XYZ trajectory. If False, the data will be read from the trajectory file with every call to the __call__ routine.
            :type reload: bool, optional, default=False

            :param verbose: If True, switch on the routine verbosity and print more logging
            :type verbose: bool, optional, default=False
        '''
        if not (isinstance(CVs, list) or isinstance(CVs, tuple)):
            CVs = [CVs]
        for i, CV in enumerate(CVs):
            assert isinstance(CV, CollectiveVariable), ''
        self.CVs = CVs
        self.coords_key = coords_key
        if name is None:
            name = '/'.join([CV.name for CV in CVs])
        TrajectoryReader.__init__(self, name=name, start=start, stride=stride, end=end, reload=reload, verbose=verbose)

    @property
    def ncvs(self):
        return len(self.CVs)

    def _read(self, fn: str):
        '''
            Wrapper to decide whether to use _read_xyz (of .xyz files) or _read_h5 (for .h5 files) to read trajectory file based on the extension. 

            :param fn: file containing the trajectory coordinates
            :type fn: str

            :raises NotImplementedError: if fn has neither .xyz nor .h5 extension.
            
            :return: trajectory data representing CV samples
            :rtype: np.ndarray
        '''
        if fn.endswith('.xyz'):
            return self._read_xyz(fn)
        if fn.endswith('.h5'):
            return self._read_h5(fn)
        else:
            raise NotImplementedError('Extension in file %s not recognized or supported (yet).' %(fn))
    
    def _read_xyz(self, fn: str):
        '''
            Routine to read .xyz trajectory file and for each frame extract coordinates and compute CV.

            :param fn: Name of XYZ file containing the trajectory coordinates
            :type fn: str

            :raises ValueError: if no data could be extracted from trajectory, i.e. because start:end:stride were chosen to restrictive.
            :raises AssertionError: if the various CVs computed from the trajectory do not have matching number of samples.
            
            :return: trajectory data representing CV samples
            :rtype: np.ndarray
        '''
        cvdata = None
        xyz = read(fn, index=slice(self.start,self.end,self.stride))
        for i, CV in enumerate(self.CVs):
            data = []
            for atoms in xyz:
                data.append(CV.compute(atoms, deriv=False))
            if len(data)==0:
                raise ValueError('No data for CV(%s) could be read from trajectory %s. Are you sure you did not choose start:end:stride to restrictive?' %(CV.name,fn)) # not sure this is needed with ase.io.read
            if cvdata is None:
                assert i==0
                cvdata = np.zeros([len(data), len(self.CVs)])
                cvdata[:,0] = data
            else:
                assert len(data)==len(cvdata[:,0]), 'Trajectory for CV(%s) has %i time steps, while first CV has %i. They should both have an equal number.' %(CV.name, len(data), len(cvdata[:,0]))
                cvdata[:,i] = data
        return cvdata

    def _read_h5(self, fn: str):
        '''
            Routine to read .h5 trajectory file and for each frame extract coordinates and compute CV.

            :param fn: Name of HDF5 file containing the trajectory coordinates
            :type fn: str

            :raises ValueError: if no data could be extracted from trajectory, i.e. because start:end:stride were chosen to restrictive.
            :raises AssertionError: if the various CVs computed from the trajectory do not have matching number of samples.
            
            :return: trajectory data representing CV samples
            :rtype: np.ndarray
        '''
        cvdata = None
        f = h5.File(fn, mode = 'r')
        numbers = f['system/numbers']
        Nsteps = len(f['/trajectory/time'])
        for i, CV in enumerate(self.CVs):
            data = []
            for itime in range(Nsteps):
                if self.coords_key is not None:
                    positions = f[self.coords_key][itime,:,:]
                else:
                    positions = f['/trajectory/pos'][itime,:,:]
                try:
                    cell = f['trajectory/cell']
                except:
                    cell = None
                atoms = Atoms(numbers=numbers, positions=positions/angstrom)
                if cell is not None:
                    atoms.set_pbc(True)
                    atoms.set_cell(cell/angstrom)
                data.append(CV.compute(atoms, deriv=False))
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
    '''
        Child class of TrajectoryReader to read the precomputed CVs from data sets stored in HDF5 file(s)
    '''
    def __init__(self, keys: list, units: list=[], name: str=None, start: int=0, stride: int=1, end: int=-1, reload: bool=False, verbose: bool=False):
        '''        
            :param keys: Represent the keys of data sets in HDF5 from which to read CV values.
            :type keys: list

            :param units: List of units for each CV that needs to be read. If set to [], each CV will be assigned unit 'au' (i.e. atomic unit)
            :type units: list of str, optional, default=[]

            :param name: a name for printing/logging purposes
            :type name: str, optional, default=None
        
            :param start: index of starting point in trajectory to extract CV values, can be used to crop out initial equilibration time.
            :type start: int, optional, default=0

            :param stride: integer defining how to subsample the trajectory CV data. If set to a value larger than 1, only take sub samples every 'stride' steps. If set to 1, sample all trajectory steps. Can be used to decrease correlations between subsequent samples.
            :type stride: int, optional, default=1

            :param end: index of last trajectory step to be sampled. Can be used to crop out the final part of a simulation trajectory. If set to -1, take all trajectory samples, i.e. no cropping.
            :type end: int, optional, default=-1

            :param reload: If True, the first time the __call__ routine is called, the data that is read/computed from the trajectory will be written in numpy format to a file (with name equal to original trajectory file name appended with '.reload' at the end). The next time the __call__ routine is called again afterwards, the trajectory data will read directly from the .reload file. This results in a considerable speedup in the case the CV is computed from a XYZ trajectory. If False, the data will be read from the trajectory file with every call to the __call__ routine.
            :type reload: bool, optional, default=False

            :param verbose: If True, switch on the routine verbosity and print more logging
            :type verbose: bool, optional, default=False
        '''
        if len(keys)==0:
            raise ValueError('No dataset keys defined, received empty list for argument keys.')
        if len(units)==0:
            self.print('No units defined, set all units to au')
            units = ['au',]*len(keys)
        assert len(keys)==len(units), 'keys and units should be list of same length'
        self.keys = keys
        TrajectoryReader.__init__(self, units=units, name=name, start=start, stride=stride, end=end, reload=reload, verbose=verbose)

    @property
    def ncvs(self):
        return len(self.keys)
    
    def _read(self, fn: str):
        '''
            Read CV samples from the data set(s) in the HDF5 trajectory file defined by the given name

            :param fn: name of HDF5 file containing CV trajectory samples
            :type fn: str

            :raises ValueError: if no data could be extracted from trajectory, i.e. because start:end:stride were chosen to restrictive.
            :raises AssertionError: if the various CVs computed from the trajectory do not have matching number of samples.

            :return: trajectory data representing CV samples
            :rtype: np.ndarray
        '''
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
    '''
        Child class of TrajectoryReader to read the CV samples from a COLVAR file. COLVAR files (as used by e.g. Plumed) are just numpy readable text files representing arrays in which the columns represent different CVs and the rows represent the various timesteps.
    '''
    def __init__(self, indices: list, units: list=[], name: str=None, start: int=0, stride: int=1, end: int=-1, reload: bool=False, verbose: bool=False):
        '''
            :param indices: Represents the indices of the columns in COLVAR file from which to read the CV values.
            :type indices: list

            :param units: List of units for each CV that needs to be read. If set to [], each CV will be assigned unit 'au' (i.e. atomic unit)
            :type units: list of str, optional, default=[]

            :param name: a name for printing/logging purposes
            :type name: str, optional, default=None
        
            :param start: index of starting point in trajectory to extract CV values, can be used to crop out initial equilibration time.
            :type start: int, optional, default=0

            :param stride: integer defining how to subsample the trajectory CV data. If set to a value larger than 1, only take sub samples every 'stride' steps. If set to 1, sample all trajectory steps. Can be used to decrease correlations between subsequent samples.
            :type stride: int, optional, default=1

            :param end: index of last trajectory step to be sampled. Can be used to crop out the final part of a simulation trajectory. If set to -1, take all trajectory samples, i.e. no cropping.
            :type end: int, optional, default=-1

            :param reload: If True, the first time the __call__ routine is called, the data that is read/computed from the trajectory will be written in numpy format to a file (with name equal to original trajectory file name appended with '.reload' at the end). The next time the __call__ routine is called again afterwards, the trajectory data will read directly from the .reload file. This results in a considerable speedup in the case the CV is computed from a XYZ trajectory. If False, the data will be read from the trajectory file with every call to the __call__ routine.
            :type reload: bool, optional, default=False

            :param verbose: If True, switch on the routine verbosity and print more logging
            :type verbose: bool, optional, default=False
        '''
        if len(indices)==0:
            raise ValueError('No column indices defined, received empty list for argument indices.')
        if len(units)==0:
            self.print('No units defined, set all units to au')
            units = ['au',]*len(indices)
        assert len(indices)==len(units), 'indices and units should be list of same length'
        self.indices = indices
        TrajectoryReader.__init__(self, units=units, name=name, start=start, stride=stride, end=end, reload=reload, verbose=verbose)
    
    @property
    def ncvs(self):
        return len(self.indices)

    def _read(self, fn):
        '''
            Read CV samples from the COLVAR file defined by the given name

            :param fn: name of COLVAR file containing CV trajectory samples
            :type fn: str

            :raises ValueError: if no data could be extracted from trajectory, i.e. because start:end:stride were chosen to restrictive.
            :raises AssertionError: if the various CVs computed from the trajectory do not have matching number of samples.

            :return: trajectory data representing CV samples
            :rtype: np.ndarray
        '''
        data = np.loadtxt(fn)
        cvdata = None
        for i, (index, unit) in enumerate(zip(self.indices,self.units)):
            if len(data.shape)==1:
                if index==0:
                    col = self._slice(data)*parse_unit(unit)
                    assert len(col.shape)==1, 'Something went wrong in the slicing of the colvar data.'
                else:
                    raise ValueError('You specified a column index larger then 0, but could only read a single column from colvar file %s.' %(fn))
            else:
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


class ASEExtendedXYZReader(TrajectoryReader):
    '''
        Child class of TrajectoryReader to read a series of CVs, defined in keys, from an extended XYZ file using ASE. 
    '''
    def __init__(self, keys: list, units: list=[], name: str=None, start: int=0, stride: int=1, end: int=-1, reload: bool=False, verbose: bool=False):
        '''
            :param keys: list of ASE trajectory info keys to be read from the title of each frame in the XYZ file. For each key defined, the value of frame.info[key] will be extracted from the trajectory constructed with ase.io.read routine.
            :type keys: list
        
            :param units: List of units for each CV that needs to be read. If set to [], each CV will be assigned unit 'au' (i.e. atomic unit)
            :type units: list of str, optional, default=[]

            :param name: a name for printing/logging purposes
            :type name: str, optional, default=None
        
            :param start: index of starting point in trajectory to extract CV values, can be used to crop out initial equilibration time.
            :type start: int, optional, default=0

            :param stride: integer defining how to subsample the trajectory CV data. If set to a value larger than 1, only take sub samples every 'stride' steps. If set to 1, sample all trajectory steps. Can be used to decrease correlations between subsequent samples.
            :type stride: int, optional, default=1

            :param end: index of last trajectory step to be sampled. Can be used to crop out the final part of a simulation trajectory. If set to -1, take all trajectory samples, i.e. no cropping.
            :type end: int, optional, default=-1

            :param reload: If True, the first time the __call__ routine is called, the data that is read/computed from the trajectory will be written in numpy format to a file (with name equal to original trajectory file name appended with '.reload' at the end). The next time the __call__ routine is called again afterwards, the trajectory data will read directly from the .reload file. This results in a considerable speedup in the case the CV is computed from a XYZ trajectory. If False, the data will be read from the trajectory file with every call to the __call__ routine.
            :type reload: bool, optional, default=False

            :param verbose: If True, switch on the routine verbosity and print more logging
            :type verbose: bool, optional, default=False
        '''
        if len(keys)==0:
            raise ValueError('No dataset keys defined, received empty list for argument keys.')
        if len(units)==0:
            self.print('No units defined, set all units to au')
            units = ['au',]*len(keys)
        assert len(keys)==len(units), 'keys and units should be list of same length'
        self.keys = keys
        TrajectoryReader.__init__(self, units=units, name=name, start=start, stride=stride, end=end, reload=reload, verbose=verbose)

    @property
    def ncvs(self):
        return len(self.keys)

    def _read(self, fn: str):
        '''
            Routine to read .xyz trajectory file and for each frame extract the CV corresponding to the given keys in the XYZ frame title.

            :param fn: Name of XYZ file containing the trajectory
            :type fn: str

            :raises ValueError: if no data could be extracted from trajectory, i.e. because start:end:stride were chosen to restrictive.
            :raises AssertionError: if the various CVs computed from the trajectory do not have matching number of samples.
            
            :return: trajectory data representing CV samples
            :rtype: np.ndarray
        '''
        traj = read(fn, index=slice(self.start,self.end,self.stride))
        nsamples = len(traj)
        cvdata = np.zeros([nsamples, len(self.keys)])
        for iframe, frame in enumerate(traj):
            for ikey, key in enumerate(self.keys):
                cvdata[iframe,ikey] = frame.info[key]*parse_unit(self.units[ikey])
        return cvdata