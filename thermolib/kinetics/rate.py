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

from ..tools import blav

import numpy as np
import sys, os
import matplotlib.pyplot as pp

__all__ = ['RateFactorEquilibrium']

class BaseRateFactor(object):
    '''
        An abstract class for inheriting child classes to compute the prefactor :math:`A` required for the computation of the rate constant of the process of crossing a transition state:

        .. math::

            \\begin{aligned}
                k &= A\\cdot\\frac{ e^{-\\beta F(q_{TS})} }{ \\int_{-\\infty}^{q_{TS}}e^{-\\beta F(q)}dq } \\\\

                A &= \\frac{1}{2}\\left\\langle|\\dot{Q}|\\right\\rangle_{TS}
            \\end{aligned}
    '''
    def __init__(self, CV, CV_TS_lims, temp, CV_unit='au'):
        '''
            :param CV: a function that computes the value (and gradient) of the collective variable
            :type CV: callable

            :param CV_TS_lims: the lower and upper boundaries for determining whether a certain frame of the trajectory corresponds with the transition state (TS). In other words, the condition CV(R)=CV_TS is replaced by CV_TS_lims[0]<=CV(R)<=CV_TS_lims[1]
            :type CV_TS_lims: list(float)
            
            :param masses: the masses of the atoms (should be consistently indexed as the coords argument parsed to the compute_contribution routine).
            :type masses: np.ndarray(float)

            :param temp: temperature
            :type temp: float

            :param CV_unit: the unit for printing the collective variable
            :type CV_unit: str
        '''
        self.CV = CV
        self.CV_TS_lims = CV_TS_lims
        self.CV_unit = CV_unit
        self.temp = temp
        self.beta = 1./(boltzmann*temp)
        self.Natoms = None
        self.As = []
        self.A = None
        self.A_err = None
        self._finished = False

    def read_results(self, fn, A_unit='1/second'):
        data = np.loadtxt(fn)
        self.As = data*parse_unit(A_unit)
        self._finished = True

    def process_trajectory(self, *args, **kwargs):
        raise NotImplementedError

    def compute_contribution(self, *args, **kwargs):
        raise NotImplementedError

    def finish(self, fn=None):
        self.As = np.array(self.As)
        data = self.As*second
        if fn is not None:
            np.savetxt(fn, data, header='A [au/s]')
        self._finished = True

    def result_no_statistics(self):
        assert self._finished, "Reading trajectory data is not finished yet"
        self.A = self.As[~np.isnan(self.As)].mean()
        self.A_err = np.nan
        return self.A

    def result_blav(self, blocksizes=None, fitrange=[0,-1], exponent=1, plot=True, verbose=True, plot_ac=False, ac_range=None, acft_plot_range=None):
        'Compute rate factor A and estimate its error with block averaging'
        assert self._finished, "Reading trajectory data is not finished yet"
        As = self.As[~np.isnan(self.As)]
        print('Number of TS samples = ', len(As))
        if blocksizes is None:
            blocksizes = np.arange(1,len(As)//2+1,1)
        if plot:
            fn_A = 'blav_rate_A.png'
        self.A, self.A_err, Acorrtime = blav(As, blocksizes=blocksizes, fitrange=fitrange, exponent=exponent, fn_plot=fn_A, unit='1e12/s', plot_ac=plot_ac, ac_range=ac_range, acft_plot_range=acft_plot_range)
        if verbose:
            print('Rate factor directly with block averaging:')
            print('------------------------------------------')
            print('  A = %.3e +- %.3e %s/s (%i TS samples, int. autocorr. time = %.3f timesteps, exponent = %.3f)' %(self.A/(parse_unit(self.CV_unit)/second), self.A_err/(parse_unit(self.CV_unit)/second), self.CV_unit, len(As), Acorrtime, exponent))
            print()
        return self.A, self.A_err
        
    def result_bootstrapping(self, nboot, verbose=True):
        'Compute rate factor A and estimate error with bootstrapping'
        assert self._finished, "Reading trajectory data is not finished yet"
        data = self.As[~np.isnan(self.As)]
        Ndata = len(data)
        As = []
        for iboot in range(nboot):
            indices = np.random.random_integers(0, high=Ndata-1, size=Ndata)
            As.append(data[indices].mean())
        As = np.array(As)
        self.A, self.A_err = As.mean(), As.std()
        if verbose:
            print('Rate factor with bootstrapping (nboot=%i):' %nboot)
            print('------------------------------------------')
            print('  A = %.3e +- %.3e %s/s (%i TS samples)' %(self.A/(parse_unit(self.CV_unit)/second), self.A_err/(parse_unit(self.CV_unit)/second), self.CV_unit, Ndata))
            print()
        return self.A, self.A_err
    
    def results_overview(self, nboot=1000):
        assert self._finished, "Reading trajectory data is not finished yet"
        print('###############  RESULTS  ###############')
        A = self.result_no_statistics()
        print('Without error estimation:')
        print('-------------------------')
        print('  A =  %.3e %s/s' %(A/(parse_unit(self.CV_unit)/second), self.CV_unit))
        print()
        A, A_err = self.result_blav()
        A, A_err = self.result_bootstrapping(nboot)



class RateFactorEquilibrium(BaseRateFactor):
    '''
        Class to compute the factor :math:`A` required for the computation of the rate constant :math:`k` of the process of crossing a transition state:

        .. math::

            \\begin{aligned}
                k &= A\\cdot\\frac{ e^{-\\beta F(q_{TS})} }{ \\int_{-\\infty}^{q_{TS}}e^{-\\beta F(q)}dq } \\\\

                A &= \\frac{1}{2}\\left\\langle|\\dot{Q}|\\right\\rangle_{TS}
            \\end{aligned}
        
        Herein, the subscript TS refers to the fact that the average has to be taken only at configurational states corresponding to the transition state (TS). Furthermore, the average contains an integral over configurational phase space as well as momenta. The integral over momenta can either be performed analytical or numerical by taking momentum samples according to a certain distribution. 
        
            * When performing the momentum integration analytical, the above expression simplifies to:

                .. math::

                    A = \\sqrt{\\frac{kT}{2\\pi}} \\cdot \\left\\langle\\left||\\vec{\\nabla}_x Q\\right||\\right\\rangle_{TS}

                where the average is now only taken in configurational space for states corresponding to the transition state (hence still the subscript TS). Furtheremore, :math:`\\vec{\\nabla}_x Q` represents the gradient of the collective variable :math:`Q` towards the mass-weighted cartesian coordinates.

            * When performing the momentum integral numerically, the expression for A can be rewritten as:

                .. math::

                    A = \\frac{1}{2}\\int_{\\vec{x}\\in TS} \\left[\\int_{-\\infty}^{+\\infty} \\left|\\dot{Q}(\\vec{x},\\vec{P})\\right| p_p(\\vec{P}) d\\vec{P}\\right]p_x(\\vec{x})d\\vec{x}
            
                where :math:`\\dot{Q}(\\vec{x},\\vec{P})` indicates that the time derivative of :math:`Q` depends on both configurational state indicated by its mass-weighted cartesian coordinates :math:`\\vec{x}` as well as the mass-weighted momenta :math:`\\vec{P}`. Furthermore :math:`p_p(\\vec{P})` represents the momentum probability distribution which will need to be specified (eg. the Maxwell-Boltzmann distribution). The integral over :math:`\\vec{x}` is computed using the samples :math:`\\vec{x}_i` taken from the MD trajectory while the integration over :math:`\\vec{P}` is computed numerically by taking random samples :math:`\\vec{P}_k` from the given momentum distribution :math:`p_p`:

                .. math::
                
                    \\begin{aligned}
                        A &= \\frac{1}{N_s}\\sum_{i=1}^{N_s}A(\\vec{x}_i) \\\\

                        A(x_i) &=  \\frac{1}{2N_p}\\sum_{k=1}^{N_p}\\left|\\dot{Q}\\left(\\vec{x}_i,\\vec{P}_k\\right)\\right|
                    \\end{aligned}
            
        These computational methods are implemented in the :meth:`thermolib.thermodynamics.rate.RateFactorEquilibrium.proces_trajectory` routine.
    '''

    def process_trajectory(self, fn_xyz, sub=slice(None,None,None), finish=True, fn_samples=None, momenta='analytical', Nmomenta=500, verbose=False):
        '''
            Process the given XYZ trajectory and store T and N values.

            :param fn_xyz: filename of the trajectory from which the rate factor will be computed.
            :type fn_xyz: str
            
            :param sub: slice object to subsample the xyz trajectory. For more information see https://molmod.github.io/molmod/reference/io.html#module-molmod.io.xyz
            :type sub: slice, optional, default=(None,None,None)

            :param finish: when set to True, the finish routine will be called after processing the trajectory, which will finalize storing the data. If multiple trajectory from different files need to be read, set finish to False for all but the last trajectory.
            :type finish: bool, optional, default=True

            :param fn_samples: write the rate factor samples (i.e. the As array) to the given file. This feature is switched off by specifying None.
            :type fn_samples: str, optional, default=None

            :param momenta: specify how to compute the momentum part of the phase space integral in computing the As samples (see description above). The following options are available:
            
                * **analytical** -- compute the momentum integral analytically, is hence the fastest method
                * **MB** -- compute the momentum integral numerical by taking random samples for the velocity from the Maxwell-Boltzmann distribution.

            :type momenta: string, optional, default=analytical

            :param Nmomenta: the number of momentum samples taken from the given distribution in case of numerical momentum integration. This keyword is only relevant when momenta is not set to analytical.
            :type Nmomenta: float, optional, default=500

            :param verbose: increase verbosity by setting to True.
            :type verbose: bool, optional, default=False
        '''
        #initialization
        xyzreader = XYZReader(fn_xyz, sub=sub)
        masses = np.array([pt[Z].mass for Z in xyzreader.numbers])
        Natoms = len(masses)
        masses3 = np.array([masses, masses, masses]).T.reshape([3*Natoms, 1]).flatten()
        if self.Natoms is None:
            self.Natoms = Natoms
            self.masses = masses3
            self._inv_sqrt_masses = 1.0/np.sqrt(masses3)
        else:
            assert self.Natoms==len(masses), "Incompatible number of atoms in file %s with respect to previously read files" %fn_xyz
            assert (self.masses==masses3).all(), "Incompatible masses in file %s with respect to previously read files" %fn_xyz
        
        if verbose:
            print('Estimating rate factor from trajectory %s for TS=[%.3f,%.3f] %s using %s momentum integration' %(fn_xyz, self.CV_TS_lims[0]/parse_unit(self.CV_unit), self.CV_TS_lims[1]/parse_unit(self.CV_unit), self.CV_unit, momenta))
        for i, (title, coords) in enumerate(xyzreader):
            #framenumber = int(title.split()[2].rstrip(','))
            self._compute_contribution(coords, momenta=momenta, Nmomenta=Nmomenta, verbose=verbose)
        
        if finish:
            if verbose:
                print('Finishing')
            self.finish(fn=fn_samples)

    def _compute_contribution(self, coords, momenta='analytical', Nmomenta=500, verbose=False):
        '''
            Compute the contributions to the rate factor of the current frame given by **coords**. These contributions are stored in self.As. Afterwards, <|DQ|>_TS is computed from these contributions as well as some statistics to estimate the error (or at least get an idea of the error).

            :param coords: 3N-dimensional vector containing the cartesian coordinates
            :type coords: np.ndarray

            :param momenta: specify how to compute the momentum part of the phase space integral in computing the As samples. The following options are available:
            
                * **analytical** -- compute the momentum integral analytically, is hence the fastest method
                * **MB** -- compute the momentum integral numerical by taking random samples for the velocity from the Maxwell-Boltzmann distribution.
            
            :type momenta: string, optional

            :param verbose: increase verbosity by setting to True, defaults to True
            :type verbose: bool, optional
        '''
        assert (coords.shape[0]==self.Natoms and coords.shape[1]==3)
        Q, DQ = self.CV.compute(coords)
        DQ = DQ.flatten()
        Ai = np.nan
        if self.CV_TS_lims[0]<=Q<=self.CV_TS_lims[1]:
            if momenta=='analytical':
                tmp = DQ*self._inv_sqrt_masses #divide each atom coord vector with sqrt of its mass by means of numpy broadcasting
                Ai = np.linalg.norm(tmp)/np.sqrt(2*np.pi*self.beta)
            elif momenta in ['MB']:
                Ai = 0.0
                for k in range(Nmomenta):
                    Ai += 0.5*abs(np.dot(DQ, self._random_velocity(distribution=momenta))) #Pi/mi=vi
                Ai /= Nmomenta
            else:
                raise ValueError('Value %s for momenta is not supported. See documentation for more information.' %momenta)
        self.As.append(Ai)

    def _random_velocity(self, distribution='MB'):
        '''
            Routine to draw a random velocity vector for each atom from the distribution specified in distribution argument. Temperature and masses are defined in the attributes of self.
        '''
        if distribution=='MB':
            v = np.random.normal(loc=0.0, scale=1.0, size=3*self.Natoms)
            v *= np.sqrt(boltzmann*self.temp)*self._inv_sqrt_masses
        else:
            raise ValueError('Distribution %s not supported' %distribution)
        return v
