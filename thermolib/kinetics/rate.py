#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 - 2019 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
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

class BaseRateFactor(object):
    '''
        Class to compute the prefactor A required for the computation of
        the rate constant of the process of crossing a transition state:

            k = A * exp(-beta*F_TS)/int(exp(-beta*F(q)),q=-inf...q_TS)
    '''
    def __init__(self, fn_xyz, CV, CV_TS_lims, temp, CV_unit='au'):
        '''
            fn_xyz         filename of the trajectory from which the rate
                           factor will be computed.

            CV             a function that computes the value (and gradient) of
                           the collective variable

            CV_TS_lims     the lower and upper boundaries for determining
                           whether a certain frame of the trajectory
                           corresponds with the transition state (TS). In other
                           words, the condition CV(R)=CV_TS is replaced by
                           CV_TS_lims[0]<=CV(R)<=CV_TS_lims[1]

            masses         the masses of the atoms (should be consistently
                           indexed as the coords argument parsed to the
                           compute_contribution routine.

            temp           temperature

            CV_unit        the unit for printing the collective variable

        '''

        self.fn_xyz = fn_xyz
        self.xyzreader = XYZReader(fn_xyz)
        masses = np.array([pt[Z].mass for Z in self.xyzreader.numbers])
        self.Natoms = len(masses)
        self.masses = np.array([masses, masses, masses]).T.reshape([3*self.Natoms, 1]).flatten()
        self.CV = CV
        self.CV_TS_lims = CV_TS_lims
        self.CV_unit = CV_unit
        self.temp = temp
        self.beta = 1./(boltzmann*temp)
        self._inv_sqrt_masses = 1.0/np.sqrt(self.masses)
        self.Ts = []
        self.Ns = []
        self.T = None
        self.N = None
        self.A = None
        self.N_err = None
        self.T_err = None
        self.A_err = None

    def read_results(self, fn):
        data = np.loadtxt(fn)
        self.Ts = data[:,0]/second
        self.Ns = data[:,1]

    def process_trajectory(self, fn_out='rate_data.txt', verbose=False):
        raise NotImplementedError

    def compute_contribution(self, framenumber, coords, verbose=False):
        raise NotImplementedError

    def finish(self, fn_out='rate_data.txt'):
        raise NotImplementedError

    def result_no_statistics(self):
        self.T = self.Ts.mean()
        self.N = self.Ns.mean()
        self.A = self.T/self.N
        self.N_err = np.nan
        self.T_err = np.nan
        self.A_err = np.nan
        return self.A

    def result_blav_alternative(self, blocksizes=None, fitrange=[0,-1], exponent=1, plot=True, verbose=True, plot_auto_correlation_range=None):
        'Compute rate factor A=T/N through estimates of T and N separately and estimate error with block averaging'
        if blocksizes is None:
            blocksizes = np.arange(1,len(self.Ts)//2+1,1)
        if plot:
            fn_T = 'blav_rate_T.png'
            fn_N = 'blav_rate_N.png'
        print('Computing value and error of A=<T>/<N> using block averaging')
        print('Applying blockaveraging on T samples')
        self.T, self.T_err, Tcorrtime = blav(self.Ts, blocksizes=blocksizes, fitrange=fitrange, exponent=exponent, fn_plot=fn_T, unit='1e12/s', plot_auto_correlation_range=plot_auto_correlation_range)
        print('Applying blockaveraging on N samples')
        self.N, self.N_err, Ncorrtime = blav(self.Ns, blocksizes=blocksizes, fitrange=fitrange, exponent=exponent, fn_plot=fn_N, unit='1', plot_auto_correlation_range=plot_auto_correlation_range)
        self.A, self.A_err = self.T/self.N, abs(self.T/self.N)*np.sqrt((self.T_err/self.T)**2+(self.N_err/self.N)**2)
        if verbose:
            print('Rate factor with block averaging:')
            print('---------------------------------')
            print('  T = %.3e +- %.3e %s/s (int. autocorr. time = %.3f timesteps, exponent = %.3f)' %(self.T/(parse_unit(self.CV_unit)/second), self.T_err/(parse_unit(self.CV_unit)/second), self.CV_unit, Tcorrtime, exponent))
            print('  N = %.3e +- %.3e au   (int. autocorr. time = %.3f timesteps, exponent = %.3f)' %(self.N, self.N_err, Ncorrtime, exponent))
            print('')
            print('  Resulting A=T/N (error propagation: dA=|T/N|*sqrt((dT/T)**2+(dN/N)**2)')
            print('  A = %.3e +- %.3e %s/s' %(self.A/(parse_unit(self.CV_unit)/second), self.A_err/(parse_unit(self.CV_unit)/second), self.CV_unit))
            print()
        return self.A, self.A_err

    def result_blav(self, blocksizes=None, fitrange=[0,-1], exponent=1, plot=True, verbose=True, plot_auto_correlation_range=None):
        'Compute rate factor A directly'
        mask = self.Ns>0
        As = self.Ts[mask]
        print('Number of samples in TS = ', len(As))
        if blocksizes is None:
            blocksizes = np.arange(1,len(As)//2+1,1)
        if plot:
            fn_A = 'blav_rate_A.png'
        self.A, self.A_err, Acorrtime = blav(As, blocksizes=blocksizes, fitrange=fitrange, exponent=exponent, fn_plot=fn_A, unit='1e12/s', plot_auto_correlation_range=plot_auto_correlation_range)
        if verbose:
            print('Rate factor directly with block averaging:')
            print('---------------------------------')
            print('  A = %.3e +- %.3e %s/s (%i samples, int. autocorr. time = %.3f timesteps, exponent = %.3f)' %(self.A/(parse_unit(self.CV_unit)/second), self.A_err/(parse_unit(self.CV_unit)/second), self.CV_unit, len(As), Acorrtime, exponent))
            print()
        return self.A, self.A_err
    
    def result_bootstrapping(self, nboot, verbose=True):
        'Compute rate factor and estimate error with bootstrapping'
        Ts = []
        Ns = []
        for iboot in range(nboot):
            indices = np.random.random_integers(0, high=len(self.Ts)-1, size=len(self.Ts))
            T = self.Ts[indices].mean()
            Ts.append(T)
            indices = np.random.random_integers(0, high=len(self.Ns)-1, size=len(self.Ns))
            N = self.Ns[indices].mean()
            Ns.append(N)
        self.T, self.T_err = np.array(Ts).mean(), np.array(Ts).std()
        self.N, self.N_err = np.array(Ns).mean(), np.array(Ns).std()
        self.A, self.A_err = self.T/self.N, abs(self.T/self.N)*np.sqrt((self.T_err/self.T)**2+(self.N_err/self.N)**2)
        if verbose:
            print('Rate factor with bootstrapping (nboot=%i):' %nboot)
            print('------------------------------------------')
            print('  T = %.3e +- %.3e %s/s' %(T/(parse_unit(self.CV_unit)/second), T_err/(parse_unit(self.CV_unit)/second), self.CV_unit))
            print('  N = %.3e +- %.3e au' %(N,N_err))
            print('')
            print('  Resulting A=T/N (error propagation: dA=|T/N|*sqrt((dT/T)**2+(dN/N)**2)')
            print('  A = %.3e +- %.3e %s/s' %(A/(parse_unit(self.CV_unit)/second), A_err/(parse_unit(self.CV_unit)/second), self.CV_unit))
            print()
        return self.A, self.A_err

    def results_overview(self):
        print('###############  RESULTS  ###############')
        T, N, A, T_err, N_err, A_err = self.result_no_statistics()
        print('Without error estimation (whole simulation as one sample):')
        print('----------------------------------------------------------')
        print('  T =  %.3e %s/s' %(T/(parse_unit(self.CV_unit)/second), self.CV_unit))
        print('  N =  %.3e au' %N)
        print('')
        print('  Resulting A=T/N')
        print('  A =  %.3e %s/s' %(A/(parse_unit(self.CV_unit)/second), self.CV_unit))
        print()
        T, N, A, T_err, N_err, A_err = self.result_blav()
        nboot = 100
        T, N, A, T_err, N_err, A_err = self.result_bootstrapping(nboot)




class RateFactorEquilibrium(BaseRateFactor):
    '''
        Class to compute the factor A required for the computation of
        the rate constant of the process of crossing a transition state. This
        implementation only works for trajectories from equilbrium simulations
        (such as Umbrella sampling, but not Metadynamics):

            A = sqrt(k*T/(2*pi)) * <|DQ|>_TS

        This expression includes an analytic integration over the momentum
        degrees of freedom (which results in the factor sqrt(k*T/(2*pi)). The
        factor is given by the expression

             A = sqrt(k*T/(2*pi)) * <|DQ|>_TS = T/N

        with

             T = sqrt(k*T/(2*pi)) * <delta(Q(R)-q_TS)*|DQ|>_umb
             N = <delta(Q(R)-q_TS)>_umb

        in which DQ represents the gradient of Q(R) towards R, and R represent
        mass-weighted cartesian coordinates.
    '''

    def process_trajectory(self, fn_out='rate_data.txt', verbose=True):
        if verbose:
            print('Estimating rate factor from trajectory %s for TS=[%.3f,%.3f] %s' %(self.fn_xyz, self.CV_TS_lims[0]/parse_unit(self.CV_unit), self.CV_TS_lims[1]/parse_unit(self.CV_unit), self.CV_unit))
        for i, (title, coords) in enumerate(self.xyzreader):
            framenumber = int(title.split()[2].rstrip(','))
            self._compute_contribution(framenumber, coords, verbose=verbose)
        if verbose:
            print()
        self._finish(fn_out=fn_out)

    def _compute_contribution(self, framenumber, coords, verbose=False):
        '''
            Compute the contributions to T and N of the current frame given by
            coords. These contributions are stored in self.Ns and self.Ts.
            Afterwards, <|DQ|>_TS is computed from these contributions as well as
            some statistics to estimate the error (or at least get an idea of the
            error).
        '''
        assert (coords.shape[0]==self.Natoms and coords.shape[1]==3)
        Q, DQ = self.CV.compute(coords)
        DQ = DQ.T.reshape([3*self.Natoms, 1]).flatten()
        T, N = 0.0, 0.0
        if self.CV_TS_lims[0]<=Q<=self.CV_TS_lims[1]:
            tmp = DQ*self._inv_sqrt_masses #divide each atom coord vector with sqrt of its mass by means of numpy broadcasting
            T = np.linalg.norm(tmp)/np.sqrt(2*np.pi*self.beta)
            N = 1.0
        self.Ts.append(T)
        self.Ns.append(N)

    def _finish(self, fn_out='rate_data.txt'):
        self.Ts = np.array(self.Ts)
        self.Ns = np.array(self.Ns)
        data = np.zeros([len(self.Ts), 2], float)
        data[:,0] = self.Ts*second
        data[:,1] = self.Ns
        np.savetxt(fn_out, data, header='T [au/s], N [-]')




class RateFactorAlternative(BaseRateFactor):
    '''
        Class to compute the prefactor A required for the computation of the
        rate constant of the process of crossing a transition state. This
        implementation is applicable to any tranjectory and the expression for
        A is as follows:

            A = 0.5*<|dot(Q)|>_TS

        This expression of the prefactor does not include an analytical
        integration over the momenta. Therefore, this integral will be included
        numerically in this class using random momenta taken from the
        Maxwell-Boltzmann distribution. The factor 0.5 originiates from the
        fact that we want to compute the rate of going towards the product side
        once system is in the TS. Due to symmetry reasons wo do this by
        computing the absolute value of the rate of change of q (i.e. dot(Q))
        and take half to only account for TS --> P. The prefactor is given by
        the expression

             A = 0.5*<|dot(Q)|>_TS = T/N

        with

             T = 0.5*sum(1/Np*sum( abs(dot(q)_nk)*exp(-beta*Vn) , k=1..Np), n=1..Nr)
             N = sum( exp(-beta*Vn), n=1..Nr)

        Herein, the summation runs over the samples vec{Rn} and vec{Pk} and
        dot(q)_nk represents the time derivative of the collective variable,
        evaluated at vec{Rn} and vec{Pk}, which can be expressed as:

             dot(q)_nk = sum(dq/dRi*Pi/mi, i=1..3Nat)

        with dq/dRi the partial derivative of Q with respect to the i-th
        component of the 3N-dimensional cartesian coordinate vector, evaluated
        at vec{Rn}, mi the i-th component of the 3N-dimensional mass vector, and
        Pi represents the k-th component of the 3N-dimensional momentum vector
        vec{Pk}.
    '''
    def __init__(self, fn_xyz, fn_ener, CV, CV_TS_lims, temp, Nmomenta=1000):
        '''
            *Arguments*

            CV              a function that computes the value (and gradient) of
                            the collective variable

            CV_TS_lims      the lower and upper boundaries for determining
                            whether a certain frame of the trajectory
                            corresponds with the transition state (TS). In other
                            words, the condition CV(R)=CV_TS is replaced by
                            CV_TS_lims[0]<=CV(R)<=CV_TS_lims[1]

            masses          the masses of the atoms, assumed to be a
                            N-dimensional vector, but will be converted to its
                            3N dimensional analogue.

            temp            the temperature

            Nframes         The number of frames in the trajectory to be
                            processed.

            **Keyword Arguments**

            Nmomenta        The number of momenta samples taken from the
                            Maxwell-Boltzmann distribution for the numerical
                            integration over the momenta.
        '''
        BaseRateFactor.__init__(self, fn_xyz, CV, CV_TS_lims, temp)
        self.fn_ener = fn_ener
        self.data_ener = np.loadtxt(fn_ener) #read data from .ener file, potential energies are stored in 5th column (columnindex=4)
        self.Nmomenta = Nmomenta
        self.dotQs = [] #list keeping track of 1/Np*sum(abs(dot(q)_nk,k=1..Np)
        self.Vs = [] #list keeping track of V_n

    def process_trajectory(self, fn_out='data.traj', verbose=False):
        if verbose:
            print('Estimating rate factor from trajectory %s for TS=[%.3f,%.3f] %s' %(self.fn_xyz, self.CV_TS_lims[0]/parse_unit(self.CV_unit), self.CV_TS_lims[1]/parse_unit(self.CV_unit), self.CV_unit))
            print('    Reading potential energies from %s ...' %self.fn_ener)
        self.dict_ener = {} #convert to dictionairy because the data in fn_ener is not necessairy in increasing order of step number
        for i, time, K, temp, V, Cons, UsedTime in self.data_ener:
            self.dict_ener[i] = [time, K, temp, V, Cons, UsedTime] #duplicates overwrite previous values
        if verbose:
            print('    Processing trajectory')
        Nframes = 0
        Ncontrib = 0
        for i, (title, coords) in enumerate(self.xyzreader):
            framenumber = int(title.split()[2].rstrip(','))
            found = self._compute_contribution(framenumber, coords, verbose=verbose)
            Nframes += 1
            if found: Ncontrib += 1
        if verbose:
            print('        found %i contributions out of %i frames' %(Ncontrib, Nframes))
        self._finish(fn_out=fn_out)

    def _compute_contribution(self, framenumber, coords, verbose=False):
        '''
            Check whether the current frame defined by the given coordinates and
            energy correspond to a sample in the transition state. If so,
            compute its contribution to the nominator T and denominater N (see
            description above).

            *Arguments*

            framenumber     the index of the frame, solely for logging purposes

            coords          3N-dimensional vector containing the cartesian
                            coordinates

            energy          potential energy of the given frame
        '''
        assert (coords.shape[0]==self.Natoms and coords.shape[1]==3)
        Q, DQ = self.CV.compute(coords)
        DQ = DQ.T.reshape([3*self.Natoms, 1]).flatten()
        found = False
        if self.CV_TS_lims[0]<=Q<=self.CV_TS_lims[1]:
            if framenumber not in self.dict_ener.keys():
                #if verbose:
                #    print('Found frame (i=%i) with  CV = %.6f au    but could not read potential energy from dict_ener ==> skipped' %(framenumber, Q))
                return False
            V = self.dict_ener[framenumber][3]
            #if verbose:
            #    print('Found frame (i=%i) with  CV = %.6f au    and    Epot = %.6e au' %(framenumber, Q, V))
            dotQ = 0.0
            for k in range(self.Nmomenta):
                dotQ += 0.5*abs(np.dot(DQ, self._velocity_from_MB())) #Pi/mi=vi
            dotQ /= self.Nmomenta
            self.dotQs.append(dotQ)
            self.Vs.append(V)
            found = True
        return found

    def _velocity_from_MB(self):
        '''
            Routine to draw a random velocity vector for each atom from a
            Maxwell Boltzmann distribution. Temperature and masses are defined
            in the attributes of self.
        '''
        v = np.random.normal(loc=0.0, scale=1.0, size=3*self.Natoms)
        v *= np.sqrt(boltzmann*self.temp)*self._inv_sqrt_masses
        return v

    def _finish(self, fn_out='data.txt'):
        self.dotQs = np.array(self.dotQs)
        self.Vs = np.array(self.Vs)-min(self.Vs)
        self.Ts = self.dotQs*np.exp(-self.beta*self.Vs)
        self.Ns = np.exp(-self.beta*self.Vs)
        data = np.zeros([len(self.Ts), 2], float)
        data[:,0] = self.Ts*second
        data[:,1] = self.Ns
        np.savetxt(fn_out, data, header='T [au/s], N [-]')
