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

from ..tools import blav, h5_read_dataset
from ..error import GaussianDistribution, LogGaussianDistribution, Propagator

import numpy as np
import matplotlib.pyplot as pp

__all__ = ['BaseRateFactor', 'RateFactorEquilibrium']

class BaseRateFactor(object):
    '''
        An abstract class for inheriting child classes to compute the prefactor :math:`A` required for the computation of the rate constant of the process of crossing a transition state:

        .. math::

                k &= A\\cdot\\frac{ e^{-\\beta F(q_{TS})} }{ \\int_{-\\infty}^{q_{TS}}e^{-\\beta F(q)}dq } \\\\
                A &= \\frac{1}{2}\\left\\langle|\\dot{Q}|\\right\\rangle_{TS}
    '''
    def __init__(self, CV, CV_TS_lims, temp, CV_unit='au'):
        '''
            :param CV: definition of the collective variable as function of the atomic coordinates of the system
            :type CV: insgtance of child of :py:class:`CollectiveVariable <thermolib.thermodynamics.cv.CollectiveVariable>`

            :param CV_TS_lims: the lower and upper boundaries for determining whether a certain frame of the trajectory corresponds with the transition state (TS). In other words, the condition CV(R)=CV_TS is replaced by CV_TS_lims[0]<=CV(R)<=CV_TS_lims[1]
            :type CV_TS_lims: list[float]

            :param temp: temperature of the simulation from which the rate factor will be computed
            :type temp: float

            :param CV_unit: the unit for printing the collective variable
            :type CV_unit: str
        '''
        assert isinstance(CV_TS_lims,list) and len(CV_TS_lims)==2 and isinstance(CV_TS_lims[0],float) and isinstance(CV_TS_lims[1],float), 'CV_TS_lims needs to be list of two float values'
        self.CV = CV
        self.CV_TS_lims = CV_TS_lims
        self.CV_unit = CV_unit
        self.temp = temp
        self.beta = 1./(boltzmann*temp)
        self.Natoms = None
        self.As = []
        self.A = None
        self.A_dist = None
        self._finished = False

    def read_results(self, fn, A_unit='au/s'):
        '''
            Read the samples for the A property from the given file

            :param fn: file name from which to read the A samples
            :type fn: str

            :param A_unit: unit in which the A samples were stored in the file
            :type A_unit: str, optional, default='au/s'
        '''
        data = np.loadtxt(fn)
        self.As = data*parse_unit(A_unit)
        self._finished = True

    def process_trajectory(self, *args, **kwargs):
        '''
            To be implemented in a child class
        '''
        raise NotImplementedError

    def _compute_contribution(self, *args, **kwargs):
        '''
            To be implemented in a child class
        '''
        raise NotImplementedError

    def finish(self, fn=None):
        '''
            Finish extracting/reading samples of the A property and optionally write all samples to a given file.

            :param fn: If not None, name of file to which the A samples will be written for later reuse
            :type fn: _tstrype_, optional, default=None
        '''
        self.As = np.array(self.As)
        data = self.As*second
        if fn is not None:
            np.savetxt(fn, data, header='A [au/s]')
        self._finished = True

    def result_no_statistics(self):
        '''
            Store and return the mean A value from the stored samples

            :return: mean A value
            :rtype: float
        '''
        assert self._finished, "Reading trajectory data is not finished yet"
        self.A = self.As[~np.isnan(self.As)].mean()
        self.A_dist = None
        return self.A

    def result_blav(self, fn=None, plot=False, blocksizes=None, fitrange=[1,np.inf], model_function=None, plot_ylims=None, verbose=True):
        '''
            Store and return the error distribution of the stored A samples. Error is estimated using the block averaging (blav) method. For more details on this blav procedure as well as the meaning of the arguments `blocksizes`, `fitrange`, and `model_function`, see documentation of the routine :py:meth:`_wrap <thermolib.tools._blav>`.


            :param fn: If not None, name of file to store a plot of the block averaging procedure to
            :type fn: str, optional, default=None

            :param plot: If True, make plot of the block averaging procedure. Is ignored if fn is not None.
            :type plot: bool, optional, default=False

            :param plot_ylims: Limits for the y-axis in the error plot.
            :type plot_ylims: list or None, optional, default=None

            :param verbose: If True, turn on verbosity of logging during this routine
            :type verbose: bool, optional, default=True

            :return: Distribution of A samples
            :rtype: :py:class:`GaussianDistribution <thermolib.error.GaussianDistribution>`
        '''
        assert self._finished, "Reading trajectory data is not finished yet"
        As = self.As[~np.isnan(self.As)]
        if blocksizes is None:
            blocksizes = np.arange(1,int(len(As)/2), 1)
        self.A = As.mean()
        A_std, Acorrtime = blav(As, blocksizes=blocksizes, fitrange=fitrange, model_function=model_function, plot=plot, fn_plot=fn, unit='1/s', plot_ylims=plot_ylims)
        self.A_dist = GaussianDistribution(self.A, A_std) 
        
        if verbose:
            print('Rate factor with block averaging:')
            print('---------------------------------')
            print('  A = %s (%i TS samples, int. autocorr. time = %.3f timesteps)' %(self.A_dist.print(unit='%s/s' %self.CV_unit, do_scientific=True), len(As), Acorrtime))
            print()
        return self.A, self.A_dist
        
    def result_bootstrapping(self, nboot, verbose=True):
        '''
            Store and return the error distribution of the stored A samples. Error is estimated using the bootstrapping method.

            :param nboot: number of bootstrapping cycles
            :type nboot: int

            :param verbose: If True, turn on verbosity of logging during this routine
            :type verbose: bool, optional, default=True

            :return: Distribution of A samples
            :rtype: :py:class:`GaussianDistribution <thermolib.error.GaussianDistribution>`
        '''
        assert self._finished, "Reading trajectory data is not finished yet"
        data = self.As[~np.isnan(self.As)]
        Ndata = len(data)
        As = []
        for iboot in range(nboot):
            indices = np.random.random_integers(0, high=Ndata-1, size=Ndata)
            As.append(data[indices].mean())
        As = np.array(As)
        self.A, A_std = As.mean(), As.std()
        self.A_dist = GaussianDistribution(self.A, A_std) 
        if verbose:
            print('Rate factor with bootstrapping (nboot=%i):' %nboot)
            print('------------------------------------------')
            print('  A = %s (%i TS samples)' %(self.A_dist.print(unit='%s/s' %self.CV_unit, do_scientific=True), Ndata))
            print()
        return self.A, self.A_dist

    def compute_rate(self, fep, propagator=Propagator(target_distribution=None), verbose=False):
        '''
            Combine the kinetic information of the rate factor encoded in the current instance with the thermodynamic information encoded in the free energy profile fep to obtain a rate constant according to the formula 

            .. math::

                k &= A\\cdot\\frac{ e^{-\\beta F(q_{TS})} }{ \\int_{-\\infty}^{q_{TS}}e^{-\\beta F(q)}dq } \\\\
                A &= \\frac{1}{2}\\left\\langle|\\dot{Q}|\\right\\rangle_{TS}

            :param fep: free energy profile to extract required thermodynamic free energy information
            :type fep: :py:class:`SimpleFreeEnergyProfile <thermolib.thermodynamics.fep.SimpleFreeEnergyProfile>`

            :param propagator: a Propagator used for error propagation. Can be usefull if one wants to adjust the error propagation settings (such as the number of random samples taken, or the desired distribution of the targeted error). See documentation on the :py:class:`Propagator <thermolig.error.Propagator>` class for more info.
            :type propagator: instance of :py:class:`Propagator <thermolib.error.Propagator>`, optional, default=Propagator(target_distribution=None)

            :param verbose: If True, turns on verbosity
            :type verbose: bool, optional, default=False

            :return: list of forward & backward rate constants and phenomenological free energy barriers and potentially the corresponding error distributions (if both the current instance and the fep instance have associated error distributions)
            :rtype: forward k, distribution of forward k, forward dF, distribution of forward dF, backard k, distribution of backward k, backward dF, distribution of backward dF
        '''
        if not (self.A_dist is None or fep.ts.F_dist is None or fep.R.Z_dist is None or fep.P.Z_dist is None):
            def fun_k(A,Fts,Z):
                return A*np.exp(-Fts/(boltzmann*fep.T))/Z
            def fun_F(A,Fts,Z):
                return -np.log(fep.beta*planck*fun_k(A,Fts,Z))/fep.beta
            #setup propagator for forward k and dF
            propagator.reset()
            propagator.gen_args_samples(self.A_dist, fep.ts.F_dist, fep.R.Z_dist)
            #calculate k_forward
            propagator.calc_fun_values(fun_k)
            k_forward = propagator.get_distribution(target_distribution=LogGaussianDistribution)
            #calculate dF_forward
            propagator.calc_fun_values(fun_F)
            dF_forward = propagator.get_distribution(target_distribution=GaussianDistribution)
            #setup propagator for backward k and dF
            propagator.reset()
            propagator.gen_args_samples(self.A_dist, fep.ts.F_dist, fep.P.Z_dist)
            #calculate k_backward
            propagator.calc_fun_values(fun_k)
            k_backward = propagator.get_distribution(target_distribution=LogGaussianDistribution)
            #calculate dF_backward
            propagator.calc_fun_values(fun_F)
            dF_backward = propagator.get_distribution(target_distribution=GaussianDistribution)
            if verbose:
                print('k_F  = %s' %k_forward.print(unit='1/s', do_scientific=True))
                print('dF_F = %s' %dF_forward.print(unit='kjmol'))
                print('k_B  = %s' %k_backward.print(unit='1/s', do_scientific=True))
                print('dF_B = %s' %dF_backward.print(unit='kjmol'))
            return k_forward.mean(), k_forward, dF_forward.mean(), dF_forward, k_backward.mean(), k_backward, dF_backward.mean(), dF_backward
        else:
            k_forward   = self.A*np.exp(-fep.ts.F/(boltzmann*fep.T))/fep.R.Z
            k_backward  = self.A*np.exp(-fep.ts.F/(boltzmann*fep.T))/fep.P.Z
            dF_forward  = boltzmann*fep.T*np.log(boltzmann*fep.T/(planck*k_forward ))
            dF_backward = boltzmann*fep.T*np.log(boltzmann*fep.T/(planck*k_backward))
            if verbose:
                print('k_F  = %.3e 1/s'    %(k_forward*second))
                print('dF_F = %.3f kJ/mol' %(dF_forward/kjmol))
                print('k_B  = %.3e 1/s'    %(k_backward*second))
                print('dF_B = %.3f kJ/mol' %(dF_backward/kjmol))
            return k_forward, None, dF_forward, None, k_backward, None, dF_backward, None


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
            Process the given XYZ trajectory to compute and store A samples.

            :param fn_xyz: filename of the trajectory from which the rate factor will be computed.
            :type fn_xyz: str
            
            :param sub: slice object to subsample the xyz trajectory. For more information see https://wiki.fysik.dtu.dk/ase/ase/io/io.html
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
        trajectory = read(fn_xyz, index=sub)
        masses = trajectory[0].get_masses() * amu
        Natoms = len(masses)
        masses3 = np.array([masses, masses, masses]).T.reshape([3*Natoms, 1]).flatten()
        if self.Natoms is None:
            self.Natoms = Natoms
            self.masses = masses3
            self._inv_sqrt_masses = 1.0 / np.sqrt(masses3)
        else:
            assert self.Natoms==len(masses), "Incompatible number of atoms in file %s with respect to previously read files" %fn_xyz
            assert (self.masses==masses3).all(), "Incompatible masses in file %s with respect to previously read files" %fn_xyz
        
        if verbose:
            print('Estimating rate factor from trajectory %s for TS=[%.3f,%.3f] %s using %s momentum integration' %(fn_xyz, self.CV_TS_lims[0]/parse_unit(self.CV_unit), self.CV_TS_lims[1]/parse_unit(self.CV_unit), self.CV_unit, momenta))
        for atoms in trajectory:
            self._compute_contribution(atoms, momenta=momenta, Nmomenta=Nmomenta)
        
        if finish:
            if verbose:
                print('Finishing')
            self.finish(fn=fn_samples)

    def process_trajectory_h5(self, fn_h5, sub=slice(None,None,None), finish=True, fn_samples=None, momenta='analytical', Nmomenta=500, verbose=False):
        '''
            Process the given XYZ trajectory to compute and store A samples.

            :param fn_h5: filename of the trajectory from which the rate factor will be computed.
            :type fn_h5: str
            
            :param sub: slice object to subsample the xyz trajectory.
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
        masses = h5_read_dataset(fn_h5, '/system/masses/')
        Natoms = len(masses)
        masses3 = np.array([masses, masses, masses]).T.reshape([3*Natoms, 1]).flatten()
        if self.Natoms is None:
            self.Natoms = Natoms
            self.masses = masses3
            self._inv_sqrt_masses = 1.0/np.sqrt(masses3)
        else:
            assert self.Natoms==len(masses), "Incompatible number of atoms in file %s with respect to previously read files" %fn_h5
            assert (self.masses==masses3).all(), "Incompatible masses in file %s with respect to previously read files" %fn_h5
        
        if verbose:
            print('Estimating rate factor from trajectory %s for TS=[%.3f,%.3f] %s using %s momentum integration' %(fn_h5, self.CV_TS_lims[0]/parse_unit(self.CV_unit), self.CV_TS_lims[1]/parse_unit(self.CV_unit), self.CV_unit, momenta))
        
        for coords in h5_read_dataset(fn_h5, '/trajectory/pos/'):
            self._compute_contribution(coords, momenta=momenta, Nmomenta=Nmomenta)
        
        if finish:
            if verbose:
                print('Finishing')
            self.finish(fn=fn_samples)

    def _compute_contribution(self, atoms, momenta='analytical', Nmomenta=500):
        '''
            Compute a sample for the A rate factor from the current frame given by **coords**. These samples are stored in self.As. Afterwards, <|DQ|>_TS is computed from these contributions as well as some statistics to estimate the error (or at least get an idea of the error).

            :param coords: 3N-dimensional vector containing the cartesian coordinates
            :type coords: np.ndarray

            :param momenta: specify how to compute the momentum part of the phase space integral in computing the As samples. The following options are available:
            
                * **analytical** -- compute the momentum integral analytically, is hence the fastest method
                * **MB** -- compute the momentum integral numerical by taking random samples for the velocity from the Maxwell-Boltzmann distribution.
            
            :type momenta: string, optional, default='analytical'

            :param Nmomenta: number of random samples taken from the Maxwell-Boltzmann distribution upon selection of the MB method for momenta parameter. This parameter is ignored if momenta='analytical'
            :type momenta: int, optional, default=500
        '''
        Q, DQ = self.CV.compute(atoms)
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

            :param distribution: the distribution from which to take random samples for the momenta. Currently, only 'MB' (for Maxwell-Boltzmann distribution at the temperature stored in self.temp) is implemented.
            :type distribution: str, optional, default='MB'
        '''
        if distribution=='MB':
            v = np.random.normal(loc=0.0, scale=1.0, size=3*self.Natoms)
            v *= np.sqrt(boltzmann*self.temp)*self._inv_sqrt_masses
        else:
            raise ValueError('Distribution %s not supported' %distribution)
        return v
