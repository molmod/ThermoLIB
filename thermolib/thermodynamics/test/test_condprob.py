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


import numpy as np
import os
from molmod.units import kjmol
import matplotlib.pyplot as pl

from thermolib.thermodynamics.fep import BaseFreeEnergyProfile
from thermolib.thermodynamics.condprob import ConditionalProbability1D1D

np.random.seed(5)

def test_transform1D1D_linear(plot=False):
    '''
        Analytical test:    F(x) -> F(y) with y = h(x) = x/3 - 0.5
                            F(x) = F(h^(-1)(x)) = F(3*y + 3/2)
                            F(y) = F(h^(-1)(y)) - kT ln(dx/dy)
    '''

    x = np.arange(-3.5, 3.51, 0.01)
    f = (x**2 - 4)**2
    fes = BaseFreeEnergyProfile(x, f*kjmol, temp=300, f_unit='kjmol')
    #fes.plot('F.png')
    #pl.close()

    cv = np.random.uniform(low=-3, high=3, size=10000)
    np.savetxt('test_colvar', np.array([cv/3.-0.5, cv]).T)

    y = np.arange(-3.5, 3.51, 0.01)
    cond_prob = ConditionalProbability1D1D(y, x)
    cond_prob.process_trajectory_cvs('test_colvar', col_q1=0, col_cv=1)
    cond_prob.finish()

    fes_y = cond_prob.transform(fes, cv_label='y')
    fes_y.set_ref(ref='min')

    g = ((3*y + 1.5)**2 - 4)**2

    if plot:
        pl.plot(fes_y.cvs, fes_y.fs/kjmol, label='Transformed')
        pl.plot(y, g, '--', label='Analytical')
        pl.xlim([-2, 2])
        pl.ylim([-1, 30])
        pl.xlabel('y')
        pl.ylabel('F (kJ/mol)')
        pl.legend()
        pl.show()

    assert (np.abs(fes_y.fs[~np.isnan(fes_y.fs)]/kjmol-g[~np.isnan(fes_y.fs)]) < 0.75).all()
    os.remove('test_colvar')


def test_transform1D1D_nonlinear(plot=False):
    '''
        Analytical test:    F(x) -> F(y) with y = h(x) = 2*tan(x)
                            F(x) = F(h^(-1)(y)) = F(arctan(0.5*y))
                            F(y) = F(h^(-1)(y)) - kT ln(dx/dy)
    '''

    x = np.arange(-1.2, 1.21, 0.01)
    f = 50*(x**2 - 0.5)**2
    fes = BaseFreeEnergyProfile(x, f*kjmol, temp=300, f_unit='kjmol')

    cv = np.random.uniform(low=-1.2, high=1.2, size=100000)
    np.savetxt('test_colvar', np.array([2*np.tan(cv), cv]).T)

    y = np.arange(-3.3, 3.31, 0.02)
    cond_prob = ConditionalProbability1D1D(y, x)
    cond_prob.process_trajectory_cvs('test_colvar', col_q1=0, col_cv=1)
    cond_prob.finish()

    fes_y = cond_prob.transform(fes, cv_label='y')
    fes_y.set_ref(ref='min')

    g = 50*(np.arctan(0.5*y)**2 - 0.5)**2
    g -= 1/fes.beta/kjmol * np.log(0.5*np.cos(np.arctan(0.5*y))**2)
    g -= np.amin(g)

    if plot:
        pl.plot(fes_y.cvs, fes_y.fs/kjmol, label='Transformed')
        pl.plot(y, g, '--', label='Analytical')
        pl.ylim([-1, 20])
        pl.xlabel('y')
        pl.ylabel('F (kJ/mol)')
        pl.legend()
        pl.show()

    assert (np.abs(fes_y.fs[~np.isnan(fes_y.fs)]/kjmol-g[~np.isnan(fes_y.fs)]) < 1).all()
    os.remove('test_colvar')


if __name__=='__main__':

    test_transform1D1D_linear(plot=True)
    test_transform1D1D_nonlinear(plot=True)
