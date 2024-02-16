# ThermoLIB

ThermoLIB is a library for the application of Statistical Physics, Thermodynamics and/or kinetic theory to molecular simulations. The library consists of several sub modules.

## Thermodynamics

Module for reading, transforming and manipulating free energy profiles. Functions include:

* the construction of 1D free energy profiles and 2D free energy surfaces from a single simulation, or from multiple (pobbibly biased) simulations using WHAM including a (correlated) error estimation.
* transformation of one collective variable to another
* projection of 2D profiles to 1D profiles
* "de"-projection of 1D to 2D profiles
* estimation of thermodynamics properties derivable from the free energy surface (e.g. free energy of macrostate defined as the integral over part of the CV-space, free energy of reaction, ...)

## Kinetics

Module for computing the rate factor required in transition state theory required to compute the reaction rate constant. This factor is related to the time derivative of the collective variable in the transition state.

# Dependencies

* Cython
* numpy
* scipy
* molmod
* sklearn
* matplotlib

# Therms of use

ThermoLIB is developed by prof. Louis Vanduyfhuys at the Center for Molecular Modeling under supervision of prof. Veronique Van Speybroeck. Usage of ThermoLIB should be requested with prof. Van Speybroeck and/or prof. Vanduyfhuys.

Copyright (C) 2019 - 2023 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise stated.