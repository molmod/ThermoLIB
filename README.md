ThermoLIB is a library for the application of Statistical Physics, Thermodynamics and/or kinetic theory to molecular simulations. The library consists of several sub modules. ThermoLIB is developed by prof. Louis Vanduyfhuys at the Center for Molecular Modeling under supervision of prof. Veronique Van Speybroeck.

# Submodules

## Thermodynamics

Module for reading, transforming and manipulating free energy profiles. Functions include:

* the construction of 1D free energy profiles and 2D free energy surfaces from a single simulation, or from multiple (pobbibly biased) simulations using WHAM     including analytical error estimation.
* transformation of one collective variable to another
* projection of 2D profiles to 1D profiles
* "de"-projection of 1D to 2D profiles

## Kinetics

Module for computing the rate factor required in transition state theory required to compute the reaction rate constant. This factor is related to the time derivative of the collective variable in the transition state.

# Dependencies

* numpy
* scipy
* molmod
* sklearn
* matplotlib

Copyright (C) 2019 - 2021 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise stated.