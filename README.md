# ThermoLIB

ThermoLIB is a library for the application of Statistical Physics, Thermodynamics and/or kinetic theory to molecular simulations. The library consists of several sub modules. Documentation on installation and usage can be found at https://github.ugent.be/pages/lvduyfhu/ThermoLIB/.

## Submodules

### Thermodynamics

Module for reading, transforming and manipulating free energy profiles. Functions include computing free energy profile from input trajectory by means of a histogram, transformation of one collective variable to another, projection of 2D profiles to 1D profiles, "de"-projection of 1D to 2D profiles, derivation of (meta)stable macrostates.

### Kinetics

Module for computing the rate factor required in transition state theory required to compute the reaction rate constant. This factor is related to the time derivative of the collective variable in the transition state.

## Dependencies

* numpy
* scipy
* molmod
* sklearn
* matplotlib

## Terms of use

ThermoLIB is developed by Louis Vanduyfhuys at the Center for Molecular Modeling under supervision of prof. Veronique Van Speybroeck. Usage of this package should be authorized by prof. Van Van Speybroeck.

Copyright (C) 2019 - 2021 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise stated.
