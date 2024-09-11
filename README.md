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

# Dependencies & Installation

ThermoLIB has the following dependencies:

* Cython
* numpy
* scipy
* molmod
* scikit-learn
* matplotlib
* h5py
* ase

As molmod is currently no longer maintained, it might result in conflicting package versions with some of the new versions of the above packages. Therefore, below I show how to set up a conda environment with confirmed non-conflicting and working versions of all dependencies above. 

    conda create -n thermolib python==3.8.5
    conda activate thermolib
    pip install numpy==1.22.0
    pip install matplotlib==3.3.4
    pip install scipy==1.6.3
    pip install scikit-learn==0.24.2
    pip install cython==0.29.23
    pip install h5py==2.10.0
    pip install ase==3.22.1
    pip install git+https://github.com/molmod/molmod.git
    pip install git+https://github.ugent.be/lvduyfhu/ThermoLIB.git
    
If a certain version is no longer available in the future, the user will have to manually test which later version is still compatible. In the near future, the required functions from molmod will be merged into thermolib so that molmod no longer becomes a dependency.

# Therms of use

ThermoLIB is developed by prof. Louis Vanduyfhuys at the Center for Molecular Modeling under supervision of prof. Veronique Van Speybroeck. Usage of ThermoLIB should be requested with prof. Van Speybroeck and/or prof. Vanduyfhuys.

Copyright (C) 2019 - 2023 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise stated.