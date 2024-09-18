![ThermoLIB](./doc/logo_thermolib_light.png#gh-light-mode-only)
![ThermoLIB](./doc/logo_thermolib_dark.png#gh-dark-mode-only)
[![License](https://flat.badgen.net/static/LICENSE/GNU%20GPL%20(v3)/red)](https://github.ugent.be/lvduyfhu/ThermoLIB/blob/master/LICENSE)
![Python](https://flat.badgen.net/static/Python/3.8/blue)
[![Documentation](https://flat.badgen.net/static/Documentation/molmod.github.io/green)](https://molmod.github.io/thermolib)
# What is ThermoLIB?

ThermoLIB is a library developed at the [Center for Molecular Modeling (CMM)](https://molmod.ugent.be/) for the application of Statistical Physics and/or Thermodynamics to molecular simulations. The library consists of several sub modules:

*  **Thermodynamics** - Module for reading, constructing, transforming and manipulating free energy profiles. Functions include construction of free energy profiles (FEPs) from histogram(s) (including error estimation), identification of (meta)stable macrostates and computation of their free energy, transformation of FEPs from one collective variable (CV) to another, (de)projection of an free energy surface (FES) to a lower/higher dimensional FES.

*  **Kinetics** - Module for computing the rate constant of a process/reaction using transition state theory (TST). Next to integrated reactant macrostate free energy and transition state microstate free energy (which are both computed using the routines in the thermodynamics module), the rate constant also requires a prefactor related to the time derivative of the collective variable in the transition state.

# How to install?

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

# Documentation

More information on the features of ThermoLIB and how to use them, including tutorials can be found at TODO.

# Therms of use

ThermoLIB is developed by prof. Louis Vanduyfhuys at the Center for Molecular Modeling under supervision of prof. Veronique Van Speybroeck. Usage of ThermoLIB should be requested with prof. Van Speybroeck or prof. Vanduyfhuys.

Copyright (C) 2019 - 2024 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise stated.