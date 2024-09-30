<img src='https://github.ugent.be/lvduyfhu/ThermoLIB/blob/master/logo_thermolib_light.png#gh-light-mode-only' width='200'>
<img src='https://github.ugent.be/lvduyfhu/ThermoLIB/blob/master/logo_thermolib_dark.png#gh-dark-mode-only' width='200'>

[![License](https://flat.badgen.net/static/LICENSE/GNU%20GPL%20(v3)/red)](https://github.ugent.be/lvduyfhu/ThermoLIB/blob/master/LICENSE)
![Python](https://flat.badgen.net/static/Python/3.8/blue)
[![Documentation](https://flat.badgen.net/static/Documentation/GitHUB%20Pages/green)](https://github.ugent.be/pages/lvduyfhu/ThermoLIB/)
# What is ThermoLIB?

ThermoLIB is a library developed at the [Center for Molecular Modeling (CMM)](https://molmod.ugent.be/) for the application of Statistical Physics and/or Thermodynamics to molecular simulations. The library consists of:

*  ***Thermodynamics*** - Module for constructing, manipulating and post-processing of 1D free energy profiles (FEP) and 2D free energy surfaces (FES). Functions include 

    - construction of FEPs using WHAM including error estimatoin
    - definition of (meta)stable micro- and macrostates
    - transformation of FEPs from one collective variable to another
    - (de)projection of a FES to a lower/higher dimensional FES.

*  ***Kinetics*** - Module for computing the rate constant of a process/reaction using transition state theory (TST). Functions include:

    - compute rate constant (which is theoretically indepedent of used CV)
    - compute associated phenomenological free energy barrier
    - error propagation from FEP to rate constant/phenomenological barrier

# Documentation

More information on how to use ThermoLIB, including tutorials, can be found in its manual at [https://github.ugent.be/pages/lvduyfhu/ThermoLIB/](https://github.ugent.be/pages/lvduyfhu/ThermoLIB/).

# How to install?

ThermoLIB has the following dependencies:

* [Cython](http://cython.org/)
* [numpy](http://numpy.org/)
* [scipy](http://www.scipy.org/)
* [molmod](https://molmod.github.io/molmod/)
* [scikit-learn](https://scikit-learn.org/)
* [matplotlib](http://matplotlib.sourceforge.net)
* [h5py](https://www.h5py.org/)
* [ase](https://wiki.fysik.dtu.dk/ase/)

As [molmod](https://molmod.github.io/molmod/) is currently no longer maintained, it might result in conflicting package versions with some of the new versions of the above packages. Therefore, below I show how to set up a conda environment with confirmed non-conflicting and working versions of all dependencies above. 

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

# Terms of use

ThermoLIB is developed by prof. Louis Vanduyfhuys at the Center for Molecular Modeling under supervision of prof. Veronique Van Speybroeck. Usage of ThermoLIB should be requested with prof. Van Speybroeck or prof. Vanduyfhuys.

Copyright (C) 2019 - 2024 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise stated.