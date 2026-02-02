<img src='https://github.com/molmod/ThermoLIB/blob/master/logo_thermolib_light.png#gh-light-mode-only' width='200'>
<img src='https://github.com/molmod/ThermoLIB/blob/master/logo_thermolib_dark.png#gh-dark-mode-only' width='200'>

[![License](https://flat.badgen.net/static/LICENSE/GNU%20GPL%20(v3)/red)](https://github.com/molmod/ThermoLIB/blob/master/LICENSE)
![Python](https://flat.badgen.net/static/Python/3.14/blue)
[![Documentation](https://flat.badgen.net/static/Documentation/GitHUB%20Pages/green)](https://molmod.github.io/ThermoLIB/)
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

A full illustration on the use of ThermoLIB can be find in [the preprint on Arxiv](https://doi.org/10.48550/arXiv.2601.23071). More information on how to use ThermoLIB, including tutorials, can be found in its manual at [https://molmod.github.io/ThermoLIB/](https://molmod.github.io/ThermoLIB/).

# How to cite ThermoLIB

If you use ThermoLIB in your research, please cite it as follows:

    M. Bocus, L. Vanduyfhuys, 2026, 10.48550/arXiv.2601.23071

# How to install?

ThermoLIB has the following dependencies:

* [Cython](http://cython.org/)
* [numpy](http://numpy.org/)
* [scipy](http://www.scipy.org/)
* [scikit-learn](https://scikit-learn.org/)
* [matplotlib](http://matplotlib.sourceforge.net)
* [h5py](https://www.h5py.org/)
* [ase](https://wiki.fysik.dtu.dk/ase/) (Version 3.23.0 or newer)

To **install ThermoLIB** using pip either download the master branch as zip, go to your desired pip environment and run 

    pip install ThermoLIB-master.zip

or if you have properly configured your ssh to access GitHub.com using a key through the defined alias 'github', you can simply run:

    pip install git+ssh://github/molmod/ThermoLIB.git

# Terms of use

ThermoLIB is mainly developed by prof. Louis Vanduyfhuys at the Center for Molecular Modeling. Usage of ThermoLIB should be requested with prof. Vanduyfhuys.

Copyright (C) 2019 - 2026 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise stated.