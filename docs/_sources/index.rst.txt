.. ThermoLIB documentation master file, created by
   sphinx-quickstart on Fri Jun 25 22:32:09 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#####################
Welcome to ThermoLIB!
#####################

ThermoLIB is a library developed at the `Center for Molecular Modeling <https://molmod.ugent.be/>`_ (CMM) for the application of Statistical Physics and/or Thermodynamics to molecular simulations. The library consists of two main sub modules:

*  **Thermodynamics** - Module for reading, constructing, transforming and manipulating free energy profiles

*  **Kinetics** - Module for computing the rate constant of a process/reaction using transition state theory (TST)

.. admonition:: Citing ThermoLIB

   If you used ThermoLIB in your research, please refer to ThermoLIB as follows:
   
      M. Bocus, L. Vanduyfhuys, *J. Chem. Inf. Model.*, **2026**, `10.1021/acs.jcim.6c02199 <https://doi.org/10.1021/acs.jcim.6c02199>`_


This documentation contains instructions on how to install ThermoLIB, a user guide on all functionallities of the library as well as extensive tutorials illustrating the use of ThermoLIB in practice. Finally, there is also a reference guide included listing the call signatures of all available functions and classes. The documentation is structured as follows:
   
.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   ig.rst
   ug.rst
   tutorials.rst
   rg.rst

.. admonition:: Support

   For support and help you cannot extract from the guides and tuturials below, please contact us at `louis.vanduyfhuys@ugent.be <louis.vanduyfhuys@ugent.be>`_. It might be useful to check if the routine/class you are having troubles with has a verbose/verbosity keyword argument that you can use to give more logging output on what ThermoLIB is doing.