.. _seclab_ig:

Installation Guide
##################

Here you can find information on the packages required for a successfull installation of ThermoLIB as well as details on how to download and install ThermoLIB on a Linux system. On Windows 11, ThermoLIB was tested for `Windows Subsystem for Linux - version 2 <https://learn.microsoft.com/en-us/windows/wsl/install>`_. For a pure Windows-based installation, we cannot garantue any assistance but we are of course interested to hear about your adventures.

Dependencies
************

Some software packages should be installed before ThermoLIB can be installed and/or used. It is recommended to use the software package management of your Linux distribution to install these dependencies. The following software must be installed:

*  Python3 (including the development files): http://www.python.org/
*  Cython: http://cython.org/
*  Numpy: http://numpy.org/
*  Scipy: http://www.scipy.org/
*  matplotlib: http://matplotlib.sourceforge.net
*  scikit-learn: https://scikit-learn.org/
*  Atomic Simulation environment (ASE): https://wiki.fysik.dtu.dk/ase/ (Version 3.23.0 or newer)
*  h5py: https://www.h5py.org/

In order to get the LaTeX support in the plots made by ThermoLIB, you also need the following packages installed

*  latex: https://www.latex-project.org/
*  cm-super: https://www.ctan.org/tex-archive/fonts/ps-type1/cm-super/

which can be installed on Ubuntu using the following command:

.. code-block:: bash

    sudo apt-get install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super

Download and install ThermoLIB
******************************

Direct install using pip
========================

You can install ThermoLIB directly in a conda environment using pip. If you want to create a new conda environment for ThermoLIB, you can do so as follows:

.. code-block:: bash

    conda create -n thermolib python
    conda activate thermolib

**Researchers of Ghent University** can then access and install **the latest development version** of ThermoLIB through `<https://github.ugent.be>`_ as follows:

.. code-block:: bash

    pip install git+https://github.ugent.be/lvduyfhu/ThermoLIB.git

(**TODO**) Everybody can access and install **the latest stable release** of ThermoLIB through github.com as follows:

.. code-block:: bash

    pip install git+https://github.com/XXX/XXX.git

Source code download and install
================================

If you want to have a copy of the source code for further implementation and advanced testing, you can clone the Git repository. This also allows you to upload your own changes in the form of a pull request. Git is free and open-source distributed revision control system to easily handle programming projects shared between several people. Further information about git (including downloads and tutorials) can be found `here <http://git-scm.com/>`_. To clone the ThermoLIB repository, go to your favorite directory for source code (e.g. ``~/build``) and execute the following commands.

**For researchers of Ghent University**:

.. code-block:: bash

    git clone https://github.ugent.be/lvduyfhu/ThermoLIB.git
    cd thermolib

**For researchers outside Ghent University**:

.. code-block:: bash

    git clone git://github.com/XXX/TODO.git thermolib
    cd thermolib

The source code can be updated with the latest patches with the following command:

.. code-block:: bash

    git pull

Once you downloaded the source code and installed all required packages, ThermoLIB can be installed. To install ThermoLIB in the active conda environment, you can use pip. For that just go into the ThermoLIB main directory (e.g. ``~/build/ThermoLIB```) and run:

.. code-block:: bash

    pip install .

Update ThermoLIB
****************

To update ThermoLIB to the latest version, either go to your conda environment and run the pip install command again:

.. code-block:: bash

    pip install git+https://github.ugent.be/lvduyfhu/ThermoLIB.git #for Ugent users only
    pip install git+https://github.com/XXX/XXX.git #for all users

or go to your local copy of the source code, pull the latest changes from the git repository and install:

.. code-block:: bash

    cd thermolib
    git pull
    pip install .

Test installation
*****************

To test if you successfully installed ThermoLIB, fire up a python terminal and try to import the ThermoLIB package. In a Linux-based operating system, you can open a terminal and type (might be different for other operating systems):

.. code-block::

    python
    >>> import thermolib

A more elaborate test is to run one or more of the tutorials. If you do not get an error, you have (probably) successfully installed ThermoLIB!