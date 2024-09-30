.. _seclab_ig:

Installation Guide
##################

Here you can find information on the packages required for a successfull installation of ThermoLIB as well as details on how to download and install ThermoLIB on a Linux system. On Windows 11, ThermoLIB was tested for `Windows Subsystem for Linux - version 2 <https://learn.microsoft.com/en-us/windows/wsl/install>`_. For a pure Windows-based installation, we cannot garantue any assistance but we are of course interested to hear about your adventures.

Dependencies
************

Some software packages should be installed before ThermoLIB can be installed and/or used. It is recommended to use the software package management of your Linux distribution to install these dependencies. The following software must be installed:

*  Python3 <= 3.8 (including the development files): http://www.python.org/
*  Cython: http://cython.org/
*  Numpy: http://numpy.scipy.org/
*  Scipy: http://www.scipy.org/
*  matplotlib: http://matplotlib.sourceforge.net
*  scikit-learn: https://scikit-learn.org/
*  Atomic Simulation environment (ASE): https://wiki.fysik.dtu.dk/ase/
*  h5py: https://www.h5py.org/

Finally, ThermoLIB currently also requires a working build of the `MolMod <http://molmod.github.com/molmod/>`_ pacakge, which could normally be installed using pip as follows:

.. code-block:: bash

    pip install git+https://github.com/molmod/molmod.git

Unfortunately, MolMod is no longer maintained and updated, which might give rise to conflicts when installing molmod on top of recent versions of the packages given above. Therefore, we give below instructions on how to create a working conda environment with specific versions of these packages that should all be compatible with each other and with molmod and ThermoLIB:

.. code-block:: bash

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

You can directly install ThermoLIB (e.g. in the conda environment detailed above) using pip. **Researchers of Ghent University** can access it through `<https://github.ugent.be>`_ as follows:

.. code-block:: bash

    pip install git+https://github.ugent.be/lvduyfhu/ThermoLIB.git

**Researchers outside Ghent University** can access it through github.com as follows (TODO):

Developers download
===================

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

Once you downloaded the source code and installed all required packages, ThermoLIB can be installed. To install ThermoLIB irrespective of any (conda) environment, simply go to the directory in which you extracted the ThermoLIB source files (the directory containing the file ``setup.py``) and run the following command:

.. code-block:: bash

    python setup.py install

If you want to install ThermoLIB only in the current conda environment, you can use pip. For that just go into the ThermoLIB main directory (e.g. ``~/build/ThermoLIB```) and run:

.. code-block:: bash

    pip install .

Test installation
*****************

To test if you successfully installed ThermoLIB, fire up a python terminal and try to import the ThermoLIB package. In a Linux-based operating system, you can open a terminal and type (might be different for other operating systems):

.. code-block::

    python
    >>> import thermolib

If you do not get an error, you have (probably) successfully installed ThermoLIB!