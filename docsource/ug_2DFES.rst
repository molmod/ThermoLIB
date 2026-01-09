.. _seclab_ug_construct2D:

*************************************************
Constructing 2D histograms & free energy surfaces
*************************************************

To construct a 2D histogram, we will proceed very similarly as was the case for a 1D histogram, but instead use routines from the :py:class:`Histogram2D <thermolib.thermodynamics.histogram.Histogram2D>` class. We again make a distinction between constructing a histogram from a single simulation or from multiple (possibly biased) simulations using WHAM. The 2D histograms that are constructed in ThermoLIB use **the convention of xy indexing** (see `documentation on the numpy.meshgrid routine <https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html>`_).

.. _seclab_ug_construct2D_hist:

Constructing a 2D histogram ...
===============================

... from single simulation
--------------------------

To construct a 2D Histogram from a series of cv data coming from a single simulation, we can simply do:

.. code-block:: python

    from thermolib.thermodynamics.histogram import Histogram2D
    hist = Histogram2D.from_single_trajectory(cv_data, bins) #assumes cv_data and bins is in atomic units

Herein, cv_data represents a 2D numpy array containing the samples of the 2 CVs given in its two columns. If we still need to extract this array from simulations files, we can again do so using the readers defined in the trajectory module. For example, if we have a ``COLVAR.dat`` file available containing the samples of the first cv as its first column and those of the second cv as its second column, we can simply do:

.. code-block:: python

    from thermolib.thermodynamic.trajectory import ColVarReader
    reader = ColVarReader([0,1]) #assumes both CVs are in atomic units in the trajectory files
    cv_data = reader('COLVAR.dat')

.. _seclab_ug_2Dhistogram_wham:

... from multiple simulations using WHAM
----------------------------------------

In order to reconstruct a 2D Histogram using data coming from multiple (possibly biased) simulations, we again use the procedure composed of first reading the WHAM input file to properly identify which trajectories can be found where and which bias potentials are associated, followed by actually reading all trajectory files. To further illustrate, we consider two possible scenarios.

.. _seclab_ug_2Dhistogram_wham_scenario1:

SCENARIO 1 (2D bias => 2D FES)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this case, multiple simulations were done, each biased using a 2D biasing potential in terms of 2 CVs of interest (e.g. with umbrella sampling), and we want to construct a free energy surface in terms of these same two CVs. Let's assume we have indeed done such 2D umbrella simulation with harmonic bias potentials equidistantly separated on a CV1 grid from -1 `A` to +1 `A` with a 0.2 `A` spacing and a CV2 grid from 0 `deg` to 180 `deg` with a 20 `deg` spacing. Each harmonic bias potential has a force constant of 150 `kjmol/angstrom**2` along CV1 and a force constant of 50 `kjmol` along CV2. This results in 11*10=110 simulations, each with a trajectory of samples of CV1 and CV2 stored to ``COLVAR_WXX.dat`` files. As such, we have a directory with the following siumulation data:

.. code-block::

    current dir:
    |wham_input.txt
    |COLVAR_W0.dat
    |COLVAR_W1.dat
    |COLVAR_W2.dat
    |...
    |COLVAR_W109.dat
    |COLVAR_W110.dat

Each ``COLVAR_WXX.dat`` file has two columns, the first column representing samples of the first CV (for example in `angstrom`) and the second column representing samples of the second CV (for example in `degrees`). Furthermore, the WHAM input file ``wham_input.txt`` has the following content:

.. code-block::

    T=300K
    W0  -1.0 0.0 150 50
    W1  -0.8 0.0 150 50
    ...
    W11 -1.0 20.0 150 50
    W12 -0.8 20.0 150 50
    ...
    W108 0.8 180.0 150 50
    W109 1.0 180.0 150 50

Looking for example at the second line in this file, we recognize a simulation which has gotten label W0 (for which we know that its trajectory of CV samples stored in the file ``COLVAR_W0.dat``) and has a harmonic bias potential with a minimum at CV1 value of Q01=-1.0 (which we know has unit of `angstrom`), a minimum at CV2 value of Q02=0.0 (unit of `deg`), a force constant along CV1 of KAPPA1=150 (which we know has unit of `kjmol/angstrom**2`) and a force constant along CV2 of KAPPA2=50 (unit of `kjmol`). The units corresponding to these values of Q01, Q02, KAPPA1 and KAPPA2 can be defined in the :py:meth:`read_wham_input <thermolib.tools.read_wham_input>` routine. If the order of the ``wham_input.txt`` would have been different, you can specify this using the `order_2D` argument. As such, we are now able to write the piece of code required to construct the 2D Histogram:

.. code-block:: python

    from thermolib.thermodynamics.trajectory import ColVarReader
    from thermolib.tools import read_wham_input
    from thermolib.thermodynamics.histogram import Histogram2D
    
    #define a Trajectory reader able to read 2 CVs from a single COLVAR file
    cvs_reader = ColVarReader([0,1], units=['A', 'deg'])
    
    #read all trajectories and define corresponding bias potentials
    temp, biasses, trajectories = read_wham_input(
        'wham_input.txt', cvs_reader, 'COLVAR_%s.dat', bias_potential='Parabola2D', 
        q01_unit='angstrom', q02_unit='deg', 
        kappa1_unit='kjmol/angstrom**2', kappa2_unit='kjmol'
    )
    
    #define the bin grid on which the 2D histogram needs to be computed
    bins_cv1 = np.arange(-1.0, 1.05, 0.05)*angstrom
    bins_cv2 = np.arange(0.0, 185.0, 5.0)*deg
    
    #use WHAM to construct the 2D histogram
    hist = Histogram2D.from_wham([bins_cv1,bins_cv2], trajectories, biases, temp)


The third argument of the :py:meth:`read_wham_input <thermolib.tools.read_wham_input>` routine, i.e. ``COLVAR_%s.dat``, tells ThermoLIB that the trajectory corresponding the simulation that got the label W12 in ``wham_input.txt`` can be found in the file ``COLVAR_W12.dat``. The arguments ``q01_unit``, ``kappa1_unit``, ... allow to define the units of Q01, KAPPA1, ...

.. _seclab_ug_2Dhistogram_wham_scenario2:

SCENARIO 2 (1D bias => 2D FES)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this case, we actually start from a set of simulations that were biased along only 1 CV (e.g. a 1D umbrella sampling simulation), but we are interested in constructing a 2D free energy surface along this original collective variable (denoted CV) plus an additional a posteriori determined collective variable (denoted Q). Let's assume we have done 1D umbrella sampling simulation with harmonic bias potentials equidistantly separated on a CV grid from -1 `A` to +1 `A` with a 0.2 `A` spacing. Each harmonic bias potential has a force constant of 150 `kjmol/angstrom**2` along CV. This results in 11 simulations, each with a trajectory of CV samples stored to ``COLVAR_WXX.dat`` files. In order to extract the Q samples, it could be that these Q values are also stored in the ``COLVAR_WXX.dat`` files, in which case we can define the cvs_reader similar as in :ref:`SCENARIO1 <seclab_ug_2Dhistogram_wham_scenario1>`. Let us assume, however, that this is not the case, and we instead have to a posteriori compute the Q values from the structure coordinates stored in an XYZ file for each trajectory. In this case, we can use the CVComputer class as we will see below. To proceed, we assume we have a directory with the following siumulation data:

.. code-block::

    current dir:
    |wham_input.txt
    |COLVAR_W0.dat
    |COLVAR_W1.dat
    |COLVAR_W2.dat
    |...
    |COLVAR_W10.dat
    |COLVAR_W11.dat
    |traj_W0.xyz
    |traj_W1.xyz
    |traj_W2.xyz
    |...
    |traj_W10.xyz
    |traj_W11.xyz

Each ``COLVAR_WXX.dat`` file has a single column representing samples of the CV (for example in angstrom). Each ``traj_WXX.xyz`` file represents the XYZ coordinates of the entire structure (in `angstroms` as is by convention assumed for XYY files). Furthermore, the WHAM input file ``wham_input.txt`` has the following content:

.. code-block::
    
    T=300K
    W0  -1.0 150
    W1  -0.8 150
    ...
    W10 0.8 150
    W11 1.0 150

The above ``wham_input.txt`` file clearly reveals that the bias potantial is 1D (with Q0 ranging from -1.0 to +1.0 and all with the same force constant of 150). In order to construct a 2D histogram, we can now use the following code:

.. code-block:: python

    from thermolib.thermodynamics.trajectory import ColVarReader, CVComputer
    from thermolib.thermodynamics.cv import CoordinationNumber
    from thermolib.tools import read_wham_input
    from thermolib.thermodynamic.histogram import Histogram2D
    
    #define a Trajectory reader able to read the original collective varialbe 
    #CV from a COLVAR file
    cv_reader = ColVarReader([0], units=['A'])
    
    #define a Trajectory reader able to compute the additional collective variable
    #Q as a coordination number between atoms 3 and 5 from a XYZ trajectory file
    Q = CoordinationNumber([3,5])
    q_reader = CVComputer(CV)
    
    #read all trajectories and define corresponding bias potentials
    temp, biasses, trajectories = read_wham_input(
        'wham_input.txt', [cv_reader,q_reader], ['COLVAR_%s.dat','traj_%s.xyz'], 
        bias_potential='Parabola2D', q01_unit='angstrom', kappa1_unit='kjmol/angstrom**2'
    )
    
    #define the bin grid on which the 2D histogram needs to be computed
    bins_cv1 = np.arange(-1.0, 1.05, 0.05)*angstrom
    bins_cv2 = np.arange(0.0, 1.0, 0.02)*deg
    
    #use WHAM to construct the 2D histogram
    hist = Histogram2D.from_wham([bins_cv1,bins_cv2], trajectories, biases, temp)


We now see that the third argument of the :py:meth:`read_wham_input <thermolib.tools.read_wham_input>` routine becomes a list ``['COLVAR_%s.dat', 'traj_%s.xyz']`` to indicate that the path template for the trajectory files of the two different CVs is now different.



.. _seclab_ug_construct2D_fes:

Constructing a 2D free energy surface ...
=========================================

A 2D free energy surface can be constructed using the :class:`FreeEnergySurface2D <thermolib.thermodynamics.fep.FreeEnergySurface2D>` class. Similar as for 1D FEPs, there exist several ways to construct a FES as is illustrated in the subsections below.

... using the constructor
-------------------------

.. code-block:: python
    
    from thermolib.thermodynamics.fep import FreeEnergySurface2D
    fes = FreeEnergySurface2D(cv1s, cv2s, fs, temp)
    
Herein, the arguments ``cv1s`` and ``cv2s`` represent numpy arrays specifying the grid for the two CVs respectively (in atomic units). The argument ``fs`` represents a 2D numpy array with dimension matching those of the cv1s and cv2s array **using the convention of in xy indexing** (see `documentation on the numpy.meshgrid routine <https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html>`_), and represents the free energy corresponding to the values of cv1s and cv2s. containing respectively the collective variable and free energy on a grid, which should all be defined in atomic units. Furthermore, ``temp`` represents the temperature (in ``atomic units``, hence, ``Kelvin``) at which the free energy profile is evaluated.

... reading from a text file
----------------------------

.. code-block:: python

    from thermolib.thermodynamics.fep import FreeEnergySurface2D
    fes = FreeEnergySurface2D.from_txt(fn, temp)

Herein, the argument ``fn`` represents the name of a text file containing the values of the collective variable and the corresponding free energy on a grid. By default, the values of cv1s and cv2s should be defined in atomic units in respectively the first and second column, and the values of fs in `kjmol` in the third column, all separated by one or multiple whitespaces. More control on the columns from which to read the data, the delimiter used, units in which the cvs and/or f is defined and more can be found in :py:meth:`here <thermolib.thermodynamics.fep.FreeEnergySurface2D.from_txt>`. As was also the case in the constructor routine, ``temp`` represents the temperature (in ``atomic units``, hence, ``Kelvin``) at which the free energy profile is evaluated.

... from a histogram
--------------------

The free energy profile can also be constructed from a histogram that was computed from one or more (possibly biased) molecular simulations :ref:`as detailed before <seclab_ug_construct2D_hist>`. To convert this histogram into a free energy profile, we proceed as follows:

.. code-block:: python

    from thermolib.thermodynamics.fep import FreeEnergySurface2D
    temp = 300*kelvin
    fes = FreeEnergySurface2D.from_histogram(hist, temp) #hist is a Histogram2D instance

