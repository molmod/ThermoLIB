.. _seclab_ug_micromacro:

********************************************
Extracting micro- and macrostate free energy
********************************************

Once you have a free energy profile, you can extract thermodynamic properties from them such as the CV value and free energy of microstates and macrostates. In the context of ThermoLIB, a microstate is defined as a single point on the free energy profile, while a macrostate is defined as a region on the free energy profile. The most obvious and usefull examples of both are: a transition state given by the local maximum on a FEP as a microstate, and the reactant/product defined as the region left/right of the transition state as a macrostate. The free energy of a microstate is just simply the corresponding free energy value on the FEP, the free energy of a macrostate is computed through application of boltzmann partition functions. For example, the macrostate corresponding to a reactant state, i.e. all microstates left of the transition state local maximum, is defined as follows:

.. math::

    F_R &= -k_BT\ln\left[\int_{-\infty}^{q^{TST}} e^{-\beta F(q)}dq\right]

where :math:`q^{TST}` is the value of the CV in the transition state. In case of a macrostate, it might also be usefull to know the average (and variance) off the CV in the macrostate. All of these properties can be computed from a free energy profile with ThermoLIB using the classes in the :py:mod:`state <thermolib.thermodynamics.state>` module. In the code below, we define ``ts`` as the microstate corresponding to the local maximum in the cv range of [-0.5,0.5], we define ``r`` as a microstate corresponding to the local minimum in the cv range [-inf, 0.0] and define ``R`` as a macrostate corresponding to the collection of microstates with a cv smaller than that of the ``ts`` microstate:

.. code-block:: python

    from thermolib.thermodynamics.state import Minimum, Maximum, Integrate
    from molmod.constants import boltzmann

    ts = Maximum('TransState', cv_range=[-0.5,0.5])
    r = Minimum('MicroReac', cv_range=[-np.inf, 0])
    beta = 1.0/(boltzmann*300*kelvin)
    R = Integrate('MacroReac', [-np.inf, ts], beta)

Note that the definition of the macrostate also requires specifying the temperature (through the beta variable) as can be understood from the above mathematical expression. Finally, to compute the values of these definitions for a given free energy profile, we can just call the :py:meth:`compute <thermolib.thermodynamics.state.State.compute>` routine:

.. code-block:: python

    for state in [ts, R, r]:
        state.compute(fep)

Which produces following output:

.. code-block::

    MICROSTATE TransState:
    --------------
    index =  None
    F     = 24.761 +- 1.382 kjmol
    CV    = -0.024 +- 0.134 au

    MICROSTATE MicroReac:
    --------------
    index =  None
    F     = 0.000 +- 1.388 kjmol
    CV    = -1.246 +- 0.111 au

    MACROSTATE MacroReac:
    --------------
    F       = 2.340 +- 1.449 kjmol
    CV Mean = -1.237 +- 0.021 au
    CV StD  = 0.162 +- 0.012 au

As the present case involved a free energy profile with associated error bars, you see that (1) the error is propagated to all above observables and (2) the value of ``index`` in the microstates is None (becausethe minimum is not unique and hence no unique index can be provided). The command to extract value depends on wether or not the parsed free energy profile (``fep`` in the code block above) has error bars associated to it or not:

.. code-block:: python

    from molmod.units import kjmol
    for state in [ts, r, R]:
        #without error bars
        print('%10s at: F(%s) = %s kJ/mol' %(state.name, state.cv, state.F/kjmol))
        #with error bars
        print('%10s at: F(%s) = %s kJ/mol' %(state.name, state.cv_dist.print(), state.F_dist.print(unit='kjmol')))

Alternatively, you can print the microstate/macrostate information after it has been parse the fep using its print routine:

.. code-block:: python

    for state in [ts, r, R]:
        state.print()

In case your free energy profile features a simple structure of a single local maximum (the transition state) separating a clear macrostate valey at its left (the reactant state) and right side (the product state), one can do all the above automatically by using :class:`the SimpleFreeEnergyProfile <thermolib.thermodynamics.fep.SimpleFreeEnergyProfile>` child class and run its :py:meth:`process_states <thermolib.thermodynamics.fep.SimpleFreeEnergyProfile.process_states>` routine. In order to achieve this, one can either define the free energy profile from the start as a SimpleFreeEnergyProfile when converting a histogram to a fep:

.. code-block:: python
    
    fep = SimpleFreeEnergyProfile.from_histogram(hist, temp)
    
or a posteriori convert a BaseFreeEnergyProfile to a SimpleFreeEnergyProfile:

.. code-block:: python
    
    fep = SimpleFreeEnergyProfile.from_base(fep)

Once the :class:`the SimpleFreeEnergyProfile <thermolib.thermodynamics.fep.SimpleFreeEnergyProfile>` is available, we can auto detect the reactant, transition and product state as follows:

.. code-block:: python

    fep.process_states()

In this case, ThermoLIB will look for the transition state as the global maximum (i.e. over the entire CV range), the reactant state will be defined as the collection of all states to its left and the product state as the collection of all states to its right. If you want to narrow the cv range for where to look for the transition state (e.g. because the states at the edges of the cv space have free energies even higher then the central local maximum) and/or you want to limit the ranges over which to integrate for the reactant/product macrostates, you can do so by specifying the ``lims`` argument:

.. code-block:: python

    #look for the transition state maximum in the region -0.2,0.2.
    #integrate reactant state over all microstates left of the transition state
    #integrate product states over all microstates right to the transition state
    fep.process_states(lims=[-np.inf, -0.2, 0.2, np.inf])

    #look for the transition state maximum in the region -0.2,0.2.
    #integrate reactant state over microstates with a cv value between -1.0 and the transition state
    #integrate product states over microstates with a cv value between transition state and 0.8:
    fep.process_states(lims=[-1.0, -0.2, 0.2, 0.8])

You can then access the states in several ways:

.. code-block:: python

    #access the transition state specifically
    fep.ts.print()
    #access the reactant/product local minima (microstates) respectively:
    fep.r.print()
    fep.p.print()
    #access the reactant/product integrated macrostates respectively:
    fep.R.print()
    fep.P.print()
    #access the microstates/macrostates one by one
    for state in fep.microstates:
        state.print()
    for state in fep.macrostates:
        state.print()
    #print all info on all micro and macrostates:
    fep.print_states()

Finally, if you have processed the states of a :class:`the SimpleFreeEnergyProfile <thermolib.thermodynamics.fep.SimpleFreeEnergyProfile>` and you plot the fep, the plot will also contain to the micro and macrostate information:

.. code-block:: python

    fep = SimpleFreeEnergyProfile.from_histogram(hist, temp)
    fep.process_states(lims=[-np.inf, -0.2,0.2, np.inf])
    fep.plot()

This will produce a plot similar to the one given below:

.. image:: fep_with_states.png
    :width: 650

